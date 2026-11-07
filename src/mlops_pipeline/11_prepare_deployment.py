
import json
import os
import shutil
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient


def load_config():
    project_root = Path(__file__).resolve().parents[2]

    with open(project_root / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return project_root, config


def clean_dataframe(df, config):
    df = df.copy()

    target_column = config["data"]["target_column"]
    id_column = config["data"]["id_column"]

    if id_column in df.columns:
        df = df.drop(columns=[id_column])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    if target_column in df.columns:
        df = df.drop(columns=[target_column])

    return df


def get_production_model(client, model_name):
    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    production_versions = [
        version for version in versions
        if version.current_stage == "Production"
    ]

    if not production_versions:
        raise RuntimeError(
            f"No Production version found for '{model_name}'."
        )

    if len(production_versions) > 1:
        raise RuntimeError(
            f"Multiple Production versions found for '{model_name}'."
        )

    return production_versions[0]


def prepare_deployment():
    project_root, config = load_config()

    # Configure MLflow using the project's database
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or config["mlflow"]["tracking_uri"]
    )

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    model_name = config["registry"]["model_name"]

    print("[INFO] MLflow tracking URI:", mlflow.get_tracking_uri())

    # Find the existing Production model
    model_version = get_production_model(client, model_name)

    version = str(model_version.version)
    run_id = model_version.run_id
    model_source = model_version.source

    model_id = model_source.rsplit("/", 1)[-1]

    print("\n[INFO] Production model found")
    print("[INFO] Model name :", model_name)
    print("[INFO] Version    :", version)
    print("[INFO] Run ID     :", run_id)
    print("[INFO] Model ID   :", model_id)

    # Locate the model on the local Mac filesystem
    local_model_path = (
        project_root
        / "mlruns"
        / "1"
        / "models"
        / model_id
        / "artifacts"
    )

    # Use the preprocessor saved with the Production run
    local_preprocessor_path = (
        project_root
        / "mlruns"
        / "1"
        / run_id
        / "artifacts"
        / "preprocessing_pipeline_v2"
        / "retrained_preprocessor_v2.pkl"
    )

    if not local_preprocessor_path.is_file():
        local_preprocessor_path = (
            project_root
            / "mlruns"
            / "1"
            / run_id
            / "artifacts"
            / "preprocessing_pipeline"
            / "preprocessor.pkl"
        )

    if not local_preprocessor_path.is_file():
        local_preprocessor_path = project_root / config["paths"]["preprocessor"]

    if not (local_model_path / "MLmodel").is_file():
        raise FileNotFoundError(
            f"Production model artifact not found: {local_model_path}"
        )

    if not local_preprocessor_path.is_file():
        raise FileNotFoundError(
            f"Production preprocessor not found: {local_preprocessor_path}"
        )

    deployment_dir = project_root / "deployment"

    temp_dir = Path(tempfile.mkdtemp(
        prefix="deployment_staging_",
        dir=str(project_root),
    ))

    backup_dir = None

    try:
        # Load the existing Production model locally
        print("\n[INFO] Loading Production model...")

        model = mlflow.sklearn.load_model(
            str(local_model_path)
        )

        print("[SUCCESS] Production model loaded.")

        # Load the matching Production preprocessor
        print("[INFO] Loading Production preprocessor...")

        preprocessor = joblib.load(
            local_preprocessor_path
        )

        print("[SUCCESS] Production preprocessor loaded.")

        # Verify that preprocessing output matches model input
        raw_data_path = project_root / config["data"]["raw_path"]
        df = pd.read_csv(raw_data_path)

        sample = df.head(5)
        X_sample = clean_dataframe(sample, config)

        transformed = preprocessor.transform(X_sample)

        expected_features = model.n_features_in_

        if transformed.shape[1] != expected_features:
            raise RuntimeError(
                f"Feature mismatch: preprocessor produces "
                f"{transformed.shape[1]} features, "
                f"but model expects {expected_features}."
            )

        print(
            f"[SUCCESS] Feature compatibility verified: "
            f"{transformed.shape[1]} features."
        )

        # Verify predictions
        predictions = model.predict(transformed)

        if len(predictions) != len(sample):
            raise RuntimeError("Prediction verification failed.")

        print("[SUCCESS] Model prediction verification passed.")

        # Save deployment artifacts into temporary directory
        model_path = temp_dir / "model.pkl"
        preprocessor_path = temp_dir / "preprocessor.pkl"

        joblib.dump(model, model_path)
        joblib.dump(preprocessor, preprocessor_path)

        print("[SUCCESS] Model saved:", model_path)
        print("[SUCCESS] Preprocessor saved:", preprocessor_path)

        # Create metadata
        metadata = {
            "model_name": model_name,
            "model_version": version,
            "model_stage": model_version.current_stage,
            "run_id": run_id,
            "model_source": model_source,
            "model_id": model_id,
            "preprocessor_artifact": (
                "preprocessing_pipeline_v2/retrained_preprocessor_v2.pkl"
                if "preprocessing_pipeline_v2" in str(local_preprocessor_path)
                else "preprocessing_pipeline/preprocessor.pkl"
            ),
            "tracking_uri": tracking_uri,
            "source": "MLflow local artifacts",
            "expected_features": int(expected_features),
        }

        metadata_path = temp_dir / "model_metadata.json"

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print("[SUCCESS] Model metadata saved.")

        # Verify saved artifacts can be loaded again
        print("\n[INFO] Verifying deployment files...")

        verified_model = joblib.load(model_path)
        verified_preprocessor = joblib.load(preprocessor_path)

        verified_output = verified_preprocessor.transform(X_sample)
        verified_predictions = verified_model.predict(verified_output)

        if len(verified_predictions) != 5:
            raise RuntimeError(
                "Final deployment verification failed."
            )

        print("[SUCCESS] All deployment artifacts verified.")

        # Preserve any existing deployment
        if deployment_dir.exists():
            index = 1

            while True:
                candidate = project_root / f"deployment_backup_{index}"

                if not candidate.exists():
                    backup_dir = candidate
                    break

                index += 1

            print("[INFO] Preserving previous deployment:", backup_dir)
            deployment_dir.rename(backup_dir)

        # Move the completed bundle into deployment/
        try:
            temp_dir.rename(deployment_dir)

        except Exception:
            if backup_dir is not None and backup_dir.exists():
                if not deployment_dir.exists():
                    backup_dir.rename(deployment_dir)

            raise

        print("\n========================================")
        print("[SUCCESS] DEPLOYMENT PREPARATION COMPLETED")
        print("========================================")
        print("deployment/")
        print("├── model.pkl")
        print("├── preprocessor.pkl")
        print("└── model_metadata.json")

        if backup_dir is not None:
            print("[INFO] Previous deployment backup:", backup_dir)

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    prepare_deployment()
