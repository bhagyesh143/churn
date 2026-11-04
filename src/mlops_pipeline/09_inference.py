
import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient


def configure_mlflow(config):
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or config["mlflow"].get("tracking_uri")
    )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


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


def run_inference_demo():
    print("[INFO] Starting ML Pipeline Production Inference...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)

    model_name = config["registry"]["model_name"]

    # Load the existing Production model
    try:
        print("[INFO] Finding the Production model in MLflow...")

        client = MlflowClient()

        production_versions = client.get_latest_versions(
            model_name,
            stages=["Production"]
        )

        if not production_versions:
            print("[ERROR] No Production model found.")
            return False

        production_version = production_versions[0]

        model_id = production_version.source.rsplit("/", 1)[-1]

        local_model_path = os.path.abspath(
            os.path.join(
                "mlruns",
                "1",
                "models",
                model_id,
                "artifacts"
            )
        )

        print(f"[INFO] Production version: {production_version.version}")
        print(f"[INFO] Model ID: {model_id}")
        print(f"[INFO] Loading model from: {local_model_path}")

        if not os.path.isfile(
            os.path.join(local_model_path, "MLmodel")
        ):
            print("[ERROR] Production model artifact not found.")
            return False

        model = mlflow.sklearn.load_model(local_model_path)

        print("[SUCCESS] Production model loaded successfully!")

    except Exception as error:
        print(f"[ERROR] Could not load Production model: {error}")
        return False

    # Check required files
    preprocessor_path = config["paths"]["preprocessor"]
    raw_data_path = config["data"]["raw_path"]

    if not os.path.exists(preprocessor_path):
        print(f"[ERROR] Preprocessor not found: {preprocessor_path}")
        return False

    if not os.path.exists(raw_data_path):
        print(f"[ERROR] Raw dataset not found: {raw_data_path}")
        return False

    # Load preprocessor and dataset
    preprocessor = joblib.load(preprocessor_path)
    df = pd.read_csv(raw_data_path)

    # Select five sample customers
    sample = df.sample(n=5, random_state=7)

    X_sample = clean_dataframe(sample, config)
    X_sample = preprocessor.transform(X_sample)

    # Generate predictions
    predictions = model.predict(X_sample)
    probabilities = model.predict_proba(X_sample)[:, 1]

    result = sample.copy()
    result["Predicted_Churn"] = predictions
    result["Churn_Probability"] = np.round(probabilities, 4)

    print("\n--- ML Pipeline Production Inference Results ---")
    print(
        result[
            ["Predicted_Churn", "Churn_Probability"]
        ].to_string(index=False)
    )
    print("-----------------------------------------------\n")

    # Save results
    os.makedirs("outputs", exist_ok=True)

    output_path = config["paths"]["inference_output"]
    result.to_csv(output_path, index=False)

    print(f"[SUCCESS] Predictions saved to {output_path}")

    return True


if __name__ == "__main__":
    passed = run_inference_demo()
    sys.exit(0 if passed else 1)
