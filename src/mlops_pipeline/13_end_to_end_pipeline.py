
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCRIPTS = [
    "01_load_data.py",
    "02_validate_data.py",
    "03_preprocessing.py",
    "04_validate_outputs.py",
    "05_train_registry.py",
    "06_model_evaluation.py",
    "07_quality_gate.py",
    "08_automate_lifecycle.py",
    "09_inference.py",
    "10_failure_injection.py",
    "11_prepare_deployment.py",
]

def main():
    print("=" * 65)
    print("CHURN PREDICTION - END-TO-END MLOPS PIPELINE")
    print("=" * 65)

    for script in SCRIPTS:
        script_path = ROOT / "src" / "mlops_pipeline" / script

        if not script_path.exists():
            print(f"\n[ERROR] Script not found: {script_path}")
            sys.exit(1)

        print(f"\n{'=' * 65}")
        print(f"Running: {script}")
        print("=" * 65)

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=ROOT
        )

        if result.returncode != 0:
            print(f"\n[FAILED] Pipeline stopped at {script}")
            sys.exit(result.returncode)

        print(f"[SUCCESS] {script}")

    print("\n" + "=" * 65)
    print("END-TO-END PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 65)

if __name__ == "__main__":
    main()
