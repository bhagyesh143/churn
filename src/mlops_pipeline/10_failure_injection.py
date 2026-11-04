import os
import json
import shutil
import pandas as pd
import yaml


# ---------------------------------------------------------
# Load configuration
# ---------------------------------------------------------
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


RAW_DATA = config["data"]["raw_path"]

ARTIFACTS_DIR = "artifacts"
REPORTS_DIR = "reports"

FAILURE_REPORT = os.path.join(
    ARTIFACTS_DIR,
    "failure_simulation_report.json"
)

CORRUPTED_REPORT = os.path.join(
    ARTIFACTS_DIR,
    "corrupted_simulation_errors.csv"
)


os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ---------------------------------------------------------
# Failure Injection
# ---------------------------------------------------------
def inject_failures():

    print("\n========== FAILURE INJECTION ==========")

    failures = []
    errors = []

    # -----------------------------------------------------
    # Failure 1: Missing raw dataset
    # -----------------------------------------------------
    print("\n[TEST 1] Checking raw dataset availability...")

    if not os.path.exists(RAW_DATA):
        failures.append({
            "failure_type": "Missing Raw Dataset",
            "status": "DETECTED",
            "message": f"Raw dataset not found: {RAW_DATA}"
        })

        errors.append({
            "test": "Missing Raw Dataset",
            "expected": "Dataset should exist",
            "result": "FAIL"
        })

        print("[PASS] Missing dataset failure detected.")
    else:
        print("[PASS] Raw dataset exists.")

    # -----------------------------------------------------
    # Failure 2: Invalid target column
    # -----------------------------------------------------
    print("\n[TEST 2] Checking target column...")

    try:
        df = pd.read_csv(RAW_DATA)

        target_column = config["data"]["target_column"]

        if target_column not in df.columns:

            failures.append({
                "failure_type": "Missing Target Column",
                "status": "DETECTED",
                "message": f"Target column '{target_column}' not found."
            })

            errors.append({
                "test": "Missing Target Column",
                "expected": target_column,
                "result": "FAIL"
            })

            print("[PASS] Missing target-column failure detected.")

        else:
            print(
                f"[PASS] Target column '{target_column}' exists."
            )

    except Exception as e:

        failures.append({
            "failure_type": "Dataset Read Failure",
            "status": "DETECTED",
            "message": str(e)
        })

        errors.append({
            "test": "Dataset Read",
            "expected": "CSV should load successfully",
            "result": "FAIL"
        })

        print("[PASS] Dataset read failure detected.")

    # -----------------------------------------------------
    # Failure 3: Missing values
    # -----------------------------------------------------
    print("\n[TEST 3] Checking missing-value failure...")

    try:
        df = pd.read_csv(RAW_DATA)

        missing_count = int(df.isnull().sum().sum())

        if missing_count > 0:

            failures.append({
                "failure_type": "Missing Values",
                "status": "DETECTED",
                "message": f"{missing_count} missing values detected."
            })

            errors.append({
                "test": "Missing Values",
                "expected": 0,
                "actual": missing_count,
                "result": "FAIL"
            })

            print(
                f"[PASS] Missing-value failure detected: "
                f"{missing_count}"
            )

        else:
            print("[PASS] No missing values found.")

    except Exception as e:

        print(f"[INFO] Missing-value test skipped: {e}")

    # -----------------------------------------------------
    # Failure 4: Invalid target values
    # -----------------------------------------------------
    print("\n[TEST 4] Checking target-value failure...")

    try:
        df = pd.read_csv(RAW_DATA)

        target_column = config["data"]["target_column"]

        if target_column in df.columns:

            valid_values = {"Yes", "No", 1, 0, "1", "0"}

            invalid_values = [
                value
                for value in df[target_column].dropna().unique()
                if value not in valid_values
            ]

            if invalid_values:

                failures.append({
                    "failure_type": "Invalid Target Values",
                    "status": "DETECTED",
                    "message": (
                        f"Invalid target values found: "
                        f"{invalid_values}"
                    )
                })

                errors.append({
                    "test": "Invalid Target Values",
                    "expected": ["Yes", "No"],
                    "actual": invalid_values,
                    "result": "FAIL"
                })

                print(
                    "[PASS] Invalid target-value failure detected."
                )

            else:
                print("[PASS] Target values are valid.")

    except Exception as e:

        print(f"[INFO] Target-value test skipped: {e}")

    # -----------------------------------------------------
    # Failure 5: Duplicate records
    # -----------------------------------------------------
    print("\n[TEST 5] Checking duplicate-record failure...")

    try:
        df = pd.read_csv(RAW_DATA)

        duplicate_count = int(df.duplicated().sum())

        if duplicate_count > 0:

            failures.append({
                "failure_type": "Duplicate Records",
                "status": "DETECTED",
                "message": (
                    f"{duplicate_count} duplicate records found."
                )
            })

            errors.append({
                "test": "Duplicate Records",
                "expected": 0,
                "actual": duplicate_count,
                "result": "FAIL"
            })

            print(
                f"[PASS] Duplicate-record failure detected: "
                f"{duplicate_count}"
            )

        else:
            print("[PASS] No duplicate records found.")

    except Exception as e:

        print(f"[INFO] Duplicate test skipped: {e}")

    # -----------------------------------------------------
    # Save corrupted simulation errors
    # -----------------------------------------------------
    if errors:

        error_df = pd.DataFrame(errors)

        error_df.to_csv(
            CORRUPTED_REPORT,
            index=False
        )

    else:

        pd.DataFrame(
            columns=["test", "expected", "actual", "result"]
        ).to_csv(
            CORRUPTED_REPORT,
            index=False
        )

    # -----------------------------------------------------
    # Create failure simulation report
    # -----------------------------------------------------
    report = {
        "pipeline": "Churn Prediction MLOps",
        "failure_injection": True,
        "total_tests": 5,
        "failures_detected": len(failures),
        "status": (
            "FAILURES_DETECTED"
            if failures
            else "NO_FAILURES_DETECTED"
        ),
        "failures": failures
    }

    with open(
        FAILURE_REPORT,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=4
        )

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------
    print("\n========================================")
    print("FAILURE INJECTION COMPLETED")
    print("========================================")

    print(f"Failures detected : {len(failures)}")
    print(f"Report saved      : {FAILURE_REPORT}")
    print(f"Error CSV saved   : {CORRUPTED_REPORT}")

    return report


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    inject_failures()