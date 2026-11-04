# LAB 9: Failure Handling and Testing

## 1. Objective
To test the Churn Prediction MLOps pipeline by identifying data-related failures and recording them in reports.

## 2. Failure Scenarios
- Missing raw dataset
- Missing target column
- Missing values in the dataset
- Invalid target values
- Duplicate records

## 3. Failure Detection
The failure injection module checks the dataset for the above conditions and records detected issues.

## 4. Reports Generated
- `artifacts/failure_simulation_report.json`
- `artifacts/corrupted_simulation_errors.csv`

## 5. Automation
The GitHub Actions workflow `lab9_failure_tests.yml` runs the failure injection script and verifies that the reports are generated.

## 6. Conclusion
The Lab 9 failure-handling module helps identify data-quality problems and generates reports to support reliable execution of the Churn Prediction MLOps pipeline.