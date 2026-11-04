\# MLflow CI Setup



\## 1. Objective

To configure MLflow experiment tracking and model registration for the Churn Prediction MLOps project.



\## 2. Configuration

The project uses `config.yaml` to configure MLflow tracking and the experiment name.



\- Tracking URI: `sqlite:///mlflow.db`

\- Experiment name: `Telco\_Churn\_Prediction`

\- Registered model: `Telco\_Churn\_Production\_Model`



\## 3. Model Training and Tracking

The training pipeline trains a Random Forest classifier, records evaluation metrics, logs model artifacts, and registers the trained model through MLflow.



\## 4. Continuous Integration

GitHub Actions checks out the repository, sets up Python, installs dependencies, and executes the configured Lab 9 failure-handling tests.



\## 5. Important Notes

\- Keep database backup files out of Git.

\- Install project dependencies using `requirements.txt`.

\- Keep configuration values consistent with the pipeline code.

\- MLflow database files generated locally do not need to be committed.



\## 6. Conclusion

MLflow supports experiment tracking and model registration, while GitHub Actions automates project checks.

