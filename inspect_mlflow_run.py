
import mlflow
from mlflow import MlflowClient

client = MlflowClient()

run_id = "5d44f167302344bba50acd0f2e1475d9"

run = client.get_run(run_id)

print("RUN ID:", run.info.run_id)
print("STATUS:", run.info.status)
print("ARTIFACT URI:", run.info.artifact_uri)

print("\nARTIFACTS:")

artifacts = client.list_artifacts(run_id)

for artifact in artifacts:
    print(f"{artifact.path} | is_dir={artifact.is_dir}")

    if artifact.is_dir:
        children = client.list_artifacts(run_id, artifact.path)

        for child in children:
            print(f"  └── {child.path} | is_dir={child.is_dir}")
