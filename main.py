import mlflow
import yaml
from pathlib import Path
import os


def workflow():
    try:
        exp = mlflow.get_experiment_by_name("test")
        exp_id = exp.experiment_id
    except AttributeError:
        exp_id = mlflow.create_experiment("test", artifact_location="mlruns")
    with mlflow.start_run(experiment_id=exp_id) as active_run:
        mlflow.set_tracking_uri("mlruns")
        print("Launching process...")
        print(os.getcwd())
        process_run = mlflow.run(".", "preprocess", parameters={"config_path": "./configs/preprocess.yaml"}, 
        env_manager="local", build_image=True, experiment_id=exp_id)
        process_run = mlflow.tracking.MlflowClient().get_run(process_run.run_id)
        processed_uri = Path(process_run.info.artifact_uri)
        print(processed_uri)


if __name__ == "__main__":
    #mlflow.set_experiment("test")
    workflow()