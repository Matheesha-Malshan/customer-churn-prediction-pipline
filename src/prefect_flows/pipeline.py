from prefect import task,flow 

import yaml
from src.data.clean_data import clean_data_flow
from src.features.engineer_features import smote_flow
from src.models.tune_optuna import run_optimization
from src.models.train import model_train
from src.models.evaluate import evaluate_model


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

@task
def clean_data_flow_run(data_path=config["data"]["filltered_data"]):
    return clean_data_flow(data_path)

@task
def engineer_features_flow(data_path=config["data"]["cleaned_data"]):
    return smote_flow(data_path)

@task
def run_optimization_flow():
    return run_optimization()

@task
def model_train_flow(data_path=config["data"]["balanced_data"]):
    return model_train(data_path)

@task
def model_evaluate_flow(model_path=config["model"]["model_path"], data_path=config["data"]["balanced_data"]):
    return evaluate_model(model_path, data_path)

@flow
def run_pipeline():
    print(clean_data_flow_run())
    #print(model_train_flow())
    #print(model_evaluate_flow())

run_pipeline()