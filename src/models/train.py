import lightgbm as lgb
import pandas as pd
from prefect import task, flow
import yaml
import joblib
import os

from src.utils.helpers import split_data  

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

@task
def load_data(data_path: str):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Churn"])  
    y = data["Churn"]
    return X, y

@task
def prepare_datasets(X, y):
    x_train,x_val,x_test,y_train,y_val,y_test=split_data(X, y)

    train_data = lgb.Dataset(x_train, label=y_train)
    valid_data = lgb.Dataset(x_val, label=y_val,reference=train_data)
    return train_data, valid_data

@task
def model_train(data_path: str):
    X, y = load_data(data_path)
    train_data, valid_data = prepare_datasets(X, y)

    model = lgb.train(
    params=config['params'],
    train_set=train_data,
    valid_sets=[valid_data],
    num_boost_round=50,
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )
    os.makedirs(os.path.dirname(config["model"]["model_path"]), exist_ok=True)
    joblib.dump(model,config["model"]["model_path"])

    return model

if __name__ == "__main__":
    model = model_train(config["data"]["balanced_data"])
    print("Training completed.")


