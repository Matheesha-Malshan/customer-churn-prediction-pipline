import lightgbm as lgb
import yaml
import optuna
import pandas as pd
from prefect import task, flow

from src.utils.helpers import split_data  


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

@task
def objective(trial):
    data_path = config["data"]["balanced_data"]
    X, y = pd.read_csv(data_path).drop(columns=["Churn"]), pd.read_csv(data_path)["Churn"]
    x_train,x_val,x_test,y_train,y_val,y_test= split_data(X, y)
    
    train_data = lgb.Dataset(x_train, label=y_train)
    valid_data = lgb.Dataset(x_val, label=y_val, reference=train_data)

    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    model = lgb.train(
        param,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=100,
        callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )
    return model

@task
def run_optimization():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    print("Best params:", study.best_params)
    print("Best score:", 1.0 - study.best_value)

if __name__ == "__main__":
    run_optimization()
