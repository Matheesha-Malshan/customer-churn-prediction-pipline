import yaml
from prefect import task, flow
import pandas as pd
import numpy as np


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

@task
def read_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)

@flow
def procced_data_flow() -> pd.DataFrame:
    return read_data(data_path=config["data"]["balanced_data"])

if __name__ == "__main__":
    df = procced_data_flow()
    print(df)


