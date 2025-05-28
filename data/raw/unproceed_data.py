import yaml
from prefect import task, flow
import pandas as pd

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

@task
def read_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)

@flow
def data_flow() -> pd.DataFrame:
    return read_data(data_path=config["data"]["data_path"])

if __name__ == "__main__":
    df = data_flow()
    print(df)