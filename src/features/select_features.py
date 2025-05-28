import yaml
from prefect import task, flow
import pandas as pd


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

@task
def read_filltered_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)

@task
def correlation_matrix(filtered_data: pd.DataFrame) -> pd.DataFrame:
    corr_matrix = filtered_data.corr(method='spearman')
    return corr_matrix

@flow
def correlation_data_flow(data_path: str) -> pd.DataFrame:
    df = read_filltered_data(data_path)
    corr_matrix = correlation_matrix(df)
    return corr_matrix

if __name__ == "__main__":
    data_path = config["data"]["filltered_data"]
    corr_matrix = correlation_data_flow(data_path)
    print(corr_matrix)


