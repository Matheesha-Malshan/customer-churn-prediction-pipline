import yaml
from prefect import task, flow
import pandas as pd
from sklearn.ensemble import IsolationForest
import os


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

@task
def read_filtered_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)

@task
def remove_outliers(filtered_data: pd.DataFrame) -> pd.DataFrame:
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_pred = iso_forest.fit_predict(filtered_data)
    mask = iso_pred == 1
    cleaned_data = filtered_data[mask]
    return cleaned_data

@task
def clean_data_flow(data_path: str) -> pd.DataFrame:
    df = read_filtered_data(data_path)
    cleaned_df = remove_outliers(df)

    save_dir = '../'
    save_path = os.path.join(save_dir,'cleaned_data.csv')
    os.makedirs(save_dir, exist_ok=True)
    cleaned_df.to_csv(save_path, index=False)

    return cleaned_df

if __name__ == "__main__":
    data_path = config["data"]["filltered_data"]
    cleaned = clean_data_flow(data_path)
    print(cleaned)



