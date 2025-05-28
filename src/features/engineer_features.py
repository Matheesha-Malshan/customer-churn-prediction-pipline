import yaml
from prefect import task, flow
import pandas as pd
import os

from imblearn.over_sampling import SMOTE


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

@task
def read_cleaned_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path)

@task
def smote(X_cleaned: pd.DataFrame) -> pd.DataFrame:
    y=X_cleaned["Churn"]
    x=X_cleaned.drop("Churn",axis=1)
    X_resampled,y_resampled =SMOTE().fit_resample(x, y)
    balanced_data=pd.concat([ X_resampled,y_resampled],axis=1)
    return balanced_data

@flow
def smote_flow(data_path: str) -> pd.DataFrame:
    df = read_cleaned_data(data_path)
    balanced_df = smote(df)
    
    save_dir = '../'
    save_path = os.path.join(save_dir,'imbalnced_handle_data.csv')
    os.makedirs(save_dir, exist_ok=True)
    balanced_df.to_csv(save_path, index=False)

    return balanced_df

if __name__ == "__main__":
    data_path = config["data"]["cleaned_data"]
    cleaned = smote_flow(data_path)
    print(cleaned)

