import pandas as pd
import joblib
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,classification_report
from src.utils.helpers import split_data
import json

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def evaluate_model(model_path, data_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    _, _, X_test, _, _, y_test = split_data(X, y)

    model = joblib.load(model_path)
    y_pred = (model.predict(X_test) > 0.5).astype(int)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    class_report_dict = classification_report(y_test, y_pred, output_dict=True)

    with open("output\classification_report.json", "w") as f:
        json.dump(class_report_dict, f, indent=4)

    return results
    return results

if __name__=="__main__":
    evaluate_model(config["model"]["model_path"],config["data"]["balanced_data"])
   





