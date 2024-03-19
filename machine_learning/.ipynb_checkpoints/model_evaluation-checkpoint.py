from os import environ
from numpy import load
import pandas as pd
import numpy as np
import tempfile

import onnx
import torch
from torch import nn
import copy
import torch.optim as optim
import tqdm
import datetime
import mlflow
import re

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

def model_evaluate_metrics(y, y_pred, metrics):
    metrics_dict = {}
    for metric in metrics:
        metrics_dict[metric.__name__] = metric(y, y_pred)

    return metrics_dict


def model_evaluation(mlflow_tracking_uri='http://mlflow-v4-redhat-ods-applications.apps.io.mos-paas.de.eviden.com'):
    with open("model_uri.txt", "r") as file:
        new_model_uri = file.read()
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    X_eval = pd.DataFrame(load("X_eval.npy"))
    y_eval = pd.DataFrame(load("y_eval.npy"))
    
    X_eval_tensor = torch.tensor(X_eval.values, dtype=torch.float32).to(device)
    y_eval_tensor = torch.tensor(y_eval.values, dtype=torch.float32).to(device).reshape(-1,1)
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    client = mlflow.MlflowClient()
    all_models = client.search_registered_models()
    all_model_names_aliases = {}
    for model in all_models:
        all_model_names_aliases[model.latest_versions[0].name] = model.aliases
    
    eval_model_uris = {}
    for model_name in all_model_names_aliases.keys():
        if all_model_names_aliases[model_name] != {}:
            model_versions = client.search_model_versions(f"name='{model_name}'")
            for model in model_versions:
                for key in all_model_names_aliases[model_name].keys():       
                    if model.version == all_model_names_aliases[model_name][key]:
                        eval_model_uris[f"{model.source}/{model.version}"] = all_model_names_aliases[model_name]
    eval_metric = {}
    print(eval_model_uris)
    
    uri_pattern = r'(.*)/\d+$'

    for eval_model_uri_long in eval_model_uris.keys():
        
        eval_model_uri = re.sub(uri_pattern, r'\1', eval_model_uri_long)
        
        model_name_long = eval_model_uri.split('/')[-1]
        model_name = '-'.join(model_name_long.split('-')[:-2])

        run_id = eval_model_uri.split('/')[-3]
        with mlflow.start_run(run_id=run_id) as run:
            model = mlflow.pytorch.load_model(model_uri=eval_model_uri)
            
            model.eval()
            y_pred = model(X_eval_tensor)
            y_pred_rounded = torch.round(y_pred)
            
            y_pred_numpy = y_pred_rounded.detach().numpy()
            y_eval_numpy = y_eval_tensor.detach().numpy()

            # evaluate and log metrics
            metrics = model_evaluate_metrics(
                y_eval_numpy,
                y_pred_numpy,
                metrics=[accuracy_score, precision_score, recall_score, f1_score],
            )

            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            eval_metric[eval_model_uri_long] = metrics['accuracy_score']

            mlflow.end_run()

    for uri_long in eval_model_uris.keys():
        uri = re.sub(uri_pattern, r'\1', uri_long)
        client.delete_registered_model_alias(uri.split('/')[-1], next(iter(eval_model_uris[uri_long])))

    champion_model_uri_long = max(eval_metric, key=eval_metric.get)
    print(eval_metric)
    print(champion_model_uri_long)
    version_pattern = r'/(\d+)$'
    match = re.search(version_pattern, champion_model_uri_long)
    champion_version = match.group(1)
    champion_model_uri = re.sub(uri_pattern, r'\1', champion_model_uri_long)

    # client = mlflow.MlflowClient()
    # model_version = client.get_latest_versions(name=champion_model_uri.split('/')[-1])[0].version
    # print(model_version)
    client.set_registered_model_alias(champion_model_uri.split('/')[-1], "champion", champion_version)
    print(new_model_uri)
    print(champion_model_uri)
    with open("output.txt", "w") as file:
        if new_model_uri == champion_model_uri:
            file.write("new_champion")
        else:
            file.write("keep_champion")


if __name__ == '__main__':
    model_evaluation(mlflow_tracking_uri='http://mlflow-v4-redhat-ods-applications.apps.io.mos-paas.de.eviden.com')