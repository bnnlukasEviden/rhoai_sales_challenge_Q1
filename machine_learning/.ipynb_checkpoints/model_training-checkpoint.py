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

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn


class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(121, 121)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(121, 121)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(121, 121)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(121, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x


def model_train(model, X_train, y_train, X_val, y_val):
    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epoch_count = int(environ.get('epoch_count', '20'))
    batch_size = int(environ.get('batch_size', '200'))

    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None

    for epoch in range(epoch_count):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )

        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)

        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc, model


def train_x_val(mlflow_tracking_uri):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'training the model on {device}')

    X = pd.DataFrame(load("X_train.npy"))
    y = pd.DataFrame(load("y_train.npy"))

    split_number = int(environ.get('split_number', '2'))
    sss = StratifiedKFold(n_splits=split_number, random_state=None, shuffle=False)

    best_acc_deep = - np.inf   # init to negative infinity
    best_model_deep = None

    for train_index, test_index in sss.split(X, y):
        Xtrain = X.iloc[train_index]
        Xtest = X.iloc[test_index]
        ytrain = y.iloc[train_index]
        ytest = y.iloc[test_index]

        X_train_tensor = torch.tensor(Xtrain.values, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(ytrain.values, dtype=torch.float32).to(device).reshape(-1,1)
        X_test_tensor = torch.tensor(Xtest.values, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(ytest.values, dtype=torch.float32).to(device).reshape(-1, 1)

        model_d = Deep()
        acc_deep, model_deep = model_train(model_d, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
        if acc_deep > best_acc_deep:
            best_acc_deep = acc_deep
            best_model_deep = model_deep
        print("Accuracy (deep): %.2f" % acc_deep)

    now = datetime.datetime.now()
    date = now.strftime("%Y%m%d%H%M")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(f"rain_prediction_model_experiment")
    with mlflow.start_run(run_name=f"rain_prediction_model-{date}") as run:
        print('artifact uri:', mlflow.get_artifact_uri())
        mlflow.pytorch.log_model(model_d, "rain_prediction_model", registered_model_name="rain_prediction_model")
        
        client = mlflow.MlflowClient()
        model_version = client.get_latest_versions(name="rain_prediction_model")[0].version
        model_uri = client.get_latest_versions(name="rain_prediction_model")[0].source
        client.set_registered_model_alias("rain_prediction_model", "challenger", model_version)
        
        mlflow.end_run()
        
    torch.onnx.export(best_model_deep, torch.randn(363, 121, requires_grad=True).to(device), "rain_prediction_model.onnx")
    
    with open("model_uri.txt", "w") as file:
        file.write(model_uri)


if __name__ == '__main__':
    train_x_val(mlflow_tracking_uri='http://mlflow-v4-redhat-ods-applications.apps.io.mos-paas.de.eviden.com')