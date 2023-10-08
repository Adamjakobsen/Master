import sys
import optuna
import logging
import argparse
import deepxde as dde
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import torch
import scipy.io
import os
import json
import GPUtil
from datetime import datetime
from heartpinn import CustomPointCloud, PINN
from utils import *

device = 'cuda:' + sys.argv[1]
def objective(trial):
    torch.cuda.set_device(device)
    dde.config.default_device=device
    dde.config.set_default_float("float32")
    config=load_config()
    initializer=config["TRAINING"]["initializer"]
    activation=config["TRAINING"]["activation"]
    num_domain=trial.suggest_int("num_domain", 1000, 100000)
    num_boundary=trial.suggest_int("num_boundary", 100, 10000)
    resampling_period = trial.suggest_int("resampling_period", 100, 50000)
    #n_neurons=trial.suggest_int("n_neurons", 32, 256)
    #n_layers=trial.suggest_int("n_layers", 1, 6)
    n_neurons=config["TRAINING"]["n_neurons"]
    n_layers=config["TRAINING"]["n_layers"]
    lr=trial.suggest_float("lr", 1e-6, 1e-1, log=True)
    n_epochs=50000
    #Define model
    pinn=PINN()
    # Load the data and scale
    X, X_boundary, v = pinn.get_data()
    scaler = MinMaxScaler()
    X= scaler.fit_transform(X)
    # Split the data into training and testing sets (80/20)
    X_train, X_test, v_train, v_test = train_test_split(X, v, test_size=0.9, random_state=42)
    data_list = [X_train, X_test, v_train, v_test]
    # Initialize the model
    geomtime = pinn.geotime()
    observe_v = dde.PointSetBC(X_train, v_train, component=0)
    ic = pinn.IC(X_train, v_train)
    bc = pinn.BC(geomtime)
    input_data = [bc, ic, observe_v]
    PDE= pinn.pde2d_vm
    data = dde.data.TimePDE(geomtime,
                            PDE,
                            input_data,
                            num_domain=num_domain,
                            num_boundary=num_boundary)
    
    net = dde.maps.FNN([3] + n_layers * [n_neurons] +
                       [2], activation, initializer)
    

    model = dde.Model(data, net)
    resampler = dde.callbacks.PDEPointResampler(period=resampling_period,pde_points=True,bc_points=True)
    weights= config["TRAINING"]["weights2"]
    model.compile("adam", lr=lr, loss_weights=weights)
    losshistory,train_state = model.train(epochs=n_epochs,display_every=1000)
    v_pred_test = model.predict(X_test)[:,0]
    # If they are not 1D, reshape or index them
    if len(v_test.shape) > 1:
        v_test = v_test.reshape(-1)
    if len(v_pred_test.shape) > 1:
        v_pred_test = v_pred_test.reshape(-1)
    RMSE = np.sqrt(np.mean((v_test - v_pred_test)**2))
    return RMSE

if __name__ == "__main__":
    study_name="heartpinn"
    storage="sqlite:///heartpinn.db"
    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials=10)




