# main.py

import gc
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
from heartpinn import CustomPointCloud, PINN



def parse_arguments():
    parser = argparse.ArgumentParser(description='PINN model for 2D PDEs.')
    parser.add_argument('mode', type=str, choices=['train', 'predict', 'continue'],
                        help='Mode to run the model: train, predict, or continue.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint for prediction or continuation.')
    return parser.parse_args()

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def get_gpu_with_most_memory():
    # Get the list of all available GPU devices
    devices = GPUtil.getGPUs()
    
    # Check if any GPUs are available
    if len(devices) == 0:
        print("No available GPUs.")
        return None
    
    # Sort the devices by available memory
    devices = sorted(devices, key=lambda x: x.memoryFree, reverse=True)
    
    # Get the device with the most free memory
    best_device = devices[0]
    
    print(f"Device ID: {best_device.id}, Free Memory: {best_device.memoryFree} MB")
    
    return best_device.id

def define_model(config):
    # Load the configuration
    config = load_config()
    n_neurons = config["n_neurons"]
    n_layers = config["n_layers"]
    activation = config["activation"]
    initializer = config["initializer"]
    

    # Initialize the model
    pinn = PINN()



def main(args):
    config = load_config()
    # Initialize CUDA and other configurations
    torch.cuda.empty_cache()
    dde.config.set_random_seed(42)
    dde.config.set_default_float("float64")
    best_device_id = get_gpu_with_most_memory()
    #Set torch device
    device = torch.device(f"cuda:{best_device_id}")
    torch.cuda.set_device(device)
    dde.config.default_device=device
    torch.cuda.set_per_process_memory_fraction(0.9)

    

    

    # Load the configuration
    n_neurons = config["n_neurons"]
    n_layers = config["n_layers"]
    activation = config["activation"]
    initializer = config["initializer"]

    #n_training_phases = config["n_training_phases"]

    # Initialize the model
    pinn = PINN()

    # Load the data and scale
    X, X_boundary, v = pinn.get_data()
    scaler = MinMaxScaler()
    X= scaler.fit_transform(X)
    # Split the data into training and testing sets (80/20)
    X_train, X_test, v_train, v_test = train_test_split(X, v, test_size=0.8, random_state=42)
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
                            num_domain=3000,
                            num_boundary=100)
    lr_phase1 = config["lr_phase1"]
    net = dde.maps.FNN([3] + n_layers * [n_neurons] +
                       [2], activation, initializer)
    #save_path = os.getcwd()+f"/models/heart_model_{n_neurons}x{n_layers}_{activation}_{initializer}_lr{lr_phase1}mini_Vnorm_domain3k_rs10k"
    save_path = os.getcwd()+f"/models/heart_model_{n_neurons}x{n_layers}_{activation}_{initializer}_mini_Vnorm_domain3k_rs10k"
    
    if args.mode == "train":

        #Phase 1: Train on data only
        lr_phase1 = config["lr_phase1"]
        weights1 = config["weights1"]
        checker = dde.callbacks.ModelCheckpoint(
            save_path, save_better_only=True, period=2000)
        
        model = dde.Model(data, net)
        resampler = dde.callbacks.PDEPointResampler(period=10000,pde_points=True,bc_points=True)
        
        
        model.compile("adam", lr=lr_phase1, loss_weights=weights1)
        losshistory, train_state = model.train(
            iterations=15000, model_save_path=save_path,callbacks=[resampler])

        dde.saveplot(losshistory, train_state, issave=True, isplot=True)

        #Phase 2: Train on data, BC, IC and PDE+ODE
        lr_phase2 = config["lr_phase2"]
        weights2 = config["weights2"]
        model.compile("adam", lr=0.0005, loss_weights=weights2)
        losshistory, train_state = model.train(
            iterations=300000, model_save_path=save_path,callbacks=[resampler])
        dde.saveplot(losshistory, train_state, issave=True, isplot=True)

        #Phase 3: L-BFGS-B
        #net.regularizer = ("l2", 0) # No regularization
        #model.compile("L-BFGS-B")
        #losshistory, train_state = model.train(
        #    iterations=20000, model_save_path=save_path)
        #dde.saveplot(losshistory, train_state, issave=True, isplot=True)

        #Save the model
        model.save(save_path)






    elif args.mode == "predict":
        torch.device("cpu")
        dde.config.default_device = "cpu"

        model = dde.Model(data, net)
        model.compile("adam", lr=0.0005)
        # model.compile("adam", lr=0.0001)
        checkpoint= args.checkpoint
        model.restore(save_path+"-"+checkpoint+".pt")
        #model.restore("./models/heart_model_64x5_tanh_Glorot normal_mini_t_norm-160000.pt")
        from plot import generate_2D_animation, plot_2D
        # plot_2D(pinn, model)
        generate_2D_animation(pinn, model)
        # plot_2D_grid(data_list, pinn, model, "planar_wave")
    elif args.mode == "continue":
        ...

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
