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


def parse_arguments():
    parser = argparse.ArgumentParser(description='PINN model for 2D PDEs.')
    parser.add_argument('mode', type=str, choices=['train','train_adaptive', 'predict', 'continue','tuning', 'make_dir'],
                        help='Mode to run the model: train,train_adaptive, predict, continue,tuning, or make_dir.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint for prediction or continuation.')
    parser.add_argument('--config', type=str, default="config.json", help='Path to the configuration file.')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID to use.')
    return parser.parse_args()


def define_model(config):
    # Load the configuration
    n_neurons = config["TRAINING"]["n_neurons"]
    n_layers = config["TRAINING"]["n_layers"]
    activation = config["TRAINING"]["activation"]
    initializer = config["TRAINING"]["initializer"]
    regularizer = config["TRAINING"]["regularizer"]
    test_size = config["DATA"]["test_size"]
    num_domain = config["DATA"]["num_domain"]
    num_boundary = config["DATA"]["num_boundary"]
    resampling_period = config["DATA"]["resampling_period"]
    num_test = config["DATA"]["num_test"]
    num_init= config["DATA"]["num_init"]
    xy_slice = config["DATA"]["xy_slice"]
    t_slice = config["DATA"]["t_slice"]
    isotropic=config["DATA"]["isotropic"]
    batches=config["DATA"]["batches"]
    
    

    
    
    path_directory= make_directory(config)
    save_path = os.path.join(path_directory, "model")
    save_config(config, path_directory)

    # Initialize the model
    pinn = PINN()

    # Load the data and scale
    X_unscaled, X_boundary, v = pinn.get_data(slice_xy=xy_slice, slice_t=t_slice,long=False, isotropic=isotropic)
    #print size in mb and shape
    import sys
    print("Size of X_unscaled: %.3f MB" % (sys.getsizeof(X_unscaled) / 1e6))
    print("Shape of X_unscaled:", np.shape(X_unscaled))
    
    scaler = MinMaxScaler()
    X= scaler.fit_transform(X_unscaled)
    # Split the data into training and testing sets (80/20)
    X_train, X_test, v_train, v_test = train_test_split(X, v, test_size=test_size, random_state=42)
    data_list = [X_train, X_test, v_train, v_test]
    batch_size= int(X_train.shape[0]/batches)
    # Initialize the model
    geomtime = pinn.geotime()
    observe_v = dde.PointSetBC(X_train, v_train, component=0,batch_size=batch_size)
    X_train_unscaled = scaler.inverse_transform(X_train)
    ic = pinn.IC(X_train, v_train,X_train_unscaled)
    #bc = pinn.BC(geomtime)
    input_data = [ observe_v]
    if isotropic:
        PDE= pinn.pde2d_vm
        inputs=[3]
    else:
        PDE= pinn.pde2d_conductivities
        inputs= [7]
    #print("shape x_tr:",np.shape(X_train[::1000]))
    print("batch size:", batch_size)
    first_batch = X_train[:batch_size,:]
    print("first batch:", first_batch.shape)
    data = dde.data.TimePDE(geomtime,
                            PDE,
                            input_data,
                            num_domain=num_domain,
                            num_boundary=num_boundary,
                            anchors=X_train
                            
                            
                            )
    if type(n_neurons) == list:
        net = dde.maps.FNN(inputs + n_neurons +
                           [2], activation, initializer)
    else:
        net = dde.maps.FNN(inputs + n_layers * [n_neurons] +
                       [2], activation, initializer)
        
    #net.regularizer = (regularizer[0], regularizer[1])
    

    model = dde.Model(data, net)
    return model,net,pinn,data,geomtime,PDE, save_path,path_directory

def train_model(pinn,model, config,save_path,path_directory)-> None:
    #Phase 1: Train on data only
    optimizer=config["TRAINING"]["optimizer"]
    lr_phase1 = config["TRAINING"]["lr_phase1"]
    weights1 = config["TRAINING"]["weights1"]
    epochs_phase1 = config["TRAINING"]["epochs_phase1"]
    resampling_period = config["DATA"]["resampling_period"]
    batches=config["DATA"]["batches"]

    
    X, X_boundary, v = pinn.get_data()
    print("Shape of X", np.shape(X))
    scaler = MinMaxScaler()
    X= scaler.fit_transform(X)
    # Split the data into training and testing sets (80/20)
    X_train, X_test, v_train, v_test = train_test_split(X, v, test_size=0.8, random_state=42)

    batch_size= int(X_train.shape[0]/batches)

    resampler = dde.callbacks.PDEPointResampler(period=resampling_period,pde_points=True,bc_points=True)
    save_better = dde.callbacks.ModelCheckpoint(save_path,save_better_only=True,period=2000)
    
    model.compile("adam", lr=lr_phase1, loss_weights=weights1)
    losshistory, train_state = model.train(
            iterations=epochs_phase1,batch_size=batch_size, model_save_path=save_path,callbacks=[resampler,save_better])
    dde.saveplot(losshistory, train_state, issave=True, isplot=True,output_dir=path_directory)
    
    #Phase 2: Train on data, BC, IC and PDE+ODE
    lr_phase2 = config["TRAINING"]["lr_phase2"]
    epochs_phase2 = config["TRAINING"]["epochs_phase2"]
    weights2 = config["TRAINING"]["weights2"]
    model.compile(optimizer, lr=lr_phase2, loss_weights=weights2)
    losshistory, train_state = model.train(
        iterations=epochs_phase2,batch_size=batch_size, model_save_path=save_path,callbacks=[resampler,save_better])
    dde.saveplot(losshistory, train_state, issave=True, isplot=True,output_dir=path_directory)
    model.save(save_path)
    v_pred_test = model.predict(X_test)[:,0]
    # If they are not 1D, reshape or index them
    if len(v_test.shape) > 1:
        v_test = v_test.reshape(-1)
    if len(v_pred_test.shape) > 1:
        v_pred_test = v_pred_test.reshape(-1)
    RMSE = np.sqrt(np.mean((v_test - v_pred_test)**2))
    step=losshistory.steps[-1]
    print("step:",step ,"RMSE test", RMSE)

def train_model_adaptive_residual(pinn,model,geomtime,data,pde, config,save_path,path_directory)-> None:
    X, X_boundary, v = pinn.get_data()
    scaler = MinMaxScaler()
    X= scaler.fit_transform(X)
    # Split the data into training and testing sets (80/20)
    X_train, X_test, v_train, v_test = train_test_split(X, v, test_size=0.8, random_state=42)
    #Phase 1: Train on data only
    lr_phase1 = config["TRAINING"]["lr_phase1"]
    weights1 = config["TRAINING"]["weights1"]
    epochs_phase1 = config["TRAINING"]["epochs_phase1"]
    resampling_period = config["DATA"]["resampling_period"]
    err_threshold = config["TRAINING"]["err_threshold"]
    resampler = dde.callbacks.PDEPointResampler(period=resampling_period,pde_points=True,bc_points=True)

    model.compile("adam", lr=lr_phase1, loss_weights=weights1)
    losshistory, train_state = model.train(
            iterations=epochs_phase1, model_save_path=save_path,callbacks=[resampler])

    dde.saveplot(losshistory, train_state, issave=True, isplot=True,output_dir=path_directory)
    
    #Adaptive residual based learning
    lr_phase1 = config["TRAINING"]["lr_phase1"]
    weights2 = config["TRAINING"]["weights2"]
    epochs_phase1 = config["TRAINING"]["epochs_phase1"]
    epochs_phase2=config["TRAINING"]["epochs_phase2"]
    resampling_period = config["DATA"]["resampling_period"]
    save_best= dde.callbacks.ModelCheckpoint(save_path,save_better_only=True,period=2000)
    resampler = dde.callbacks.PDEPointResampler(period=resampling_period,pde_points=True,bc_points=True)
    X_points_colocation = geomtime.random_points(100000)
    print(np.shape(X_points_colocation))
    step=losshistory.steps[-1]
    err=1
    while err>err_threshold or int(step)<epochs_phase2:
        f = model.predict(X_points_colocation, operator=pde)
        
        
        
        err_eq = np.abs(f)
        
        err = np.mean(np.abs(err_eq))
        #x_id = np.argmax(err_eq)
        #Index og 10 largest errors
        kth_value = -1000
        x_id = np.argpartition(err_eq[0,:,0], kth_value)[kth_value:]
        print("Mean residual: %.3e" % (err))
        #print("Adding new points:", X_points_colocation[x_id], "\n")
        data.add_anchors(X_points_colocation[x_id])
        early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
        model.compile("adam", lr=lr_phase1, loss_weights=weights2)
        losshistory,train_state=model.train(iterations=10000, disregard_previous_best=False, callbacks=[early_stopping])
        
        v_pred_test = model.predict(X_test)[:,0]
        # If they are not 1D, reshape or index them
        if len(v_test.shape) > 1:
            v_test = v_test.reshape(-1)
        if len(v_pred_test.shape) > 1:
            v_pred_test = v_pred_test.reshape(-1)
        RMSE = np.sqrt(np.mean((v_test - v_pred_test)**2))
        step=losshistory.steps[-1]
        print("step:",step ,"RMSE test", RMSE)
        model.save(save_path)
        if step>=epochs_phase2:
            break   
    
    dde.saveplot(losshistory, train_state, issave=True, isplot=True,output_dir=path_directory)
    model.save(save_path)
    """
    #RMSE
    v_pred_test = model.predict(X_test)[:,0]
    
    # If they are not 1D, reshape or index them
    if len(v_test.shape) > 1:
        v_test = v_test.reshape(-1)
    if len(v_pred_test.shape) > 1:
        v_pred_test = v_pred_test.reshape(-1)
    RMSE = np.sqrt(np.mean((v_test - v_pred_test)**2))
    print("RMSE test final", RMSE)
    
    model.compile("L-BFGS-B")
    losshistory, train_state = model.train(model_save_path=save_path)
    model.save(save_path)

    #RMSE
    X, X_boundary, v = pinn.get_data()
    scaler = MinMaxScaler()
    X= scaler.fit_transform(X)
    # Split the data into training and testing sets (80/20)
    X_train, X_test, v_train, v_test = train_test_split(X, v, test_size=0.85, random_state=42)
    v_pred_test = model.predict(X_test)[:,0]
    
    # If they are not 1D, reshape or index them
    if len(v_test.shape) > 1:
        v_test = v_test.reshape(-1)
    if len(v_pred_test.shape) > 1:
        v_pred_test = v_pred_test.reshape(-1)
    RMSE = np.sqrt(np.mean((v_test - v_pred_test)**2))
    print("RMSE test L-BFGS", RMSE)

    #Phase 2: Train on data, BC, IC and PDE+ODE
    lr_phase2 = config["TRAINING"]["lr_phase2"]
    epochs_phase2 = config["TRAINING"]["epochs_phase2"]
    weights2 = config["TRAINING"]["weights2"]
    model.compile("adam", lr=lr_phase2, loss_weights=weights2)
    losshistory, train_state = model.train(
        iterations=epochs_phase2, model_save_path=save_path,callbacks=[resampler])
    dde.saveplot(losshistory, train_state, issave=True, isplot=True,output_dir=path_directory)
    model.save(save_path)
    """

#def train_batches(pinn,model, config,save_path,path_directory)-> None:



    

def run_tuning(config):
    # Initialize CUDA and other configurations
    torch.cuda.empty_cache()
    dde.config.set_random_seed(42)
    dde.config.set_default_float("float32") 
    torch.cuda.set_per_process_memory_fraction(0.9)
    # Set device to argument --gpu
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    dde.config.default_device=device
    #Print device
    print("Device:", torch.cuda.current_device())
    # Define the model
    model,net,pinn,data, save_path,path_directory = define_model(config)
    # Setup logger
    log_file = os.path.join(os.path.dirname(save_path), "experiment.log")
    setup_logger(log_file)
    
    logging.info("Experiment started.")
    logging.info(f"Configuration: {config}")
    lr = config["TRAINING"]["lr_phase1"]
    weights = config["TRAINING"]["weights2"]
    epochs= config["TRAINING"]["epochs_phase1"]
    resampling_period = config["DATA"]["resampling_period"]
    resampler = dde.callbacks.PDEPointResampler(period=resampling_period,pde_points=True,bc_points=True)

    model.compile("adam", lr=lr, loss_weights=weights)
    losshistory, train_state = model.train(
            iterations=1, model_save_path=save_path,callbacks=[resampler])
    # Load the data and scale
    X, X_boundary, v = pinn.get_data()
    scaler = MinMaxScaler()
    X= scaler.fit_transform(X)
    # Split the data into training and testing sets 
    X_train, X_test, v_train, v_test = train_test_split(X, v, test_size=config["DATA"]["test_size"], random_state=42)

    #Predict on test set
    v_pred_test=model.predict(X_test)[:, 0]
    # Check their shapes
    print("Shape of v_test:", v_test.shape)
    print("Shape of v_pred_test:", v_pred_test.shape)

    # If they are not 1D, reshape or index them
    if len(v_test.shape) > 1:
        v_test = v_test.reshape(-1)
    if len(v_pred_test.shape) > 1:
        v_pred_test = v_pred_test.reshape(-1)

    #RMSE test
    RMSE = np.sqrt(np.sum((v_test - v_pred_test)**2)/v_test.shape[0])
    print("RMSE test", RMSE)
    #Save RMSE and configuration to txt file. Create file if needed, else append
    #with open( "RMSE.txt", "a+") as f:
    #    f.write(f"RMSE: {RMSE}, Configuration: {config}\n")
    
    






def main(args):
    if args.mode == "tuning":
        config = load_config(args.config)
        run_tuning(config)
        return


    config = load_config()
    if args.mode == "make_dir":
        make_directory(config)
        return
    # Initialize CUDA and other configurations
    torch.cuda.empty_cache()
    dde.config.set_random_seed(42)
    dde.config.set_default_float("float32")
    best_device_id = get_gpu_with_most_memory()
    #Set torch device
    device = torch.device(f"cuda:{best_device_id}")
    torch.cuda.set_device(device)
    dde.config.default_device=device
    torch.cuda.set_per_process_memory_fraction(0.9)

    

    # Define the model
    model,net,pinn,data,geomtime,PDE, save_path,path_directory = define_model(config)
    # Setup logger
    log_file = os.path.join(os.path.dirname(save_path), "experiment.log")
    setup_logger(log_file)
    
    logging.info("Experiment started.")
    logging.info(f"Configuration: {config}")

    if args.mode == "train":
        # Train the model
        train_model(pinn,model, config, save_path,path_directory)
    
    elif args.mode == "train_adaptive":
        train_model_adaptive_residual(pinn,model,geomtime,data,PDE, config,save_path,path_directory)
    
    elif args.mode == "predict":
        torch.device("cpu")
        dde.config.default_device = "cpu"

        model = dde.Model(data, net)
        #model.compile("adam", lr=0.0005)
        model.compile("adam", lr=0.0001)
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