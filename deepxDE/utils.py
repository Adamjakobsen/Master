import h5py

import numpy as np
import GPUtil
import logging
import os
import json
from datetime import datetime


def get_data():
    scar_filename = "./meshes/basemesh.h5"

    v_filename = "./meshes/vm.h5"

    with h5py.File(scar_filename, "r") as scar_file:
        coordinates = scar_file["coordinates"][:]
        triangles = scar_file["topology"][:]

    t = np.arange(0, 605, 5)

    vm = np.zeros((len(t), len(coordinates)))
    i = 0

    with h5py.File(v_filename, "r") as v_file:
        group = v_file["vm"]
        datasets = list(group.keys())
        float_numbers = [float(number) for number in datasets]
        sorted_float_numbers = sorted(float_numbers)
        # sort datasets starts at t=0 end t=600 with length datasets
        datasets = [str(number) for number in sorted_float_numbers]
        for dataset in datasets:
            data = group[dataset]
            

            vm[i, :] = data[:]
            i += 1

    return coordinates, triangles, vm


def get_boundary():
    boundary_filename = "./meshes//boundary_mesh.h5"

    with h5py.File(boundary_filename, "r") as boundary_file:
        group1 = boundary_file["Mesh"]
        group2 = group1["mesh"]
        coordinates = group2["geometry"][:]
        triangles = group2["topology"][:]

    return coordinates, triangles

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

def make_directory(config):
    now = datetime.now()
    now = now.strftime("%Y-%m-%d %H:%M")
    name = f"{now}_{config['TRAINING']['n_neurons']}x{config['TRAINING']['n_layers']}_{config['TRAINING']['activation']}_{config['TRAINING']['initializer']}_numdom{config['DATA']['num_domain']}_rs{config['DATA']['resampling_period']}"
    path_directory = os.path.join("./experiments/", name)
    if not os.path.exists(path_directory):
        os.makedirs(path_directory)
    # After path_directory is created
    with open("temp_path_directory.txt", "w") as f:
        f.write(path_directory)
    config['path_directory'] = path_directory
    save_config(config, path_directory)
    return path_directory

def save_config(config, path):
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG for detailed logs
        format="%(asctime)s [%(levelname)s]: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
