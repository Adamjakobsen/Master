import itertools
import subprocess
import os
from datetime import datetime
import json
import GPUtil
from utils import *

# Define the hyperparameter space
"""
hyperparameter_grid = {
    'lr_phase1': [0.001, 0.0005, 0.0001, 0.00005],
    'batch_size': [256, 512, 1024],
    'activation': ['tanh', 'silu'],
    'num_domain': [1000, 10000],
    'resampling_period': [1000, 10000]
}
"""
hyperparameter_grid = {
    'lr_phase1': [0.001],
    'batch_size': [256],
    'activation': ['tanh', 'silu'],
    'num_domain': [1000],
    'resampling_period': [1000]
}

# Generate all combinations of hyperparameters
keys, values = zip(*hyperparameter_grid.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Get the number of available GPUs
num_gpus = len(GPUtil.getGPUs())

# Run experiments
for i, experiment in enumerate(experiments):
    gpu_id = i % num_gpus  # Distribute the experiments across all GPUs

    # Create a unique folder for this experiment
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{now}_lr{experiment['lr_phase1']}_batch{experiment['batch_size']}_activation{experiment['activation']}_domain{experiment['num_domain']}_resample{experiment['resampling_period']}"
    experiment_folder = os.path.join("./experiments", experiment_name)
    os.makedirs(experiment_folder, exist_ok=True)

    # Create a new config file for this experiment
    config = load_config()  # Assuming load_config() loads the default config
    config['TRAINING'].update(experiment)
    config_path = os.path.join(experiment_folder, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # Create an output file for this experiment
    output_file_path = os.path.join(experiment_folder, "output.log")

    # Run the experiment on the selected GPU
    cmd = f"DDE_BACKEND=pytorch python main.py tuning --config {config_path} --gpu {gpu_id}"
    with open(output_file_path, "w") as output_file:
        subprocess.Popen(cmd, shell=True, executable="/bin/bash", stdout=output_file, stderr=subprocess.STDOUT)
