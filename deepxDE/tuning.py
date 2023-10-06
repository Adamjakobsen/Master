import itertools
import subprocess
import os
from datetime import datetime
import json

# Define the hyperparameter space
hyperparameter_grid = {
    'lr_phase1': [0.001, 0.01],
    'n_neurons': [32, 64],
    'n_layers': [2, 4],
    'num_domain': [100, 200],
    'resampling_period': [10, 20]
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
    experiment_name = f"{now}_lr{experiment['lr_phase1']}_neurons{experiment['n_neurons']}_layers{experiment['n_layers']}_domain{experiment['num_domain']}_resample{experiment['resampling_period']}"
    experiment_folder = os.path.join("./experiments", experiment_name)
    os.makedirs(experiment_folder, exist_ok=True)

    # Create a new config file for this experiment
    config = load_config()  # Assuming load_config() loads the default config
    config['TRAINING'].update(experiment)
    config_path = os.path.join(experiment_folder, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # Run the experiment on the selected GPU
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python main.py --config {config_path}"
    subprocess.Popen(cmd, shell=True, executable="/bin/bash")