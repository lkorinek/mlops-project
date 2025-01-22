import os
import sys

import yaml

import wandb

sys.path.append(os.path.realpath("./src/mlops_project"))
from hydra import compose, initialize
from train import train


def create_and_run_sweep(config_path):
    print(f"Creating W&B sweep with config: {config_path}")

    with open(config_path, "r") as file:
        sweep_config = yaml.safe_load(file)

    sweep_id = wandb.sweep(sweep=sweep_config, project="wandb-sweep")

    # Wrapper function to pass additional arguments
    with initialize(version_base="1.1", config_path="../configs/train_config"):
        cfg = compose(config_name="default_config.yaml")

    print(f"Training config: {cfg}")

    def train_wrapper():
        train(cfg)

    # Run the W&B agent
    wandb.agent(sweep_id, function=train_wrapper, count=50)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_sweep.py <path_to_sweep_config>")
        sys.exit(1)

    config_path = sys.argv[1]
    create_and_run_sweep(config_path)
