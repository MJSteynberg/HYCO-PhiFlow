"""
Hydra-based experiment runner.

Usage:
    python run_hydra.py experiment=burgers_experiment
    python run_hydra.py experiment=burgers_experiment trainer_params.epochs=200
    python run_hydra.py +experiment=smoke_experiment run_params.mode=[generate,train]
"""

import os
import sys
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generator import run_generation
from src.factories.trainer_factory import TrainerFactory
from src.evaluation import Evaluator

import numpy as np

print(np.__version__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    config = OmegaConf.to_container(cfg, resolve=True)
    config["project_root"] = str(PROJECT_ROOT)
    tasks = config["run_params"]["mode"]

    if isinstance(tasks, str):
        tasks = [tasks]

    # Execute tasks
    for task in tasks:
        if task == "generate":
            run_generation(config)

        elif task == "train":
            # Use factory to create trainer
            trainer = TrainerFactory.create_trainer(config)
            trainer.train()

        elif task == "evaluate":
            evaluator = Evaluator(config)
            evaluator.evaluate()

        else:
            print(f"Warning: Unknown task '{task}'")


if __name__ == "__main__":
    main()
