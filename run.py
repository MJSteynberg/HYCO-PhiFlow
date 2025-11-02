"""
Hydra-based experiment runner with structured logging.

Usage:
    python run.py --config-name=burgers_experiment
    python run.py --config-name=burgers_experiment trainer_params.epochs=200
    python run.py --config-name=smoke_experiment run_params.mode=[generate,train]
"""

import os
import sys
import logging
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
from src.utils.logger import setup_logger

# Setup root logger
logger = setup_logger("hyco_phiflow", level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    config = OmegaConf.to_container(cfg, resolve=True)
    config["project_root"] = str(PROJECT_ROOT)

    # Configure logging from config
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("root_level", "INFO"))
    logger.setLevel(log_level)

    # Apply module-specific log levels
    module_levels = log_config.get("module_levels", {})
    for module_name, level_str in module_levels.items():
        module_logger = logging.getLogger(module_name)
        module_logger.setLevel(getattr(logging, level_str))

    tasks = config["run_params"]["mode"]

    if isinstance(tasks, str):
        tasks = [tasks]

    logger.info(f"Starting HYCO-PhiFlow with tasks: {tasks}")
    logger.info(
        f"Experiment: {config['run_params'].get('experiment_name', 'unknown')}"
    )

    # Execute tasks
    for task in tasks:
        logger.info(f"=== Starting task: {task} ===")

        if task == "generate":
            logger.info("Running data generation")
            run_generation(config)

        elif task == "train":
            logger.info("Running training")
            # Use factory to create trainer
            trainer = TrainerFactory.create_trainer(config)
            trainer.train()

        elif task == "evaluate":
            logger.info("Running evaluation")
            evaluator = Evaluator(config)
            evaluator.evaluate()

        else:
            logger.warning(f"Unknown task '{task}', skipping")

        logger.info(f"=== Completed task: {task} ===")

    logger.info("All tasks completed successfully")


if __name__ == "__main__":
    main()
