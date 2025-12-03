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

from src.data.data_generator import DataGenerator
from src.factories.trainer_factory import TrainerFactory
from src.factories.dataloader_factory import DataLoaderFactory
from src.evaluation import Evaluator
from src.utils.logger import setup_logger

# Setup root logger
logger = setup_logger("hyco_phiflow", level=logging.INFO)

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("add", lambda x, y: x + y, replace=True)
OmegaConf.register_new_resolver("subtract", lambda x, y: x - y, replace=True)
OmegaConf.register_new_resolver("divide", lambda x, y: x // y if isinstance(x, int) and isinstance(y, int) else x / y, replace=True)

# Convenience resolver for total synthetic epochs in standalone mode
# Usage: ${total_synthetic_epochs:${trainer.synthetic.epochs},${trainer.hybrid.cycles},${trainer.hybrid.warmup}}
OmegaConf.register_new_resolver(
    "total_synthetic_epochs",
    lambda epochs, cycles, warmup: epochs * (cycles + warmup),
    replace=True
)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    config = OmegaConf.to_container(cfg, resolve=True)
    config["project_root"] = str(PROJECT_ROOT)

    tasks = config["general"]["tasks"]


    logger.info(f"Starting HYCO-PhiFlow with tasks: {tasks}")
    logger.info(
        f"Experiment: {config['general'].get('experiment_name', 'unknown')}"
    )

    # Execute tasks
    for task in tasks:
        logger.info(f"=== Starting task: {task} ===")

        if task == "generate":
            logger.info("Running data generation")
            data_generator = DataGenerator(config)
            data_generator.generate_data()

        elif task == "train":
            logger.info("Running training")
            mode = config["general"]["mode"]

            if mode == "synthetic":
                # Create PhiML Dataset first (to get num_channels)
                logger.info("Creating PhiML dataset...")
                dataset = DataLoaderFactory.create_phiml(
                    config,
                    sim_indices=config['trainer']['train_sim'],
                )

                # Create trainer with num_channels from dataset
                logger.info(f"Creating trainer with {dataset.num_channels} channels...")
                trainer = TrainerFactory.create_trainer(config, num_channels=dataset.num_channels)

                # Train with PhiML dataset
                num_epochs = config["trainer"]['synthetic']["epochs"]
                logger.info(f"Starting training for {num_epochs} epochs...")
                trainer.train(dataset=dataset, num_epochs=num_epochs)
                
            elif mode == "physical":
                # Create PhiML Dataset for physical training
                logger.info("Creating PhiML dataset...")
                dataset = DataLoaderFactory.create_phiml(
                    config,
                    sim_indices=config['trainer']['train_sim'],
                    enable_augmentation=False,
                    percentage_real_data=1.0
                )

                # Train with PhiML dataset
                # Use batch_size from config (larger batches = faster but more memory)
                num_epochs = config["trainer"]['physical']["epochs"]
                batch_size = config['trainer'].get('batch_size', 8)
                
                # Create trainer
                logger.info("Creating physical trainer...")
                trainer = TrainerFactory.create_trainer(config)
                
                logger.info(f"Starting training for {num_epochs} epochs with batch_size={batch_size}...")
                trainer.train(dataset=dataset, num_epochs=num_epochs)
                
            elif mode == "hybrid":
                # Phase 3: Hybrid training with data augmentation
                # HybridTrainer manages data creation internally
                # Just call train() with no arguments
                logger.info("Starting hybrid training...")
                trainer = TrainerFactory.create_trainer(config)
                trainer.train()
            
            else:
                raise ValueError(f"Unknown mode: {mode}")


        elif task == "evaluate":
            logger.info("Running evaluation")
            from src.evaluation import Evaluator
            evaluator = Evaluator(config)
            evaluator.evaluate()

        else:
            logger.warning(f"Unknown task '{task}', skipping")

        logger.info(f"=== Completed task: {task} ===")

    logger.info("All tasks completed successfully")


if __name__ == "__main__":
    main()
