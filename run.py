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
from src.factories.dataloader_factory import DataLoaderFactory
from src.evaluation import Evaluator
from src.utils.logger import setup_logger

# Setup root logger
logger = setup_logger("hyco_phiflow", level=logging.INFO)


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
            run_generation(config)

        elif task == "train":
            logger.info("Running training")
            # Use factory to create trainer (Phase 1 API)
            trainer = TrainerFactory.create_trainer(config)
            # Create data loader/dataset based on model type using NEW DataLoaderFactory
            mode = config["general"]["mode"]
            
            if mode == "synthetic":
                # Create PhiML Dataset (pure PhiML pipeline - no PyTorch!)
                logger.info("Creating PhiML dataset...")
                dataset = DataLoaderFactory.create_phiml(
                    config,
                    sim_indices=config['trainer']['train_sim'],
                    enable_augmentation=False,
                    percentage_real_data=1.0
                )

                logger.info(f"Dataset created: {len(dataset)} samples")

                # Train with PhiML dataset (no DataLoader wrapper!)
                num_epochs = config["trainer"]['synthetic']["epochs"]
                logger.info(f"Starting training for {num_epochs} epochs...")
                trainer.train(dataset=dataset, num_epochs=num_epochs)
                
            elif mode == "physical":
                # Create FieldDataset for physical training using DataLoaderFactory
                dataset = DataLoaderFactory.create(
                    config,
                    mode='field',
                    batch_size=None,  # Physical models don't use batching
                )
                
                # Train with explicit data passing (Phase 1 API)
                # Physical typically uses single epoch (optimization per sample)
                num_epochs = config["trainer"]['physical']["epochs"]
                trainer.train(data_source=dataset, num_epochs=num_epochs)
                
            elif mode == "hybrid":
                # Phase 3: Hybrid training with data augmentation
                # HybridTrainer manages data creation internally
                # Just call train() with no arguments
                logger.info("Starting hybrid training...")
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
