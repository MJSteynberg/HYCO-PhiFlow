"""Hybrid trainer that alternates between synthetic and physical training with data augmentation."""

import os
import time
from typing import Dict, Any, List
from pathlib import Path
import torch

from phi.torch.flow import *
from phi.math import math, Tensor

from src.training.synthetic.trainer import SyntheticTrainer
from src.training.physical.trainer import PhysicalTrainer
from src.data.dataset import AccessPolicy, Dataset
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridTrainer:
    """
    Hybrid trainer that alternates between synthetic and physical model training
    with cross-model data augmentation.

    Training procedure:
    1. Warmup phase: Train synthetic model on real data only
    2. Hybrid phase: For each cycle:
       - Generate augmented data from synthetic model
       - Train physical model with real + synthetic data (weighted loss)
       - Generate augmented data from physical model
       - Train synthetic model with real + physical data (weighted loss)
       
    Loss formulation (HYCO paper):
    - Synthetic model minimizes: L + λ_syn * I
    - Physical model minimizes: L + λ_phy * I
    where L = real data loss, I = interaction (generated data) loss
    """

    def __init__(
        self,
        config: Dict[str, Any],
        synthetic_model,
        physical_model,
    ):
        """
        Initialize hybrid trainer.

        Args:
            config: Full configuration dictionary
            synthetic_model: Initialized synthetic model
            physical_model: Initialized physical model
        """
        self.config = config
        self.synthetic_model = synthetic_model
        self.physical_model = physical_model

        self._parse_config(config)

        # Determine maximum rollout steps needed (for dataset creation)
        synthetic_config = config['trainer']['synthetic']
        rollout_scheduler = synthetic_config.get('rollout_scheduler', None)
        if rollout_scheduler:
            max_rollout_steps = rollout_scheduler.get('end', config['trainer'].get('rollout_steps', 4))
        else:
            max_rollout_steps = synthetic_config.get('rollout_steps', config['trainer'].get('rollout_steps', 4))

        # Create dataset with cache sized for all training sims
        self.dataset = Dataset(
            config=config,
            train_sim=config['trainer']['train_sim'],
            rollout_steps=max_rollout_steps,
            max_cached_sims=len(config['trainer']['train_sim']) + 2,  # Cache all + buffer
        )

        # Create individual trainers
        self.synthetic_trainer = SyntheticTrainer(config, synthetic_model)
        self.physical_trainer = PhysicalTrainer(config, physical_model)

        # Optional: cache directory for debugging augmented data
        self.cache_dir = Path(self.augmentation_config.get('cache_dir', 'data/cache_phiml'))
        self.save_augmented = self.augmentation_config.get('save_augmented', False)
        if self.save_augmented:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"HybridTrainer initialized: "
            f"cycles={self.cycles}, warmup={self.warmup_cycles}, "
            f"alpha={self.alpha}"
        )

    def _parse_config(self, config: Dict[str, Any]):
        """Parse hybrid training configuration."""
        hybrid_config = config['trainer']['hybrid']

        self.cycles = hybrid_config['cycles']
        self.warmup_cycles = hybrid_config.get('warmup', 0)
        self.augmentation_config = hybrid_config.get('augmentation', {})
        self.alpha = self.augmentation_config.get('alpha', 1.0)

        # Get epochs for each model
        self.synthetic_epochs = config['trainer']['synthetic']['epochs']
        self.physical_epochs = config['trainer']['physical']['epochs']

        # Loss scaling config
        loss_scaling = hybrid_config.get('loss_scaling', {})
        
        self.synthetic_loss_config = {
            'real_weight': loss_scaling.get('synthetic', {}).get('real_weight', 1.0),
            'interaction_weight': loss_scaling.get('synthetic', {}).get('interaction_weight', 1.0),
            'proportional': loss_scaling.get('synthetic', {}).get('proportional', False),
        }
        
        self.physical_loss_config = {
            'real_weight': loss_scaling.get('physical', {}).get('real_weight', 1.0),
            'interaction_weight': loss_scaling.get('physical', {}).get('interaction_weight', 0.1),
            'proportional': loss_scaling.get('physical', {}).get('proportional', False),
        }

    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute hybrid training.

        Returns:
            Dictionary with training results
        """
        results = {
            "cycles": [],
            "synthetic_losses": [],
            "physical_losses": [],
            "cycle_times": [],
        }

        logger.info("=" * 60)
        logger.info("Starting Hybrid Training")
        logger.info("=" * 60)

        # Warmup phase: Train on real data only
        if self.warmup_cycles > 0:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"WARMUP PHASE: Training on real data only ({self.warmup_cycles} cycles)")
            logger.info(f"{'=' * 60}\n")

            for cycle in range(self.warmup_cycles):
                logger.info(f"\n--- Warmup Cycle {cycle + 1}/{self.warmup_cycles} ---")
                cycle_start = time.time()

                # Configure dataset for real data only
                self.dataset.access_policy = AccessPolicy.REAL_ONLY
                self.dataset.alpha = 1.0

                # Train synthetic model on real data
                self.synthetic_trainer.train(
                    self.dataset,
                    num_epochs=self.synthetic_epochs,
                    start_epoch=cycle * self.synthetic_epochs,
                    verbose=verbose
                )

                cycle_time = time.time() - cycle_start
                logger.info(f"Warmup cycle {cycle + 1} completed in {cycle_time:.2f}s")

        # Hybrid phase: Alternating training with augmentation
        logger.info(f"\n{'=' * 60}")
        logger.info(f"HYBRID PHASE: Training with data augmentation ({self.cycles} cycles)")
        logger.info(f"{'=' * 60}\n")

        # Configure schedulers to span all hybrid cycles
        total_synthetic_epochs = self.cycles * self.synthetic_epochs
        self.synthetic_trainer.set_total_epochs_for_hybrid(total_synthetic_epochs)

        # Set loss scaling once (doesn't change between cycles)
        self.physical_trainer.set_loss_scaling(
            real_weight=self.physical_loss_config['real_weight'],
            interaction_weight=self.physical_loss_config['interaction_weight'],
            proportional=self.physical_loss_config['proportional']
        )
        self.synthetic_trainer.set_loss_scaling(
            real_weight=self.synthetic_loss_config['real_weight'],
            interaction_weight=self.synthetic_loss_config['interaction_weight'],
            proportional=self.synthetic_loss_config['proportional']
        )

        # Configure dataset for both real and generated data
        self.dataset.access_policy = AccessPolicy.BOTH
        self.dataset.alpha = self.alpha

        for cycle in range(self.cycles):
            logger.info(f"\n--- Hybrid Cycle {cycle + 1}/{self.cycles} ---")
            cycle_start = time.time()

            # Generate synthetic data (batched for efficiency)
            synthetic_trajectories = self._generate_synthetic_data_batched()
            self.dataset.set_augmented_trajectories(synthetic_trajectories)
            
            if self.save_augmented:
                self._save_augmented_data(synthetic_trajectories, f"synthetic_cycle_{cycle}")

            # Train physical model
            physical_results = self.physical_trainer.train(
                self.dataset,
                num_epochs=self.physical_epochs,
                verbose=verbose
            )

            # Generate physical data
            physical_trajectories = self._generate_physical_data_batched()
            self.dataset.set_augmented_trajectories(physical_trajectories)
            
            if self.save_augmented:
                self._save_augmented_data(physical_trajectories, f"physical_cycle_{cycle}")

            # Train synthetic model
            synthetic_results = self.synthetic_trainer.train(
                self.dataset,
                num_epochs=self.synthetic_epochs,
                start_epoch=cycle * self.synthetic_epochs,
                verbose=verbose
            )

            cycle_time = time.time() - cycle_start

            results["cycles"].append(cycle + 1)
            results["synthetic_losses"].append(synthetic_results["final_loss"])
            results["physical_losses"].append(physical_results["final_loss"])
            results["cycle_times"].append(cycle_time)

            logger.info(
                f"Cycle {cycle + 1} completed in {cycle_time:.2f}s - "
                f"Synthetic loss: {synthetic_results['final_loss']:.6f}, "
                f"Physical loss: {physical_results['final_loss']:.6f}"
            )

        logger.info("\n" + "=" * 60)
        logger.info("Hybrid Training Completed")
        logger.info("=" * 60)

        return results

    def _generate_synthetic_data_batched(self) -> List[Tensor]:
        """
        Generate augmented trajectories using the synthetic model (batched for efficiency).
        
        Batches all simulations together for GPU parallelization.
        """
        train_sims = self.config['trainer']['train_sim']
        num_steps = self.config['data']['trajectory_length']
        
        # Collect all initial states from cache (no disk I/O)
        initial_states = []
        for sim_idx in train_sims:
            sim_data = self.dataset._load_simulation(sim_idx)
            initial_states.append(sim_data.time[0])
        
        # Stack into batch for parallel processing
        batched_initial = math.stack(initial_states, batch('batch'))
        
        # Batched rollout (GPU parallelized)
        with torch.no_grad():
            states = [batched_initial]
            current_state = batched_initial
            for _ in range(num_steps - 1):
                next_state = self.synthetic_model(current_state)
                states.append(next_state)
                current_state = next_state
        
        # Stack time dimension
        trajectory_batched = math.stack(states, batch('time'))
        trajectory_batched = math.stop_gradient(trajectory_batched)
        
        # Unstack back to list of individual trajectories
        return [trajectory_batched.batch[i] for i in range(len(train_sims))]

    def _generate_physical_data_batched(self) -> List[Tensor]:
        """
        Generate augmented trajectories using the physical model (batched for efficiency).
        """
        train_sims = self.config['trainer']['train_sim']
        num_steps = self.config['data']['trajectory_length'] - 1
        
        # Collect all initial states from cache
        initial_states = []
        for sim_idx in train_sims:
            sim_data = self.dataset._load_simulation(sim_idx)
            initial_states.append(sim_data.time[0])
        
        # Stack into batch
        batched_initial = math.stack(initial_states, batch('batch'))
        
        # Batched rollout using physical model
        with torch.no_grad():
            trajectory_batched = self.physical_model.rollout(
                batched_initial,
                self.physical_model.params,
                num_steps
            )
        
        trajectory_batched = math.stop_gradient(trajectory_batched)
        
        # Unstack back to list
        return [trajectory_batched.batch[i] for i in range(len(train_sims))]

    def _save_augmented_data(self, trajectories: List[Tensor], prefix: str):
        """
        Save augmented trajectories to cache directory (for debugging).

        Args:
            trajectories: List of trajectory tensors
            prefix: Prefix for filenames
        """
        for i, trajectory in enumerate(trajectories):
            cache_path = self.cache_dir / f"{prefix}_sim_{i:04d}.npz"
            math.save(str(cache_path), trajectory)

        logger.debug(f"Saved {len(trajectories)} augmented trajectories to {self.cache_dir}")