"""Hybrid trainer that alternates between synthetic and physical training with data augmentation."""

import os
import time
from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm

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
    1. Warmup phase: Train both models on real data only
    2. Hybrid phase: For each cycle:
       - Train synthetic model
       - Generate augmented data from synthetic model
       - Train physical model with mixed real+synthetic data
       - Generate augmented data from physical model
       - Train synthetic model with mixed real+physical data
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
        # Use the maximum from synthetic rollout scheduler if enabled
        synthetic_config = config['trainer']['synthetic']
        rollout_scheduler = synthetic_config.get('rollout_scheduler', None)
        if rollout_scheduler:
            max_rollout_steps = rollout_scheduler.get('end', config['trainer'].get('rollout_steps', 4))
        else:
            max_rollout_steps = synthetic_config.get('rollout_steps', config['trainer'].get('rollout_steps', 4))

        # Create dataset directly with maximum rollout steps
        self.dataset = Dataset(
            config=config,
            train_sim=config['trainer']['train_sim'],
            rollout_steps=max_rollout_steps,
        )

        # Create individual trainers
        self.synthetic_trainer = SyntheticTrainer(config, synthetic_model)
        self.physical_trainer = PhysicalTrainer(config, physical_model)

        # Cache directory for augmented data
        self.cache_dir = Path(self.augmentation_config.get('cache_dir', 'data/cache_phiml'))
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

        logger.info("="*60)
        logger.info("Starting Hybrid Training")
        logger.info("="*60)

        # Warmup phase: Train on real data only
        if self.warmup_cycles > 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"WARMUP PHASE: Training on real data only ({self.warmup_cycles} cycles)")
            logger.info(f"{'='*60}\n")

            for cycle in range(self.warmup_cycles):
                logger.info(f"\n--- Warmup Cycle {cycle + 1}/{self.warmup_cycles} ---")
                cycle_start = time.time()

                # Configure dataset for real data only
                self.dataset.access_policy = AccessPolicy.REAL_ONLY
                self.dataset.alpha = 1.0

                # Train synthetic model
                synthetic_results = self.synthetic_trainer.train(
                    self.dataset,
                    num_epochs=self.synthetic_epochs,
                    start_epoch=cycle * self.synthetic_epochs,
                    verbose=verbose
                )

                cycle_time = time.time() - cycle_start

        # Hybrid phase: Alternating training with augmentation
        logger.info(f"\n{'='*60}")
        logger.info(f"HYBRID PHASE: Training with data augmentation ({self.cycles} cycles)")
        logger.info(f"{'='*60}\n")

        for cycle in range(self.cycles):
            cycle_start = time.time()

            synthetic_trajectories = self._generate_synthetic_data()
            self._save_augmented_data(synthetic_trajectories, f"synthetic_cycle_{cycle}")

            self.dataset.set_augmented_trajectories(synthetic_trajectories)
            self.dataset.access_policy = AccessPolicy.BOTH
            self.dataset.alpha = self.alpha

            physical_results = self.physical_trainer.train(
                self.dataset,
                num_epochs=self.physical_epochs,
                verbose=verbose
            )

            physical_trajectories = self._generate_physical_data()
            self._save_augmented_data(physical_trajectories, f"physical_cycle_{cycle}")

            self.dataset.set_augmented_trajectories(physical_trajectories)
            self.dataset.access_policy = AccessPolicy.BOTH
            self.dataset.alpha = self.alpha

            synthetic_results_2 = self.synthetic_trainer.train(
                self.dataset,
                num_epochs=self.synthetic_epochs,
                start_epoch=(self.warmup_cycles + cycle) * self.synthetic_epochs + self.synthetic_epochs,
                verbose=verbose
            )

            cycle_time = time.time() - cycle_start

            results["cycles"].append(cycle + 1)
            results["synthetic_losses"].append(synthetic_results_2["final_loss"])
            results["physical_losses"].append(physical_results["final_loss"])
            results["cycle_times"].append(cycle_time)

        logger.info("\n" + "="*60)
        logger.info("Hybrid Training Completed")
        logger.info("="*60)

        return results

    def _generate_synthetic_data(self) -> List[Tensor]:
        """
        Generate augmented trajectories using the synthetic model.

        Returns:
            List of trajectory tensors
        """
        trajectories = []

        # Generate trajectories from each training simulation
        for sim_idx in self.config['trainer']['train_sim']:
            # Load simulation data
            sim_path = os.path.join(
                self.config['data']['data_dir'],
                f"sim_{sim_idx:04d}.npz"
            )
            sim_data = math.load(sim_path)

            # Take first state as initial condition
            initial_state = sim_data.time[0]

            # Rollout using synthetic model (no batch dimension needed for single rollout)
            trajectory = self._rollout_synthetic_single(
                initial_state,
                self.config['data']['trajectory_length']
            )

            trajectories.append(trajectory)

        logger.info(f"Generated {len(trajectories)} synthetic trajectories")
        return trajectories

    def _generate_physical_data(self) -> List[Tensor]:
        """
        Generate augmented trajectories using the physical model.

        Returns:
            List of trajectory tensors
        """
        trajectories = []

        # Generate trajectories from each training simulation
        for sim_idx in self.config['trainer']['train_sim']:
            # Load simulation data
            sim_path = os.path.join(
                self.config['data']['data_dir'],
                f"sim_{sim_idx:04d}.npz"
            )
            sim_data = math.load(sim_path)

            # Take first state as initial condition
            initial_state = sim_data.time[0]

            # Add batch dimension using stack (more robust than expand)
            initial_state_batched = math.stack([initial_state], batch('batch'))

            # Rollout using physical model with current learned params
            trajectory_batched = self.physical_model.rollout(
                initial_state_batched,
                self.physical_model.params,
                self.config['data']['trajectory_length'] - 1
            )

            # Remove batch dimension
            trajectory = trajectory_batched.batch[0]

            # Detach from computational graph
            trajectory = math.stop_gradient(trajectory)

            trajectories.append(trajectory)

        logger.info(f"Generated {len(trajectories)} physical trajectories")
        return trajectories

    def _rollout_synthetic_single(self, initial_state: Tensor, num_steps: int) -> Tensor:
        """
        Rollout the synthetic model for multiple steps (single trajectory, no batch).

        Args:
            initial_state: Initial state tensor (x, y?, field)
            num_steps: Number of steps to rollout

        Returns:
            Trajectory tensor (time, x, y?, field)
        """
        states = [initial_state]
        current_state = initial_state

        for _ in range(num_steps - 1):
            next_state = self.synthetic_model(current_state)
            states.append(next_state)
            current_state = next_state

        # Stack along time dimension
        trajectory = math.stack(states, batch('time'))

        # Detach from computational graph to prevent gradient issues
        trajectory = math.stop_gradient(trajectory)

        return trajectory

    def _save_augmented_data(self, trajectories: List[Tensor], prefix: str):
        """
        Save augmented trajectories to cache directory.

        Args:
            trajectories: List of trajectory tensors
            prefix: Prefix for filenames
        """
        for i, trajectory in enumerate(trajectories):
            cache_path = self.cache_dir / f"{prefix}_sim_{i:04d}.npz"
            math.save(str(cache_path), trajectory)

        logger.info(f"Saved {len(trajectories)} augmented trajectories to {self.cache_dir}")
