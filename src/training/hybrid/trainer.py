"""Hybrid trainer that alternates between synthetic and physical training with data augmentation."""

import os
import time
import copy
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import torch

from phi.torch.flow import *
from phi.math import math, Tensor

from src.training.synthetic.trainer import SyntheticTrainer
from src.training.physical.trainer import PhysicalTrainer
from src.data.dataset import AccessPolicy, Dataset
from src.data.sparsity import SparsityConfig, TemporalSparsityConfig, SpatialSparsityConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridTrainer:
    """
    Hybrid trainer that alternates between synthetic and physical model training
    with cross-model data augmentation using random initial conditions.

    Training procedure:
    1. Warmup phase: Train synthetic model on real data only
    2. Hybrid phase: For each cycle:
       - Generate random ICs from physical model
       - Rollout physical model from random ICs → physical trajectories
       - Train synthetic model with real + physical data (weighted loss)
       - Generate NEW random ICs from physical model
       - Rollout synthetic model from random ICs → synthetic trajectories
       - Train physical model with real + synthetic data (weighted loss)

    Loss formulation (HYCO paper):
    - Synthetic model minimizes: L + λ_syn * I
    - Physical model minimizes: L + λ_phy * I
    where L = real data loss, I = interaction (generated data) loss

    Key feature: Both models train on trajectories from the same IC distribution,
    ensuring consistent exploration of the state space.
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

        # Parse sparsity configuration
        self.sparsity_config = self._parse_sparsity_config(config)

        # Determine maximum rollout steps needed (for dataset creation)
        synthetic_config = config['trainer']['synthetic']
        rollout_scheduler = synthetic_config.get('rollout_scheduler', None)
        if rollout_scheduler:
            max_rollout_steps = rollout_scheduler.get('end', config['trainer'].get('rollout_steps', 4))
        else:
            max_rollout_steps = synthetic_config.get('rollout_steps', config['trainer'].get('rollout_steps', 4))

        # Create dataset with cache sized for all training sims and temporal sparsity
        self.dataset = Dataset(
            config=config,
            train_sim=config['trainer']['train_sim'],
            rollout_steps=max_rollout_steps,
            max_cached_sims=len(config['trainer']['train_sim']) + 2,  # Cache all + buffer
            temporal_sparsity=self.sparsity_config.temporal if self.sparsity_config else None
        )

        # Create individual trainers with sparsity config
        self.synthetic_trainer = SyntheticTrainer(config, synthetic_model, sparsity_config=self.sparsity_config)
        self.physical_trainer = PhysicalTrainer(config, physical_model, sparsity_config=self.sparsity_config)

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

        # Augmentation configuration
        self.num_augment_trajectories = self.augmentation_config.get('num_trajectories', 3)
        self.augment_trajectory_length = self.augmentation_config.get(
            'trajectory_length',
            config['data']['trajectory_length']
        )

        # New: Trajectory generation strategy for hybrid training
        # Full trajectory length for physical model (used to train synthetic)
        self.physical_trajectory_length = self.augmentation_config.get(
            'physical_trajectory_length',
            config['data']['trajectory_length']
        )
        # Short trajectory length for synthetic model (used to train physical)
        self.synthetic_trajectory_length = self.augmentation_config.get(
            'synthetic_trajectory_length',
            self.augment_trajectory_length
        )

        # Get epochs and batch size for each model
        self.synthetic_epochs = config['trainer']['synthetic']['epochs']
        self.physical_epochs = config['trainer']['physical']['epochs']
        self.batch_size = config['trainer']['batch_size']

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

        # Loss-based model reversion config
        reversion_config = hybrid_config.get('reversion', {})
        self.enable_reversion = reversion_config.get('enabled', False)
        self.revert_synthetic = reversion_config.get('synthetic', True)
        self.revert_physical = reversion_config.get('physical', True)
        # Minimum relative improvement required to keep new parameters (e.g., 0.01 = 1%)
        self.min_improvement = float(reversion_config.get('min_improvement', 0.0))

    def _parse_sparsity_config(self, config: Dict[str, Any]) -> SparsityConfig:
        """Parse sparsity configuration from Hydra config."""
        if 'sparsity' not in config:
            return SparsityConfig()

        sparsity = config['sparsity']

        temporal = TemporalSparsityConfig(**sparsity.get('temporal', {}))
        spatial = SpatialSparsityConfig(**sparsity.get('spatial', {}))

        sparsity_config = SparsityConfig(temporal=temporal, spatial=spatial)

        if temporal.enabled or spatial.enabled:
            logger.info("Sparsity configuration:")
            if temporal.enabled:
                logger.info(f"  Temporal: {temporal.mode} mode")
            if spatial.enabled:
                logger.info(f"  Spatial: {spatial.mode} mode")

        return sparsity_config

    # =========================================================================
    # Model State Management (for loss-based reversion)
    # =========================================================================

    def _save_synthetic_state(self) -> Dict[str, Any]:
        """Save the current state of the synthetic model's network."""
        # Use PyTorch's state_dict for network weights
        return copy.deepcopy(self.synthetic_model.network.state_dict())

    def _restore_synthetic_state(self, state: Dict[str, Any]):
        """Restore the synthetic model's network to a saved state."""
        self.synthetic_model.network.load_state_dict(state)

    def _save_physical_state(self) -> Tensor:
        """Save the current state of the physical model's parameters."""
        return math.stop_gradient(self.physical_model.params)

    def _restore_physical_state(self, state: Tensor):
        """Restore the physical model's parameters to a saved state."""
        self.physical_model.params = state

    def _evaluate_synthetic_loss(self) -> float:
        """
        Evaluate synthetic model loss on real data only (no training).

        Returns:
            Average MSE loss on real data
        """
        self.dataset.access_policy = AccessPolicy.REAL_ONLY
        self.dataset.alpha = 1.0

        # Disable training mode for evaluation (no noise injection)
        self.synthetic_model.training = False

        total_loss = 0.0
        num_batches = 0
        rollout_steps = self.synthetic_trainer.rollout_steps

        with torch.no_grad():
            for batch in self.dataset.iterate_batches(self.batch_size, shuffle=False):
                if not batch.has_real:
                    continue

                current_state = batch.real_initial_state
                batch_loss = 0.0

                for t in range(min(rollout_steps, batch.real_targets.shape.get_size('time'))):
                    next_state = self.synthetic_model(current_state)
                    target_t = batch.real_targets.time[t]
                    mse = math.mean((next_state - target_t) ** 2)
                    step_loss = float(math.mean(mse, 'batch').native().item())
                    batch_loss += step_loss
                    current_state = next_state

                total_loss += batch_loss / rollout_steps
                num_batches += 1

        # Restore dataset config
        self.dataset.access_policy = AccessPolicy.BOTH
        self.dataset.alpha = self.alpha

        return total_loss / max(num_batches, 1)

    def _evaluate_physical_loss(self) -> float:
        """
        Evaluate physical model loss on real data only (no training).

        Returns:
            Average MSE loss on real data
        """
        self.dataset.access_policy = AccessPolicy.REAL_ONLY
        self.dataset.alpha = 1.0

        total_loss = 0.0
        num_batches = 0
        rollout_steps = self.physical_trainer.rollout_steps

        with torch.no_grad():
            for batch in self.dataset.iterate_batches(self.batch_size, shuffle=False):
                if not batch.has_real:
                    continue

                # Physical model rollout
                trajectory = self.physical_model.rollout(
                    batch.real_initial_state,
                    self.physical_model.params,
                    rollout_steps
                )

                # Compute loss against targets
                batch_loss = 0.0
                for t in range(min(rollout_steps, batch.real_targets.shape.get_size('time'))):
                    pred_t = trajectory.time[t + 1]  # +1 because rollout includes initial
                    target_t = batch.real_targets.time[t]
                    mse = math.mean((pred_t - target_t) ** 2)
                    step_loss = float(math.mean(mse, 'batch').native().item())
                    batch_loss += step_loss

                total_loss += batch_loss / rollout_steps
                num_batches += 1

        # Restore dataset config
        self.dataset.access_policy = AccessPolicy.BOTH
        self.dataset.alpha = self.alpha

        return total_loss / max(num_batches, 1)

    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute hybrid training.

        If reversion is enabled, models are only updated if their loss on real
        data decreases after training. Otherwise, they revert to pre-cycle state.

        Returns:
            Dictionary with training results
        """
        results = {
            "cycles": [],
            "synthetic_losses": [],
            "physical_losses": [],
            "cycle_times": [],
            "synthetic_reversions": [],  # Track which cycles had reversions
            "physical_reversions": [],
        }

        logger.info("=" * 60)
        logger.info("Starting Hybrid Training")
        if self.enable_reversion:
            logger.info(f"Loss-based reversion ENABLED (synthetic={self.revert_synthetic}, physical={self.revert_physical})")
            if self.min_improvement > 0:
                logger.info(f"  Minimum improvement threshold: {self.min_improvement:.1%}")
        logger.info("=" * 60)

        # Configure schedulers to span ALL training (warmup + hybrid cycles)
        # This must be done BEFORE any training so LR decays properly over entire run
        total_synthetic_epochs = (self.warmup_cycles + self.cycles) * self.synthetic_epochs
        self.synthetic_trainer.set_total_epochs_for_hybrid(total_synthetic_epochs)

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

        # Reset best checkpoint tracking after warmup
        # This prevents warmup-era checkpoints (trained on real data only) from
        # interfering with hybrid phase evaluation (which uses augmented data too)
        self.synthetic_trainer.reset_best_checkpoint()

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

        # Track best losses and states for reversion decisions
        best_synthetic_loss = float('inf')
        best_synthetic_state = None
        best_physical_loss = float('inf')
        best_physical_state = None
        # Track rollout to reset best loss when it changes
        last_synthetic_rollout = self.synthetic_trainer.rollout_steps

        for cycle in range(self.cycles):
            logger.info(f"\n--- Hybrid Cycle {cycle + 1}/{self.cycles} ---")
            cycle_start = time.time()

            synthetic_reverted = False
            physical_reverted = False

            # =========================================================
            # SYNTHETIC MODEL TRAINING (with optional reversion)
            # =========================================================

            # Check if rollout changed - if so, reset best loss and state
            current_rollout = self.synthetic_trainer.rollout_steps
            if current_rollout != last_synthetic_rollout:
                logger.info(f"  Rollout changed ({last_synthetic_rollout} -> {current_rollout}), resetting best synthetic loss")
                best_synthetic_loss = float('inf')
                best_synthetic_state = None
                last_synthetic_rollout = current_rollout

            # Step 1: Generate physical trajectories for synthetic training
            initial_states = self._generate_random_initial_states(self.num_augment_trajectories)
            physical_trajectories = self._generate_physical_trajectories(initial_states)
            self.dataset.set_augmented_trajectories(physical_trajectories)

            if self.save_augmented:
                self._save_augmented_data(physical_trajectories, f"physical_cycle_{cycle}")

            # Train synthetic model (trainer keeps its own best checkpoint)
            synthetic_results = self.synthetic_trainer.train(
                self.dataset,
                num_epochs=self.synthetic_epochs,
                start_epoch=cycle * self.synthetic_epochs,
                verbose=verbose
            )

            # Check if we should revert synthetic model
            if self.enable_reversion and self.revert_synthetic:
                # Get best loss achieved during this cycle (from trainer's tracking)
                best_in_cycle_loss = synthetic_results.get('best_val_loss', synthetic_results['final_loss'])
                
                # Load the best-in-cycle checkpoint to get that state
                try:
                    self.synthetic_trainer.load_checkpoint()
                except FileNotFoundError:
                    pass  # No checkpoint, use current state
                
                # Evaluate loss with best-in-cycle state
                loss_after_synthetic = self._evaluate_synthetic_loss()
                logger.info(f"  Synthetic best-in-cycle loss: {loss_after_synthetic:.6f} (from epoch {synthetic_results.get('best_epoch', '?')})")

                if best_synthetic_loss == float('inf'):
                    # First cycle - just save as best
                    best_synthetic_loss = loss_after_synthetic
                    best_synthetic_state = self._save_synthetic_state()
                    logger.info(f"  Synthetic model KEPT (first baseline): {loss_after_synthetic:.6f}")
                elif loss_after_synthetic > best_synthetic_loss:
                    # Best-in-cycle is worse than global best - revert to global best
                    logger.warning(
                        f"  REVERTING synthetic model: best-in-cycle {loss_after_synthetic:.6f} > global best {best_synthetic_loss:.6f}"
                    )
                    self._restore_synthetic_state(best_synthetic_state)
                    synthetic_reverted = True
                else:
                    # Check if improvement meets threshold
                    improvement = (best_synthetic_loss - loss_after_synthetic) / (best_synthetic_loss + 1e-10)
                    if improvement < self.min_improvement:
                        logger.warning(
                            f"  REVERTING synthetic model: improvement {improvement:.2%} < threshold {self.min_improvement:.2%} "
                            f"({best_synthetic_loss:.6f} -> {loss_after_synthetic:.6f})"
                        )
                        self._restore_synthetic_state(best_synthetic_state)
                        synthetic_reverted = True
                    else:
                        # Update global best
                        best_synthetic_loss = loss_after_synthetic
                        best_synthetic_state = self._save_synthetic_state()
                        logger.info(f"  Synthetic model KEPT: loss improved by {improvement:.2%} to {loss_after_synthetic:.6f}")

            # =========================================================
            # PHYSICAL MODEL TRAINING (with optional reversion)
            # =========================================================

            # Step 2: Generate synthetic trajectories for physical training
            initial_states = self._generate_random_initial_states(self.num_augment_trajectories)
            synthetic_trajectories = self._generate_synthetic_trajectories(initial_states)
            self.dataset.set_augmented_trajectories(synthetic_trajectories)

            if self.save_augmented:
                self._save_augmented_data(synthetic_trajectories, f"synthetic_cycle_{cycle}")

            # Train physical model
            physical_results = self.physical_trainer.train(
                self.dataset,
                num_epochs=self.physical_epochs,
                verbose=verbose
            )
            
            # Check if we should revert physical model
            if self.enable_reversion and self.revert_physical:
                loss_after_physical = self._evaluate_physical_loss()
                logger.info(f"  Physical loss after training: {loss_after_physical:.6f}")

                if best_physical_loss == float('inf'):
                    # First cycle - just save as best
                    best_physical_loss = loss_after_physical
                    best_physical_state = self._save_physical_state()
                    logger.info(f"  Physical model KEPT (first baseline): {loss_after_physical:.6f}")
                elif loss_after_physical > best_physical_loss:
                    # Loss increased - revert to global best
                    logger.warning(
                        f"  REVERTING physical model: current {loss_after_physical:.6f} > global best {best_physical_loss:.6f}"
                    )
                    self._restore_physical_state(best_physical_state)
                    physical_reverted = True
                else:
                    # Check if improvement meets threshold
                    improvement = (best_physical_loss - loss_after_physical) / (best_physical_loss + 1e-10)
                    if improvement < self.min_improvement:
                        logger.warning(
                            f"  REVERTING physical model: improvement {improvement:.2%} < threshold {self.min_improvement:.2%} "
                            f"({best_physical_loss:.6f} -> {loss_after_physical:.6f})"
                        )
                        self._restore_physical_state(best_physical_state)
                        physical_reverted = True
                    else:
                        # Update global best
                        best_physical_loss = loss_after_physical
                        best_physical_state = self._save_physical_state()
                        logger.info(f"  Physical model KEPT: loss improved by {improvement:.2%} to {loss_after_physical:.6f}")

            # =========================================================
            # Record Results
            # =========================================================

            cycle_time = time.time() - cycle_start

            results["cycles"].append(cycle + 1)
            results["synthetic_losses"].append(synthetic_results["final_loss"])
            results["physical_losses"].append(physical_results["final_loss"])
            results["cycle_times"].append(cycle_time)
            results["synthetic_reversions"].append(synthetic_reverted)
            results["physical_reversions"].append(physical_reverted)

            # Summary log
            reversion_info = ""
            if self.enable_reversion:
                reversion_info = f" [S:{'REV' if synthetic_reverted else 'OK'}, P:{'REV' if physical_reverted else 'OK'}]"

            logger.info(
                f"Cycle {cycle + 1} completed in {cycle_time:.2f}s - "
                f"Synthetic loss: {synthetic_results['final_loss']:.6f}, "
                f"Physical loss: {physical_results['final_loss']:.6f}{reversion_info}"
            )

        logger.info("\n" + "=" * 60)
        logger.info("Hybrid Training Completed")
        logger.info("=" * 60)

        # Log reversion summary if enabled
        if self.enable_reversion:
            syn_revs = sum(results["synthetic_reversions"])
            phy_revs = sum(results["physical_reversions"])
            logger.info(f"Reversion summary: Synthetic={syn_revs}/{self.cycles}, Physical={phy_revs}/{self.cycles}")

        return results

    def _generate_random_initial_states(self, num_ics: int) -> List[Tensor]:
        """
        Generate random initial conditions from the physical model.

        Args:
            num_ics: Number of initial conditions to generate

        Returns:
            List of initial state tensors
        """
        batched_ics = self.physical_model.get_initial_state(batch_size=num_ics)
        return list(math.unstack(batched_ics, 'batch'))

    def _generate_physical_trajectories(self, initial_states: List[Tensor]) -> List[Tensor]:
        """
        Generate trajectories from physical model using given initial conditions.

        Args:
            initial_states: List of initial state tensors

        Returns:
            List of trajectory tensors (each with time dimension)
        """
        # Stack initial states into batch
        batched_initial = math.stack(initial_states, batch('batch'))

        # Rollout using physical model
        with torch.no_grad():
            trajectory_batched = self.physical_model.rollout(
                batched_initial,
                self.physical_model.params,
                self.physical_trajectory_length - 1
            )

        # Apply stop_gradient and unstack
        trajectory_batched = math.stop_gradient(trajectory_batched)

        return list(math.unstack(trajectory_batched, 'batch'))

    def _generate_synthetic_trajectories(self, initial_states: List[Tensor]) -> List[Tensor]:
        """
        Generate trajectories from synthetic model using given initial conditions.

        Args:
            initial_states: List of initial state tensors

        Returns:
            List of trajectory tensors (each with time dimension)
        """
        trajectories = []

        # Disable training mode for inference (no noise injection)
        self.synthetic_model.training = False

        with torch.no_grad():
            for initial_state in initial_states:
                states = [initial_state]
                current = initial_state

                # Rollout for trajectory_length - 1 steps
                for _ in range(self.synthetic_trajectory_length - 1):
                    next_state = self.synthetic_model(current)
                    states.append(next_state)
                    current = next_state

                # Stack along time dimension
                trajectory = math.stack(states, batch('time'))
                trajectory = math.stop_gradient(trajectory)
                trajectories.append(trajectory)

        return trajectories

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