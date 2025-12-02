"""PhiML trainer for synthetic models with LR scheduling and weighted loss support."""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
from phi.torch.flow import *
from phiml import math as phimath
from phiml.nn import update_weights
from phiml import nn as phiml_nn

from src.utils.logger import get_logger
from src.data.sparsity import SparsityConfig, SpatialMask

logger = get_logger(__name__)


class SyntheticTrainer:
    """
    PhiML trainer for synthetic models with weighted loss support.
    
    Supports separate weighting for real data loss (L) and interaction loss (I)
    for hybrid training: total_loss = real_weight * L + interaction_weight * I
    """

    def __init__(self, config: Dict[str, Any], model, sparsity_config: SparsityConfig = None):
        self.model = model
        self.config = config

        self._parse_config(config)
        self._setup_optimizer()
        self._setup_scheduler()

        self.best_val_loss = float("inf")
        self.best_epoch = 0

        # Loss scaling for hybrid training
        self.real_loss_weight = 1.0
        self.interaction_loss_weight = 1.0
        self.proportional_scaling = False

        # Spatial sparsity setup
        self.sparsity_config = sparsity_config or SparsityConfig()
        self._spatial_mask = None  # Lazy initialization

        logger.info(f"SyntheticTrainer ready: lr={self.learning_rate}, scheduler={self.scheduler_type}")

    def _parse_config(self, config: Dict[str, Any]):
        """Parse training configuration."""
        synthetic_config = config['trainer']['synthetic']
        self.epochs = synthetic_config['epochs']
        self.learning_rate = synthetic_config['learning_rate']
        self.batch_size = config['trainer']['batch_size']
        self.rollout_steps = synthetic_config.get('rollout_steps', config['trainer'].get('rollout_steps', 4))
        
        # Rollout scheduler config
        self.rollout_scheduler = synthetic_config.get('rollout_scheduler', None)
        if self.rollout_scheduler:
            self.rollout_start = self.rollout_scheduler.get('start', 1)
            self.rollout_end = self.rollout_scheduler.get('end', self.rollout_steps)
            self.rollout_strategy = self.rollout_scheduler.get('strategy', 'linear')
            self.rollout_exponent = float(self.rollout_scheduler.get('exponent', 2.0))
            logger.info(f"Rollout scheduler: {self.rollout_start} -> {self.rollout_end} ({self.rollout_strategy})")
            self.rollout_steps = self.rollout_start
        
        # Total epochs for scheduling (may be overridden for hybrid training)
        self.rollout_total_epochs = self.epochs
        
        self.scheduler_type = synthetic_config.get('scheduler', 'cosine')

        # Checkpoint path
        model_path_dir = config["model"]["synthetic"]["model_path"]
        model_save_name = config["model"]["synthetic"]["model_save_name"]
        self.checkpoint_path = Path(model_path_dir) / f"{model_save_name}"
        os.makedirs(model_path_dir, exist_ok=True)

    def _setup_optimizer(self):
        """Setup PhiML optimizer."""
        self.optimizer = phiml_nn.adam(self.model.network, learning_rate=self.learning_rate)
        
        # if hasattr(torch, 'compile'):
        #     logger.info("Compiling model network with torch.compile...")
        #     self.model.network = torch.compile(self.model.network)

    def _setup_scheduler(self):
        """Setup LR scheduler."""
        if self.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        elif self.scheduler_type == 'step':
            self.scheduler = StepLR(self.optimizer, step_size=self.epochs // 3, gamma=0.1)
        elif self.scheduler_type == 'exponential':
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.99)
        else:
            self.scheduler = None

    def _get_spatial_mask(self, spatial_shape: Shape) -> Optional[SpatialMask]:
        """Get or create spatial mask."""
        if self._spatial_mask is None and self.sparsity_config.spatial.enabled:
            self._spatial_mask = SpatialMask(self.sparsity_config.spatial, spatial_shape)
            logger.info(f"Spatial mask initialized: {self._spatial_mask.visible_fraction:.1%} visible")
        return self._spatial_mask

    def _get_rollout_steps(self, epoch: int) -> int:
        """Calculate rollout steps based on current epoch."""
        if not self.rollout_scheduler:
            return self.rollout_steps

        # Use rollout_total_epochs for progress calculation
        total_epochs = self.rollout_total_epochs
        
        # Calculate progress (0 to 1)
        progress = min(epoch / max(total_epochs, 1), 1.0)
        
        # Apply strategy
        if self.rollout_strategy == 'exponential':
            progress = progress ** self.rollout_exponent

        # Calculate number of steps in the schedule
        num_steps = self.rollout_end - self.rollout_start + 1
        
        # Calculate current step index
        step_index = int(progress * num_steps)

        # Calculate current rollout steps
        current_rollout = self.rollout_start + step_index
        
        return min(current_rollout, self.rollout_end)

    def set_loss_scaling(self, real_weight: float, interaction_weight: float, 
                         proportional: bool = False):
        """Configure loss scaling for hybrid training."""
        self.real_loss_weight = real_weight
        self.interaction_loss_weight = interaction_weight
        self.proportional_scaling = proportional

    def set_total_epochs_for_hybrid(self, total_epochs: int):
        """
        Configure total epochs for both rollout and LR scheduling across hybrid cycles.
        
        Args:
            total_epochs: Total epochs across all hybrid cycles
        """
        self.rollout_total_epochs = total_epochs
        
        # Recreate LR scheduler with correct T_max
        if self.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_epochs)
        elif self.scheduler_type == 'step':
            self.scheduler = StepLR(self.optimizer, step_size=total_epochs // 3, gamma=0.1)
        elif self.scheduler_type == 'exponential':
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.99)
        
        logger.info(f"Reconfigured for hybrid training: {self.scheduler_type} with total_epochs={total_epochs}")

    def _train_batch(self, separated_batch) -> float:
        """
        Train on a batch with separate real/generated loss weighting.

        Computes: total_loss = real_weight * L + interaction_weight * I
        where L is loss on real data and I is loss on generated data.
        """
        rollout_steps = self.rollout_steps

        # Get spatial mask if enabled (only initialize once)
        spatial_mask = None
        if separated_batch.has_real:
            spatial_mask = self._get_spatial_mask(separated_batch.real_initial_state.shape.spatial)

        def compute_loss(init_state, targets, apply_spatial_mask: bool = False):
            """Compute MSE loss with optional spatial masking."""
            current_state = init_state
            total_loss = phimath.tensor(0.0)
            for t in range(rollout_steps):
                next_state = self.model(current_state)
                target_t = targets.time[t]

                # Apply spatial mask ONLY if requested (for real data)
                if apply_spatial_mask and spatial_mask is not None:
                    step_loss = spatial_mask.compute_masked_mse(next_state, target_t)
                else:
                    step_loss = phimath.mean((next_state - target_t)**2)

                total_loss += step_loss
                current_state = next_state
            return phimath.mean(total_loss, 'batch') / float(rollout_steps)
    
        def combined_loss_function():
            real_loss = phimath.tensor(0.0)
            interaction_loss = phimath.tensor(0.0)
            
            if separated_batch.has_real:
                # Apply spatial mask for real data
                real_loss = compute_loss(
                    separated_batch.real_initial_state,
                    separated_batch.real_targets,
                    apply_spatial_mask=True
                )
            
            if separated_batch.has_generated:
                # NO spatial mask for generated/interaction data
                interaction_loss = compute_loss(
                    separated_batch.generated_initial_state,
                    separated_batch.generated_targets,
                    apply_spatial_mask=False
                )
            
            # Apply proportional scaling if enabled
            i_weight = self.interaction_loss_weight
            if self.proportional_scaling and separated_batch.has_real and separated_batch.has_generated:
                ratio = phimath.stop_gradient(real_loss / (interaction_loss + 1e-8))
                i_weight = self.interaction_loss_weight * ratio
            
            return self.real_loss_weight * real_loss + i_weight * interaction_loss
        
        loss = update_weights(self.model, self.optimizer, combined_loss_function)
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        return loss

    def train(self, dataset, num_epochs: int, start_epoch: int = 0, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute training for specified number of epochs.
        
        Args:
            dataset: Dataset instance with iterate_batches method
            num_epochs: Number of epochs to train
            start_epoch: Starting epoch number (for hybrid training across cycles)
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary with training results
        """
        results = {
            "train_losses": [],
            "epochs": [],
            "num_epochs": num_epochs,
            "best_epoch": 0,
            "best_val_loss": float("inf"),
        }

        pbar = tqdm(
            range(start_epoch, start_epoch + num_epochs), 
            desc="Synthetic Training", 
            unit="epoch", 
            disable=not verbose
        )

        for epoch in pbar:
    
            # Update rollout steps if scheduler is active
            if self.rollout_scheduler:
                new_rollout_steps = self._get_rollout_steps(epoch)
                if new_rollout_steps != self.rollout_steps:
                    self.rollout_steps = new_rollout_steps
                    self.best_val_loss = float("inf")

            start_time = time.time()
            epoch_loss = 0.0
            num_batches = 0

            for separated_batch in dataset.iterate_batches(self.batch_size, shuffle=True):
                batch_loss = self._train_batch(separated_batch)
                epoch_loss += batch_loss
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else epoch_loss
            results["train_losses"].append(float(avg_epoch_loss))
            results["epochs"].append(epoch + 1)

            loss_value = float(avg_epoch_loss)
            if loss_value < self.best_val_loss:
                self.best_val_loss = loss_value
                self.best_epoch = epoch + 1
                results["best_epoch"] = self.best_epoch
                results["best_val_loss"] = self.best_val_loss
                self.save_checkpoint(epoch=epoch, loss=avg_epoch_loss)

            epoch_time = time.time() - start_time
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.learning_rate
            pbar.set_postfix({
                "loss": f"{loss_value:.6f}",
                "best": f"{self.best_val_loss:.6f}",
                "lr": f"{current_lr:.2e}",
                "rollout": f"{self.rollout_steps}",
                "time": f"{epoch_time:.2f}s"
            })

        results["final_loss"] = results["train_losses"][-1] if results["train_losses"] else 0.0
        return results

    def save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint."""
        self.model.save(str(self.checkpoint_path))

    def load_checkpoint(self, path: Path = None):
        """Load model checkpoint."""
        path = path or self.checkpoint_path
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        self.model.load(str(path))
        logger.info(f"Loaded checkpoint from {path}")