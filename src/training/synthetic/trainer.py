# src/training/synthetic/trainer.py

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any, List

# Import tensor trainer (new hierarchy)
from src.training.tensor_trainer import TensorTrainer
from torch.utils.data import DataLoader


# Import logging
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SyntheticTrainer(TensorTrainer):
    """
    Tensor-based trainer for synthetic models using DataManager pipeline.

    Uses TensorDataset for efficient cached data loading with no runtime
    Field conversions. All conversions happen once during caching.

    Inherits from TensorTrainer to get PyTorch-specific functionality.

    Phase 1 Migration: Now receives model externally, data passed via train().
    """

    def __init__(self, config: Dict[str, Any], model: nn.Module):
        """
        Initializes the trainer with external model.

        Args:
            config: Full configuration dictionary
            model: Pre-created synthetic model (e.g., UNet)
        """
        # Initialize base trainer with model
        super().__init__(config, model)

        # --- Loss function ---
        self.loss_fn = nn.MSELoss()  # Simple MSE for tensor-based training

    def _train_epoch(self, data_source: DataLoader) -> float:
        """
        OPTIMIZED: Removed redundant time tracking, cleaner structure.
        """
        self.model.train()
        total_loss = 0.0

        # Optional: Add progress bar
        # pbar = tqdm(data_source, desc="Training", leave=False)
        
        for batch in data_source:
            self.optimizer.zero_grad(set_to_none=True)  # More efficient

            # Compute loss using shared method
            loss = self._compute_batch_loss(batch)

            # AMP backward pass
            self.scaler.scale(loss).backward()
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        # Update scheduler (moved to base class logic if needed)
        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / len(data_source)

    def _compute_batch_loss(self, batch) -> torch.Tensor:
        """
        OPTIMIZED: Better memory management, clearer autoregressive loop.
        """
        initial_state, rollout_targets = batch
        
        # Non-blocking transfer
        initial_state = initial_state.to(self.device, non_blocking=True)
        rollout_targets = rollout_targets.to(self.device, non_blocking=True)

        # Enforce BVTS-only inputs: initial_state and rollout_targets must be
        # BVTS-shaped tensors: [B, V, 1, H, W] and [B, V, T, H, W]. If not,
        # raise a descriptive error to guide migration.
        if not (initial_state.dim() == 5 and rollout_targets.dim() == 5):
            raise ValueError(
                f"Expected BVTS tensors for SyntheticTrainer (initial: 5D, targets: 5D), "
                f"got initial.dim()={initial_state.dim()}, targets.dim()={rollout_targets.dim()}. "
                "Ensure datasets and DataManager produce BVTS-shaped tensors [B,V,T,H,W]."
            )

        num_steps = rollout_targets.shape[2]

        # BVTS autoregressive loop: current_state is [B, V, 1, H, W]
        with torch.amp.autocast(enabled=True, device_type=self.device.type):
            current_state = initial_state
            total_loss = 0.0

            for t in range(num_steps):
                # Predict next frame as BVTS [B, V, 1, H, W]
                prediction = self.model(current_state)

                # prediction: [B, V, 1, H, W] -> squeeze time dim for comparison
                pred_frame = prediction[:, :, 0]

                # target frame: rollout_targets[:, :, t] -> [B, V, H, W]
                target_frame = rollout_targets[:, :, t]

                # Accumulate loss (compare per-channel tensors)
                total_loss += self.loss_fn(pred_frame, target_frame)

                # Next input is the predicted frame (keep as [B,V,1,H,W])
                current_state = prediction

            avg_loss = total_loss / float(num_steps)

        return avg_loss
