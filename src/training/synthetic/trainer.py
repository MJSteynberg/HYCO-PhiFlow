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

        num_steps = rollout_targets.shape[1]
        
        # Use AMP context
        with torch.amp.autocast(enabled=True, device_type=self.device.type):
            current_state = initial_state
            total_loss = 0.0

            for t in range(num_steps):
                # Predict next state
                prediction = self.model(current_state)
                target = rollout_targets[:, t]

                # Accumulate loss
                total_loss += self.loss_fn(prediction, target)

                current_state = prediction

            # Average over timesteps
            avg_loss = total_loss / num_steps
        
        return avg_loss
