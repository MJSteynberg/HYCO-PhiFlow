# src/training/synthetic/trainer.py

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm

# Import tensor trainer (new hierarchy)
from src.training.tensor_trainer import TensorTrainer

# Import the model registry
from src.models import ModelRegistry

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

        # --- Derive all parameters from config ---
        self.data_config = config["data"]
        self.model_config = config["model"]["synthetic"]
        self.trainer_config = config["trainer_params"]

        # --- Data specifications ---
        self.field_names: List[str] = self.data_config["fields"]
        self.dset_name = self.data_config["dset_name"]
        self.data_dir = self.data_config["data_dir"]

        # --- Checkpoint path ---
        model_save_name = self.model_config["model_save_name"]
        model_path_dir = self.model_config["model_path"]
        self.checkpoint_path = Path(model_path_dir) / f"{model_save_name}.pth"
        os.makedirs(model_path_dir, exist_ok=True)
        logger.debug(f"Checkpoint path set to: {self.checkpoint_path}")

        # --- Loss function ---
        self.loss_fn = nn.MSELoss()  # Simple MSE for tensor-based training

        # --- Learning rate scheduler (optional, updates per epoch) ---
        use_scheduler = self.trainer_config.get("use_scheduler", True)
        if use_scheduler:
            epochs = self.trainer_config.get("epochs", 1)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs
            )
            logger.debug(
                f"Created CosineAnnealingLR scheduler with T_max={epochs} epochs"
            )
        else:
            self.scheduler = None

        # --- Memory monitoring (optional, enabled by config) ---
        enable_memory_monitoring = self.trainer_config.get(
            "enable_memory_monitoring", False
        )
        if enable_memory_monitoring:
            try:
                from src.utils.memory_monitor import EpochPerformanceMonitor

                verbose_batches = self.trainer_config["memory_monitor_batches"]
                self.memory_monitor = EpochPerformanceMonitor(
                    enabled=True,
                    verbose_batches=verbose_batches,
                    device=0 if torch.cuda.is_available() else -1,
                )
                logger.info(
                    f"Performance monitoring enabled (verbose for first {verbose_batches} batches)"
                )
            except ImportError:
                logger.warning(
                    "Could not import EpochPerformanceMonitor. Monitoring disabled."
                )
                self.memory_monitor = None
        else:
            self.memory_monitor = None

    def _train_epoch_with_data(self, data_source):
        """
        Runs one epoch of autoregressive training using provided data source.

        Args:
            data_source: DataLoader with batches of (initial_state, rollout_targets)

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0

        # Use memory monitor if available
        if self.memory_monitor is not None:
            self.memory_monitor.on_epoch_start()

        for batch_idx, batch in enumerate(data_source):
            # Track batch start time
            if self.memory_monitor is not None:
                self.memory_monitor.on_batch_start(batch_idx)

            self.optimizer.zero_grad()

            # Compute loss using shared method
            avg_rollout_loss = self._compute_batch_loss(batch)
            avg_rollout_loss.backward()

            self.optimizer.step()

            # Track batch completion with memory monitor
            if self.memory_monitor is not None:
                self.memory_monitor.on_batch_end(batch_idx, avg_rollout_loss.item())

            total_loss += avg_rollout_loss.item()

        # Track epoch completion
        if self.memory_monitor is not None:
            self.memory_monitor.on_epoch_end()

        # Update learning rate scheduler once per epoch (outside batch loop)
        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / len(data_source)
        return avg_loss

    def _compute_batch_loss(self, batch) -> torch.Tensor:
        """
        Compute loss for a single batch.

        Used by both training and validation through parent class.

        Args:
            batch: Tuple of (initial_state, rollout_targets) from TensorDataset

        Returns:
            Loss tensor for the batch
        """
        initial_state, rollout_targets = batch
        initial_state = initial_state.to(self.device)
        rollout_targets = rollout_targets.to(self.device)

        # Autoregressive rollout
        batch_size = initial_state.shape[0]
        num_steps = rollout_targets.shape[1]

        current_state = initial_state  # [B, C_all, H, W] - all fields
        total_step_loss = 0.0

        for t in range(num_steps):
            # Predict next state (model returns all fields)
            prediction = self.model(current_state)  # [B, C_all, H, W]

            target_all = rollout_targets[:, t, :, :, :]  # [B, C_all, H, W]

            step_loss = self.loss_fn(prediction, target_all)
            total_step_loss += step_loss

            # Use full prediction (all fields) as input for next timestep
            current_state = prediction

        # Average loss over timesteps
        avg_rollout_loss = total_step_loss / num_steps
        return avg_rollout_loss

    def _unpack_tensor_to_dict(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Slice a concatenated tensor into individual field tensors.

        Args:
            tensor: Tensor with all fields concatenated on channel dimension
                   Shape: [B, C, H, W] or [B, T, C, H, W]

        Returns:
            Dictionary mapping field names to their tensor slices
        """
        output_dict = {}
        for field_name, (start_ch, end_ch) in self.channel_map.items():
            if tensor.dim() == 4:  # [B, C, H, W]
                output_dict[field_name] = tensor[:, start_ch:end_ch, :, :]
            elif tensor.dim() == 5:  # [B, T, C, H, W]
                output_dict[field_name] = tensor[:, :, start_ch:end_ch, :, :]
        return output_dict
