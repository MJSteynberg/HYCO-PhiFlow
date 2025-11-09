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

        # --- Loss function ---
        self.loss_fn = nn.MSELoss()  # Simple MSE for tensor-based training

        # --- AMP ---
        self.scaler = torch.amp.GradScaler(enabled=True)

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

    def _train_epoch(self, data_source):
        """
        Runs one epoch of autoregressive training using provided data source.

        Args:
            data_source: DataLoader with batches of (initial_state, rollout_targets)

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0


        for batch_idx, batch in enumerate(data_source):
            # Track batch start time

            self.optimizer.zero_grad()

            # Compute loss using shared method
            avg_rollout_loss = self._compute_batch_loss(batch)

            self.scaler.scale(avg_rollout_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += avg_rollout_loss.item()

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
        with torch.amp.autocast(enabled=True, device_type=self.device.type):
            initial_state, rollout_targets = batch
            initial_state = initial_state.to(self.device, non_blocking=True)
            rollout_targets = rollout_targets.to(self.device, non_blocking=True)

            # Autoregressive rollout
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

    # def _unpack_tensor_to_dict(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    #     """
    #     Slice a concatenated tensor into individual field tensors.

    #     Args:
    #         tensor: Tensor with all fields concatenated on channel dimension
    #                Shape: [B, C, H, W] or [B, T, C, H, W]

    #     Returns:
    #         Dictionary mapping field names to their tensor slices
    #     """
    #     output_dict = {}
    #     for field_name, (start_ch, end_ch) in self.channel_map.items():
    #         if tensor.dim() == 4:  # [B, C, H, W]
    #             output_dict[field_name] = tensor[:, start_ch:end_ch, :, :]
    #         elif tensor.dim() == 5:  # [B, T, C, H, W]
    #             output_dict[field_name] = tensor[:, :, start_ch:end_ch, :, :]
    #     return output_dict
