# src/training/synthetic/trainer.py

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from torch.utils.data import DataLoader


# Import logging
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SyntheticTrainer():
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
        super().__init__()

        # Parse configuration and setup trainer
        self.model = model
        self._parse_config(config)
        self._setup()
        # self.model = torch.compile(self.model.to(self.device))]
        self.model = self.model.to(self.device)

        # Validation state tracking
        self.best_val_loss = float("inf")
        self.best_epoch = 0

        # --- Loss function ---
        self.loss_fn = nn.MSELoss()  # Simple MSE for tensor-based training

        # --- Try to load checkpoint if exists ---
        if Path(self.checkpoint_path).exists():
            try:
                checkpoint = self.load_checkpoint(self.checkpoint_path)
                logger.info(
                    f"Loaded checkpoint from {self.checkpoint_path} at epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}"
                )
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

    def _parse_config(self, config):
        """
        Parse configuration for synthetic trainer.

        Extracts relevant parameters from the config dictionary
        and sets up internal variables.
        """

        # --- Data specifications ---
        self.field_names: List[str] = config["data"]["fields"]
        self.data_dir = config["data"]["data_dir"]
        self.dset_name = config["data"]["dset_name"]
        
        # --- Checkpoint path ---
        model_path_dir = config["model"]["synthetic"]["model_path"]
        model_save_name = config["model"]["synthetic"]["model_save_name"]
        
        self.checkpoint_path = Path(model_path_dir) / f"{model_save_name}.pth"
        os.makedirs(model_path_dir, exist_ok=True)

        # PyTorch-specific initialization
        self.device = torch.device(config['trainer']['device'] if torch.cuda.is_available() else "cpu")
        self.epochs = config['trainer']['synthetic']['epochs']
        self.learning_rate = config['trainer']['synthetic']['learning_rate']

    def _setup(self):

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scaler = torch.amp.GradScaler(enabled=True)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)


    ####################
    # Training Methods #
    ####################

    def train(self, data_source: DataLoader, num_epochs: int, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute training for specified number of epochs with provided data.

        Args:
            data_source: PyTorch DataLoader yielding (input, target) tuples
            num_epochs: Number of epochs to train

        Returns:
            Dictionary with training results including losses and metrics
        """
        results = {
            "train_losses": [],
            "epochs": [],
            "num_epochs": num_epochs,
            "best_epoch": 0,
            "best_val_loss": float("inf"),
        }

        # Create progress bar for epochs (disable if suppress_training_logs is True)
        disable_tqdm = not verbose
        pbar = tqdm(
            range(num_epochs), desc="Training", unit="epoch", disable=disable_tqdm
        )


        for epoch in pbar:
            start_time = time.time()
            # Training
            self.model.train()
            train_loss = 0.0

            for batch in data_source:
                self.optimizer.zero_grad(set_to_none=True)  # More efficient
                loss = self._compute_loss(batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                train_loss += loss.item()
            # Update scheduler (moved to base class logic if needed)
            if self.scheduler is not None:
                self.scheduler.step()

            results["train_losses"].append(train_loss / len(data_source))
            results["epochs"].append(epoch + 1)

            # Track best model (based on train loss if no validation)
            if train_loss / len(data_source) < self.best_val_loss:
                self.best_val_loss = train_loss / len(data_source)
                self.best_epoch = epoch + 1
                results["best_epoch"] = self.best_epoch
                results["best_val_loss"] = self.best_val_loss

                self.save_checkpoint(
                    epoch=epoch,
                    loss=train_loss / len(data_source),
                    optimizer_state=(
                        self.optimizer.state_dict() if self.optimizer else None
                    )
                )
            epoch_time = time.time() - start_time

            # Update progress bar
            postfix_dict = {
                "train_loss": f"{train_loss / len(data_source):.6f}",
                "time": f"{epoch_time:.2f}s",
            }

            postfix_dict["best_epoch"] = self.best_epoch

            pbar.set_postfix(postfix_dict)

        final_loss = results["train_losses"][-1]

        results["final_loss"] = final_loss
        return results

    def _compute_loss(self, batch) -> torch.Tensor:
        """
        OPTIMIZED: Better memory management, clearer autoregressive loop.
        """
        initial_state, rollout_targets = batch
        
        # Non-blocking transfer
        initial_state = initial_state.to(self.device, non_blocking=True)
        rollout_targets = rollout_targets.to(self.device, non_blocking=True)
        num_steps = rollout_targets.shape[2]
        # BVTS autoregressive loop: current_state is [B, V, 1, H, W]
        with torch.amp.autocast(enabled=True, device_type=self.device.type):
            current_state = initial_state
            total_loss = 0.0

            for t in range(num_steps):
                # Predict next frame as BVTS [B, V, 1, H, W]
                prediction = self.model(current_state)
                total_loss += self.loss_fn(prediction[:, :, 0], rollout_targets[:, :, t])
                current_state = prediction

            avg_loss = total_loss / float(num_steps)

        return avg_loss
    
    #############
    # Utilities #
    #############
    


    def save_checkpoint(
        self,
        epoch: int,
        loss: float,
        optimizer_state: Optional[Dict] = None,
        additional_info: Optional[Dict] = None,
    ):
        """
        Save PyTorch model checkpoint.

        Args:
            epoch: Current epoch number
            loss: Current loss value
            optimizer_state: Optimizer state dict (optional)
            additional_info: Any additional information to save (optional)
            is_best: If True, also save as 'best.pth'

        Raises:
            ValueError: If checkpoint_path is not set
            RuntimeError: If model is not initialized
        """

        # Build checkpoint dictionary
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "loss": loss,
        }

        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state

        if additional_info is not None:
            checkpoint.update(additional_info)

        torch.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(
        self, path: Optional[Path] = None, strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load PyTorch model checkpoint.

        Args:
            path: Path to checkpoint. If None, uses self.checkpoint_path
            strict: Whether to strictly enforce that keys in checkpoint match
                   keys in model (passed to model.load_state_dict)

        Returns:
            Checkpoint dictionary containing epoch, loss, config, etc.

        Raises:
            ValueError: If no checkpoint path provided
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If model is not initialized
        """

        if path is None:
            path = self.checkpoint_path

        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        return checkpoint

    def get_parameter_count(self) -> int:
        """
        Get total number of model parameters.

        Returns:
            Total parameter count, or 0 if model not initialized
        """
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())

    def get_trainable_parameter_count(self) -> int:
        """
        Get number of trainable parameters.

        Returns:
            Trainable parameter count, or 0 if model not initialized
        """
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def print_model_summary(self):
        """
        Print model summary information.

        Displays:
        - Model type
        - Total parameters
        - Trainable parameters
        - Non-trainable parameters
        - Device (CPU/CUDA)
        """
        if self.model is None:
            logger.warning("Model not initialized")
            return

        total_params = self.get_parameter_count()
        trainable_params = self.get_trainable_parameter_count()

        logger.info("=" * 60)
        logger.info("MODEL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Model type: {self.model.__class__.__name__}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
        logger.info(f"Device: {self.device}")
        logger.info("=" * 60)
