"""
Tensor Trainer

This module provides the base class for PyTorch tensor-based trainers.
All PyTorch-specific functionality lives here, including:
- Device management (CPU/GPU)
- Model checkpointing
- Parameter counting
- Epoch-based training loop structure
"""

from abc import abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os

from src.training.abstract_trainer import AbstractTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TensorTrainer(AbstractTrainer):
    """
    Base class for PyTorch tensor-based trainers.

    NEW ARCHITECTURE (Phase 1):
    - Model is passed in __init__, not created internally
    - Data is passed to train() method, not held internally
    - Trainers are persistent across training calls
    - Optimizer state is preserved

    Provides all PyTorch-specific functionality:
    - Device management (CPU/GPU)
    - Model checkpoint saving/loading
    - Model parameter counting and summary
    - Common epoch-based training loop structure
    - Validation support (optional)

    Subclasses should implement:
    - _train_epoch_with_data(): Train for one epoch using provided data_source
    - _compute_batch_loss(): Compute loss for a single batch (optional, for validation)

    The train() method accepts data explicitly and should not be overridden
    in most cases.

    Attributes:
        config: Full configuration dictionary
        device: PyTorch device (CPU or CUDA)
        model: The PyTorch neural network model (passed in __init__)
        optimizer: PyTorch optimizer
        checkpoint_path: Path to model checkpoint file
        best_val_loss: Best validation loss seen so far
    """

    def __init__(self, config: Dict[str, Any], model: nn.Module):
        """
        Initialize tensor trainer with model.

        Args:
            config: Full configuration dictionary containing all settings
            model: Pre-created PyTorch model
        """
        super().__init__(config)

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

        # PyTorch-specific initialization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Store model and move to device
        self.model = torch.compile(model.to(self.device))

        # Create optimizer for the model
        self.optimizer = self._create_optimizer()

        # --- AMP ---
        self._setup_amp_scaler(enabled=True)

        # --- Scheduler ---
        epochs = self.trainer_config["epochs"]
        self._setup_scheduler('cosine', T_max=epochs)

        # Validation state tracking
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer for the model.

        Can be overridden by subclasses for custom optimizer configuration.
        Default: Adam with learning rate from config.

        Returns:
            PyTorch optimizer instance
        """
        learning_rate = self.config['trainer_params']['learning_rate']
        return torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    @abstractmethod
    def _train_epoch(self, data_source: DataLoader) -> float:
        """
        Train for one epoch using provided data source.

        This method should:
        1. Iterate through batches from data_source
        2. Perform forward pass
        3. Compute loss
        4. Perform backward pass and optimization
        5. Return average epoch loss

        Args:
            data_source: DataLoader yielding (input, target) tuples

        Returns:
            Average loss for the epoch
        """
        pass

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

        # Only log if not suppressed in config
        if verbose:
            logger.info(f"Training on {self.device} for {num_epochs} epochs")

        # Create progress bar for epochs (disable if suppress_training_logs is True)
        disable_tqdm = not verbose
        pbar = tqdm(
            range(num_epochs), desc="Training", unit="epoch", disable=disable_tqdm
        )

        for epoch in pbar:
            start_time = time.time()

            # Training
            train_loss = self._train_epoch(data_source)
            results["train_losses"].append(train_loss)
            results["epochs"].append(epoch + 1)

            # Track best model (based on train loss if no validation)
            if train_loss < self.best_val_loss:
                self.best_val_loss = train_loss
                self.best_epoch = epoch + 1
                results["best_epoch"] = self.best_epoch
                results["best_val_loss"] = self.best_val_loss

                self.save_checkpoint(
                    epoch=epoch,
                    loss=train_loss,
                    optimizer_state=(
                        self.optimizer.state_dict() if self.optimizer else None
                    )
                )

            epoch_time = time.time() - start_time

            # Update progress bar
            postfix_dict = {
                "train_loss": f"{train_loss:.6f}",
                "time": f"{epoch_time:.2f}s",
            }

            postfix_dict["best_epoch"] = self.best_epoch

            pbar.set_postfix(postfix_dict)

        final_loss = results["train_losses"][-1]

        # Only log if not suppressed in config
        if verbose:
            logger.info(
                f"Training Complete! Best Epoch: {results['best_epoch']}, Final Loss: {final_loss:.6f}"
            )

        results["final_loss"] = final_loss
        return results
    
    # In TensorTrainer, add these methods:

    def _validate_epoch(self, data_source: DataLoader) -> float:
        """
        Run validation on provided data source.
        Subclasses must implement _compute_batch_loss().
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in data_source:
                loss = self._compute_batch_loss(batch)
                total_loss += loss.item()
        
        self.model.train()
        return total_loss / len(data_source)

    @abstractmethod
    def _compute_batch_loss(self, batch) -> torch.Tensor:
        """
        Compute loss for a single batch.
        Must be implemented by subclasses.
        """
        pass

    def _setup_amp_scaler(self, enabled: bool = True):
        """Setup automatic mixed precision scaler."""
        self.scaler = torch.amp.GradScaler(enabled=enabled)
        
    def _setup_scheduler(self, scheduler_type: str = 'cosine', **kwargs):
        """Setup learning rate scheduler."""
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, **kwargs
            )
        # Add other scheduler types as needed

    # =========================================================================
    # PyTorch-Specific Utilities
    # =========================================================================

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
            "config": self.config,
        }

        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state

        if additional_info is not None:
            checkpoint.update(additional_info)

        # Save regular checkpoint
        torch.save(checkpoint, self.checkpoint_path)
        logger.debug(f"Saved checkpoint to {self.checkpoint_path}")

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

    def move_model_to_device(self):
        """Move model to the configured device (CPU or CUDA)."""
        if self.model is not None:
            self.model = self.model.to(self.device)
            logger.info(f"Model moved to {self.device}")

    def set_train_mode(self):
        """Set model to training mode."""
        if self.model is not None:
            self.model.train()

    def set_eval_mode(self):
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()
