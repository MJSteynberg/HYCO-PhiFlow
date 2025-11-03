"""
Tensor Trainer

This module provides the base class for PyTorch tensor-based trainers.
All PyTorch-specific functionality lives here, including:
- Device management (CPU/GPU)
- Model checkpointing
- Parameter counting
- Epoch-based training loop structure

This separates tensor-based concerns from field-based concerns,
providing a cleaner architecture than the previous BaseTrainer.
"""

from abc import abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

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

        # PyTorch-specific initialization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Store model and move to device (allow None for testing)
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = None
        
        # Create optimizer for the model (only if model exists)
        self.optimizer = self._create_optimizer() if model is not None else None
        
        # Checkpoint path (can be set by subclass)
        self.checkpoint_path: Optional[Path] = None

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
        learning_rate = self.config.get("trainer_params", {}).get("learning_rate", 0.001)
        return torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    @abstractmethod
    def _train_epoch_with_data(self, data_source: DataLoader) -> float:
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
                        Note: NO weights - all samples treated equally!

        Returns:
            Average loss for the epoch
        """
        pass

    def train(self, data_source: DataLoader, num_epochs: int) -> Dict[str, Any]:
        """
        Execute training for specified number of epochs with provided data.

        NEW SIGNATURE: Data is passed explicitly, not held internally.
        
        Args:
            data_source: PyTorch DataLoader yielding (input, target) tuples
            num_epochs: Number of epochs to train
        
        Returns:
            Dictionary with training results including losses and metrics
        """
        if self.model is None:
            raise RuntimeError("Model must be initialized before training")

        results = {
            "train_losses": [],
            "epochs": [],
            "num_epochs": num_epochs,
            "best_epoch": 0,
            "best_val_loss": float("inf")
        }

        logger.info(f"Training on {self.device} for {num_epochs} epochs")

        # Get checkpoint configuration
        save_best_only = self.config.get("trainer_params", {}).get("save_best_only", True)
        checkpoint_freq = self.config.get("trainer_params", {}).get("checkpoint_freq", 10)

        # Create progress bar for epochs
        pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")

        for epoch in pbar:
            start_time = time.time()
            
            # Training
            train_loss = self._train_epoch_with_data(data_source)
            results["train_losses"].append(train_loss)
            results["epochs"].append(epoch + 1)

            # Track best model (based on train loss if no validation)
            if train_loss < self.best_val_loss:
                self.best_val_loss = train_loss
                self.best_epoch = epoch + 1
                results["best_epoch"] = self.best_epoch
                results["best_val_loss"] = self.best_val_loss
                
                # Save best model (only if checkpoint_path is set)
                if self.checkpoint_path is not None:
                    self.save_checkpoint(
                        epoch=epoch,
                        loss=train_loss,
                        optimizer_state=self.optimizer.state_dict() if self.optimizer else None,
                        is_best=True
                    )

            epoch_time = time.time() - start_time
            
            # Update progress bar
            postfix_dict = {
                "train_loss": f"{train_loss:.6f}",
                "time": f"{epoch_time:.2f}s"
            }
            
            if self.best_epoch > 0:
                postfix_dict["best_epoch"] = self.best_epoch
            
            pbar.set_postfix(postfix_dict)

            # Periodic checkpoint (if not using best_only)
            if not save_best_only and checkpoint_freq > 0 and (epoch + 1) % checkpoint_freq == 0:
                if self.checkpoint_path is not None:
                    self.save_checkpoint(
                        epoch=epoch,
                        loss=train_loss,
                        optimizer_state=(
                            self.optimizer.state_dict() if self.optimizer else None
                        ),
                    )

        final_loss = results["train_losses"][-1]
        logger.info(f"Training Complete! Best Epoch: {results['best_epoch']}, Final Loss: {final_loss:.6f}")

        results["final_loss"] = final_loss
        return results

    # =========================================================================
    # PyTorch-Specific Utilities
    # =========================================================================

    def save_checkpoint(
        self,
        epoch: int,
        loss: float,
        optimizer_state: Optional[Dict] = None,
        additional_info: Optional[Dict] = None,
        is_best: bool = False,
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
        if self.checkpoint_path is None:
            raise ValueError("checkpoint_path not set")

        if self.model is None:
            raise RuntimeError("Model not initialized")

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

        # Save best checkpoint if specified
        if is_best:
            best_path = Path(self.checkpoint_path).parent / "best.pth"
            torch.save(checkpoint, best_path)
            logger.debug(f"Saved best checkpoint to {best_path}")

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
        if self.model is None:
            raise RuntimeError("Model not initialized")

        if path is None:
            path = self.checkpoint_path

        if path is None:
            raise ValueError("No checkpoint path provided")

        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        logger.debug(f"Loaded checkpoint from {path}")
        if "epoch" in checkpoint:
            logger.debug(f"  Epoch: {checkpoint['epoch']}")
        if "loss" in checkpoint:
            logger.debug(f"  Loss: {checkpoint['loss']:.6f}")

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
