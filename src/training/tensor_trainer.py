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

    Provides all PyTorch-specific functionality:
    - Device management (CPU/GPU)
    - Model checkpoint saving/loading
    - Model parameter counting and summary
    - Common epoch-based training loop structure
    - Validation support (optional)

    Subclasses must implement:
    - _create_model(): Create and initialize the PyTorch model
    - _create_data_loaders(): Create train and validation DataLoaders
    - _train_epoch(): Train for one epoch and return loss
    - _compute_batch_loss(): Compute loss for a single batch (used for validation)

    The train() method can be overridden if needed, but a default
    implementation is provided for standard epoch-based training with
    optional validation.

    Attributes:
        config: Full configuration dictionary
        device: PyTorch device (CPU or CUDA)
        model: The PyTorch neural network model
        optimizer: PyTorch optimizer
        train_loader: PyTorch DataLoader for training
        val_loader: PyTorch DataLoader for validation (optional)
        checkpoint_path: Path to model checkpoint file
        best_val_loss: Best validation loss seen so far
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize tensor trainer.

        Args:
            config: Full configuration dictionary containing all settings
        """
        super().__init__(config)

        # PyTorch-specific initialization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # To be set by subclasses
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.checkpoint_path: Optional[Path] = None

        # Validation state tracking
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """
        Create and return the PyTorch model.

        This method should:
        1. Get model specifications from self.config
        2. Instantiate the appropriate model class
        3. Move model to self.device if needed
        4. Set self.model
        5. Return the model

        Returns:
            PyTorch model instance
        """
        pass

    @abstractmethod
    def _create_data_loaders(self):
        """
        Create and return train and validation DataLoaders.

        This method should:
        1. Get data specifications from self.config
        2. Create training dataset and DataLoader
        3. Create validation dataset and DataLoader (if val_sim specified)
        4. Set self.train_loader and self.val_loader
        
        Note: val_loader can be None if no validation data specified.
        """
        pass

    @abstractmethod
    def _compute_batch_loss(self, batch) -> torch.Tensor:
        """
        Compute loss for a single batch.
        
        Used by both training and validation to ensure same loss computation.
        Subclasses define the batch structure and how to compute loss.
        
        Args:
            batch: A batch from the DataLoader (structure defined by subclass)
            
        Returns:
            Loss tensor for the batch
        """
        pass

    @abstractmethod
    def _train_epoch(self) -> float:
        """
        Train for one epoch.

        This method should:
        1. Iterate through batches from self.dataloader
        2. Perform forward pass
        3. Compute loss
        4. Perform backward pass and optimization
        5. Return average epoch loss

        Returns:
            Average loss for the epoch
        """
        pass

    def _validate_epoch(self) -> Optional[float]:
        """
        Run validation for one epoch.
        
        Returns validation loss if validation data exists, None otherwise.
        Uses the same loss computation as training (_compute_batch_loss).
        
        Returns:
            Average validation loss, or None if no validation data
        """
        if self.val_loader is None or len(self.val_loader) == 0:
            return None
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                loss = self._compute_batch_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        avg_val_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_val_loss

    def train(self) -> Dict[str, Any]:
        """
        Execute epoch-based training loop with optional validation.

        Default implementation of standard epoch-based training.
        If val_loader exists, runs validation and tracks best model.
        Subclasses can override for custom training logic.

        Returns:
            Dictionary with training results including losses and metrics
        """
        if self.model is None or self.train_loader is None:
            raise RuntimeError(
                "Model and train_loader must be initialized before training"
            )

        results = {
            "train_losses": [],
            "val_losses": [],
            "epochs": [],
            "best_epoch": 0,
            "best_val_loss": float('inf')
        }
        
        has_validation = self.val_loader is not None and len(self.val_loader) > 0
        validate_every = self.config["trainer_params"].get("validate_every", 1)

        logger.info(f"Training on {self.device}")
        if has_validation:
            logger.debug(f"Validation every {validate_every} epoch(s)")

        # Create progress bar for epochs
        pbar = tqdm(range(self.get_num_epochs()), desc="Training", unit="epoch")

        for epoch in pbar:
            start_time = time.time()
            
            # Training
            train_loss = self._train_epoch()
            results["train_losses"].append(train_loss)
            results["epochs"].append(epoch + 1)

            # Validation (if enabled and at right frequency)
            val_loss = None
            if has_validation and (epoch + 1) % validate_every == 0:
                val_loss = self._validate_epoch()
                results["val_losses"].append(val_loss)
                
                # Track best model based on validation loss
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch + 1
                    results["best_epoch"] = self.best_epoch
                    results["best_val_loss"] = self.best_val_loss
                    
                    # Save best model (only if checkpoint_path is set)
                    if self.checkpoint_path is not None:
                        self.save_checkpoint(
                            epoch=epoch,
                            loss=val_loss,
                            optimizer_state=self.optimizer.state_dict() if self.optimizer else None,
                            is_best=True
                        )

            epoch_time = time.time() - start_time
            
            # Update progress bar with train/val loss and time
            postfix_dict = {
                "train_loss": f"{train_loss:.6f}",
                "time": f"{epoch_time:.2f}s"
            }
            
            if val_loss is not None:
                postfix_dict["val_loss"] = f"{val_loss:.6f}"
                
            if self.best_epoch > 0:
                postfix_dict["best_epoch"] = self.best_epoch
            
            pbar.set_postfix(postfix_dict)

            # Periodic checkpoint (if not using best_only)
            save_best_only = self.config["trainer_params"].get("save_best_only", True)
            checkpoint_freq = self.get_checkpoint_frequency()
            
            if not save_best_only and checkpoint_freq > 0 and (epoch + 1) % checkpoint_freq == 0:
                self.save_checkpoint(
                    epoch=epoch,
                    loss=train_loss,
                    optimizer_state=(
                        self.optimizer.state_dict() if self.optimizer else None
                    ),
                )

        logger.info(f"Training Complete! Best Epoch: {results['best_epoch']}, " + 
                   (f"Val Loss: {results['best_val_loss']:.6f}, " if has_validation else "") +
                   f"Train Loss: {results['train_losses'][-1]:.6f}")

        return results

    def get_num_epochs(self) -> int:
        """Get number of training epochs from config."""
        return self.config["trainer_params"]["epochs"]

    def get_print_frequency(self) -> int:
        """Get how often to print progress (in epochs)."""
        return self.config["trainer_params"]["print_freq"]

    def get_checkpoint_frequency(self) -> int:
        """Get how often to save checkpoints (in epochs)."""
        return self.config["trainer_params"]["checkpoint_freq"]

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
