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

from src.training.abstract_trainer import AbstractTrainer


class TensorTrainer(AbstractTrainer):
    """
    Base class for PyTorch tensor-based trainers.

    Provides all PyTorch-specific functionality:
    - Device management (CPU/GPU)
    - Model checkpoint saving/loading
    - Model parameter counting and summary
    - Common epoch-based training loop structure

    Subclasses must implement:
    - _create_model(): Create and initialize the PyTorch model
    - _create_data_loader(): Create the PyTorch DataLoader
    - _train_epoch(): Train for one epoch and return loss

    The train() method can be overridden if needed, but a default
    implementation is provided for standard epoch-based training.

    Attributes:
        config: Full configuration dictionary
        device: PyTorch device (CPU or CUDA)
        model: The PyTorch neural network model
        optimizer: PyTorch optimizer
        dataloader: PyTorch DataLoader
        checkpoint_path: Path to model checkpoint file
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
        self.dataloader: Optional[DataLoader] = None
        self.checkpoint_path: Optional[Path] = None

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
    def _create_data_loader(self) -> DataLoader:
        """
        Create and return the PyTorch DataLoader.

        This method should:
        1. Get data specifications from self.config
        2. Create or get dataset instance
        3. Create DataLoader with appropriate settings
        4. Set self.dataloader
        5. Return the DataLoader

        Returns:
            PyTorch DataLoader instance
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

    def train(self) -> Dict[str, Any]:
        """
        Execute epoch-based training loop.

        Default implementation of standard epoch-based training.
        Subclasses can override for custom training logic.

        Returns:
            Dictionary with training results including losses and metrics
        """
        if self.model is None or self.dataloader is None:
            raise RuntimeError(
                "Model and dataloader must be initialized before training"
            )

        results = {"losses": [], "epochs": []}

        print(f"\n{'='*60}")
        print(f"Starting Training on {self.device}")
        print(f"{'='*60}\n")

        for epoch in range(self.get_num_epochs()):
            epoch_loss = self._train_epoch()

            results["losses"].append(epoch_loss)
            results["epochs"].append(epoch)

            # Print progress
            if (epoch + 1) % self.get_print_frequency() == 0:
                print(
                    f"Epoch [{epoch+1}/{self.get_num_epochs()}], Loss: {epoch_loss:.6f}"
                )

            # Save checkpoint (if enabled)
            checkpoint_freq = self.get_checkpoint_frequency()
            if checkpoint_freq > 0 and (epoch + 1) % checkpoint_freq == 0:
                self.save_checkpoint(
                    epoch=epoch,
                    loss=epoch_loss,
                    optimizer_state=(
                        self.optimizer.state_dict() if self.optimizer else None
                    ),
                )

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Final Loss: {results['losses'][-1]:.6f}")
        print(f"{'='*60}\n")

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
        print(f"Saved checkpoint to {self.checkpoint_path}")

        # Save best checkpoint if specified
        if is_best:
            best_path = Path(self.checkpoint_path).parent / "best.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint to {best_path}")

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

        print(f"Loaded checkpoint from {path}")
        if "epoch" in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if "loss" in checkpoint:
            print(f"  Loss: {checkpoint['loss']:.6f}")

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
            print("Model not initialized")
            return

        total_params = self.get_parameter_count()
        trainable_params = self.get_trainable_parameter_count()

        print("\n" + "=" * 60)
        print("MODEL SUMMARY")
        print("=" * 60)
        print(f"Model type: {self.model.__class__.__name__}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        print(f"Device: {self.device}")
        print("=" * 60 + "\n")

    def move_model_to_device(self):
        """Move model to the configured device (CPU or CUDA)."""
        if self.model is not None:
            self.model = self.model.to(self.device)
            print(f"Model moved to {self.device}")

    def set_train_mode(self):
        """Set model to training mode."""
        if self.model is not None:
            self.model.train()

    def set_eval_mode(self):
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()
