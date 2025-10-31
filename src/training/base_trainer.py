"""
Base Trainer Class

This module provides an abstract base class for all trainers,
containing shared functionality and enforcing a consistent interface.

Features:
- Device management (CPU/GPU)
- Model checkpoint saving/loading
- Model parameter counting
- Configuration management
- Consistent interface via abstract methods

The base trainer is meant to be subclassed by specific trainer implementations
like SyntheticTrainer and PhysicalTrainer.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import torch
import torch.nn as nn


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.
    
    Provides common functionality:
    - Model loading/saving
    - Device management
    - Config parsing
    - Checkpoint handling
    - Model summary information
    
    Subclasses must implement:
    - _create_model(): Create and initialize the model
    - _create_data_loader(): Create the data loader
    - _train_epoch(): Train for one epoch
    - train(): Main training loop
    
    Attributes:
        config: Full configuration dictionary
        device: PyTorch device (CPU or CUDA)
        model: The neural network model (set by subclass)
        checkpoint_path: Path to model checkpoint file
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base trainer.
        
        Args:
            config: Full configuration dictionary containing all settings
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # To be set by subclasses
        self.model: Optional[nn.Module] = None
        self.checkpoint_path: Optional[Path] = None
    
    @abstractmethod
    def _create_model(self):
        """
        Create model instance.
        
        Must be implemented by subclass to instantiate the appropriate model
        (synthetic or physical) and set self.model.
        
        Returns:
            Model instance
        """
        pass
    
    @abstractmethod
    def _create_data_loader(self):
        """
        Create data loader.
        
        Must be implemented by subclass to create the appropriate data loader
        for the training task.
        
        Returns:
            DataLoader instance
        """
        pass
    
    @abstractmethod
    def _train_epoch(self):
        """
        Train one epoch.
        
        Must be implemented by subclass to define the training logic for
        a single epoch (forward pass, loss computation, backward pass).
        
        Returns:
            Training metrics for the epoch (e.g., loss, accuracy)
        """
        pass
    
    @abstractmethod
    def train(self):
        """
        Main training loop.
        
        Must be implemented by subclass to orchestrate the full training
        process across all epochs.
        
        Returns:
            Training results/metrics
        """
        pass
    
    def save_checkpoint(
        self,
        epoch: int,
        loss: float,
        optimizer_state: Optional[Dict] = None,
        additional_info: Optional[Dict] = None,
        is_best: bool = False
    ):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            loss: Current loss value
            optimizer_state: Optimizer state dict (optional)
            additional_info: Any additional information to save (optional)
            is_best: If True, also save as 'best.pth'
        
        Raises:
            ValueError: If checkpoint_path is not set
        """
        if self.checkpoint_path is None:
            raise ValueError("checkpoint_path not set")
        
        # Build checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'config': self.config,
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        if additional_info is not None:
            checkpoint.update(additional_info)
        
        # Save regular checkpoint
        torch.save(checkpoint, self.checkpoint_path)
        
        # Save best checkpoint if specified
        if is_best:
            best_path = self.checkpoint_path.parent / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint to {best_path}")
    
    def load_checkpoint(
        self, 
        path: Optional[Path] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint. If None, uses self.checkpoint_path
            strict: Whether to strictly enforce that keys in checkpoint match
                   keys in model (passed to model.load_state_dict)
            
        Returns:
            Checkpoint dictionary containing epoch, loss, config, etc.
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        if path is None:
            path = self.checkpoint_path
        
        if path is None:
            raise ValueError("No checkpoint path provided")
        
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        print(f"Loaded checkpoint from {path}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'loss' in checkpoint:
            print(f"  Loss: {checkpoint['loss']:.6f}")
        
        return checkpoint
    
    def get_parameter_count(self) -> int:
        """
        Get total number of model parameters.
        
        Returns:
            Total parameter count
        """
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
    
    def get_trainable_parameter_count(self) -> int:
        """
        Get number of trainable parameters.
        
        Returns:
            Trainable parameter count
        """
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def print_model_summary(self):
        """
        Print model summary information.
        
        Displays:
        - Total parameters
        - Trainable parameters
        - Device (CPU/CUDA)
        """
        total_params = self.get_parameter_count()
        trainable_params = self.get_trainable_parameter_count()
        
        print("\n" + "="*60)
        print("MODEL SUMMARY")
        print("="*60)
        print(f"Model type: {self.model.__class__.__name__}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        print(f"Device: {self.device}")
        print("="*60 + "\n")
    
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
