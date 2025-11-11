"""
Training Module

This module provides the trainer hierarchy for HYCO-PhiFlow:

- AbstractTrainer: Minimal common interface for all trainers
- TensorTrainer: Base class for PyTorch tensor-based trainers
- FieldTrainer: Base class for PhiFlow field-based trainers

Concrete implementations:
- SyntheticTrainer: Trains synthetic (data-driven) models using tensors
- PhysicalTrainer: Trains physical models using field optimization
- HybridTrainer: Alternates between synthetic and physical training with augmentation
"""

from src.training.abstract_trainer import AbstractTrainer
from src.training.tensor_trainer import TensorTrainer
from src.training.field_trainer import FieldTrainer
from src.training.hybrid import HybridTrainer

__all__ = [
    "AbstractTrainer",
    "TensorTrainer",
    "FieldTrainer",
    "HybridTrainer",
]
