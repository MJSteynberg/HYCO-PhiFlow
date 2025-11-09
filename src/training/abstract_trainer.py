"""
Abstract Trainer

This module provides a minimal abstract base class that all trainers must implement.
This is the foundation of the new trainer hierarchy that separates concerns between
tensor-based (PyTorch) and field-based (PhiFlow) trainers.

Key Principle:
- Only includes functionality that ALL trainers need
- No PyTorch-specific code (that goes in TensorTrainer)
- No PhiFlow-specific code (that goes in FieldTrainer)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class AbstractTrainer(ABC):
    """
    Minimal interface that all trainers must implement.

    This abstract class defines only the essential contract that every trainer
    must fulfill, regardless of whether it's tensor-based, field-based, or hybrid.

    Design Philosophy:
    - Keep it minimal - only what's truly common
    - No assumptions about model type (PyTorch vs PhiFlow)
    - No assumptions about training paradigm (epochs vs optimization runs)
    - Provide clean extension points for different trainer types

    Attributes:
        config: Full configuration dictionary containing all settings
        project_root: Root directory of the project

    Subclasses:
        TensorTrainer: For PyTorch tensor-based models
        FieldTrainer: For PhiFlow field-based models
        HybridTrainer: For combined training strategies
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize abstract trainer.

        Args:
            config: Full configuration dictionary containing all settings.
                   This should include data, model, and trainer parameters.
        """
        self._config = config
        self._project_root = config.get("project_root", ".")

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Execute training and return results.

        This is the main entry point for training. Each trainer type implements
        its own training logic appropriate to its paradigm:
        - TensorTrainer: Epoch-based gradient descent
        - FieldTrainer: Optimization-based parameter inference
        - HybridTrainer: Coordinated training of multiple models

        Returns:
            Dictionary containing training results, metrics, and any other
            relevant information. The exact contents depend on the trainer type,
            but should typically include:
            - 'loss': Final or average loss value(s)
            - 'epochs' or 'iterations': Number of training steps completed
            - Any trainer-specific metrics
        """
        pass

    @property
    def config(self) -> Dict[str, Any]:
        """
        Return configuration used for this trainer.

        Useful for logging, debugging, and ensuring reproducibility.

        Returns:
            Configuration dictionary
        """
        return self._config
    
    @config.setter
    def config(self, new_config: Dict[str, Any]):
        """
        Set a new configuration for the trainer.

        Args:
            new_config: New configuration dictionary to replace the existing one.
        """
        self._config = new_config

    @property
    def project_root(self) -> str:
        """
        Get the project root directory.

        Returns:
            Project root path as string
        """
        return self._project_root
    
    @project_root.setter
    def project_root(self, new_root: str):
        """
        Set a new project root directory.

        Args:
            new_root: New project root path as string
        """
        self._project_root = new_root
