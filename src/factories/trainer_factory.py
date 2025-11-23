"""Factory for creating trainers with Phase 1 API."""

from typing import Dict, Any, List
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from phi.math import math, Tensor

from src.training.abstract_trainer import AbstractTrainer
from src.training.synthetic.trainer import SyntheticTrainer
from src.training.physical.trainer import PhysicalTrainer

# HybridTrainer imported lazily to avoid circular dependency
from src.factories.model_factory import ModelFactory
from src.factories.dataloader_factory import DataLoaderFactory
from src.utils.logger import get_logger
import warnings
from src.training.hybrid import HybridTrainer

logger = get_logger(__name__)


class TrainerFactory:
    """
    Factory for creating trainer instances with Phase 1 API.

    Phase 1: Creates models and data externally, passes to trainers.
    Trainers now receive:
    - SyntheticTrainer(config, model)
    - PhysicalTrainer(config, model, learnable_params)
    - HybridTrainer(config, synthetic_model, physical_model, learnable_params)
    """

    _trainers = {
        "synthetic": SyntheticTrainer,
        "physical": PhysicalTrainer,
        # "hybrid" added lazily to avoid circular import
    }

    @staticmethod
    def list_available_trainers() -> List[str]:
        """Get list of available trainer types."""
        return ["synthetic", "physical", "hybrid"]

    @staticmethod
    def create_trainer(config: Dict[str, Any], num_channels: int = None) -> AbstractTrainer:
        """
        Create trainer from config with Phase 1 API.

        Args:
            config: Configuration dictionary
            num_channels: Number of channels (required for synthetic, from dataset.num_channels)

        Returns:
            Trainer instance (AbstractTrainer subclass)

        Raises:
            ValueError: If model_type is unknown
        """
        model_type = config["general"]["mode"]
        # Create trainer based on type
        if model_type == "synthetic":
            if num_channels is None:
                raise ValueError("num_channels is required for synthetic training")
            return TrainerFactory._create_synthetic_trainer(config, num_channels)
        elif model_type == "physical":
            return TrainerFactory._create_physical_trainer(config)
        elif model_type == "hybrid":
            return TrainerFactory.create_hybrid_trainer(config, num_channels)
        else:
            raise ValueError(f"Unknown model_type '{model_type}'")

    @staticmethod
    def _create_synthetic_trainer(config: Dict[str, Any], num_channels: int) -> SyntheticTrainer:
        """
        Create SyntheticTrainer with external model.

        Args:
            config: Full configuration dictionary
            num_channels: Number of input/output channels (from dataset.num_channels)

        Returns:
            SyntheticTrainer instance
        """
        # Create model with num_channels from dataset
        model = ModelFactory.create_synthetic_model(config, num_channels=num_channels)

        # Create trainer with model
        trainer = SyntheticTrainer(config, model)

        return trainer

    # @staticmethod
    def _create_physical_trainer(config: Dict[str, Any]) -> PhysicalTrainer:
        """
        Create PhysicalTrainer with external model and learnable parameters.

        Args:
            config: Full configuration dictionary

        Returns:
            PhysicalTrainer instance
        """
        # Create model externally
        downsample_factor = config["trainer"]["physical"]["downsample_factor"]
        model = ModelFactory.create_physical_model(config, downsample_factor=downsample_factor)


    
        # Create trainer with model and params
        trainer = PhysicalTrainer(config, model)

        return trainer

    @staticmethod
    def create_hybrid_trainer(config: Dict[str, Any], num_channels: int = None):
        """
        Create a hybrid trainer that alternates between synthetic and physical training.

        Args:
            config: Full configuration dictionary
            num_channels: Number of channels (from dataset.num_channels). If None, will
                          be inferred by creating a dataset.

        Returns:
            HybridTrainer instance configured with both models
        """
        # Get num_channels from dataset if not provided
        if num_channels is None:
            dataset = DataLoaderFactory.create_phiml(
                config, sim_indices=config['trainer']['train_sim']
            )
            num_channels = dataset.num_channels

        # Create physical model first (needed for 'auto' static_fields inference)
        physical_model = ModelFactory.create_physical_model(config)

        # Create synthetic model (can infer static_fields from physical model if config says 'auto')
        synthetic_model = ModelFactory.create_synthetic_model(
            config, num_channels=num_channels, physical_model=physical_model
        )

        # Create hybrid trainer
        hybrid_trainer = HybridTrainer(
            config=config,
            synthetic_model=synthetic_model,
            physical_model=physical_model,
        )

        return hybrid_trainer

    @staticmethod
    def register_trainer(name: str, trainer_class: type):
        """
        Register a new trainer type.

        Args:
            name: Name to register the trainer under
            trainer_class: Trainer class to register
        """
        TrainerFactory._trainers[name] = trainer_class
