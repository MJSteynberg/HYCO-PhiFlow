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
from src.data import DataManager
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
    def create_trainer(config: Dict[str, Any]) -> AbstractTrainer:
        """
        Create trainer from config with Phase 1 API.

        Args:
            config: Configuration dictionary

        Returns:
            Trainer instance (AbstractTrainer subclass)

        Raises:
            ValueError: If model_type is unknown
        """
        model_type = config["general"]["mode"]
        # Create trainer based on type
        if model_type == "synthetic":
            return TrainerFactory._create_synthetic_trainer(config)
        elif model_type == "physical":
            return TrainerFactory._create_physical_trainer(config)
        elif model_type == "hybrid":
            return TrainerFactory.create_hybrid_trainer(config)
        else:
            raise ValueError(f"Unknown model_type '{model_type}'")

    @staticmethod
    def _create_synthetic_trainer(config: Dict[str, Any]) -> SyntheticTrainer:
        """
        Create PhiMLSyntheticTrainer with external model.

        Now uses PhiML trainer which supports both PyTorch and PhiML models.

        Args:
            config: Full configuration dictionary

        Returns:
            PhiMLSyntheticTrainer instance
        """
        # Create model externally
        model = ModelFactory.create_synthetic_model(config)

        # Create PhiML trainer with model (auto-detects model type)
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
        model = ModelFactory.create_physical_model(config)


    
        # Create trainer with model and params
        trainer = PhysicalTrainer(config, model)

        return trainer

    @staticmethod
    def create_hybrid_trainer(config: Dict[str, Any]):
        """
        Create a hybrid trainer that alternates between synthetic and physical training.

        Args:
            config: Full configuration dictionary

        Returns:
            HybridTrainer instance configured with both models
        """
        # Create synthetic model
        synthetic_model = ModelFactory.create_synthetic_model(config)

        # Create physical model and learnable parameters
        physical_model = ModelFactory.create_physical_model(config)

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
