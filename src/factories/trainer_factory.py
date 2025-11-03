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
from src.data import DataManager, FieldDataset
from src.factories.model_factory import ModelFactory
from src.factories.dataloader_factory import DataLoaderFactory
from src.utils.logger import get_logger
import warnings

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
        model_type = config["run_params"]["model_type"]

        available = TrainerFactory.list_available_trainers()
        if model_type not in available:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Available: {available}"
            )

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
        Create SyntheticTrainer with external model.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            SyntheticTrainer instance
        """
        # Create model externally
        model = ModelFactory.create_synthetic_model(config)
        
        # Create trainer with model
        trainer = SyntheticTrainer(config, model)
        
        return trainer

    @staticmethod
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
        
        # Extract learnable parameters from config
        learnable_params_config = config["trainer_params"].get("learnable_parameters", [])
        
        if not learnable_params_config:
            raise ValueError("No 'learnable_parameters' defined in trainer_params for physical training.")
        
        # Create learnable parameter tensors
        learnable_params: List[Tensor] = []
        for param in learnable_params_config:
            name = param["name"]
            initial_guess = param["initial_guess"]
            # Wrap in PhiFlow Tensor
            learnable_params.append(math.tensor(initial_guess))
        
        # Create trainer with model and params
        trainer = PhysicalTrainer(config, model, learnable_params)
        
        return trainer

    @staticmethod
    def create_data_loader_for_synthetic(
        config: Dict[str, Any],
        sim_indices: List[int] = None,
        batch_size: int = None,
        shuffle: bool = True,
        use_sliding_window: bool = True,
    ) -> DataLoader:
        """
        Create DataLoader for synthetic training with optional augmentation.
        
        DEPRECATED: Use DataLoaderFactory.create(config, mode='tensor') instead.
        This method is kept for backward compatibility but will be removed in a future version.
        
        Args:
            config: Full configuration dictionary
            sim_indices: Simulation indices to load (defaults to train_sim from config)
            batch_size: Batch size (defaults to config batch_size)
            shuffle: Whether to shuffle data
            use_sliding_window: Whether to use sliding window (default True for Phase 1)
            
        Returns:
            DataLoader with TensorDataset
        """
        warnings.warn(
            "create_data_loader_for_synthetic is deprecated. "
            "Use DataLoaderFactory.create(config, mode='tensor') instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Use new DataLoaderFactory
        return DataLoaderFactory.create(
            config=config,
            mode='tensor',
            sim_indices=sim_indices,
            batch_size=batch_size,
            shuffle=shuffle,
            use_sliding_window=use_sliding_window,
        )

    @staticmethod
    def create_dataset_for_physical(
        config: Dict[str, Any],
        sim_indices: List[int] = None,
        use_sliding_window: bool = True,
    ) -> FieldDataset:
        """
        Create FieldDataset for physical training (returns fields, not tensors).
        
        DEPRECATED: Use DataLoaderFactory.create(config, mode='field') instead.
        This method is kept for backward compatibility but will be removed in a future version.
        
        Args:
            config: Full configuration dictionary
            sim_indices: Simulation indices to load (defaults to train_sim from config)
            use_sliding_window: Whether to use sliding window (default True for Phase 1)
            
        Returns:
            FieldDataset (new simplified version)
        """
        warnings.warn(
            "create_dataset_for_physical is deprecated. "
            "Use DataLoaderFactory.create(config, mode='field') instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Use new DataLoaderFactory which returns FieldDataset directly
        return DataLoaderFactory.create(
            config=config,
            mode='field',
            sim_indices=sim_indices,
            use_sliding_window=use_sliding_window,
            batch_size=None,  # Physical training doesn't use batching
        )

    @staticmethod
    def generate_augmented_cache(
        config: Dict[str, Any],
        model: torch.nn.Module,
        model_type: str = 'synthetic',
        force_regenerate: bool = False,
    ) -> int:
        """
        Generate and cache augmented predictions for training.
        
        DEPRECATED: This method is deprecated and not yet updated for the new architecture.
        Cache generation is now handled differently in hybrid training.
        
        Args:
            config: Full configuration dictionary
            model: Trained model to generate predictions
            model_type: 'synthetic' or 'physical'
            force_regenerate: If True, clear existing cache and regenerate
            
        Returns:
            Number of samples generated and cached
        """
        warnings.warn(
            "generate_augmented_cache is deprecated and not yet updated for the new architecture. "
            "Cache generation is now handled differently in hybrid training.",
            DeprecationWarning,
            stacklevel=2
        )
        raise NotImplementedError(
            "generate_augmented_cache is not yet implemented in the new architecture. "
            "Augmentation is now handled directly via augmentation_config in datasets."
        )

    @staticmethod
    def create_hybrid_trainer(config: Dict[str, Any]):
        """
        Create a hybrid trainer that alternates between synthetic and physical training.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            HybridTrainer instance configured with both models
        """
        from src.training.hybrid import HybridTrainer
        
        logger.info("Creating hybrid trainer...")
        
        # Create synthetic model
        synthetic_model = ModelFactory.create_synthetic_model(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        synthetic_model.to(device)
        
        # Create physical model and learnable parameters
        physical_model = ModelFactory.create_physical_model(config)
        
        # Extract learnable parameters from config (same as _create_physical_trainer)
        learnable_params_config = config["trainer_params"].get("learnable_parameters", [])
        
        # Create learnable parameter tensors (empty list if none defined, e.g., for advection)
        learnable_params: List[Tensor] = []
        for param in learnable_params_config:
            name = param["name"]
            initial_guess = param["initial_guess"]
            # Wrap in PhiFlow Tensor
            learnable_params.append(math.tensor(initial_guess))
        
        # Create hybrid trainer
        hybrid_trainer = HybridTrainer(
            config=config,
            synthetic_model=synthetic_model,
            physical_model=physical_model,
            learnable_params=learnable_params,
        )
        
        logger.info("Hybrid trainer created successfully")
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
