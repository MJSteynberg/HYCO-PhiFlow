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
from src.data import DataManager, HybridDataset
from src.factories.model_factory import ModelFactory
from src.data.augmentation import (
    AdaptiveAugmentedDataLoader,
    CacheManager,
    generate_and_cache_predictions,
)
from src.config import AugmentationConfig, create_cache_path, get_augmentation_summary
from src.utils.logger import get_logger

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
        
        This method checks if augmentation is enabled in the config and creates
        either a standard DataLoader or an AdaptiveAugmentedDataLoader accordingly.
        
        Args:
            config: Full configuration dictionary
            sim_indices: Simulation indices to load (defaults to train_sim from config)
            batch_size: Batch size (defaults to config batch_size)
            shuffle: Whether to shuffle data
            use_sliding_window: Whether to use sliding window (default True for Phase 1)
            
        Returns:
            DataLoader (or AdaptiveAugmentedDataLoader) with HybridDataset
        """
        data_config = config["data"]
        trainer_config = config["trainer_params"]
        model_config = config["model"]["synthetic"]
        
        # Use provided values or defaults from config
        if sim_indices is None:
            sim_indices = trainer_config["train_sim"]
        if batch_size is None:
            batch_size = trainer_config["batch_size"]
        
        # Setup paths
        project_root = Path(config.get("project_root", "."))
        raw_data_dir = project_root / data_config["data_dir"] / data_config["dset_name"]
        cache_dir = project_root / data_config["data_dir"] / "cache"
        
        # Create DataManager
        data_manager = DataManager(
            raw_data_dir=str(raw_data_dir),
            cache_dir=str(cache_dir),
            config=config,
            validate_cache=data_config.get("validate_cache", True),
            auto_clear_invalid=data_config.get("auto_clear_invalid", False),
        )
        
        # Extract field specifications
        field_names = data_config["fields"]
        input_specs = model_config["input_specs"]
        output_specs = model_config["output_specs"]
        
        dynamic_fields = list(output_specs.keys())
        static_fields = [f for f in input_specs.keys() if f not in output_specs]
        
        num_predict_steps = trainer_config["num_predict_steps"]
        
        # Calculate frames needed
        if use_sliding_window:
            num_frames = None  # Load all available frames
        else:
            num_frames = num_predict_steps + 1
        
        # Create base dataset
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=sim_indices,
            field_names=field_names,
            num_frames=num_frames,
            num_predict_steps=num_predict_steps,
            dynamic_fields=dynamic_fields,
            static_fields=static_fields,
            use_sliding_window=use_sliding_window,
        )
        
        # Check if augmentation is enabled
        augmentation_config = trainer_config.get("augmentation", {})
        aug_config = AugmentationConfig(augmentation_config)
        
        if not aug_config.enabled:
            # Standard DataLoader without augmentation
            logger.info("Creating standard DataLoader (augmentation disabled)")
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False,
            )
            return data_loader
        
        # Augmentation is enabled - use AdaptiveAugmentedDataLoader
        logger.info("Creating AdaptiveAugmentedDataLoader (augmentation enabled)")
        logger.info(get_augmentation_summary(aug_config))
        
        # Create cache path
        cache_root = config.get("cache", {}).get("root", "data/cache")
        cache_root = project_root / cache_root
        experiment_name = aug_config.get_cache_config().get("experiment_name", data_config["dset_name"])
        
        # Create CacheManager
        cache_manager = CacheManager(
            cache_root=str(cache_root),
            experiment_name=experiment_name,
            auto_create=config.get("cache", {}).get("auto_create", True),
        )
        
        # Create AdaptiveAugmentedDataLoader
        strategy = aug_config.get_strategy()
        
        # Determine strategy and prepare accordingly
        if strategy == "cached":
            # Check if cache exists, if not warn user
            if not cache_manager.exists() or cache_manager.is_empty():
                logger.warning(
                    f"Cache directory is empty or doesn't exist: {cache_manager.cache_dir}\n"
                    f"No augmented data will be used. To use cached augmentation:\n"
                    f"1. Pre-generate cache using scripts/generate_cache.py, or\n"
                    f"2. Switch to 'on_the_fly' strategy in config"
                )
                # Fall back to standard DataLoader
                data_loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=0,
                    pin_memory=True if torch.cuda.is_available() else False,
                )
                return data_loader
            
            logger.info(f"Using cached augmentation strategy from {cache_manager.cache_dir}")
            logger.info(f"Cache contains {cache_manager.count_samples()} samples")
            
        elif strategy == "on_the_fly":
            logger.info("Using on-the-fly augmentation strategy (not yet fully implemented)")
            logger.warning("On-the-fly generation requires model - will use cached/memory strategy")
        
        # Create adaptive loader
        # Map config strategy names to AdaptiveAugmentedDataLoader strategy names
        strategy_map = {
            "cached": "cache",
            "on_the_fly": "on_the_fly",
        }
        loader_strategy = strategy_map.get(strategy, strategy)
        
        adaptive_loader = AdaptiveAugmentedDataLoader(
            real_dataset=dataset,
            alpha=aug_config.get_alpha(),
            generated_data=None,  # Will load from cache if strategy='cache'
            cache_dir=str(cache_manager.cache_dir) if strategy == "cached" else None,
            cache_size=aug_config.get_cache_config().get("max_memory_samples", 1000),
            strategy=loader_strategy,
            validate_count=True,
        )
        
        # Get the actual DataLoader
        loader = adaptive_loader.get_loader(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
        )
        
        # Log info about created loader
        info = adaptive_loader.get_info()
        logger.info(f"Adaptive loader info: {info}")
        
        return loader

    @staticmethod
    def create_dataset_for_physical(
        config: Dict[str, Any],
        sim_indices: List[int] = None,
        use_sliding_window: bool = True,
    ) -> HybridDataset:
        """
        Create HybridDataset for physical training (returns fields, not tensors).
        
        Args:
            config: Full configuration dictionary
            sim_indices: Simulation indices to load (defaults to train_sim from config)
            use_sliding_window: Whether to use sliding window (default True for Phase 1)
            
        Returns:
            HybridDataset with return_fields=True
        """
        data_config = config["data"]
        trainer_config = config["trainer_params"]
        
        # Use provided values or defaults from config
        if sim_indices is None:
            sim_indices = trainer_config["train_sim"]
        
        # Setup paths
        project_root = Path(config.get("project_root", "."))
        raw_data_dir = project_root / data_config["data_dir"] / data_config["dset_name"]
        cache_dir = project_root / data_config["data_dir"] / "cache"
        
        # Create DataManager
        data_manager = DataManager(
            raw_data_dir=str(raw_data_dir),
            cache_dir=str(cache_dir),
            config=config,
            validate_cache=data_config.get("validate_cache", True),
            auto_clear_invalid=data_config.get("auto_clear_invalid", False),
        )
        
        # Extract field specifications
        field_names = data_config["fields"]
        num_predict_steps = trainer_config["num_predict_steps"]
        num_frames = num_predict_steps + 1  # Initial state + targets
        
        # Create dataset with return_fields=True for PhiFlow training
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=sim_indices,
            field_names=field_names,
            num_frames=num_frames,
            num_predict_steps=num_predict_steps,
            return_fields=True,  # Return PhiFlow fields, not tensors
            use_sliding_window=use_sliding_window,
        )
        
        return dataset

    @staticmethod
    def generate_augmented_cache(
        config: Dict[str, Any],
        model: torch.nn.Module,
        model_type: str = 'synthetic',
        force_regenerate: bool = False,
    ) -> int:
        """
        Generate and cache augmented predictions for training.
        
        This method should be called before training with cached augmentation strategy
        to pre-populate the cache with model predictions.
        
        Args:
            config: Full configuration dictionary
            model: Trained model to generate predictions
            model_type: 'synthetic' or 'physical'
            force_regenerate: If True, clear existing cache and regenerate
            
        Returns:
            Number of samples generated and cached
            
        Example:
            >>> config = load_config()
            >>> model = load_pretrained_model()
            >>> num_cached = TrainerFactory.generate_augmented_cache(
            ...     config, model, model_type='synthetic'
            ... )
            >>> print(f"Cached {num_cached} predictions")
        """
        data_config = config["data"]
        trainer_config = config["trainer_params"]
        
        # Check if augmentation is enabled
        augmentation_config = trainer_config.get("augmentation", {})
        aug_config = AugmentationConfig(augmentation_config)
        
        if not aug_config.enabled:
            logger.warning("Augmentation is disabled in config, no cache to generate")
            return 0
        
        # Get augmentation parameters
        alpha = aug_config.get_alpha()
        device = aug_config.get_device()
        
        # Setup cache
        project_root = Path(config.get("project_root", "."))
        cache_root = config.get("cache", {}).get("root", "data/cache")
        cache_root = project_root / cache_root
        experiment_name = aug_config.get_cache_config().get("experiment_name", data_config["dset_name"])
        
        cache_manager = CacheManager(
            cache_root=str(cache_root),
            experiment_name=experiment_name,
            auto_create=True,
        )
        
        # Check if cache already exists
        if cache_manager.exists() and not cache_manager.is_empty() and not force_regenerate:
            num_existing = cache_manager.count_samples()
            logger.info(
                f"Cache already contains {num_existing} samples. "
                f"Use force_regenerate=True to overwrite."
            )
            return num_existing
        
        if force_regenerate and cache_manager.exists():
            logger.info("Force regenerate: clearing existing cache")
            cache_manager.clear_cache(confirm=True)
        
        # Create dataset for generation
        if model_type == "synthetic":
            # Create standard dataloader
            data_loader = TrainerFactory.create_data_loader_for_synthetic(
                config,
                shuffle=False,  # Don't shuffle for generation
            )
            # Extract dataset from dataloader
            real_dataset = data_loader.dataset
            
            # Generate predictions
            logger.info(f"Generating synthetic predictions with alpha={alpha}")
            num_cached = generate_and_cache_predictions(
                model=model,
                real_dataset=real_dataset,
                cache_manager=cache_manager,
                alpha=alpha,
                model_type='synthetic',
                device=device,
                batch_size=aug_config.get_on_the_fly_config().get('batch_size', 32),
                save_format=aug_config.get_cache_config().get('format', 'dict'),
            )
            
        elif model_type == "physical":
            # Create dataset for physical model
            dataset = TrainerFactory.create_dataset_for_physical(config)
            
            # Generate physical predictions
            logger.info(f"Generating physical predictions with alpha={alpha}")
            from src.data.augmentation import generate_physical_predictions
            
            inputs_list, targets_list = generate_physical_predictions(
                model=model,
                real_dataset=dataset,
                alpha=alpha,
                device=device,
                num_rollout_steps=aug_config.get_on_the_fly_config().get('rollout_steps', 10),
            )
            
            # Save to cache
            num_cached = len(inputs_list)
            save_format = aug_config.get_cache_config().get('format', 'dict')
            for idx, (inp, tgt) in enumerate(zip(inputs_list, targets_list)):
                cache_manager.save_sample(idx, inp, tgt, format=save_format)
            
            # Update metadata
            cache_manager.update_metadata({
                'alpha': alpha,
                'model_type': 'physical',
                'num_samples': num_cached,
                'rollout_steps': aug_config.get_on_the_fly_config().get('rollout_steps', 10),
            })
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        logger.info(f"Successfully cached {num_cached} predictions")
        
        # Validate cache
        validation = cache_manager.validate_cache(expected_count=num_cached)
        if not validation['valid']:
            logger.warning(f"Cache validation issues: {validation}")
        else:
            logger.info("Cache validation passed")
        
        return num_cached

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
