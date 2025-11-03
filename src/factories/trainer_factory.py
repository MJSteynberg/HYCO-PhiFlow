"""Factory for creating trainers with Phase 1 API."""

from typing import Dict, Any, List
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from phi.math import math, Tensor

from src.training.abstract_trainer import AbstractTrainer
from src.training.synthetic.trainer import SyntheticTrainer
from src.training.physical.trainer import PhysicalTrainer
from src.data import DataManager, HybridDataset
from src.factories.model_factory import ModelFactory


class TrainerFactory:
    """
    Factory for creating trainer instances with Phase 1 API.
    
    Phase 1: Creates models and data externally, passes to trainers.
    Trainers now receive:
    - SyntheticTrainer(config, model)
    - PhysicalTrainer(config, model, learnable_params)
    """

    _trainers = {
        "synthetic": SyntheticTrainer,
        "physical": PhysicalTrainer,
    }

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

        if model_type not in TrainerFactory._trainers:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Available: {list(TrainerFactory._trainers.keys())}"
            )

        # Create trainer based on type
        if model_type == "synthetic":
            return TrainerFactory._create_synthetic_trainer(config)
        elif model_type == "physical":
            return TrainerFactory._create_physical_trainer(config)
        elif model_type == "hybrid":
            return TrainerFactory._create_hybrid_trainer(config)
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
        Create DataLoader for synthetic training.
        
        Args:
            config: Full configuration dictionary
            sim_indices: Simulation indices to load (defaults to train_sim from config)
            batch_size: Batch size (defaults to config batch_size)
            shuffle: Whether to shuffle data
            use_sliding_window: Whether to use sliding window (default True for Phase 1)
            
        Returns:
            DataLoader with HybridDataset
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
        
        # Create dataset
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
        
        # Create DataLoader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        
        return data_loader

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
    def register_trainer(name: str, trainer_class: type):
        """
        Register a new trainer type.

        Args:
            name: Name to register the trainer under
            trainer_class: Trainer class to register
        """
        TrainerFactory._trainers[name] = trainer_class

    @staticmethod
    def list_available_trainers():
        """
        List all available trainer types.

        Returns:
            List of registered trainer names
        """
        return list(TrainerFactory._trainers.keys())

