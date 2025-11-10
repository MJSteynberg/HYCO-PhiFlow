"""
DataLoader Factory

Simplified factory for creating data loaders with minimal configuration.
Replaces complex TrainerFactory data methods with a single, clear creation method.

This factory uses ConfigHelper to extract parameters and creates either
TensorDataset or FieldDataset based on the mode parameter.
"""

from typing import List, Optional, Literal, Union
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.data import DataManager, TensorDataset, FieldDataset
from src.config import ConfigHelper
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoaderFactory:
    """
    Factory for creating data loaders with minimal configuration.

    This replaces the complex TrainerFactory data methods with a
    single, clear creation method that uses ConfigHelper to extract
    all necessary parameters.

    Key simplifications:
    - Single creation method vs 4 separate methods
    - ConfigHelper handles all config extraction
    - Clear mode parameter: 'tensor' or 'field'
    - Consistent interface for all use cases

    Example:
        >>> from src.factories import DataLoaderFactory
        >>>
        >>> # Synthetic training (returns DataLoader)
        >>> loader = DataLoaderFactory.create(
        ...     config,
        ...     mode='tensor',
        ...     shuffle=True
        ... )
        >>>
        >>> # Physical training (returns Dataset)
        >>> dataset = DataLoaderFactory.create(
        ...     config,
        ...     mode='field',
        ...     batch_size=None
        ... )
    """

    @staticmethod
    def create(
        config: dict,
        mode: Literal["tensor", "field"] = "tensor",
        sim_indices: Optional[List[int]] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        enable_augmentation: Optional[bool] = None,
        num_workers: int = 0,
    ) -> Union[DataLoader, FieldDataset]:
        """
        Create a data loader or dataset for training.

        This method:
        1. Extracts configuration using ConfigHelper
        2. Creates DataManager
        3. Determines field specifications
        4. Creates appropriate dataset (Tensor or Field)
        5. Wraps in DataLoader (for tensor mode) or returns Dataset (for field mode)

        Args:
            config: Full configuration dictionary from Hydra
            mode: 'tensor' for synthetic models, 'field' for physical models
            sim_indices: Simulation indices to use (default: from config)
            batch_size: Batch size for DataLoader (default: from config, None for field mode)
            shuffle: Whether to shuffle data (default: True)
            use_sliding_window: Use sliding window (default: from config)
            enable_augmentation: Enable augmentation (default: from config)
            num_workers: Number of DataLoader workers (default: 0)

        Returns:
            - DataLoader: For 'tensor' mode (suitable for synthetic training)
            - FieldDataset: For 'field' mode (suitable for physical training)

        Raises:
            ValueError: If mode is invalid or configuration is invalid

        Example:
            >>> # Tensor mode (synthetic training)
            >>> loader = DataLoaderFactory.create(
            ...     config,
            ...     mode='tensor',
            ...     sim_indices=[0, 1, 2],
            ...     batch_size=16,
            ...     shuffle=True
            ... )
            >>> for initial, targets in loader:
            ...     # initial: [B, C_all, H, W]
            ...     # targets: [B, T, C_dynamic, H, W]
            ...     pass
            >>>
            >>> # Field mode (physical training)
            >>> dataset = DataLoaderFactory.create(
            ...     config,
            ...     mode='field',
            ...     sim_indices=[0, 1, 2],
            ...     batch_size=None,  # Physical models don't use batching
            ... )
            >>> for initial_fields, target_fields in dataset:
            ...     # initial_fields: Dict[str, Field]
            ...     # target_fields: Dict[str, List[Field]]
            ...     pass
        """
        logger.debug(f"Creating data loader (mode={mode})...")

        # === Step 1: Extract configuration using ConfigHelper ===
        cfg = ConfigHelper(config)

        # Validate configuration
        issues = cfg.validate()
        if issues:
            raise ValueError(
                f"Invalid configuration:\n"
                + "\n".join(f"  - {issue}" for issue in issues)
            )

        # Get parameters (use provided values or fall back to config)
        sim_indices = (
            sim_indices if sim_indices is not None else cfg.get_train_sim_indices()
        )
        batch_size = batch_size if batch_size is not None else cfg.get_batch_size()
        enable_augmentation = (
            enable_augmentation
            if enable_augmentation is not None
            else cfg.is_augmentation_enabled()
        )

        num_frames = None
        num_predict_steps = cfg.get_num_predict_steps()

        logger.debug(f"  Simulations: {len(sim_indices)}")
        logger.debug(f"  Batch size: {batch_size}")
        logger.debug(f"  Augmentation: {enable_augmentation}")

        # === Step 2: Create DataManager ===
        data_manager = DataLoaderFactory._create_data_manager(config, cfg)

        # === Step 3: Get field specifications ===
        field_names = cfg.get_field_names()

        # === Step 4: Get augmentation config ===
        augmentation_config = None
        if enable_augmentation:
            augmentation_config = cfg.get_augmentation_config()
            logger.debug(f"  Augmentation mode: {augmentation_config['mode']}")
            logger.debug(f"  Augmentation alpha: {augmentation_config['alpha']}")

        # === Step 5: Create dataset based on mode ===
        if mode == "tensor":
            # Tensor mode: for synthetic (neural network) models
            dynamic_fields, static_fields = cfg.get_field_types()

            logger.debug(f"  Dynamic fields: {dynamic_fields}")
            logger.debug(f"  Static fields: {static_fields}")

            dataset = TensorDataset(
                data_manager=data_manager,
                sim_indices=sim_indices,
                field_names=field_names,
                num_frames=num_frames,
                num_predict_steps=num_predict_steps,
                dynamic_fields=dynamic_fields,
                static_fields=static_fields,
                augmentation_config=augmentation_config,
            )

            # Wrap in DataLoader for batching
            logger.debug(f"  Created TensorDataset with {len(dataset)} samples")
            logger.debug(
                f"  Creating DataLoader (batch_size={batch_size}, shuffle={shuffle})..."
            )

            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )

            logger.debug(f"DataLoader created successfully")
            return data_loader

        elif mode == "field":
            # Field mode: for physical (PDE-based) models
            logger.debug(f"  Fields: {field_names}")

            dataset = FieldDataset(
                data_manager=data_manager,
                sim_indices=sim_indices,
                field_names=field_names,
                num_frames=num_frames,
                num_predict_steps=num_predict_steps,
                augmentation_config=augmentation_config,
            )

            # Return dataset directly (no DataLoader for field mode)
            logger.debug(f"  Created FieldDataset with {len(dataset)} samples")
            logger.debug(f"FieldDataset created successfully")
            return dataset

        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'tensor' or 'field'.")

    @staticmethod
    def _create_data_manager(config: dict, cfg: ConfigHelper) -> DataManager:
        """
        Create DataManager from configuration.

        Args:
            config: Full configuration dictionary
            cfg: ConfigHelper instance

        Returns:
            Configured DataManager instance

        Note: Cache creation and validation are always enabled (hardcoded).
        """
        # Get paths
        project_root = cfg.get_project_root()
        raw_data_dir = project_root / cfg.get_raw_data_dir()
        cache_dir = project_root / cfg.get_cache_dir()

        # Get auto-clear setting
        auto_clear_invalid = cfg.should_auto_clear_invalid()

        logger.debug(f"  Raw data: {raw_data_dir}")
        logger.debug(f"  Cache: {cache_dir}")

        return DataManager(
            raw_data_dir=str(raw_data_dir),
            cache_dir=str(cache_dir),
            config=config,
            auto_clear_invalid=auto_clear_invalid,
        )

    @staticmethod
    def create_for_evaluation(
        config: dict,
        mode: Literal["tensor", "field"] = "tensor",
        sim_indices: Optional[List[int]] = None,
    ) -> Union[DataLoader, FieldDataset]:
        """
        Create a data loader/dataset for evaluation.

        Convenience method that uses evaluation-specific defaults:
        - No shuffling
        - No augmentation
        - Uses validation sim indices if not specified

        Args:
            config: Full configuration dictionary
            mode: 'tensor' or 'field'
            sim_indices: Simulation indices (default: validation sims from config)

        Returns:
            DataLoader or FieldDataset configured for evaluation
        """
        cfg = ConfigHelper(config)

        # Use validation sims if not specified
        if sim_indices is None:
            sim_indices = cfg.get_val_sim_indices()
            if not sim_indices:
                logger.warning("No validation sims specified, using train sims")
                sim_indices = cfg.get_train_sim_indices()

        return DataLoaderFactory.create(
            config=config,
            mode=mode,
            sim_indices=sim_indices,
            shuffle=False,  # Don't shuffle for evaluation
            enable_augmentation=False,  # No augmentation for evaluation
        )

    @staticmethod
    def get_info(config: dict) -> dict:
        """
        Get information about what data loader would be created.

        Useful for debugging and validation without actually creating the loader.

        Args:
            config: Full configuration dictionary

        Returns:
            Dictionary with data loader configuration information
        """
        cfg = ConfigHelper(config)

        return {
            "dataset_name": cfg.get_dataset_name(),
            "model_type": cfg.get_model_type(),
            "field_names": cfg.get_field_names(),
            "train_sims": cfg.get_train_sim_indices(),
            "val_sims": cfg.get_val_sim_indices(),
            "batch_size": cfg.get_batch_size(),
            "num_predict_steps": cfg.get_num_predict_steps(),
            "use_sliding_window": cfg.should_use_sliding_window(),
            "augmentation_enabled": cfg.is_augmentation_enabled(),
            "augmentation_config": cfg.get_augmentation_config(),
        }
