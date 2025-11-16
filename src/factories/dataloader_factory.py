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
from src.utils.logger import get_logger
from src.data.dataset_utilities import field_collate_fn, tensor_collate_fn

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
        enable_augmentation: bool = False,
        num_workers: int = 0,
        percentage_real_data: float = 1.0,
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


        # === Step 2: Create DataManager ===
        data_manager = DataLoaderFactory._create_data_manager(config)

        # === Step 5: Create dataset based on mode ===
        if mode == "tensor":

            dataset = TensorDataset(config, data_manager, enable_augmentation)

            # Wrap in DataLoader for batching
            logger.debug(f"  Created TensorDataset with {len(dataset)} samples")
            logger.debug(
                f"  Creating DataLoader (batch_size={batch_size}, shuffle={shuffle})..."
            )

            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=tensor_collate_fn,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )

            logger.debug(f"DataLoader created successfully")
            return data_loader

        elif mode == "field":

            dataset = FieldDataset(config, data_manager, enable_augmentation)

            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=field_collate_fn,
            )
            logger.debug(f"DataLoader created successfully")
            return data_loader

        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'tensor' or 'field'.")

    @staticmethod
    def _create_data_manager(config: dict) -> DataManager:
        """
        Create DataManager from configuration.

        Args:
            config: Full configuration dictionary
            cfg: ConfigHelper instance

        Returns:
            Configured DataManager instance

        Note: Cache creation and validation are always enabled (hardcoded).
        """

        return DataManager(config)


