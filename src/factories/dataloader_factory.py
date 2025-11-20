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

from src.data import DataManager, Dataset
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
    def create_phiml(
        config: dict,
        sim_indices: Optional[List[int]] = None,
        field_names: Optional[List[str]] = None,
        num_frames: Optional[int] = None,
        rollout_steps: Optional[int] = None,
        percentage_real_data: float = 1.0,
        enable_augmentation: bool = False,
    ) -> Dataset:
        """
        Create a pure PhiML dataset for training.

        This method creates a Dataset that yields PhiML tensor batches
        directly - no PyTorch dependency.

        Args:
            config: Full configuration dictionary from Hydra
            sim_indices: Simulation indices to use (default: from config)
            field_names: Field names to load (default: from config)
            num_frames: Number of frames per simulation (default: from config)
            rollout_steps: Number of prediction steps (default: from config)
            percentage_real_data: Fraction of real data to use (0.0-1.0)
            enable_augmentation: Whether to enable augmented data

        Returns:
            Dataset instance that yields PhiML tensor batches

        Example:
            >>> dataset = DataLoaderFactory.create_phiml(
            ...     config,
            ...     sim_indices=[0, 1, 2],
            ...     rollout_steps=10
            ... )
            >>> for batch in dataset.iterate_batches(batch_size=16):
            ...     # batch['initial_state']: Tensor(batch=B, x=H, y=W, vector=V)
            ...     # batch['targets']: Tensor(batch=B, time=T, x=H, y=W, vector=V)
            ...     pass
        """
        logger.debug("Creating PhiML dataset...")

        # Create DataManager
        data_manager = DataManager(config)

        # Get parameters from config if not provided
        if sim_indices is None:
            sim_indices = list(range(config['data']['num_simulations']))

        if field_names is None:
            # Try to get from fields first
            if 'fields' in config['data']:
                field_names = config['data']['fields']
            elif isinstance(config['data'].get('fields_scheme'), dict):
                field_names = config['data']['fields_scheme']['dynamic']
            else:
                # Fallback - assume velocity for now
                field_names = ['velocity']

        if rollout_steps is None:
            rollout_steps = config['trainer']['rollout_steps']

        if num_frames is None:
            num_frames = config['data'].get('num_frames', None)

        # Create Dataset
        dataset = Dataset(
            config=config,
            data_manager=data_manager,
            sim_indices=sim_indices,
            field_names=field_names,
            num_frames=num_frames,
            rollout_steps=rollout_steps,
            percentage_real_data=percentage_real_data,
            enable_augmentation=enable_augmentation
        )

        logger.debug(f"PhiML dataset created: {len(dataset)} samples")
        return dataset

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


