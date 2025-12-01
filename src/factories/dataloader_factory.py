"""
DataLoader Factory

Simplified factory for creating datasets with minimal configuration.
Uses the new streamlined Dataset class that loads PhiML tensors directly.
"""

from typing import List, Optional

from src.data.dataset import Dataset
from src.data.sparsity import TemporalSparsityConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoaderFactory:
    """
    Factory for creating data loaders with minimal configuration.

    Simplified to use the new streamlined Dataset class that doesn't
    require DataManager or complex configuration.

    The new Dataset:
    - Loads simulations directly from PhiML tensors (saved by DataGenerator)
    - Stores data in memory
    - No augmentation, no filtering - just simple iteration
    - Works with both physical and synthetic trainers

    Example:
        >>> dataset = DataLoaderFactory.create_phiml(config, sim_indices=[0, 1, 2], rollout_steps=10)
        >>> for batch in dataset.iterate_batches(batch_size=16):
        ...     # batch.initial_state: Tensor(batch, x, y?, field)
        ...     # batch.targets: Tensor(batch, time, x, y?, field)
        ...     pass
    """
    @staticmethod
    def create_phiml(
        config: dict,
        sim_indices: Optional[List[int]] = None,
        rollout_steps: Optional[int] = None,
        temporal_sparsity: Optional[TemporalSparsityConfig] = None,
        # Legacy parameters (ignored for now - keeping for API compatibility)
        field_names: Optional[List[str]] = None,
        num_frames: Optional[int] = None,
        percentage_real_data: float = 1.0,
        enable_augmentation: bool = False,
    ) -> Dataset:
        """
        Create a simplified PhiML dataset for training.

        This method creates a Dataset that yields PhiML tensor batches
        directly - no PyTorch dependency, no DataManager needed.

        Args:
            config: Full configuration dictionary from Hydra
            sim_indices: Simulation indices to use (default: from config's train_sim)
            rollout_steps: Number of prediction steps (default: from config)
            temporal_sparsity: Temporal sparsity configuration (default: from config or None)
            field_names: IGNORED (kept for API compatibility)
            num_frames: IGNORED (kept for API compatibility)
            percentage_real_data: IGNORED (kept for API compatibility)
            enable_augmentation: IGNORED (kept for API compatibility)

        Returns:
            Dataset instance that yields PhiML tensor batches

        Example:
            >>> dataset = DataLoaderFactory.create_phiml(config, sim_indices=[0, 1, 2], rollout_steps=10)
            >>> for batch in dataset.iterate_batches(batch_size=16):
            ...     # batch.initial_state: Tensor(batch, x, y?, field)
            ...     # batch.targets: Tensor(batch, time, x, y?, field)
            ...     pass
        """
        logger.debug("Creating simplified PhiML dataset...")

        # Get parameters from config if not provided
        if sim_indices is None:
            sim_indices = config['trainer'].get('train_sim', [0])

        if rollout_steps is None:
            # Try to find rollout_steps from model-specific config, with fallback
            mode = config.get('general', {}).get('mode', 'synthetic')
            if mode == 'synthetic':
                rollout_steps = config['trainer'].get('synthetic', {}).get(
                    'rollout_steps',
                    config['trainer'].get('rollout_steps', 4)
                )
            elif mode == 'physical':
                rollout_steps = config['trainer'].get('physical', {}).get(
                    'rollout_steps',
                    config['trainer'].get('rollout_steps', 4)
                )
            else:
                # For hybrid or unknown modes, use global or default
                rollout_steps = config['trainer'].get('rollout_steps', 4)

        # Extract temporal sparsity from config if not provided
        if temporal_sparsity is None and 'sparsity' in config:
            sparsity_cfg = config['sparsity']
            if 'temporal' in sparsity_cfg:
                temporal_sparsity = TemporalSparsityConfig(**sparsity_cfg['temporal'])

        # Create simplified Dataset (no DataManager, no complex config!)
        dataset = Dataset(
            config=config,
            train_sim=sim_indices,
            rollout_steps=rollout_steps,
            temporal_sparsity=temporal_sparsity
        )

        logger.debug(f"PhiML dataset created: {dataset.total_samples} samples")
        return dataset

