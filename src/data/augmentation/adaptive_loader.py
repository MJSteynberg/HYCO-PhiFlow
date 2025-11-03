"""
Adaptive augmented data loader with automatic strategy selection.

This module provides AdaptiveAugmentedDataLoader which automatically selects
the best augmentation strategy (memory/cache/on-the-fly) based on dataset
size and system resources.
"""

import logging
from typing import Optional, Literal
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

from .augmented_dataset import AugmentedTensorDataset
from .cached_dataset import CachedAugmentedDataset

logger = logging.getLogger(__name__)

# Strategy selection type
StrategyType = Literal['memory', 'cache', 'on_the_fly']


class AdaptiveAugmentedDataLoader:
    """
    Adaptive data loader that selects augmentation strategy automatically.
    
    This loader analyzes the dataset size and system resources to choose
    the most efficient augmentation strategy:
    
    - 'memory': Load all generated data into memory (fastest, high memory)
    - 'cache': Lazy load from disk with LRU cache (balanced)
    - 'on_the_fly': Generate predictions during training (slowest, low memory)
    
    Strategy Selection Logic:
    - If generated_data is provided → 'memory' strategy
    - Else if cache_dir exists with samples → 'cache' strategy  
    - Else → 'on_the_fly' strategy (requires model for generation)
    
    Args:
        real_dataset: The base dataset containing real samples
        alpha: Proportion of generated samples (e.g., 0.1 = 10%)
        generated_data: Optional pre-loaded generated predictions (for memory strategy)
        cache_dir: Optional directory with cached predictions (for cache strategy)
        cache_size: LRU cache size for cache strategy (default: 128)
        strategy: Force a specific strategy (default: auto-select)
        validate_count: Whether to validate generated sample count
        
    Returns:
        PyTorch DataLoader with the selected augmented dataset
        
    Example:
        >>> # Auto-select strategy
        >>> loader = AdaptiveAugmentedDataLoader(
        ...     real_dataset=train_data,
        ...     alpha=0.1,
        ...     cache_dir="data/cache/hybrid_generated/burgers_128"
        ... ).get_loader(batch_size=32, shuffle=True)
        >>> 
        >>> # Force memory strategy
        >>> loader = AdaptiveAugmentedDataLoader(
        ...     real_dataset=train_data,
        ...     alpha=0.1,
        ...     generated_data=preloaded_data,
        ...     strategy='memory'
        ... ).get_loader(batch_size=32)
    """
    
    def __init__(
        self,
        real_dataset: Dataset,
        alpha: float = 0.1,
        generated_data: Optional[Dataset] = None,
        cache_dir: Optional[str] = None,
        cache_size: int = 128,
        strategy: Optional[StrategyType] = None,
        validate_count: bool = True
    ):
        self.real_dataset = real_dataset
        self.alpha = alpha
        self.generated_data = generated_data
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_size = cache_size
        self.validate_count = validate_count
        
        # Select or validate strategy
        if strategy is None:
            self.strategy = self._auto_select_strategy()
        else:
            self._validate_strategy(strategy)
            self.strategy = strategy
        
        # Create the augmented dataset
        self.dataset = self._create_dataset()
        
        logger.info(
            f"AdaptiveAugmentedDataLoader initialized with '{self.strategy}' strategy: "
            f"{len(real_dataset)} real samples, alpha={alpha:.2f}"
        )
    
    def _auto_select_strategy(self) -> StrategyType:
        """
        Automatically select the best augmentation strategy.
        
        Returns:
            Selected strategy name
        """
        # Priority 1: Memory strategy if generated data is provided
        if self.generated_data is not None:
            logger.info("Auto-selected 'memory' strategy (generated_data provided)")
            return 'memory'
        
        # Priority 2: Cache strategy if cache directory exists with samples
        if self.cache_dir is not None:
            if self.cache_dir.exists():
                # Check if cache has any .pt files
                cache_files = list(self.cache_dir.glob("sample_*.pt"))
                if len(cache_files) > 0:
                    logger.info(
                        f"Auto-selected 'cache' strategy "
                        f"({len(cache_files)} cached samples found)"
                    )
                    return 'cache'
                else:
                    logger.warning(
                        f"Cache directory exists but no samples found: {self.cache_dir}"
                    )
        
        # Priority 3: On-the-fly generation (not implemented in Day 2)
        logger.info(
            "Auto-selected 'on_the_fly' strategy (no data or cache available). "
            "Note: On-the-fly generation not yet implemented."
        )
        return 'on_the_fly'
    
    def _validate_strategy(self, strategy: StrategyType) -> None:
        """
        Validate that the requested strategy is feasible.
        
        Args:
            strategy: The requested strategy
            
        Raises:
            ValueError: If strategy is not feasible with current configuration
        """
        if strategy == 'memory' and self.generated_data is None:
            raise ValueError(
                "Cannot use 'memory' strategy without generated_data. "
                "Either provide generated_data or use 'cache'/'on_the_fly' strategy."
            )
        
        if strategy == 'cache':
            if self.cache_dir is None:
                raise ValueError(
                    "Cannot use 'cache' strategy without cache_dir. "
                    "Either provide cache_dir or use 'memory'/'on_the_fly' strategy."
                )
            if not self.cache_dir.exists():
                raise ValueError(
                    f"Cache directory does not exist: {self.cache_dir}"
                )
            cache_files = list(self.cache_dir.glob("sample_*.pt"))
            if len(cache_files) == 0:
                raise ValueError(
                    f"No cached samples found in {self.cache_dir}"
                )
        
        if strategy == 'on_the_fly':
            logger.warning(
                "On-the-fly generation strategy selected but not yet implemented. "
                "Will be available after Day 3-5 implementation."
            )
    
    def _create_dataset(self) -> Dataset:
        """
        Create the appropriate augmented dataset based on strategy.
        
        Returns:
            Augmented dataset instance
        """
        if self.strategy == 'memory':
            return AugmentedTensorDataset(
                real_dataset=self.real_dataset,
                generated_data=self.generated_data,
                alpha=self.alpha,
                validate_count=self.validate_count
            )
        
        elif self.strategy == 'cache':
            return CachedAugmentedDataset(
                real_dataset=self.real_dataset,
                cache_dir=str(self.cache_dir),
                alpha=self.alpha,
                cache_size=self.cache_size,
                validate_count=self.validate_count
            )
        
        elif self.strategy == 'on_the_fly':
            # Placeholder for future implementation
            raise NotImplementedError(
                "On-the-fly generation strategy not yet implemented. "
                "Will be available after Day 3-5 implementation. "
                "Use 'memory' or 'cache' strategy for now."
            )
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def get_loader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader for the augmented dataset.
        
        Args:
            batch_size: Batch size for training
            shuffle: Whether to shuffle samples
            num_workers: Number of worker processes for data loading
            **kwargs: Additional arguments passed to DataLoader
            
        Returns:
            Configured DataLoader instance
        """
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )
    
    def get_dataset(self) -> Dataset:
        """
        Get the underlying augmented dataset.
        
        Returns:
            The augmented dataset instance
        """
        return self.dataset
    
    def get_strategy(self) -> StrategyType:
        """
        Get the selected strategy name.
        
        Returns:
            Strategy name ('memory', 'cache', or 'on_the_fly')
        """
        return self.strategy
    
    def get_info(self) -> dict:
        """
        Get information about the loader configuration.
        
        Returns:
            Dictionary with configuration details
        """
        info = {
            'strategy': self.strategy,
            'num_real': len(self.real_dataset),
            'alpha': self.alpha,
            'num_generated': int(len(self.real_dataset) * self.alpha),
            'total_samples': len(self.dataset)
        }
        
        # Add strategy-specific info
        if self.strategy == 'cache':
            cache_info = self.dataset.get_cache_info()
            info['cache'] = cache_info
        
        return info
