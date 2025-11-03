"""
Data Augmentation Module for Hybrid Training

This module provides classes for augmenting training data by combining
real simulation data with model-generated predictions.

Key Features:
- Count-based weighting (no weight-based scaling)
- Adaptive strategy selection (memory/cache/on-the-fly)
- Cache management for efficient storage
- Support for both tensor and field data formats

Classes:
    AugmentedTensorDataset: Augmented dataset for synthetic training (tensors)
    AugmentedFieldDataset: Augmented dataset for physical training (fields)
    CachedAugmentedDataset: Lazy-loading from disk cache
    AdaptiveAugmentedDataLoader: Smart strategy selector
    CacheManager: Cache lifecycle management
"""

from .augmented_dataset import (
    AugmentedTensorDataset,
    AugmentedFieldDataset,
)
from .cached_dataset import CachedAugmentedDataset
from .adaptive_loader import AdaptiveAugmentedDataLoader
from .cache_manager import CacheManager
from .generation_utils import (
    generate_synthetic_predictions,
    generate_physical_predictions,
    generate_and_cache_predictions,
)

__all__ = [
    'AugmentedTensorDataset',
    'AugmentedFieldDataset',
    'CachedAugmentedDataset',
    'AdaptiveAugmentedDataLoader',
    'CacheManager',
    'generate_synthetic_predictions',
    'generate_physical_predictions',
    'generate_and_cache_predictions',
]
