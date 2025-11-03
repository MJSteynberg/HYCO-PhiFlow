"""
Cached augmented dataset implementation for lazy loading from disk.

This module provides CachedAugmentedDataset which loads generated predictions
from disk on-demand with LRU caching for memory efficiency.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Tuple, Optional, List
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CachedAugmentedDataset(Dataset):
    """
    Augmented dataset with lazy loading from disk cache.
    
    This dataset combines real data with generated predictions loaded from disk
    on-demand. Uses LRU cache to minimize disk I/O while managing memory usage.
    Implements count-based weighting where the number of generated samples is
    proportional to alpha without using sample weights.
    
    Args:
        real_dataset: The base dataset containing real samples
        cache_dir: Directory containing cached generated predictions
        alpha: Proportion of generated samples relative to real samples (e.g., 0.1 = 10%)
        cache_size: Maximum number of cached samples in memory (LRU eviction)
        validate_count: Whether to validate expected vs actual generated sample count
        
    Returns:
        2-tuple: (input, target) - NO weights returned
        
    Example:
        >>> real_data = SomeDataset(...)
        >>> cached_data = CachedAugmentedDataset(
        ...     real_dataset=real_data,
        ...     cache_dir="data/cache/hybrid_generated/burgers_128",
        ...     alpha=0.1,
        ...     cache_size=128
        ... )
        >>> loader = DataLoader(cached_data, batch_size=32)
        >>> for inputs, targets in loader:
        ...     # All samples have implicit weight=1.0
        ...     loss = criterion(model(inputs), targets)
    """
    
    def __init__(
        self,
        real_dataset: Dataset,
        cache_dir: str,
        alpha: float = 0.1,
        cache_size: int = 128,
        validate_count: bool = True
    ):
        self.real_dataset = real_dataset
        self.cache_dir = Path(cache_dir)
        self.alpha = alpha
        self.cache_size = cache_size
        self.validate_count = validate_count
        
        # Calculate expected count from alpha
        self.num_real = len(real_dataset)
        self.expected_generated = int(self.num_real * alpha)
        
        # Discover cached files (actual count)
        self._discover_cache_files()
        
        # Use actual cache file count
        self.num_generated = len(self.cache_files)
        self.total_samples = self.num_real + self.num_generated
        
        # Validate count if requested
        if self.validate_count:
            self._validate_cache_count()
        
        # Create LRU cache for loading samples
        self._load_cached_sample = lru_cache(maxsize=cache_size)(
            self._load_sample_from_disk
        )
        
        logger.info(
            f"CachedAugmentedDataset initialized: "
            f"{self.num_real} real + {self.num_generated} generated "
            f"(alpha={alpha:.2f}, cache_size={cache_size})"
        )
    
    def _discover_cache_files(self) -> None:
        """Discover and index all cached sample files."""
        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"Cache directory not found: {self.cache_dir}"
            )
        
        # Look for .pt files (PyTorch tensors)
        self.cache_files: List[Path] = sorted(
            self.cache_dir.glob("sample_*.pt")
        )
        
        if len(self.cache_files) == 0:
            raise ValueError(
                f"No cached samples found in {self.cache_dir}. "
                f"Expected files like 'sample_000000.pt'"
            )
        
        logger.debug(f"Discovered {len(self.cache_files)} cached files")
    
    def _validate_cache_count(self) -> None:
        """Validate that cached sample count matches expected count."""
        actual_count = self.num_generated
        expected_count = self.expected_generated
        
        if abs(actual_count - expected_count) > 1:
            logger.warning(
                f"Cache count mismatch: expected ~{expected_count} "
                f"generated samples (alpha={self.alpha:.2f} * "
                f"{self.num_real} real), but found {actual_count} "
                f"cached files. This may cause imbalanced training."
            )
    
    def _load_sample_from_disk(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a single sample from disk (not cached by lru_cache decorator).
        
        Args:
            idx: Index into cache_files list (0-based)
            
        Returns:
            Tuple of (input, target) tensors
        """
        if idx >= len(self.cache_files):
            raise IndexError(
                f"Cache index {idx} out of range (0-{len(self.cache_files)-1})"
            )
        
        file_path = self.cache_files[idx]
        
        try:
            # Load the cached sample
            data = torch.load(file_path, map_location='cpu')
            
            # Handle different formats
            if isinstance(data, dict):
                # Expected format: {'input': tensor, 'target': tensor}
                input_tensor = data['input']
                target_tensor = data['target']
            elif isinstance(data, (tuple, list)) and len(data) == 2:
                # Tuple format: (input, target)
                input_tensor, target_tensor = data
            else:
                raise ValueError(
                    f"Unexpected cache file format in {file_path}. "
                    f"Expected dict with 'input'/'target' keys or 2-tuple."
                )
            
            return input_tensor, target_tensor
            
        except Exception as e:
            logger.error(f"Failed to load cached sample from {file_path}: {e}")
            raise
    
    def __len__(self) -> int:
        """Return total number of samples (real + generated)."""
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample by index.
        
        Args:
            idx: Index into combined dataset (0 to total_samples-1)
            
        Returns:
            2-tuple: (input, target) - NO weights
        """
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(
                f"Index {idx} out of range (0-{self.total_samples-1})"
            )
        
        # First num_real samples come from real dataset
        if idx < self.num_real:
            return self.real_dataset[idx]
        
        # Remaining samples come from cache
        cache_idx = idx - self.num_real
        return self._load_cached_sample(cache_idx)
    
    def clear_cache(self) -> None:
        """Clear the LRU cache, forcing reload on next access."""
        self._load_cached_sample.cache_clear()
        logger.debug("Cleared LRU cache")
    
    def get_cache_info(self) -> dict:
        """
        Get information about cache usage.
        
        Returns:
            Dictionary with cache statistics:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - maxsize: Maximum cache size
            - currsize: Current cache size
        """
        info = self._load_cached_sample.cache_info()
        return {
            'hits': info.hits,
            'misses': info.misses,
            'maxsize': info.maxsize,
            'currsize': info.currsize
        }
