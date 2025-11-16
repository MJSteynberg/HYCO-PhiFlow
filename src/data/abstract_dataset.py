"""
Abstract Dataset Base Class - Refactored for Clean Augmentation Access

Key Changes:
- All augmented data accessed via _get_augmented_sample() (not direct access)
- Subclasses control how augmented data is accessed/processed
- Maintains distinction between real and augmented data
- Clean separation of concerns between abstract and concrete classes
"""

from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from functools import lru_cache
import torch

from .data_manager import DataManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AbstractDataset(Dataset, ABC):
    """
    Abstract base class for all HYCO datasets.
    
    Supports two data categories:
    1. Real data: From actual simulations (windowed)
    2. Augmented data: Generated data (format varies by subclass)
    
    Index ranges:
    [0, num_real) → Real data
    [num_real, num_real + num_augmented) → Augmented data
    
    Key principle: Both real and augmented access go through abstract methods
    that subclasses implement according to their needs.
    """
    
    __slots__ = (
        'data_manager', 'sim_indices', 'field_names', 'num_frames',
        'num_predict_steps', 'max_cached_sims', 'access_policy',
        'num_real', 'augmented_samples', 'num_augmented',
        '_cached_load_simulation', '_total_length', '_index_mapper'
    )

    def __init__(
        self,
        data_manager: DataManager,
        sim_indices: List[int],
        field_names: List[str],
        num_frames: int,
        num_predict_steps: int,
        access_policy: str,
        num_real: int,
        augmented_samples: List[Any],
        index_mapper: Optional[Any],
        max_cached_sims: int = 5,
    ):
        """Initialize the abstract dataset."""
        if access_policy not in ("both", "real_only", "generated_only"):
            raise ValueError(
                f"Invalid access_policy: '{access_policy}'. "
                f"Must be one of ['both', 'real_only', 'generated_only']"
            )
        
        self.data_manager = data_manager
        self.sim_indices = sim_indices
        self.field_names = field_names
        self.num_frames = num_frames
        self.num_predict_steps = num_predict_steps
        self.access_policy = access_policy
        self.num_real = num_real
        self.augmented_samples = augmented_samples
        self.num_augmented = len(augmented_samples)
        self._index_mapper = index_mapper
        self.max_cached_sims = max_cached_sims
        
        # Create LRU cache
        self._cached_load_simulation = lru_cache(maxsize=max_cached_sims)(
            self._load_simulation
        )
        
        self._total_length = self._compute_length()

    # ==================== Dataset Interface ====================

    def __len__(self) -> int:
        """Return total number of samples."""
        return self._total_length

    def __getitem__(self, idx: int):
        """
        Get a sample by index.
        
        Routes to real or augmented samples based on access policy.
        Now both routes go through abstract methods.
        """
        if self.access_policy == "real_only":
            if idx >= self.num_real:
                raise IndexError(f"Index {idx} out of range [0, {self.num_real})")
            
            sample = self._get_real_sample(idx)
            return sample
        
        elif self.access_policy == "generated_only":
            if idx >= self.num_augmented:
                raise IndexError(f"Index {idx} out of range [0, {self.num_augmented})")
            
            sample = self._get_augmented_sample(idx)
            return sample
        
        else:  # 'both'
            if idx >= self._total_length:
                raise IndexError(f"Index {idx} out of range [0, {self._total_length})")
            
            if idx < self.num_real:
                sample = self._get_real_sample(idx)
                return sample
            else:
                aug_idx = idx - self.num_real
                sample = self._get_augmented_sample(aug_idx)
                return sample

    def _compute_length(self) -> int:
        """Compute total dataset length based on access policy."""
        if self.access_policy == "real_only":
            return self.num_real
        elif self.access_policy == "generated_only":
            return self.num_augmented
        else:  # 'both'
            return self.num_real + self.num_augmented

    def _get_real_sample(self, idx: int):
        """
        Get a real sample with optional filtering.
        
        Args:
            idx: Filtered index (0 to num_real-1)
        
        Returns:
            Sample in dataset-specific format
        """
        if self._index_mapper is not None:
            actual_idx = self._index_mapper.get_actual_index(idx)
        else:
            actual_idx = idx
        
        return self._extract_sample(actual_idx)

    # ==================== Abstract Methods ====================

    @abstractmethod
    def _load_simulation(self, sim_idx: int) -> Any:
        """
        Load simulation data (wrapped by LRU cache).
        
        Args:
            sim_idx: Simulation index to load
        
        Returns:
            Simulation data in format appropriate for subclass
        """
        pass

    @abstractmethod
    def _extract_sample(self, idx: int):
        """
        Extract a single sample from loaded simulations.
        
        This method should:
        1. Compute simulation and frame from index
        2. Load simulation using self._cached_load_simulation()
        3. Extract and return sample in appropriate format
        
        Args:
            idx: Actual (unfiltered) sample index
        
        Returns:
            Sample in dataset-specific format
        """
        pass

    @abstractmethod
    def _get_augmented_sample(self, idx: int):
        """
        Get an augmented sample.
        
        Subclasses implement this to handle their specific augmented format:
        - TensorDataset: Augmented = physically-generated trajectories (windowed)
        - FieldDataset: Augmented = synthetically-generated predictions (pre-windowed)
        
        Args:
            idx: Index within augmented samples (0 to num_augmented-1)
        
        Returns:
            Sample in dataset-specific format
        """
        pass

    # ==================== Utility Methods ====================

    def _compute_sim_and_frame(self, idx: int) -> Tuple[int, int]:
        """
        Compute simulation index and starting frame from a real sample index.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (sim_idx, start_frame)
        """
        samples_per_sim = self.num_frames - self.num_predict_steps
        if samples_per_sim <= 0:
            return self.sim_indices[0], 0

        sim_offset = idx // samples_per_sim
        start_frame = idx % samples_per_sim

        if sim_offset < len(self.sim_indices):
            sim_idx = self.sim_indices[sim_offset]
        else:
            sim_idx = sim_offset
            
        return sim_idx, start_frame

    def clear_cache(self):
        """Clear the LRU cache of loaded simulations."""
        self._cached_load_simulation.cache_clear()
        logger.info("Cleared LRU cache for dataset")

    def get_cache_info(self):
        """Get statistics about the LRU cache."""
        return self._cached_load_simulation.cache_info()

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the dataset."""
        cache_info = self.get_cache_info()
        
        info = {
            "total_samples": len(self),
            "real_samples": self.num_real,
            "augmented_samples": self.num_augmented,
            "access_policy": self.access_policy,
            "num_simulations": len(self.sim_indices),
            "num_frames": self.num_frames,
            "num_predict_steps": self.num_predict_steps,
            "field_names": self.field_names,
            "lru_cache_size": cache_info.maxsize,
            "lru_cache_current": cache_info.currsize,
            "lru_cache_hits": cache_info.hits,
            "lru_cache_misses": cache_info.misses,
            "filtering_active": self._index_mapper is not None,
        }
        
        return info

    def is_augmented_sample(self, idx: int) -> bool:
        """Check if an index refers to an augmented sample."""
        if self.access_policy == "generated_only":
            return True
        elif self.access_policy == "real_only":
            return False
        else:  # 'both'
            return idx >= self.num_real

    def resample_real_data(self, seed: Optional[int] = None):
        """
        Resample the subset of real data (if filtering is active).
        
        Args:
            seed: Optional random seed for reproducibility
        """
        if self._index_mapper is None:
            logger.debug("No resampling needed (percentage_real_data=1.0)")
            return
        
        self._index_mapper.resample(seed)