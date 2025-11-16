"""
Abstract Dataset Base Class - Refactored for Clean Augmentation Access

Key Changes:
- All augmented data accessed via _get_augmented_sample() (not direct access)
- Subclasses control how augmented data is accessed/processed
- Maintains distinction between real and augmented data
- Clean separation of concerns between abstract and concrete classes
"""

from typing import List, Optional, Dict, Any, Tuple
import random
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

    # ==================== Dataset Builder Utilities (migrated) ====================

    @staticmethod
    def setup_cache(
        data_manager: DataManager,
        sim_indices: List[int],
        field_names: List[str],
        num_frames: Optional[int],
        num_predict_steps: Optional[int] = None,
    ) -> int:
        """
        Validate cache and determine num_frames using the DataManager.
        This is the same logic previously in DatasetBuilder.setup_cache.
        """
        if num_frames is None:
            logger.debug("num_frames not specified, determining from first simulation...")
            first_sim_data = data_manager.load_simulation(
                sim_indices[0], field_names=field_names, num_frames=None
            )
            sample_tensor = first_sim_data["tensor_data"][field_names[0]]
            if isinstance(sample_tensor, torch.Tensor):
                if sample_tensor.dim() >= 3:
                    if sample_tensor.dim() == 5:
                        num_frames = sample_tensor.shape[2]
                    elif sample_tensor.dim() == 4:
                        num_frames = sample_tensor.shape[1]
                    elif sample_tensor.dim() == 3:
                        num_frames = 1
                    else:
                        num_frames = sample_tensor.shape[0]
                else:
                    num_frames = 1
            else:
                raise RuntimeError("First simulation tensor is not a torch.Tensor")

            logger.debug(f"Determined num_frames = {num_frames}")
            del first_sim_data

        if num_predict_steps is not None and num_frames < (num_predict_steps + 1):
            logger.warning(
                f"Discovered num_frames={num_frames} < required {num_predict_steps + 1}; attempting to reload simulation with larger frame count..."
            )
            try:
                forced = data_manager.load_simulation(
                    sim_indices[0], field_names=field_names, num_frames=(num_predict_steps + 1)
                )
                forced_tensor = forced["tensor_data"][field_names[0]]
                if isinstance(forced_tensor, torch.Tensor):
                    if forced_tensor.dim() == 5:
                        num_frames = forced_tensor.shape[2]
                    elif forced_tensor.dim() == 4:
                        num_frames = forced_tensor.shape[1]
                    elif forced_tensor.dim() == 3:
                        num_frames = 1
                    else:
                        num_frames = forced_tensor.shape[0]
                else:
                    raise RuntimeError("Forced simulation tensor is not a torch.Tensor")

                logger.debug(f"After forced reload, num_frames = {num_frames}")
            except Exception as e:
                logger.error(f"Failed to reload simulation to satisfy predict steps: {e}")
                raise

        uncached_sims = [
            sim_idx for sim_idx in sim_indices if not data_manager.is_cached(sim_idx)
        ]

        if uncached_sims:
            logger.info(f"Caching {len(uncached_sims)} simulations...")
            for i, sim_idx in enumerate(uncached_sims, 1):
                logger.debug(f"  [{i}/{len(uncached_sims)}] Caching simulation {sim_idx}...")
                _ = data_manager.load_simulation(
                    sim_idx, field_names=field_names, num_frames=num_frames
                )
            logger.debug("All simulations cached successfully!")
        else:
            logger.debug(f"All {len(sim_indices)} simulations already cached.")

        return num_frames

    @staticmethod
    def compute_sliding_window(num_frames: int, num_predict_steps: int) -> int:
        """
        Compute samples per simulation for sliding window.
        """
        if num_frames < num_predict_steps + 1:
            raise ValueError(
                f"num_frames ({num_frames}) must be >= num_predict_steps + 1 ({num_predict_steps + 1})"
            )

        samples_per_sim = num_frames - num_predict_steps
        if samples_per_sim <= 0:
            raise ValueError(
                f"Invalid sliding window: num_frames ({num_frames}) must be > num_predict_steps ({num_predict_steps})"
            )

        return samples_per_sim


class FilteringManager:
    """
    Manages percentage-based filtering of real data.

    Responsibilities:
    - Apply percentage filter to create index mapping
    - Resample indices for different epochs
    - Map filtered indices to actual indices
    """

    def __init__(
        self,
        total_samples: int,
        percentage: float,
        seed: Optional[int] = None,
    ):
        """
        Initialize filtering manager.
        """
        if not 0.0 < percentage <= 1.0:
            raise ValueError(
                f"percentage must be in (0.0, 1.0], got {percentage}"
            )

        self.total_samples = total_samples
        self.percentage = percentage
        self.num_samples = int(total_samples * percentage)

        # Generate initial index mapping
        self._active_indices = self._generate_indices(seed)

    def _generate_indices(self, seed: Optional[int]) -> List[int]:
        """Generate random indices."""
        if seed is not None:
            random.seed(seed)

        all_indices = list(range(self.total_samples))
        random.shuffle(all_indices)
        return sorted(all_indices[: self.num_samples])

    def get_actual_index(self, filtered_idx: int) -> int:
        """Map filtered index to actual index."""
        if filtered_idx >= self.num_samples:
            raise IndexError(
                f"Filtered index {filtered_idx} out of range [0, {self.num_samples})"
            )
        return self._active_indices[filtered_idx]

    def resample(self, seed: Optional[int] = None):
        """Resample the subset of data."""
        self._active_indices = self._generate_indices(seed)
        logger.debug(
            f"Resampled indices: using {self.num_samples}/{self.total_samples} "
            f"samples ({self.percentage*100:.1f}%)"
        )

    def get_info(self) -> Dict[str, Any]:
        """Get information about filtering."""
        return {
            "total_samples": self.total_samples,
            "filtered_samples": self.num_samples,
            "percentage": self.percentage,
        }