"""
Abstract Dataset Base Class - Simplified Core

Responsibilities:
- Pure dataset interface (__len__, __getitem__)
- Index routing based on access policy
- LRU caching for loaded simulations
- Basic validation and info reporting

All augmentation, filtering, and setup logic moved to utilities.
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
    
    Simplified to focus on core dataset interface:
    - Index management and routing
    - LRU-cached simulation loading
    - Access policy enforcement
    
    Subclasses must implement:
    - _load_simulation: Load raw simulation data
    - _extract_sample: Extract single sample from simulation
    - _process_augmented_sample: Convert augmented data to output format
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
        num_frames: int,  # Now required, computed by builder
        num_predict_steps: int,
        access_policy: str,
        num_real: int,  # Computed by builder
        augmented_samples: List[Any],  # Processed by builder
        index_mapper: Optional[Any],  # Optional filtering mapper
        max_cached_sims: int = 5,
    ):
        """
        Initialize the abstract dataset.
        
        Note: This constructor now takes pre-computed values from the builder.
        Use DatasetBuilder to create dataset instances.
        
        Args:
            data_manager: DataManager for loading cached data
            sim_indices: List of simulation indices
            field_names: List of field names to load
            num_frames: Total frames per simulation
            num_predict_steps: Number of prediction steps
            access_policy: 'both', 'real_only', or 'generated_only'
            num_real: Number of real samples (after filtering)
            augmented_samples: List of processed augmented samples
            index_mapper: Optional mapper for filtered indices
            max_cached_sims: LRU cache size
        """
        # Validate access policy
        if access_policy not in ("both", "real_only", "generated_only"):
            raise ValueError(
                f"Invalid access_policy: '{access_policy}'. "
                f"Must be one of ['both', 'real_only', 'generated_only']"
            )
        
        # Store parameters
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
        
        # Create LRU cache for simulation loading
        self._cached_load_simulation = lru_cache(maxsize=max_cached_sims)(
            self._load_simulation
        )
        
        # Compute total length
        self._total_length = self._compute_length()

    # ==================== Dataset Interface ====================

    def __len__(self) -> int:
        """Return total number of samples."""
        return self._total_length

    def __getitem__(self, idx: int):
        """
        Get a sample by index.
        
        Routes to real or augmented samples based on access policy.
        """
        # Route based on access policy
        if self.access_policy == "real_only":
            if idx >= self.num_real:
                raise IndexError(f"Index {idx} out of range [0, {self.num_real})")
            return self._get_real_sample(idx)
        
        elif self.access_policy == "generated_only":
            if idx >= self.num_augmented:
                raise IndexError(f"Index {idx} out of range [0, {self.num_augmented})")
            return self.augmented_samples[idx]
        
        else:  # 'both'
            if idx >= self._total_length:
                raise IndexError(f"Index {idx} out of range [0, {self._total_length})")
            
            if idx < self.num_real:
                return self._get_real_sample(idx)
            else:
                return self.augmented_samples[idx - self.num_real]

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
        # Map filtered index to actual index if filtering is active
        if self._index_mapper is not None:
            actual_idx = self._index_mapper.get_actual_index(idx)
        else:
            actual_idx = idx
        
        # Extract sample
        return self._extract_sample(actual_idx)
    
    # In abstract_dataset.py

    def add_synthetic_simulations(self, synthetic_sims: List[Dict[str, torch.Tensor]]):
        """Add synthetic simulations that will be indexed like real simulations."""
        if not hasattr(self, '_synthetic_sims'):
            self._synthetic_sims = []
        
        self._synthetic_sims.extend(synthetic_sims)
        
        samples_per_sim = self.num_frames - self.num_predict_steps
        num_synthetic_samples = len(self._synthetic_sims) * samples_per_sim
        
        self._num_real_only = self.num_real
        self.num_real = self._num_real_only + num_synthetic_samples
        
        # CRITICAL FIX: Disable filtering when synthetic data is added
        # or update the mapper's total count
        if self._index_mapper is not None:
            self._index_mapper.total_samples = self.num_real
            self._index_mapper.num_samples = self.num_real  # Use all samples
            self._index_mapper._active_indices = list(range(self.num_real))
        
        self._total_length = self._compute_length()

    def _compute_sim_and_frame(self, idx: int) -> Tuple[int, int]:
        """Compute simulation index and starting frame from a real sample index."""
        samples_per_sim = self.num_frames - self.num_predict_steps
        if samples_per_sim <= 0:
            # This can happen if num_frames is too small
            return self.sim_indices[0], 0

        sim_offset = idx // samples_per_sim
        start_frame = idx % samples_per_sim

        if sim_offset < len(self.sim_indices):
            sim_idx = self.sim_indices[sim_offset]
        else:
            # This case should ideally not be hit if logic is separated,
            # but as a fallback, we can treat it as a direct index.
            sim_idx = sim_offset
        return sim_idx, start_frame

    def clear_synthetic_simulations(self):
        """Clear all synthetic simulations."""
        if hasattr(self, '_synthetic_sims'):
            self._synthetic_sims.clear()
            if hasattr(self, '_num_real_only'):
                self.num_real = self._num_real_only
                self._total_length = self._compute_length()

    # ==================== Cache Management ====================

    def clear_cache(self):
        """Clear the LRU cache of loaded simulations."""
        self._cached_load_simulation.cache_clear()
        logger.info("Cleared LRU cache for dataset")

    def get_cache_info(self):
        """Get statistics about the LRU cache."""
        return self._cached_load_simulation.cache_info()

    # ==================== Info Methods ====================

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
    def _process_augmented_sample(self, sample: Any) -> Any:
        """
        Convert augmented sample to output format.
        
        Args:
            sample: Raw augmented sample
        
        Returns:
            Processed sample in dataset-specific format
        """
        pass