"""
Abstract Dataset Base Class

Provides common functionality for all dataset types, similar to AbstractTrainer.
Reduces code duplication and ensures consistent behavior across dataset implementations.

This base class implements:
- Common initialization logic (DataManager, simulation indices, field names)
- Cache validation (ensures all required simulations are cached)
- Sliding window indexing (creates multiple samples per simulation)
- Augmentation management (loads from memory or cache)
- LRU caching for simulations (memory-efficient lazy loading)
- Sample routing (directs requests to real vs augmented data)

Subclasses must implement:
- _load_simulation_uncached(): Load and process simulation data from cache
- _get_real_sample(): Convert loaded data to the final format (tensors or Fields)
"""

from typing import List, Optional, Dict, Any, Tuple, Union
from abc import ABC, abstractmethod
from pathlib import Path
import torch
from torch.utils.data import Dataset
from functools import lru_cache

from .data_manager import DataManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AbstractDataset(Dataset, ABC):
    """
    Abstract base class for all HYCO datasets.

    This class provides common functionality shared by TensorDataset and FieldDataset,
    following the same pattern as AbstractTrainer in the training module.

    Common Features:
    - Lazy loading with LRU cache for memory efficiency
    - Sliding window support for multiple samples per simulation
    - Augmentation management (memory or cache-based)
    - Cache validation to ensure data availability
    - Sample routing between real and augmented data

    Template Methods (must be implemented by subclasses):
    - _load_simulation_uncached(sim_idx): Load raw simulation data
    - _get_real_sample(idx): Process and return a sample in the appropriate format

    Args:
        data_manager: DataManager instance for loading cached data
        sim_indices: List of simulation indices to include
        field_names: List of field names to load
        num_frames: Number of frames per simulation (None = load all)
        num_predict_steps: Number of prediction steps for training
        use_sliding_window: If True, create multiple samples per simulation
        augmentation_config: Optional dict with augmentation settings
        access_policy: Control which data is accessible ('both', 'real_only', 'generated_only')
        max_cached_sims: LRU cache size (number of simulations in memory)

    Augmentation Config Structure:
        {
            'mode': 'memory' | 'cache',  # How augmented data is loaded
            'alpha': float,               # Proportion of augmented samples (e.g., 0.1 = 10%)
            'data': List,                 # Pre-loaded data (for 'memory' mode)
            'cache_dir': str,             # Cache directory path (for 'cache' mode)
        }

    Access Policy:
        - 'both': Dataset provides both real and augmented samples (default)
        - 'real_only': Dataset provides only real samples (ignores augmentation)
        - 'generated_only': Dataset provides only augmented samples (requires augmentation)
    """

    def __init__(
        self,
        data_manager: DataManager,
        sim_indices: List[int],
        field_names: List[str],
        num_frames: Optional[int],
        num_predict_steps: int,
        augmentation_config: Optional[Dict[str, Any]] = None,
        access_policy: str = "both",
        max_cached_sims: int = 5,
    ):
        """
        Initialize the abstract dataset.

        This constructor handles all common initialization:
        1. Store basic parameters
        2. Validate cache exists for all simulations
        3. Determine num_frames if not provided
        4. Create LRU cache for lazy loading
        5. Build sliding window index if enabled
        6. Load augmentation data if configured
        7. Apply access policy to control real vs generated data
        """
        # Store basic parameters
        self.data_manager = data_manager
        self.sim_indices = sim_indices
        self.field_names = field_names
        self.num_frames = num_frames
        self.num_predict_steps = num_predict_steps
        self.max_cached_sims = max_cached_sims

        # Validate and store access policy
        valid_policies = ["both", "real_only", "generated_only"]
        if access_policy not in valid_policies:
            raise ValueError(
                f"Invalid access_policy: '{access_policy}'. "
                f"Must be one of {valid_policies}"
            )
        self.access_policy = access_policy

        # Validate inputs
        if not sim_indices:
            raise ValueError("sim_indices cannot be empty")

        if num_frames is not None and num_frames < num_predict_steps + 1:
            raise ValueError(
                f"num_frames ({num_frames}) must be >= num_predict_steps + 1 "
                f"({num_predict_steps + 1})"
            )

        # Validate cache and determine num_frames
        self._validate_and_setup_cache()

        # Create LRU cache for simulation loading
        self._create_cached_loader()

        # Build sliding window index
        self.sample_index = []
        self._build_sliding_window_index()

        # Load augmentation if configured
        self.augmented_samples = []
        if augmentation_config:
            self._load_augmentation(augmentation_config)
            # Process augmented samples to ensure correct format
            if self.augmented_samples:
                self.augmented_samples = [
                    self._process_augmented_sample(sample)
                    for sample in self.augmented_samples
                ]

        # Calculate sample counts
        self.num_real = len(self.sample_index) 
        self.num_augmented = len(self.augmented_samples)

        # Log dataset size based on access policy
        if self.access_policy == "both":
            logger.debug(
                f"  Dataset: {self.num_real} real + {self.num_augmented} augmented = {len(self)} samples"
            )
        elif self.access_policy == "real_only":
            logger.debug(
                f"  Dataset: {self.num_real} real samples (access_policy=real_only)"
            )
        elif self.access_policy == "generated_only":
            if self.num_augmented == 0:
                logger.warning(
                    "  Dataset: access_policy=generated_only but no augmented samples available!"
                )
            logger.debug(
                f"  Dataset: {self.num_augmented} generated samples (access_policy=generated_only)"
            )

    # ==================== Cache Management ====================

    def _validate_and_setup_cache(self):
        """
        Validate that cache exists for all simulations and setup num_frames.

        This method:
        1. Determines num_frames from first simulation if not provided
        2. Checks which simulations are not cached
        3. Verifies raw data exists for uncached simulations
        4. Caches all uncached simulations upfront

        Raises:
            ValueError: If simulation data doesn't exist or cache is invalid
        """
        # Determine num_frames from first simulation if not provided
        if self.num_frames is None:
            logger.debug(
                "num_frames not specified, determining from first simulation..."
            )
            first_sim_data = self.data_manager.get_or_load_simulation(
                self.sim_indices[0], field_names=self.field_names, num_frames=None
            )
            # Extract num_frames from first field's tensor
            self.num_frames = first_sim_data["tensor_data"][self.field_names[0]].shape[
                0
            ]
            logger.debug(f"Determined num_frames = {self.num_frames}")
            del first_sim_data  # Free memory immediately

        # Check which simulations need caching
        uncached_sims = []
        for sim_idx in self.sim_indices:
            if not self.data_manager.is_cached(sim_idx):
                # Verify raw data exists
                sim_dir = self.data_manager.raw_data_dir / f"sim_{sim_idx:06d}"
                if not sim_dir.exists():
                    raise ValueError(
                        f"Simulation {sim_idx} does not exist at {sim_dir}. "
                        f"Please run data generation first."
                    )
                uncached_sims.append(sim_idx)

        # Cache all uncached simulations upfront
        if uncached_sims:
            logger.info(f"  Caching {len(uncached_sims)} simulations...")
            for i, sim_idx in enumerate(uncached_sims, 1):
                logger.debug(
                    f"    [{i}/{len(uncached_sims)}] Caching simulation {sim_idx}..."
                )
                # Load and cache the simulation
                _ = self.data_manager.get_or_load_simulation(
                    sim_idx, field_names=self.field_names, num_frames=self.num_frames
                )
            logger.debug(f"  All simulations cached successfully!")
        else:
            logger.debug(f"All {len(self.sim_indices)} simulations already cached.")

    def _create_cached_loader(self):
        """
        Create LRU-cached simulation loader.

        The loader will keep at most max_cached_sims simulations in memory,
        automatically evicting least recently used simulations when cache is full.

        This provides memory-efficient lazy loading:
        - Simulations loaded on-demand, not all at once
        - Most recently used simulations stay in memory
        - Automatic eviction when cache size limit reached
        """
        self._cached_load_simulation = lru_cache(maxsize=self.max_cached_sims)(
            self._load_simulation_uncached
        )

    def clear_cache(self):
        """
        Manually clear the LRU cache of loaded simulations.

        Useful for:
        - Freeing memory when dataset is no longer needed
        - Forcing reload of simulations after data changes
        - Debugging memory issues
        """
        self._cached_load_simulation.cache_clear()
        logger.info(f"  Cleared LRU cache for dataset")

    def get_cache_info(self):
        """
        Get statistics about the LRU cache.

        Returns:
            CacheInfo namedtuple with fields:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - maxsize: Maximum cache size
            - currsize: Current number of cached items
        """
        return self._cached_load_simulation.cache_info()

    # ==================== Sliding Window ====================

    def _build_sliding_window_index(self):
        """
        Build index mapping for sliding window samples.

        Creates a list of (sim_idx, start_frame) tuples representing each sample.
        Each sample needs:
        - 1 initial frame at start_frame
        - num_predict_steps target frames (start_frame+1 to start_frame+num_predict_steps)

        Example with 10 frames (0-9) and 3 predict steps:
        - start_frame=0: initial=0, targets=[1,2,3] ✓
        - start_frame=1: initial=1, targets=[2,3,4] ✓
        - ...
        - start_frame=6: initial=6, targets=[7,8,9] ✓ (last valid)
        - start_frame=7: initial=7, targets=[8,9,?] ✗ (insufficient frames)

        Formula: samples_per_sim = num_frames - num_predict_steps
        Valid start_frames: 0 to (num_frames - num_predict_steps - 1)
        """
        self.sample_index = []

        # Calculate samples per simulation
        samples_per_sim = self.num_frames - self.num_predict_steps

        if samples_per_sim <= 0:
            raise ValueError(
                f"Invalid sliding window: num_frames ({self.num_frames}) must be > "
                f"num_predict_steps ({self.num_predict_steps})"
            )

        # Build index for all simulations
        for sim_idx in self.sim_indices:
            for start_frame in range(samples_per_sim):
                self.sample_index.append((sim_idx, start_frame))

        logger.debug(f"  Sliding window: {samples_per_sim} samples per simulation")
        logger.debug(
            f"  Total samples: {len(self.sample_index)} (from {len(self.sim_indices)} simulations)"
        )

    # ==================== Augmentation ====================

    def _load_augmentation(self, config: Dict[str, Any]):
        """
        Load augmented samples based on configuration.

        Supports two modes:
        1. 'memory': Pre-loaded data provided in config['data']
        2. 'cache': Load from disk cache at config['cache_dir']

        Args:
            config: Augmentation configuration dictionary with:
                - mode: 'memory' or 'cache'
                - alpha: Proportion of augmented samples (optional, for logging)
                - data: Pre-loaded samples (for memory mode)
                - cache_dir: Path to cache directory (for cache mode)

        Raises:
            ValueError: If mode is unknown or required keys are missing
            FileNotFoundError: If cache directory doesn't exist
        """
        mode = config.get("mode", "cache")
        alpha = config.get("alpha", 0.0)

        logger.debug(f"  Loading augmentation (mode={mode}, alpha={alpha})...")

        if mode == "memory":
            # Pre-loaded augmented data provided
            if "data" not in config:
                raise ValueError(
                    "Augmentation mode 'memory' requires 'data' key in config"
                )
            self.augmented_samples = config["data"]
            logger.debug(
                f"    Loaded {len(self.augmented_samples)} pre-loaded augmented samples"
            )

        elif mode == "cache":
            # Load from disk cache
            if "cache_dir" not in config:
                raise ValueError(
                    "Augmentation mode 'cache' requires 'cache_dir' key in config"
                )

            cache_dir = config["cache_dir"]
            expected_count = int(self.num_real * alpha) if alpha > 0 else None

            self.augmented_samples = self._load_from_cache(cache_dir, expected_count)
            logger.debug(
                f"    Loaded {len(self.augmented_samples)} cached augmented samples"
            )

        else:
            raise ValueError(f"Unknown augmentation mode: {mode}")

    def _load_from_cache(
        self, cache_dir: str, expected_count: Optional[int] = None
    ) -> List[Any]:
        """
        Load augmented samples from cache directory.

        Expects cache files named 'sample_*.pt' containing torch-saved data.

        Args:
            cache_dir: Path to cache directory
            expected_count: Expected number of samples (for validation, optional)

        Returns:
            List of loaded samples (format depends on what was cached)

        Raises:
            FileNotFoundError: If cache directory doesn't exist
            ValueError: If no cache files found or count mismatch
        """
        cache_path = Path(cache_dir)

        if not cache_path.exists():
            raise FileNotFoundError(
                f"Augmentation cache not found: {cache_dir}\n"
                f"Please run cache generation first."
            )

        # Find all cached sample files
        cache_files = sorted(cache_path.glob("sample_*.pt"))

        if len(cache_files) == 0:
            raise ValueError(
                f"No cached samples found in {cache_dir}\n"
                f"Expected to find sample_*.pt files."
            )

        # Validate count if expected
        if expected_count is not None and len(cache_files) != expected_count:
            logger.warning(
                f"Expected {expected_count} cached samples but found {len(cache_files)}"
            )

        # Load all cached samples
        samples = []
        for file_path in cache_files:
            try:
                data = torch.load(file_path, map_location="cpu")

                # Handle different cache formats
                if isinstance(data, dict):
                    # Format: {'input': ..., 'target': ...}
                    samples.append((data["input"], data["target"]))
                else:
                    # Format: direct tuple or other format
                    samples.append(data)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                raise

        return samples

    # ==================== Dataset Interface ====================

    def __len__(self) -> int:
        """
        Return total number of samples based on access policy.

        Returns:
            - 'real_only': num_real
            - 'generated_only': num_augmented
            - 'both': num_real + num_augmented
        """
        if self.access_policy == "real_only":
            return self.num_real
        elif self.access_policy == "generated_only":
            return self.num_augmented
        else:  # 'both'
            return self.num_real + self.num_augmented

    def __getitem__(self, idx: int):
        """
        Get a training sample by index, respecting access policy.

        Access policy routing:
        - 'real_only': All indices map to real samples
        - 'generated_only': All indices map to augmented samples
        - 'both': idx < num_real → real, idx >= num_real → augmented

        Args:
            idx: Sample index (0 to len(dataset)-1)

        Returns:
            Sample in format determined by subclass implementation
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        if self.access_policy == "real_only":
            # Only serve real data
            return self._get_real_sample(idx)

        elif self.access_policy == "generated_only":
            # Only serve generated data
            if idx >= self.num_augmented:
                raise IndexError(
                    f"Generated index {idx} out of range [0, {self.num_augmented}). "
                    f"access_policy=generated_only but only {self.num_augmented} augmented samples available."
                )
            return self.augmented_samples[idx]

        else:  # 'both'
            # Route based on index (existing behavior)
            if idx < self.num_real:
                return self._get_real_sample(idx)
            else:
                aug_idx = idx - self.num_real
                return self.augmented_samples[aug_idx]

    # ==================== Utility Methods ====================

    def get_simulation_and_frame(self, idx: int) -> Tuple[int, int]:
        """
        Get simulation index and starting frame for a real sample index.

        Args:
            idx: Real sample index (0 to num_real-1)

        Returns:
            Tuple of (sim_idx, start_frame)

        Raises:
            ValueError: If idx refers to an augmented sample
        """
        if idx >= self.num_real:
            raise ValueError(
                f"Index {idx} is an augmented sample. "
                f"This method only works for real samples (idx < {self.num_real})"
            )

        
        return self.sample_index[idx]
        

    def is_augmented_sample(self, idx: int) -> bool:
        """
        Check if an index refers to an augmented sample.

        Args:
            idx: Sample index

        Returns:
            True if augmented, False if real
        """
        return idx >= self.num_real

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset.

        Returns:
            Dictionary with dataset statistics and configuration
        """
        cache_info = self.get_cache_info()

        return {
            "total_samples": len(self),
            "real_samples": self.num_real,
            "augmented_samples": self.num_augmented,
            "access_policy": self.access_policy,
            "effective_samples": len(self),  # Actual samples available based on policy
            "num_simulations": len(self.sim_indices),
            "num_frames": self.num_frames,
            "num_predict_steps": self.num_predict_steps,
            "use_sliding_window": self.use_sliding_window,
            "samples_per_sim": (
                len(self.sample_index) // len(self.sim_indices)
                if self.use_sliding_window
                else 1
            ),
            "field_names": self.field_names,
            "lru_cache_size": cache_info.maxsize,
            "lru_cache_current": cache_info.currsize,
            "lru_cache_hits": cache_info.hits,
            "lru_cache_misses": cache_info.misses,
        }

    def _process_augmented_sample(self, sample: Any) -> Any:
        """
        Process/convert an augmented sample to match the expected format.

        This method is called during augmentation loading to ensure augmented
        samples have the same format as real samples returned by _get_real_sample().

        Default implementation returns the sample unchanged (no conversion).
        Subclasses should override this method if augmented samples need format
        conversion (e.g., TensorDataset receives tensors from both real and
        augmented sources, but FieldDataset needs to convert augmented tensors
        to Fields).

        Args:
            sample: Raw augmented sample (format depends on augmentation source)

        Returns:
            Processed sample in the format expected by _get_real_sample()

        Example:
            In FieldDataset, this converts (input_tensor, target_tensor) tuples
            to (initial_fields, target_fields) format using field converters.
        """
        return sample

    # ==================== Abstract Methods (Must be Implemented by Subclasses) ====================

    @abstractmethod
    def _load_simulation_uncached(self, sim_idx: int) -> Any:
        """
        Load simulation data from cache (uncached, will be wrapped by LRU cache).

        This method is automatically wrapped with LRU caching, so it will only be
        called when the simulation is not in the LRU cache.

        Subclasses should implement the actual loading logic here:
        - For TensorDataset: Load tensors and return Dict[str, torch.Tensor]
        - For FieldDataset: Load tensors + metadata and return appropriate format

        Args:
            sim_idx: Simulation index to load

        Returns:
            Simulation data in subclass-specific format
        """
        pass

    @abstractmethod
    def _get_real_sample(self, idx: int):
        """
        Get a real (non-augmented) sample in the appropriate format.

        This method should:
        1. Determine simulation and starting frame from idx
        2. Load simulation data using self._cached_load_simulation(sim_idx)
        3. Process and return sample in the correct format

        Args:
            idx: Real sample index (0 to num_real-1)

        Returns:
            Sample in subclass-specific format:
            - TensorDataset: Tuple[torch.Tensor, torch.Tensor] (initial, targets)
            - FieldDataset: Tuple[Dict[str, Field], Dict[str, List[Field]]]
        """
        pass
