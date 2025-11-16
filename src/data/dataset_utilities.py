"""
Dataset Utilities - Shared Components

Extracted from AbstractDataset to reduce complexity:
- DatasetBuilder: Setup and validation
- AugmentationHandler: Load and process augmentation
- FilteringManager: Percentage filtering and resampling
"""

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import random
import torch
from phi.field import Field

from .data_manager import DataManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetBuilder:
    """
    Handles dataset setup and validation.
    
    Responsibilities:
    - Cache validation and warming
    - Sliding window computation
    - Coordinate all setup components
    """
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
    
    def setup_cache(
        self,
        sim_indices: List[int],
        field_names: List[str],
        num_frames: Optional[int],
        num_predict_steps: Optional[int] = None,
    ) -> int:
        """
        Validate cache and determine num_frames.
        
        Args:
            sim_indices: List of simulation indices
            field_names: List of field names
            num_frames: Optional number of frames
        
        Returns:
            Determined num_frames
        """
        # Determine num_frames if not specified
        if num_frames is None:
            logger.debug("num_frames not specified, determining from first simulation...")
            first_sim_data = self.data_manager.load_simulation(
                sim_indices[0], field_names=field_names, num_frames=None
            )
            sample_tensor = first_sim_data["tensor_data"][field_names[0]]
            # Detect BVTS canonical layout [B, C, T, *spatial] -> time at dim=2
            if isinstance(sample_tensor, torch.Tensor):
                if sample_tensor.dim() >= 3:
                    if sample_tensor.dim() == 5:
                        # BVTS with batch: [B, C, T, *spatial]
                        num_frames = sample_tensor.shape[2]
                    elif sample_tensor.dim() == 4:
                        # BVTS without batch per-field: [C, T, *spatial]
                        num_frames = sample_tensor.shape[1]
                    elif sample_tensor.dim() == 3:
                        # [C, H, W] -> single frame
                        num_frames = 1
                    else:
                        num_frames = sample_tensor.shape[0]
                else:
                    num_frames = 1
            else:
                raise RuntimeError("First simulation tensor is not a torch.Tensor")

            logger.debug(f"Determined num_frames = {num_frames}")
            del first_sim_data

        # If a predict horizon is provided and the discovered num_frames is
        # too small for at least one training window, attempt to force a
        # re-load with a larger num_frames. This handles stale/invalid cache
        # cases where metadata reports too few frames.
        if num_predict_steps is not None and num_frames < (num_predict_steps + 1):
            logger.warning(
                f"Discovered num_frames={num_frames} < required {num_predict_steps + 1}; "
                "attempting to reload simulation with larger frame count..."
            )
            try:
                # Ask DataManager to load at least the required number of frames
                forced = self.data_manager.load_simulation(
                    sim_indices[0], field_names=field_names, num_frames=(num_predict_steps + 1)
                )
                forced_tensor = forced["tensor_data"][field_names[0]]
                if isinstance(forced_tensor, torch.Tensor):
                    if forced_tensor.dim() == 5:
                        num_frames = forced_tensor.shape[2]
                    elif forced_tensor.dim() == 4:
                        # BVTS without batch per-field: [C, T, *spatial]
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
                # Let the caller handle the invalid configuration
                raise
        
        # Check and cache uncached simulations
        uncached_sims = [
            sim_idx for sim_idx in sim_indices
            if not self.data_manager.is_cached(sim_idx)
        ]
        
        if uncached_sims:
            logger.info(f"Caching {len(uncached_sims)} simulations...")
            for i, sim_idx in enumerate(uncached_sims, 1):
                logger.debug(f"  [{i}/{len(uncached_sims)}] Caching simulation {sim_idx}...")
                _ = self.data_manager.load_simulation(
                    sim_idx, field_names=field_names, num_frames=num_frames
                )
            logger.debug("All simulations cached successfully!")
        else:
            logger.debug(f"All {len(sim_indices)} simulations already cached.")
        
        return num_frames
    
    def compute_sliding_window(
        self,
        num_frames: int,
        num_predict_steps: int
    ) -> int:
        """
        Compute samples per simulation for sliding window.
        
        Args:
            num_frames: Total frames in simulation
            num_predict_steps: Number of prediction steps
        
        Returns:
            Number of samples per simulation
        
        Raises:
            ValueError: If window configuration is invalid
        """
        if num_frames < num_predict_steps + 1:
            raise ValueError(
                f"num_frames ({num_frames}) must be >= num_predict_steps + 1 "
                f"({num_predict_steps + 1})"
            )
        
        samples_per_sim = num_frames - num_predict_steps
        
        if samples_per_sim <= 0:
            raise ValueError(
                f"Invalid sliding window: num_frames ({num_frames}) must be > "
                f"num_predict_steps ({num_predict_steps})"
            )
        
        return samples_per_sim


class AugmentationHandler:
    """
    Handles loading and processing of augmented samples.
    
    Responsibilities:
    - Load from memory or cache
    - Process raw trajectory data
    - Convert to appropriate format
    """
    
    @staticmethod
    def load_augmentation(
        config: Dict[str, Any],
        num_real: int,
        num_predict_steps: int,
        field_names: List[str]
    ) -> List[Any]:
        """
        Load augmented samples based on configuration.
        
        Args:
            config: Augmentation configuration dict
            num_real: Number of real samples (for alpha calculation)
            num_predict_steps: Number of prediction steps
            field_names: List of field names
        
        Returns:
            List of augmented samples (format depends on mode)
        """
        mode = config.get("mode", "cache")
        alpha = config.get("alpha", 0.0)
        
        logger.debug(f"Loading augmentation (mode={mode}, alpha={alpha})...")
        
        if mode == "memory":
            return AugmentationHandler._load_from_memory(
                config, num_predict_steps, field_names
            )
        elif mode == "cache":
            return AugmentationHandler._load_from_cache(
                config, num_real, alpha
            )
        else:
            raise ValueError(f"Unknown augmentation mode: {mode}")
    
    @staticmethod
    def _load_from_memory(
        config: Dict[str, Any],
        num_predict_steps: int,
        field_names: List[str]
    ) -> List[Any]:
        """Load augmentation from memory."""
        if "data" not in config:
            raise ValueError("Augmentation mode 'memory' requires 'data' key")
        
        data = config["data"]
        
        # Check if data is raw trajectories or processed samples
        if AugmentationHandler._is_trajectory_data(data):
            logger.debug("  Processing raw trajectory data...")
            samples = AugmentationHandler._process_trajectory_data(
                data, num_predict_steps, field_names
            )
        else:
            # Pre-processed samples
            samples = data
        
        logger.debug(f"  Loaded {len(samples)} augmented samples from memory")
        return samples
    
    @staticmethod
    def _load_from_cache(
        config: Dict[str, Any],
        num_real: int,
        alpha: float
    ) -> List[Any]:
        """Load augmentation from cache directory."""
        if "cache_dir" not in config:
            raise ValueError("Augmentation mode 'cache' requires 'cache_dir' key")
        
        cache_dir = config["cache_dir"]
        expected_count = int(num_real * alpha) if alpha > 0 else None
        
        cache_path = Path(cache_dir)
        
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Augmentation cache not found: {cache_dir}\n"
                f"Please run cache generation first."
            )
        
        cache_files = sorted(cache_path.glob("sample_*.pt"))
        
        if len(cache_files) == 0:
            raise ValueError(
                f"No cached samples found in {cache_dir}\n"
                f"Expected to find sample_*.pt files."
            )
        
        if expected_count is not None and len(cache_files) != expected_count:
            logger.warning(
                f"Expected {expected_count} cached samples but found {len(cache_files)}"
            )
        
        samples = []
        for file_path in cache_files:
            try:
                data = torch.load(file_path, map_location="cpu")
                if isinstance(data, dict):
                    samples.append((data["input"], data["target"]))
                else:
                    samples.append(data)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                raise
        
        logger.debug(f"  Loaded {len(samples)} cached augmented samples")
        return samples
    
    @staticmethod
    def _is_trajectory_data(data: Any) -> bool:
        """Check if data is raw trajectories vs processed samples."""
        if not isinstance(data, list) or len(data) == 0:
            return False
        
        first_item = data[0]
        
        # Check if it's a list of dicts (trajectory)
        if isinstance(first_item, list):
            if len(first_item) > 0 and isinstance(first_item[0], dict):
                return True
        
        return False
    
    @staticmethod
    def _process_trajectory_data(
        trajectories: List[List[Dict[str, Field]]],
        num_predict_steps: int,
        field_names: List[str]
    ) -> List[List[Dict[str, Field]]]:
        """
        Apply sliding window to raw trajectories.
        
        Returns list of windowed states (not yet converted to final format).
        Format: List[List[Dict[str, Field]]] where inner list has length
        num_predict_steps + 1.
        """
        logger.debug(
            f"  Windowing {len(trajectories)} trajectories with "
            f"num_predict_steps={num_predict_steps}"
        )
        
        all_windows = []
        
        for traj_idx, trajectory in enumerate(trajectories):
            traj_length = len(trajectory)
            
            if traj_length < num_predict_steps + 1:
                logger.warning(
                    f"  Trajectory {traj_idx} too short ({traj_length} steps), "
                    f"need at least {num_predict_steps + 1}. Skipping."
                )
                continue
            
            # Apply sliding window
            num_windows = traj_length - num_predict_steps
            
            for window_start in range(num_windows):
                window_states = trajectory[
                    window_start : window_start + num_predict_steps + 1
                ]
                all_windows.append(window_states)
        
        logger.debug(
            f"  Created {len(all_windows)} windowed samples from "
            f"{len(trajectories)} trajectories"
        )
        
        return all_windows


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
        seed: Optional[int] = None
    ):
        """
        Initialize filtering manager.
        
        Args:
            total_samples: Total number of available samples
            percentage: Percentage of data to use (0.0 < percentage <= 1.0)
            seed: Optional random seed
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
        return sorted(all_indices[:self.num_samples])
    
    def get_actual_index(self, filtered_idx: int) -> int:
        """
        Map filtered index to actual index.
        
        Args:
            filtered_idx: Index in filtered dataset (0 to num_samples-1)
        
        Returns:
            Actual index in full dataset
        """
        if filtered_idx >= self.num_samples:
            raise IndexError(
                f"Filtered index {filtered_idx} out of range [0, {self.num_samples})"
            )
        return self._active_indices[filtered_idx]
    
    def resample(self, seed: Optional[int] = None):
        """
        Resample the subset of data.
        
        Args:
            seed: Optional random seed for reproducibility
        """
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

from dataclasses import dataclass
from typing import Dict, Union, Tuple
from phi.field import Field
from phi.geom import Box
from phi.math import Shape
from phi.field._field_math import Extrapolation


@dataclass
class FieldMetadata:
    """
    Metadata needed to reconstruct a PhiFlow Field from a tensor.

    This stores all the information required to convert a PyTorch tensor
    back into its original Field representation.

    Attributes:
        domain: The physical domain (Box) for the field
        resolution: The spatial resolution (Shape with x, y dimensions)
        extrapolation: Boundary condition (e.g., 'periodic', 'zero-gradient', etc.)
        field_type: 'centered' or 'staggered'
        spatial_dims: Names of spatial dimensions (e.g., ['x', 'y'])
        channel_dims: Names of channel dimensions (e.g., ['vector'])
    """

    domain: Box
    resolution: Shape
    extrapolation: Union[Extrapolation, str]
    field_type: str  # 'centered' or 'staggered'
    spatial_dims: Tuple[str, ...]
    channel_dims: Tuple[str, ...]

    @classmethod
    def from_field(cls, field: Field) -> "FieldMetadata":
        """
        Extract metadata from an existing Field.

        Args:
            field: The Field to extract metadata from

        Returns:
            FieldMetadata object
        """
        # Determine field type using is_staggered property
        field_type = "staggered" if field.is_staggered else "centered"

        return cls(
            domain=field.bounds,
            resolution=field.resolution,
            extrapolation=field.extrapolation,
            field_type=field_type,
            spatial_dims=tuple(field.shape.spatial.names),
            channel_dims=(
                tuple(field.shape.channel.names) if field.shape.channel else ()
            ),
        )

    @classmethod
    def from_cache_metadata(
        cls, cached_meta: Dict, domain: Box, resolution: Shape
    ) -> "FieldMetadata":
        """
        Reconstruct FieldMetadata from cached metadata dictionary.

        Args:
            cached_meta: Dictionary containing field metadata from cache
            domain: The physical domain (must be provided externally)
            resolution: The spatial resolution (must be provided externally)

        Returns:
            FieldMetadata object
        """
        # Parse extrapolation from string
        extrap_str = cached_meta.get("extrapolation", "ZERO")

        # Map common extrapolation strings to PhiFlow objects
        from phi.math import extrapolation as extrap_module

        extrapolation_map = {
            "ZERO": extrap_module.ZERO,
            "BOUNDARY": extrap_module.BOUNDARY,
            "PERIODIC": extrap_module.PERIODIC,
            "zero-gradient": extrap_module.ZERO_GRADIENT,
            "ZERO_GRADIENT": extrap_module.ZERO_GRADIENT,
        }

        # Try to parse the extrapolation
        if extrap_str in extrapolation_map:
            extrapolation = extrapolation_map[extrap_str]
        else:
            # Try to extract the extrapolation name from a string like "<ZERO>"
            for key in extrapolation_map:
                if key in extrap_str.upper():
                    extrapolation = extrapolation_map[key]
                    break
            else:
                extrapolation = extrap_module.ZERO  # Default fallback

        # Determine field type (default to centered if not specified)
        field_type = cached_meta.get("field_type", "centered")

        return cls(
            domain=domain,
            resolution=resolution,
            extrapolation=extrapolation,
            field_type=field_type,
            spatial_dims=tuple(cached_meta.get("spatial_dims", ["x", "y"])),
            channel_dims=tuple(cached_meta.get("channel_dims", [])),
        )


def create_field_metadata_from_model(
    model, field_names: list[str], field_types: Dict[str, str] = None
) -> Dict[str, FieldMetadata]:
    """
    Create FieldMetadata for each field from a PhysicalModel instance.

    This is useful for physical trainers that need to convert tensors
    back to Fields for use with the model.

    Args:
        model: PhysicalModel instance with domain, resolution attributes
        field_names: List of field names (e.g., ['velocity', 'density'])
        field_types: Optional dict mapping field names to types ('centered' or 'staggered')
                    Defaults to 'centered' for all fields

    Returns:
        Dictionary mapping field names to FieldMetadata

    Example:
        >>> model = BurgersModel(domain=Box(...), resolution=spatial(x=128, y=128), ...)
        >>> metadata = create_field_metadata_from_model(model, ['velocity'], {'velocity': 'staggered'})
    """
    from phi.math import extrapolation

    field_types = field_types or {}

    metadata_dict = {}
    for field_name in field_names:
        field_type = field_types.get(field_name, "centered")

        # Determine channel dimensions based on common field types
        if "velocity" in field_name.lower():
            channel_dims = ("vector",)  # Velocity is typically a vector field
        else:
            channel_dims = ()  # Scalar field

        metadata_dict[field_name] = FieldMetadata(
            domain=model.domain,
            resolution=model.resolution,
            extrapolation=extrapolation.PERIODIC,  # Default, may need to be configurable
            field_type=field_type,
            spatial_dims=tuple(model.resolution.names),
            channel_dims=channel_dims,
        )

    return metadata_dict

    

from phi.torch.flow import stack, batch
from typing import List, Dict, Tuple
from phi.field import Field

# In dataset_utilities.py or dataloader_factory.py

def tensor_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Collate per-sample tensors [V, T, H, W] into batches [B, V, T, H, W].
    
    Args:
        batch: List of (initial, targets) tuples, each [V, T, H, W]
    
    Returns:
        (batched_initial, batched_targets) each [B, V, T, H, W]
    """
    initials, targets = zip(*batch)

    logger.info(f"Device tensor_collate_fn: batching {len(batch)} samples on device {initials[0].device, targets[0].device}, ")
    
    # Stack along new batch dimension
    batched_initial = torch.stack(initials, dim=0)  # [B, V, T_init, H, W]
    batched_targets = torch.stack(targets, dim=0)   # [B, V, T_pred, H, W]
    
    return batched_initial, batched_targets


def field_collate_fn(field_batch: List[Tuple[Dict[str, Field], Dict[str, List[Field]]]]):
    """
    Collate per-sample Fields into batched Fields.
    
    Args:
        batch: List of (initial_fields, target_fields) tuples
    
    Returns:
        (batched_initial, batched_targets) with batch dimension in Fields
    """

    initial_fields, target_fields = zip(*field_batch)
    
    # Stack initial fields
    field_names = initial_fields[0].keys()
    stacked_initial = {}
    stacked_targets = {}
    for name in field_names:
        initial = [sample[name] for sample in initial_fields]
        targets = [sample[name] for sample in target_fields]
        stacked_initial[name] = stack(initial, batch('batch'))
        stacked_targets[name] = stack(targets, batch('batch'))
  
    return stacked_initial, stacked_targets
