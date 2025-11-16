"""
FieldDataset - Refactored for Synthetic Prediction Augmentation

Key Changes:
- Augmented data = synthetically-generated prediction trajectories
- Trajectories stored in cache format: {'tensor_data': {field: [C, T, H, W]}}
- _get_augmented_sample() applies windowing to trajectories
- Clear distinction maintained between real and synthetically-generated data
"""

from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from phi.math import Shape
from phi.field._field_math import Extrapolation
import torch
from phi.field import Field
from phi.torch.flow import *

from .abstract_dataset import AbstractDataset
from .data_manager import DataManager
from .abstract_dataset import FilteringManager
from .augmentation_manager import AugmentationHandler
from src.utils.logger import get_logger
from phi.geom import Box
from phi.math import spatial

logger = get_logger(__name__)


@dataclass
class FieldMetadata:
    """
    Metadata needed to reconstruct a PhiFlow Field from a tensor.
    """

    domain: Box
    resolution: Shape
    extrapolation: Union[Extrapolation, str]
    field_type: str  # 'centered' or 'staggered'
    spatial_dims: Tuple[str, ...]
    channel_dims: Tuple[str, ...]

    @classmethod
    def from_field(self, field: Field) -> "FieldMetadata":
        field_type = "staggered" if field.is_staggered else "centered"
        return self(
            domain=field.bounds,
            resolution=field.resolution,
            extrapolation=field.extrapolation,
            field_type=field_type,
            spatial_dims=tuple(field.shape.spatial.names),
            channel_dims=(tuple(field.shape.channel.names) if field.shape.channel else ()),
        )

    @classmethod
    def from_cache_metadata(
        self, cached_meta: Dict, domain: Box, resolution: Shape
    ) -> "FieldMetadata":
        from phi.math import extrapolation as extrap_module

        extrap_str = cached_meta.get("extrapolation", "ZERO")
        extrapolation_map = {
            "ZERO": extrap_module.ZERO,
            "BOUNDARY": extrap_module.BOUNDARY,
            "PERIODIC": extrap_module.PERIODIC,
            "zero-gradient": extrap_module.ZERO_GRADIENT,
            "ZERO_GRADIENT": extrap_module.ZERO_GRADIENT,
        }

        if extrap_str in extrapolation_map:
            extrapolation = extrapolation_map[extrap_str]
        else:
            for key in extrapolation_map:
                if key in extrap_str.upper():
                    extrapolation = extrapolation_map[key]
                    break
            else:
                extrapolation = extrap_module.ZERO

        field_type = cached_meta.get("field_type", "centered")

        return self(
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
    """
    from phi.math import extrapolation

    field_types = field_types or {}

    metadata_dict = {}
    for field_name in field_names:
        field_type = field_types.get(field_name, "centered")
        if "velocity" in field_name.lower():
            channel_dims = ("vector",)
        else:
            channel_dims = ()

        metadata_dict[field_name] = FieldMetadata(
            domain=model.domain,
            resolution=model.resolution,
            extrapolation=extrapolation.PERIODIC,
            field_type=field_type,
            spatial_dims=tuple(model.resolution.names),
            channel_dims=channel_dims,
        )

    return metadata_dict


class FieldDataset(AbstractDataset):
    """
    PyTorch Dataset that returns PhiFlow Fields for physical training.
    
    Augmentation source: Synthetically-generated prediction trajectories
    - Stored as cache-format dicts: {'tensor_data': {field: [C, T, H, W]}}
    - Windowed on-the-fly during access
    - Remain distinguishable via _is_augmented_prediction()
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        data_manager: DataManager,
        enable_augmentation: bool,
    ):
        """Initialize FieldDataset."""
        self._field_metadata_cache = None
        self._num_augmented_trajectories = 0
        super().__init__(config, data_manager, enable_augmentation)
        
        self._log_dataset_info()

    # ==================== Setup ====================
    
    def _setup_dataset(
        self,
        data_manager: DataManager,
        sim_indices: List[int],
        field_names: List[str],
        num_frames: Optional[int],
        rollout_steps: int,
        cache_dir: str,
        percentage_real_data: float,
        enable_augmentation: bool,
    ) -> Tuple[int, int, List[Any], Optional[FilteringManager]]:
        """Setup dataset components."""
        # Use helper methods from AbstractDataset instead of DatasetBuilder
        num_frames = AbstractDataset.setup_cache(
            data_manager, sim_indices, field_names, num_frames, rollout_steps
        )
        samples_per_sim = AbstractDataset.compute_sliding_window(
            num_frames, rollout_steps
        )
        total_real_samples = len(sim_indices) * samples_per_sim
        # Setup filtering
        index_mapper = None
        if percentage_real_data < 1.0:
            index_mapper = FilteringManager(total_real_samples, percentage_real_data)
            num_real = index_mapper.num_samples
            logger.debug(
                f"  Filtering enabled: using {num_real}/{total_real_samples} "
                f"samples ({percentage_real_data*100:.1f}%)"
            )
        else:
            num_real = total_real_samples
        
        # Setup augmentation (if provided)
        augmented_samples = []
        if enable_augmentation:
            logger.debug(
                f"  Loading augmented samples from cache "
                f"({cache_dir}, alpha={percentage_real_data})..."
            )
            augmented_samples = AugmentationHandler.load_augmentation(
                cache_dir, num_real, percentage_real_data
            )  # Already in correct format
        
        return num_frames, num_real, augmented_samples, index_mapper
    
    def _log_dataset_info(self):
        """Log dataset information."""
        if self.access_policy == "both":
            logger.debug(
                f"  Dataset: {self.num_real} real + {self.num_augmented} "
                f"augmented = {len(self)} samples"
            )
        elif self.access_policy == "real_only":
            logger.debug(f"  Dataset: {self.num_real} real samples")
        elif self.access_policy == "generated_only":
            if self.num_augmented == 0:
                logger.warning(
                    "  Dataset: access_policy=generated_only but no augmented samples!"
                )
            logger.debug(f"  Dataset: {self.num_augmented} generated samples")

    # ==================== Implementation of Abstract Methods ====================
    
    def _load_simulation(self, sim_idx: int) -> Dict[str, Any]:
        """Load simulation data from cache."""
        full_data = self.data_manager.load_simulation(
            sim_idx, field_names=self.field_names, num_frames=self.num_frames
        )
        return full_data
    
    def _extract_sample(self, idx: int) -> Tuple[Dict[str, Field], Dict[str, List[Field]]]:
        """
        Extract a windowed sample from real simulations as Fields.
        
        Returns:
            Tuple of (initial_fields, target_fields)
        """
        sim_idx, start_frame = self._compute_sim_and_frame(idx)
        data = self._cached_load_simulation(sim_idx)
        
        # Move to GPU if needed (temporary)
        for name in data["tensor_data"]:
            data["tensor_data"][name] = data["tensor_data"][name].to("cuda")

        # Reconstruct field metadata
        field_metadata = self._reconstruct_metadata(data)
        
        # Convert initial state (single frame)
        initial_fields = self._tensors_to_fields(
            data, field_metadata, start_frame, start_frame + 1
        )
        
        # Convert target rollout (multiple frames)
        target_start = start_frame + 1
        target_end = start_frame + 1 + self.rollout_steps
        target_fields = self._tensors_to_fields(
            data, field_metadata, target_start, target_end
        )

        return initial_fields, target_fields
    
    def _get_augmented_sample(self, idx: int) -> Tuple[Dict[str, Field], Dict[str, List[Field]]]:
        """
        Get a synthetically-generated prediction sample.
        
        UPDATED: Now handles trajectory format with windowing.
        Trajectories are stored as cache-format dicts that can be indexed.
        
        Args:
            idx: Index within augmented samples
        
        Returns:
            Tuple of (initial_fields, target_fields) as Fields
        """
        # Compute which trajectory and which window within that trajectory
        samples_per_traj = self.num_frames - self.rollout_steps + 1
        if samples_per_traj <= 0:
            raise ValueError(
                f"Invalid configuration: num_frames={self.num_frames} must be > "
                f"rollout_steps={self.rollout_steps}"
            )
        
        traj_idx = idx // samples_per_traj
        window_start = idx % samples_per_traj
        
        if traj_idx >= self._num_augmented_trajectories:
            raise IndexError(
                f"Augmented index {idx} out of range "
                f"(trajectory {traj_idx} >= {self._num_augmented_trajectories})"
            )
        # Get the trajectory data (in cache format)
        trajectory_data = self.augmented_samples[traj_idx]
        
        # Extract tensor_data
        if isinstance(trajectory_data, dict) and 'tensor_data' in trajectory_data:
            tensor_data = trajectory_data['tensor_data']
        else:
            # Assume it's already tensor_data
            tensor_data = trajectory_data
        
        # Move to GPU if needed (temporary)
        for field_name in self.field_names:
            if isinstance(tensor_data[field_name], torch.Tensor):
                tensor_data[field_name] = tensor_data[field_name].to("cuda")
        
        # Get field metadata for conversion
        field_metadata = self._get_field_metadata()
        
        # Convert windowed portion to Fields
        # tensor_data format: {field_name: tensor[C, T, H, W]}
        
        # Extract initial state (single frame at window_start)
        initial_fields = self._tensors_to_fields(
            trajectory_data,
            field_metadata,
            window_start,
            window_start + 1
        )
        # Extract target trajectory
        target_start = window_start + 1
        target_end = window_start + 1 + self.rollout_steps
        target_fields = self._tensors_to_fields(
            trajectory_data,
            field_metadata,
            target_start,
            target_end
        )
        
        return initial_fields, target_fields

    # ==================== Augmentation Management ====================

    def set_augmented_trajectories(
        self,
        trajectory_rollouts: List[Dict[str, torch.Tensor]]
    ):
        """
        Set augmented data from synthetically-generated prediction trajectories.
        
        NEW BEHAVIOR:
        - Input: List of cache-format dicts with tensor trajectories
        - Format: [{'tensor_data': {field_name: tensor[C, T, H, W]}}]
        - Trajectories can be windowed using the same indexing logic
        
        Args:
            trajectory_rollouts: List of trajectory dicts in cache format
        """
        if not trajectory_rollouts:
            self.augmented_samples = []
            self.num_augmented = 0
            self._num_augmented_trajectories = 0
            self._total_length = self._compute_length()
            return
        
        logger.debug(f"Setting {len(trajectory_rollouts)} synthetic prediction trajectories...")
        
        # Store trajectories directly (already in cache format)
        self.augmented_samples = trajectory_rollouts
        self._num_augmented_trajectories = len(trajectory_rollouts)
        
        # Calculate total augmented samples (with windowing)
        # Each trajectory of length T can produce (T - rollout_steps) windows
        first_traj = trajectory_rollouts[0]
        
        # Extract tensor_data to check trajectory length
        if isinstance(first_traj, dict) and 'tensor_data' in first_traj:
            first_field = list(first_traj['tensor_data'].values())[0]
        else:
            first_field = list(first_traj.values())[0]
        
        traj_length = first_field.shape[1]  # T dimension in [C, T, H, W]
        
        samples_per_trajectory = traj_length - self.rollout_steps
        if samples_per_trajectory <= 0:
            logger.warning(
                f"Trajectory length {traj_length} too short for "
                f"rollout_steps={self.rollout_steps}"
            )
            self.num_augmented = 0
        else:
            self.num_augmented = self._num_augmented_trajectories * samples_per_trajectory
        
        self._total_length = self._compute_length()
        logger.debug(
            f"Set {self._num_augmented_trajectories} synthetic trajectories "
            f"({self.num_augmented} windowed samples, {samples_per_trajectory} per trajectory)"
        )

    def clear_augmented_trajectories(self):
        """Clear all augmented trajectories."""
        self.augmented_samples = []
        self.num_augmented = 0
        self._num_augmented_trajectories = 0
        self._total_length = self._compute_length()
        logger.debug("Cleared augmented trajectories")

    # ==================== Helper Methods ====================
    
    def _reconstruct_metadata(self, data: Dict[str, Any]) -> Dict[str, FieldMetadata]:
        """Reconstruct FieldMetadata from cached metadata."""

        
        field_metadata_dict = data["metadata"]["field_metadata"]
        field_metadata = {}
        
        for name, meta in field_metadata_dict.items():
            if "bounds_lower" in meta and "bounds_upper" in meta:
                lower = meta["bounds_lower"]
                upper = meta["bounds_upper"]
                
                if len(lower) == 2:
                    domain = Box(x=(lower[0], upper[0]), y=(lower[1], upper[1]))
                elif len(lower) == 3:
                    domain = Box(
                        x=(lower[0], upper[0]),
                        y=(lower[1], upper[1]),
                        z=(lower[2], upper[2])
                    )
                else:
                    domain = Box(x=1, y=1)
            else:
                raise ValueError(
                    f"Invalid cache format for field '{name}'. "
                    f"Missing bounds information."
                )
            
            sample_tensor = data["tensor_data"][name]
            tensor_shape = sample_tensor.shape
            spatial_dims = meta["spatial_dims"]
            
            # Determine spatial offset depending on tensor layout
            if isinstance(sample_tensor, torch.Tensor):
                if sample_tensor.dim() == 4:
                    spatial_offset = 2  # [C, T, H, W]
                elif sample_tensor.dim() == 3:
                    spatial_offset = 1  # [C, H, W]
                else:
                    spatial_offset = 2  # Fallback
            else:
                spatial_offset = 2

            resolution_sizes = {
                dim: tensor_shape[i + spatial_offset] for i, dim in enumerate(spatial_dims)
            }
            resolution = spatial(**resolution_sizes)
            
            field_metadata[name] = FieldMetadata.from_cache_metadata(
                meta, domain, resolution
            )
        
        return field_metadata

    def _tensors_to_fields(
        self,
        data: Dict[str, Any],
        field_metadata: Dict[str, FieldMetadata],
        start_frame: int,
        end_frame: int
    ) -> Dict[str, Field]:
        """
        Convert windowed tensors to Fields.
        
        Args:
            data: Data dict with 'tensor_data'
            field_metadata: Metadata for reconstruction
            start_frame: Start of time window
            end_frame: End of time window (exclusive)
            
        Returns:
            Dict mapping field names to Field objects
        """
        fields_dict = {}
        
        for name in self.field_names:
            sample_tensor = data["tensor_data"][name]  # [C, T, H, W]
            
            # Validate shape
            assert sample_tensor.dim() == 4, \
                f"Expected tensor [C, T, H, W], got {sample_tensor.shape}"
            
            # Slice time window
            field_meta = field_metadata[name]
            window_tensor = sample_tensor[:, start_frame:end_frame, :, :]  # [C, T_window, H, W]
            
            # Create PhiML tensor with explicit dimensions
            phiml_tensor = math.tensor(
                window_tensor, 
                channel("vector"), 
                batch("time"), 
                spatial(*field_meta.spatial_dims)
            )
            
            # Create Field
            window_field = CenteredGrid(
                phiml_tensor, 
                field_meta.extrapolation, 
                bounds=field_meta.domain
            )
            
            fields_dict[name] = window_field
        
        return fields_dict
        
    def _get_field_metadata(self) -> Dict[str, FieldMetadata]:
        """Get field metadata (cached)."""
        if self._field_metadata_cache is not None:
            return self._field_metadata_cache
        
        first_sim_idx = self.sim_indices[0]
        data = self._cached_load_simulation(first_sim_idx)
        
        self._field_metadata_cache = self._reconstruct_metadata(data)
        return self._field_metadata_cache

    # ==================== Utility Methods ====================

    def _is_augmented_prediction(self, idx: int) -> bool:
        """
        Check if a sample index corresponds to an augmented prediction.
        
        Args:
            idx: Global sample index
        
        Returns:
            True if from augmented prediction, False if real
        """
        if self.access_policy == "generated_only":
            return True
        elif self.access_policy == "real_only":
            return False
        else:  # 'both'
            return idx >= self.num_real

    def get_sample_source(self, idx: int) -> str:
        """
        Get the source of a sample.
        
        Args:
            idx: Sample index
        
        Returns:
            'real' or 'synthetic_generated'
        """
        return 'synthetic_generated' if self._is_augmented_prediction(idx) else 'real'

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FieldDataset(\n"
            f"  simulations={len(self.sim_indices)},\n"
            f"  samples={len(self)} (real={self.num_real}, aug={self.num_augmented}),\n"
            f"  fields={len(self.field_names)},\n"
            f"  frames={self.num_frames},\n"
            f"  predict_steps={self.rollout_steps}\n"
            f")"
        )