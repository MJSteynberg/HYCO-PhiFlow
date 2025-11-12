"""
Field Dataset for Physical Model Training

Returns PhiFlow Fields suitable for physics-based simulation.
Parallel structure to TensorDataset with Field-specific implementations.
"""

from typing import List, Optional, Dict, Any, Tuple
import torch
from phi.field import Field
from phi.geom import Box
from phiml.math import spatial

from .abstract_dataset import AbstractDataset
from .data_manager import DataManager
from .dataset_utilities import DatasetBuilder, AugmentationHandler, FilteringManager
from src.utils.field_conversion import FieldMetadata, make_converter, make_batch_converter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FieldDataset(AbstractDataset):
    """
    PyTorch Dataset that returns PhiFlow Fields for physical training.
    
    Returns samples in format:
    - initial_fields: Dict[field_name, Field] - all fields at starting timestep
    - target_fields: Dict[field_name, List[Field]] - all fields for T steps
    
    Args:
        data_manager: DataManager for loading cached data
        sim_indices: List of simulation indices
        field_names: List of field names to load
        num_frames: Number of frames per simulation (None = auto-detect)
        num_predict_steps: Number of prediction steps
        augmentation_config: Optional augmentation configuration
        access_policy: 'both', 'real_only', or 'generated_only'
        max_cached_sims: LRU cache size
        move_to_gpu: If True, move tensors to GPU before Field conversion
        percentage_real_data: Percentage of real data to use (0.0 < p <= 1.0)
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
        percentage_real_data: float = 1.0,
    ):
        """Initialize FieldDataset with validation and setup."""
        # Store field-specific attributes
        self._field_metadata_cache = None
        # Setup dataset using builder
        logger.debug("Setting up FieldDataset...")
        num_frames, num_real, augmented_samples, index_mapper = self._setup_dataset(
            data_manager, sim_indices, field_names, num_frames, num_predict_steps,
            augmentation_config, percentage_real_data
        )
        
        # Call parent constructor with pre-computed values
        super().__init__(
            data_manager=data_manager,
            sim_indices=sim_indices,
            field_names=field_names,
            num_frames=num_frames,
            num_predict_steps=num_predict_steps,
            access_policy=access_policy,
            num_real=num_real,
            augmented_samples=augmented_samples,
            index_mapper=index_mapper,
            max_cached_sims=max_cached_sims,
        )
        
        # Log dataset info
        self._log_dataset_info()
    
    # ==================== Setup ====================
    
    def _setup_dataset(
        self,
        data_manager: DataManager,
        sim_indices: List[int],
        field_names: List[str],
        num_frames: Optional[int],
        num_predict_steps: int,
        augmentation_config: Optional[Dict[str, Any]],
        percentage_real_data: float,
    ) -> Tuple[int, int, List[Any], Optional[FilteringManager]]:
        """
        Setup dataset components.
        
        Returns:
            Tuple of (num_frames, num_real, augmented_samples, index_mapper)
        """
        builder = DatasetBuilder(data_manager)
        
        # Setup cache and determine num_frames
        num_frames = builder.setup_cache(sim_indices, field_names, num_frames)
        
        # Compute sliding window
        samples_per_sim = builder.compute_sliding_window(num_frames, num_predict_steps)
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
        
        # Setup augmentation
        augmented_samples = []
        if augmentation_config:
            raw_samples = AugmentationHandler.load_augmentation(
                augmentation_config, num_real, num_predict_steps, field_names
            )
            # Process each augmented sample
            augmented_samples = [
                self._process_augmented_sample(sample) for sample in raw_samples
            ]
        
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
    
    def _load_simulation(self, sim_idx: int) -> Dict[str, torch.Tensor]:
        """
        Load simulation data from the data manager."""
        # Otherwise load real simulation from cache
        full_data = self.data_manager.get_or_load_simulation(
            sim_idx, field_names=self.field_names, num_frames=self.num_frames
        )
        
        return full_data
    
    
    
    def _extract_sample(self, idx: int) -> Tuple[Dict[str, Field], Dict[str, List[Field]]]:
        """
        Extract a single sample as Fields.
        
        Args:
            idx: Actual (unfiltered) sample index
        
        Returns:
            Tuple of (initial_fields, target_fields)
            - initial_fields: Dict[field_name, Field]
            - target_fields: Dict[field_name, List[Field]]
        """
        # Compute simulation and frame
        sim_idx, start_frame = self._compute_sim_and_frame(idx)
        
        # Load simulation
        data = self._cached_load_simulation(sim_idx)
        
        # Reconstruct field metadata
        field_metadata = self._reconstruct_metadata(data)
        
        # Convert initial state to Fields
        initial_fields = self._tensors_to_fields(
            data, field_metadata, start_frame, start_frame + 1
        )
        # Extract single timestep
        initial_fields = {name: fields[0] for name, fields in initial_fields.items()}
        
        # Convert target rollout to Fields
        target_start = start_frame + 1
        target_end = start_frame + 1 + self.num_predict_steps
        target_fields = self._tensors_to_fields(
            data, field_metadata, target_start, target_end
        )
        
        return initial_fields, target_fields
    
    def _process_augmented_sample(self, sample: Any) -> Tuple[Dict[str, Field], Dict[str, List[Field]]]:
        """
        Convert augmented sample to Field format.
        
        Handles both:
        - Pre-processed (input_tensor, target_tensor) tuples
        - Raw trajectory windows (List[Dict[str, Field]])
        
        Args:
            sample: Augmented sample (tuple or trajectory window)
        
        Returns:
            Tuple of (initial_fields, target_fields) as Fields
        """
        # Check if already Fields
        if isinstance(sample, tuple) and len(sample) == 2:
            initial, targets = sample
            if isinstance(initial, dict) and all(isinstance(f, Field) for f in initial.values()):
                # Already in Field format
                return initial, targets
            elif isinstance(initial, torch.Tensor):
                # Tensor format - need to convert
                return self._convert_tensor_sample(initial, targets)
        
        # Handle trajectory window
        if isinstance(sample, list) and len(sample) > 0:
            if isinstance(sample[0], dict):
                # Raw trajectory window
                return self._convert_trajectory_window(sample)
        
        # Fallback
        return sample
    
    def _convert_tensor_sample(
        self,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor
    ) -> Tuple[Dict[str, Field], Dict[str, List[Field]]]:
        """
        Convert tensor-based sample to Fields.
        
        Args:
            input_tensor: [C_all, H, W]
            target_tensor: [T, C_all, H, W]
        
        Returns:
            Tuple of (initial_fields, target_fields)
        """
        # Move to GPU if configured
        if torch.cuda.is_available():
            if not input_tensor.is_cuda:
                input_tensor = input_tensor.cuda()
            if not target_tensor.is_cuda:
                target_tensor = target_tensor.cuda()
        
        # Get field metadata
        field_metadata = self._get_field_metadata()
        
        # Create batch converter
        batch_converter = make_batch_converter(field_metadata)
        
        # Convert input
        input_with_batch = input_tensor.unsqueeze(0)  # [1, C_all, H, W]
        initial_fields = batch_converter.tensor_to_fields_batch(input_with_batch)
        
        # Remove batch dimension
        for name, field in initial_fields.items():
            if "batch" in field.shape:
                initial_fields[name] = field.batch[0]
        
        # Convert targets
        if target_tensor.dim() == 3:
            target_tensor = target_tensor.unsqueeze(0)
        
        num_timesteps = target_tensor.shape[0]
        target_fields = {name: [] for name in self.field_names}
        
        for t in range(num_timesteps):
            timestep_tensor = target_tensor[t].unsqueeze(0)
            timestep_fields = batch_converter.tensor_to_fields_batch(timestep_tensor)
            
            for name, field in timestep_fields.items():
                if "batch" in field.shape:
                    target_fields[name].append(field.batch[0])
                else:
                    target_fields[name].append(field)
        
        return initial_fields, target_fields
    
    def _convert_trajectory_window(
        self,
        window_states: List[Dict[str, Field]]
    ) -> Tuple[Dict[str, Field], Dict[str, List[Field]]]:
        """
        Convert trajectory window to Field format.
        
        Args:
            window_states: List of states [Dict[str, Field]]
                         Length = num_predict_steps + 1
        
        Returns:
            Tuple of (initial_fields, target_fields)
        """
        # First state is initial
        initial_fields = window_states[0]
        
        # Remaining states are targets
        target_states = window_states[1:]
        
        # Reorganize from List[Dict] to Dict[List]
        target_fields = {}
        for field_name in self.field_names:
            target_fields[field_name] = [state[field_name] for state in target_states]
        
        return initial_fields, target_fields
    
    # ==================== Helper Methods ====================
    
    def _reconstruct_metadata(self, data: Dict[str, Any]) -> Dict[str, FieldMetadata]:
        """
        Reconstruct FieldMetadata from cached metadata.
        
        Args:
            data: Data dictionary with 'metadata' key
        
        Returns:
            Dictionary mapping field names to FieldMetadata
        """
        field_metadata_dict = data["metadata"]["field_metadata"]
        field_metadata = {}
        
        for name, meta in field_metadata_dict.items():
            # Reconstruct domain
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
            
            # Extract resolution
            tensor_shape = data["tensor_data"][name].shape  # [T, C, H, W]
            spatial_dims = meta["spatial_dims"]
            resolution_sizes = {
                dim: tensor_shape[i + 2] for i, dim in enumerate(spatial_dims)
            }
            resolution = spatial(**resolution_sizes)
            
            # Create FieldMetadata
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
    ) -> Dict[str, List[Field]]:
        """
        Convert tensors to Fields for a frame range.
        
        Args:
            data: Data dictionary with 'tensor_data'
            field_metadata: Field metadata dictionary
            start_frame: Starting frame index
            end_frame: Ending frame index (exclusive)
        
        Returns:
            Dictionary mapping field names to list of Fields
        """
        fields_dict = {}
        
        for name in self.field_names:
            # Extract tensors
            field_tensors = data["tensor_data"][name][start_frame:end_frame]
            
            # Move to GPU if configured
            if torch.cuda.is_available():
                if not field_tensors.is_cuda:
                    field_tensors = field_tensors.cuda()
            
            # Create converter
            field_meta = field_metadata[name]
            converter = make_converter(field_meta)
            
            # Convert each timestep
            fields_list = []
            for t in range(len(field_tensors)):
                tensor_t = field_tensors[t:t+1]
                field_t = converter.tensor_to_field(tensor_t, field_meta, time_slice=0)
                fields_list.append(field_t)
            
            fields_dict[name] = fields_list
        
        return fields_dict
    
    def _get_field_metadata(self) -> Dict[str, FieldMetadata]:
        """
        Get field metadata (cached).
        
        Returns:
            Dictionary mapping field names to FieldMetadata
        """
        if self._field_metadata_cache is not None:
            return self._field_metadata_cache
        
        # Load from first simulation
        first_sim_idx = self.sim_indices[0]
        data = self._cached_load_simulation(first_sim_idx)
        
        self._field_metadata_cache = self._reconstruct_metadata(data)
        return self._field_metadata_cache
    
    # ==================== Utility Methods ====================

    def _compute_sim_and_frame(self, idx: int) -> Tuple[int, int]:
        """
        Compute simulation index and starting frame from sample index.
        Handles both real and synthetic simulations.
        """
        samples_per_sim = self.num_frames - self.num_predict_steps
        
        # Check if this is a synthetic simulation
        if hasattr(self, '_num_real_only'):
            real_samples = self._num_real_only
            if idx >= real_samples:
                # This is a synthetic simulation - compute offset within synthetic data
                synthetic_sample_idx = idx - real_samples
                sim_offset = synthetic_sample_idx // samples_per_sim
                start_frame = synthetic_sample_idx % samples_per_sim
                sim_idx = len(self.sim_indices) + sim_offset
                return sim_idx, start_frame
        
        # Real simulation
        sim_offset = idx // samples_per_sim
        start_frame = idx % samples_per_sim
        sim_idx = self.sim_indices[sim_offset]
        return sim_idx, start_frame
        
    
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
    
    def get_field_info(self) -> Dict[str, Any]:
        """Get field configuration information."""
        return {
            "field_names": self.field_names,
            "num_fields": len(self.field_names),
        }
    
    def get_sample_info(self, idx: int = 0) -> Dict[str, Any]:
        """
        Get information about a sample's structure.
        
        Args:
            idx: Sample index to inspect
        
        Returns:
            Dictionary with sample structure information
        """
        if idx >= self.num_real:
            raise ValueError(
                f"Index {idx} is augmented, cannot inspect from real data"
            )
        
        initial_fields, target_fields = self._extract_sample(idx)
        
        info = {
            "initial_fields": list(initial_fields.keys()),
            "target_fields": list(target_fields.keys()),
            "num_initial_fields": len(initial_fields),
            "num_target_fields": len(target_fields),
            "target_timesteps": {
                name: len(fields) for name, fields in target_fields.items()
            },
        }
        
        # Add shape info
        if initial_fields:
            first_name = list(initial_fields.keys())[0]
            first_field = initial_fields[first_name]
            info["field_type"] = type(first_field).__name__
            info["spatial_shape"] = first_field.shape.spatial
        
        return info
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FieldDataset(\n"
            f"  simulations={len(self.sim_indices)},\n"
            f"  samples={len(self)} (real={self.num_real}, aug={self.num_augmented}),\n"
            f"  fields={len(self.field_names)},\n"
            f"  frames={self.num_frames},\n"
            f"  predict_steps={self.num_predict_steps},\n"
            f"  move_to_gpu={torch.cuda.is_available()}\n"
            f")"
        )