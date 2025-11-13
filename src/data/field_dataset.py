"""
FieldDataset - Refactored for Synthetic Prediction Augmentation

Key Changes:
- Augmented data = synthetically-generated predictions (pre-windowed)
- Predictions stored as pre-converted Fields (no repeated conversion)
- _get_augmented_sample() returns pre-windowed Field pairs
- Clear distinction maintained between real and synthetically-generated data
"""

from typing import List, Optional, Dict, Any, Tuple
import torch
from phi.field import Field

from .abstract_dataset import AbstractDataset
from .data_manager import DataManager
from .dataset_utilities import DatasetBuilder, AugmentationHandler, FilteringManager
from src.utils.field_conversion import FieldMetadata, make_converter, make_batch_converter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FieldDataset(AbstractDataset):
    """
    PyTorch Dataset that returns PhiFlow Fields for physical training.
    
    Augmentation source: Synthetically-generated predictions
    - Stored as pre-converted Field pairs (initial_fields, target_fields)
    - Already windowed (no windowing needed)
    - Remain distinguishable via _is_augmented_prediction()
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
        """Initialize FieldDataset."""
        self._field_metadata_cache = None
        
        logger.debug("Setting up FieldDataset...")
        num_frames, num_real, augmented_samples, index_mapper = self._setup_dataset(
            data_manager, sim_indices, field_names, num_frames, num_predict_steps,
            augmentation_config, percentage_real_data
        )
        
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
        """Setup dataset components."""
        builder = DatasetBuilder(data_manager)
        
        num_frames = builder.setup_cache(sim_indices, field_names, num_frames)
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
        
        # Setup augmentation (if provided)
        augmented_samples = []
        if augmentation_config:
            raw_samples = AugmentationHandler.load_augmentation(
                augmentation_config, num_real, num_predict_steps, field_names
            )
            # Process samples if needed
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
    
    def _load_simulation(self, sim_idx: int) -> Dict[str, Any]:
        """Load simulation data from cache."""
        full_data = self.data_manager.get_or_load_simulation(
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
        
        # Reconstruct field metadata
        field_metadata = self._reconstruct_metadata(data)
        
        # Convert initial state
        initial_fields = self._tensors_to_fields(
            data, field_metadata, start_frame, start_frame + 1
        )
        initial_fields = {name: fields[0] for name, fields in initial_fields.items()}
        
        # Convert target rollout
        target_start = start_frame + 1
        target_end = start_frame + 1 + self.num_predict_steps
        target_fields = self._tensors_to_fields(
            data, field_metadata, target_start, target_end
        )
        
        return initial_fields, target_fields
    
    def _get_augmented_sample(self, idx: int) -> Tuple[Dict[str, Field], Dict[str, List[Field]]]:
        """
        Get a synthetically-generated prediction sample.
        
        These are already in Field format and pre-windowed, so just return directly.
        
        Args:
            idx: Index within augmented samples
        
        Returns:
            Tuple of (initial_fields, target_fields) as Fields
        """
        if idx >= len(self.augmented_samples):
            raise IndexError(
                f"Augmented index {idx} out of range [0, {len(self.augmented_samples)})"
            )
        
        # Augmented samples are already (initial_fields, target_fields) tuples
        return self.augmented_samples[idx]

    # ==================== Augmentation Management ====================

    def set_augmented_predictions(
        self,
        tensor_predictions: List[Tuple[torch.Tensor, torch.Tensor]]
    ):
        """
        Set augmented predictions with pre-conversion to Fields.
        
        Converts all tensor predictions to Fields in a single batch operation,
        eliminating repeated conversions on every dataset access.
        
        Args:
            tensor_predictions: List of (initial_tensor, target_tensor) tuples
        """
        if not tensor_predictions:
            self.augmented_samples = []
            self.num_augmented = 0
            self._total_length = self._compute_length()
            return
        
        logger.debug(f"Converting {len(tensor_predictions)} synthetic predictions to Fields...")
        
        # Get field metadata and create batch converter ONCE
        field_metadata = self._get_field_metadata()
        batch_converter = make_batch_converter(field_metadata)
        
        # Convert all predictions
        field_predictions = []
        for idx, (initial_tensor, target_tensor) in enumerate(tensor_predictions):
            if idx % 50 == 0 and idx > 0:
                logger.debug(f"  Converted {idx}/{len(tensor_predictions)} predictions...")
            
            fields = self._convert_tensor_to_fields_optimized(
                initial_tensor, target_tensor, batch_converter, field_metadata
            )
            field_predictions.append(fields)
        
        # Store converted Fields
        self.augmented_samples = field_predictions
        self.num_augmented = len(field_predictions)
        self._total_length = self._compute_length()
        
        logger.debug(f"Set {self.num_augmented} augmented predictions (as Fields)")

    def clear_augmented_predictions(self):
        """Clear all augmented predictions."""
        self.augmented_samples = []
        self.num_augmented = 0
        self._total_length = self._compute_length()
        logger.debug("Cleared augmented predictions")

    def _convert_tensor_to_fields_optimized(
        self,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        batch_converter,
        field_metadata: Dict
    ) -> Tuple[Dict[str, Field], Dict[str, List[Field]]]:
        """
        Optimized conversion using pre-created batch_converter.
        
        Args:
            input_tensor: Initial state [C_all, H, W]
            target_tensor: Target states [T, C_all, H, W]
            batch_converter: Pre-created batch converter (reused)
            field_metadata: Pre-fetched field metadata (reused)
        
        Returns:
            Tuple of (initial_fields, target_fields)
        """
        # Move to GPU if available
        if torch.cuda.is_available():
            if not input_tensor.is_cuda:
                input_tensor = input_tensor.cuda()
            if not target_tensor.is_cuda:
                target_tensor = target_tensor.cuda()
        
        # Convert initial state
        input_with_batch = input_tensor.unsqueeze(0)
        initial_fields = batch_converter.tensor_to_fields_batch(input_with_batch)
        
        # Remove batch dimension
        for name, field in initial_fields.items():
            if "batch" in field.shape:
                initial_fields[name] = field.batch[0]
        
        # Convert target states
        if target_tensor.dim() == 3:
            target_tensor = target_tensor.unsqueeze(0)
        
        num_timesteps = target_tensor.shape[0]
        target_fields = {name: [] for name in field_metadata.keys()}
        
        for t in range(num_timesteps):
            timestep_tensor = target_tensor[t].unsqueeze(0)
            timestep_fields = batch_converter.tensor_to_fields_batch(timestep_tensor)
            
            for name, field in timestep_fields.items():
                if "batch" in field.shape:
                    target_fields[name].append(field.batch[0])
                else:
                    target_fields[name].append(field)
        
        return initial_fields, target_fields

    # ==================== Helper Methods ====================
    
    def _reconstruct_metadata(self, data: Dict[str, Any]) -> Dict[str, FieldMetadata]:
        """Reconstruct FieldMetadata from cached metadata."""
        from phi.geom import Box
        from phi.math import spatial
        
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
            
            tensor_shape = data["tensor_data"][name].shape
            spatial_dims = meta["spatial_dims"]
            resolution_sizes = {
                dim: tensor_shape[i + 2] for i, dim in enumerate(spatial_dims)
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
    ) -> Dict[str, List[Field]]:
        """Convert tensors to Fields for a frame range."""
        fields_dict = {}
        
        for name in self.field_names:
            field_tensors = data["tensor_data"][name][start_frame:end_frame]
            
            if torch.cuda.is_available():
                if not field_tensors.is_cuda:
                    field_tensors = field_tensors.cuda()
            
            field_meta = field_metadata[name]
            converter = make_converter(field_meta)
            
            fields_list = []
            for t in range(len(field_tensors)):
                tensor_t = field_tensors[t:t+1]
                field_t = converter.tensor_to_field(tensor_t, field_meta, time_slice=0)
                fields_list.append(field_t)
            
            fields_dict[name] = fields_list
        
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
            f"  predict_steps={self.num_predict_steps}\n"
            f")"
        )