"""
Field Dataset for Physical Model Training

Returns PhiFlow Fields suitable for physics-based simulation.
Inherits common functionality from AbstractDataset.

This dataset is specifically designed for physical (PDE-based) models that
operate on PhiFlow Field objects. It handles:
- Loading tensors from DataManager cache
- Reconstructing PhiFlow Fields from tensors and metadata
- Preserving field properties (grid, extrapolation, etc.)
"""

from typing import List, Optional, Dict, Any, Tuple
import torch
from phi.field import Field
from phi.geom import Box
from phiml.math import spatial

from .abstract_dataset import AbstractDataset
from .data_manager import DataManager
from src.utils.field_conversion import FieldMetadata, make_converter


class FieldDataset(AbstractDataset):
    """
    PyTorch Dataset that returns PhiFlow Fields for physical training.

    Inherits from AbstractDataset:
    - Lazy loading with LRU cache
    - Sliding window support
    - Optional augmentation (built-in)
    - Cache validation and management

    Additional features specific to field mode:
    - Field reconstruction from tensors and metadata
    - Metadata preservation (grid, extrapolation, boundary conditions)
    - GPU memory management for tensors before conversion

    The dataset returns samples in the format expected by physical models:
    - initial_fields: Dict[field_name, Field] - all fields at starting timestep
    - target_fields: Dict[field_name, List[Field]] - all fields for next T steps

    Note: Unlike TensorDataset, FieldDataset returns ALL fields as targets
    (no static/dynamic distinction) because physical models predict all fields.

    Args:
        data_manager: DataManager instance for loading cached data
        sim_indices: List of simulation indices to include
        field_names: List of field names to load (e.g., ['velocity', 'density'])
        num_frames: Number of frames per simulation (None = load all)
        num_predict_steps: Number of prediction steps
        use_sliding_window: If True, create multiple samples per simulation
        augmentation_config: Optional augmentation configuration dict
        max_cached_sims: LRU cache size (number of simulations in memory)
        move_to_gpu: If True, move tensors to GPU before Field conversion

    Returns:
        Tuple[Dict[str, Field], Dict[str, List[Field]]]:
            - initial_fields: All fields at starting timestep
            - target_fields: All fields for next T timesteps

    Example:
        >>> dataset = FieldDataset(
        ...     data_manager=data_manager,
        ...     sim_indices=[0, 1, 2],
        ...     field_names=['velocity', 'density'],
        ...     num_frames=None,  # Load all frames
        ...     num_predict_steps=10,
        ...     use_sliding_window=True,
        ... )
        >>> initial_fields, target_fields = dataset[0]
        >>> print(initial_fields.keys())  # dict_keys(['velocity', 'density'])
        >>> print(len(target_fields['velocity']))  # 10 timesteps
    """

    def __init__(
        self,
        data_manager: DataManager,
        sim_indices: List[int],
        field_names: List[str],
        num_frames: Optional[int],
        num_predict_steps: int,
        use_sliding_window: bool = False,
        augmentation_config: Optional[Dict[str, Any]] = None,
        access_policy: str = "both",
        max_cached_sims: int = 5,
        move_to_gpu: bool = True,
    ):
        """
        Initialize the FieldDataset.

        Validates inputs and calls parent constructor to handle
        common initialization.
        """
        # Store field-specific attributes
        self.move_to_gpu = move_to_gpu and torch.cuda.is_available()

        # Initialize field metadata cache (will be populated when needed)
        self._field_metadata_cache = None

        # Call parent constructor (handles common initialization)
        super().__init__(
            data_manager=data_manager,
            sim_indices=sim_indices,
            field_names=field_names,
            num_frames=num_frames,
            num_predict_steps=num_predict_steps,
            use_sliding_window=use_sliding_window,
            augmentation_config=augmentation_config,
            access_policy=access_policy,
            max_cached_sims=max_cached_sims,
        )

    # ==================== Implementation of Abstract Methods ====================

    def _load_simulation_uncached(self, sim_idx: int) -> Dict[str, Any]:
        """
        Load simulation data with metadata for Field reconstruction.

        This method is automatically wrapped with LRU caching by the parent class,
        so it only runs when the simulation is not in the cache.

        For FieldDataset, we need both tensors and metadata to reconstruct Fields.

        Args:
            sim_idx: Simulation index to load

        Returns:
            Dictionary with:
            - 'tensor_data': Dict[field_name, torch.Tensor] with shape [T, C, H, W]
            - 'metadata': Dict with field metadata for reconstruction
        """
        # Load full data structure from DataManager (includes metadata)
        full_data = self.data_manager.get_or_load_simulation(
            sim_idx, field_names=self.field_names, num_frames=self.num_frames
        )

        # For FieldDataset, we need both tensor_data and metadata
        # Keep the full structure (unlike TensorDataset which only keeps tensors)
        return full_data

    def _get_real_sample(
        self, idx: int
    ) -> Tuple[Dict[str, Field], Dict[str, List[Field]]]:
        """
        Get a real (non-augmented) sample as PhiFlow Fields.

        This method:
        1. Determines simulation and starting frame from index
        2. Loads simulation data using LRU-cached loader
        3. Reconstructs FieldMetadata objects from cached metadata
        4. Converts tensors to Fields using appropriate converters
        5. Returns initial fields and target field sequences

        Args:
            idx: Real sample index (0 to num_real-1)

        Returns:
            Tuple of (initial_fields, target_fields) where:
            - initial_fields: Dict[field_name, Field] for starting timestep
            - target_fields: Dict[field_name, List[Field]] for next T timesteps
        """
        # Get simulation and starting frame (inherited utility method)
        sim_idx, start_frame = self.get_simulation_and_frame(idx)

        # Load simulation data (uses LRU cache from parent class)
        # This returns both tensor_data and metadata
        data = self._cached_load_simulation(sim_idx)

        # === Reconstruct FieldMetadata from cached metadata ===
        field_metadata_dict = data["metadata"]["field_metadata"]
        field_metadata = {}

        for name, meta in field_metadata_dict.items():
            # Reconstruct domain (Box) from bounds
            if "bounds_lower" in meta and "bounds_upper" in meta:
                lower = meta["bounds_lower"]
                upper = meta["bounds_upper"]

                # Create Box with correct dimensions
                if len(lower) == 2:
                    domain = Box(x=(lower[0], upper[0]), y=(lower[1], upper[1]))
                elif len(lower) == 3:
                    domain = Box(
                        x=(lower[0], upper[0]),
                        y=(lower[1], upper[1]),
                        z=(lower[2], upper[2]),
                    )
                else:
                    # Fallback for unexpected dimensions
                    domain = Box(x=1, y=1)
            else:
                raise ValueError(
                    f"Invalid cache format for field '{name}'. "
                    f"Missing 'bounds_lower' or 'bounds_upper'. "
                    f"Please clear cache and regenerate data."
                )

            # Extract resolution from tensor shape
            tensor_shape = data["tensor_data"][name].shape  # [T, C, H, W]
            spatial_dims = meta["spatial_dims"]
            resolution_sizes = {
                dim: tensor_shape[i + 2] for i, dim in enumerate(spatial_dims)
            }
            resolution = spatial(**resolution_sizes)

            # Create FieldMetadata object
            field_metadata[name] = FieldMetadata.from_cache_metadata(
                meta, domain, resolution
            )

        # === Convert initial state (start_frame) to Fields ===
        initial_fields = {}

        for name in self.field_names:
            # Extract tensor for starting frame
            tensor = data["tensor_data"][name][start_frame : start_frame + 1]

            # Move to GPU if configured
            if self.move_to_gpu:
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.cuda() if not tensor.is_cuda else tensor
                else:
                    tensor = torch.from_numpy(tensor).cuda()

            # Create converter and convert to Field
            converter = make_converter(field_metadata[name])
            initial_fields[name] = converter.tensor_to_field(
                tensor, field_metadata[name], time_slice=0
            )

        # === Convert target rollout to Fields ===
        target_fields = {}
        target_start = start_frame + 1
        target_end = start_frame + 1 + self.num_predict_steps

        for field_name in self.field_names:
            # Extract tensors for target timesteps
            field_tensors = data["tensor_data"][field_name][target_start:target_end]

            # Move to GPU if configured
            if self.move_to_gpu:
                if isinstance(field_tensors, torch.Tensor):
                    field_tensors = (
                        field_tensors.cuda()
                        if not field_tensors.is_cuda
                        else field_tensors
                    )
                else:
                    field_tensors = torch.from_numpy(field_tensors).cuda()

            # Create converter for this field type
            field_meta = field_metadata[field_name]
            field_converter = make_converter(field_meta)

            # Convert each timestep to a Field
            fields_list = []
            for t in range(len(field_tensors)):
                tensor_t = field_tensors[t : t + 1]
                field_t = field_converter.tensor_to_field(
                    tensor_t, field_meta, time_slice=0
                )
                fields_list.append(field_t)

            target_fields[field_name] = fields_list

        return initial_fields, target_fields

    def _get_field_metadata(self) -> Dict[str, FieldMetadata]:
        """
        Get field metadata for all fields in the dataset.

        This method loads metadata from the first simulation if not already cached.
        Used for converting augmented samples (tensors) to Fields.

        Returns:
            Dictionary mapping field names to FieldMetadata objects
        """
        if self._field_metadata_cache is not None:
            return self._field_metadata_cache

        # Load first simulation to extract metadata
        first_sim_idx = self.sim_indices[0]
        data = self._cached_load_simulation(first_sim_idx)

        # Reconstruct FieldMetadata from cached metadata
        field_metadata_dict = data["metadata"]["field_metadata"]
        field_metadata = {}

        for name, meta in field_metadata_dict.items():
            # Reconstruct domain (Box) from bounds
            if "bounds_lower" in meta and "bounds_upper" in meta:
                lower = meta["bounds_lower"]
                upper = meta["bounds_upper"]

                # Create Box with correct dimensions
                if len(lower) == 2:
                    domain = Box(x=(lower[0], upper[0]), y=(lower[1], upper[1]))
                elif len(lower) == 3:
                    domain = Box(
                        x=(lower[0], upper[0]),
                        y=(lower[1], upper[1]),
                        z=(lower[2], upper[2]),
                    )
                else:
                    # Fallback for unexpected dimensions
                    domain = Box(x=1, y=1)
            else:
                raise ValueError(
                    f"Invalid cache format for field '{name}'. "
                    f"Missing 'bounds_lower' or 'bounds_upper'. "
                    f"Please clear cache and regenerate data."
                )

            # Extract resolution from tensor shape
            tensor_shape = data["tensor_data"][name].shape  # [T, C, H, W]
            spatial_dims = meta["spatial_dims"]
            resolution_sizes = {
                dim: tensor_shape[i + 2] for i, dim in enumerate(spatial_dims)
            }
            resolution = spatial(**resolution_sizes)

            # Create FieldMetadata object
            field_metadata[name] = FieldMetadata.from_cache_metadata(
                meta, domain, resolution
            )

        # Cache for future use
        self._field_metadata_cache = field_metadata
        return field_metadata

    def _process_augmented_sample(
        self, sample: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[Dict[str, Field], Dict[str, List[Field]]]:
        """
        Convert tensor-based augmented sample to Field format.

        This method handles the conversion of synthetic model predictions (tensors)
        to the Field format expected by physical model training. It uses the existing
        field conversion infrastructure (BatchConcatenationConverter) to split
        concatenated tensors back into individual Fields.

        Args:
            sample: Tuple of (input_tensor, target_tensor) where:
                - input_tensor: [C_all, H, W] - concatenated input fields
                - target_tensor: [T, C_all, H, W] - concatenated target fields

        Returns:
            Tuple of (initial_fields, target_fields) where:
                - initial_fields: Dict[field_name, Field] - initial state
                - target_fields: Dict[field_name, List[Field]] - target trajectory
        """
        from src.utils.field_conversion import make_batch_converter
        from src.utils.logger import get_logger

        logger = get_logger(__name__)

        input_tensor, target_tensor = sample

        # Move tensors to GPU if configured
        if self.move_to_gpu:
            if isinstance(input_tensor, torch.Tensor):
                input_tensor = (
                    input_tensor.cuda() if not input_tensor.is_cuda else input_tensor
                )
            if isinstance(target_tensor, torch.Tensor):
                target_tensor = (
                    target_tensor.cuda() if not target_tensor.is_cuda else target_tensor
                )

        # Debug: log tensor shapes
        logger.debug(f"Processing augmented sample:")
        logger.debug(f"  Input tensor shape: {input_tensor.shape}")
        logger.debug(f"  Target tensor shape: {target_tensor.shape}")
        logger.debug(f"  Input tensor device: {input_tensor.device}")
        logger.debug(f"  Target tensor device: {target_tensor.device}")

        # Get field metadata
        field_metadata = self._get_field_metadata()

        # Create batch converter for splitting concatenated tensors
        batch_converter = make_batch_converter(field_metadata)

        logger.debug(f"  Expected total channels: {batch_converter.total_channels}")
        logger.debug(f"  Field channel counts: {batch_converter.channel_counts}")

        # Convert input tensor to initial fields
        # Input format: [C_all, H, W] - need to add batch dim for converter
        input_with_batch = input_tensor.unsqueeze(0)  # [1, C_all, H, W]
        initial_fields = batch_converter.tensor_to_fields_batch(input_with_batch)

        # Remove batch dimension from fields (they have shape [1, ...])
        for name, field in initial_fields.items():
            # Fields returned by converter may have batch dimension, remove it
            if "batch" in field.shape:
                initial_fields[name] = field.batch[0]

        # Convert target tensor to target fields
        # Target format can be either:
        # - [T, C_all, H, W] for multi-timestep predictions
        # - [C_all, H, W] for single-timestep predictions (from synthetic model)

        # Check if target is single-timestep or multi-timestep
        if target_tensor.dim() == 3:
            # Single timestep: [C_all, H, W]
            # Treat as T=1 by adding time dimension
            target_tensor = target_tensor.unsqueeze(0)  # [1, C_all, H, W]
            logger.debug(
                f"  Target is single-timestep, expanded to: {target_tensor.shape}"
            )

        num_timesteps = target_tensor.shape[0]
        target_fields = {name: [] for name in self.field_names}

        for t in range(num_timesteps):
            # Extract timestep: shape [C_all, H, W]
            timestep_tensor = target_tensor[t]

            logger.debug(f"  Timestep {t} tensor shape: {timestep_tensor.shape}")

            # Add batch dimension: [C_all, H, W] -> [1, C_all, H, W]
            timestep_tensor = timestep_tensor.unsqueeze(0)

            logger.debug(f"  Timestep {t} after unsqueeze: {timestep_tensor.shape}")

            # Convert to fields
            timestep_fields = batch_converter.tensor_to_fields_batch(timestep_tensor)

            # Append to each field's list (remove batch dim)
            for name, field in timestep_fields.items():
                if "batch" in field.shape:
                    target_fields[name].append(field.batch[0])
                else:
                    target_fields[name].append(field)

        return initial_fields, target_fields

    # ==================== Additional Utility Methods ====================

    def get_field_info(self) -> Dict[str, Any]:
        """
        Get information about field configuration.

        Returns:
            Dictionary with field names and counts
        """
        return {
            "field_names": self.field_names,
            "num_fields": len(self.field_names),
            "move_to_gpu": self.move_to_gpu,
        }

    def get_sample_info(self, idx: int = 0) -> Dict[str, Any]:
        """
        Get information about a sample's structure.

        Useful for debugging and understanding data format.

        Args:
            idx: Sample index to inspect (default: 0)

        Returns:
            Dictionary with sample structure information
        """
        if idx >= self.num_real:
            raise ValueError(
                f"Index {idx} is augmented, cannot inspect structure from real data"
            )

        # Get a sample
        initial_fields, target_fields = self._get_real_sample(idx)

        # Extract information
        info = {
            "initial_fields": list(initial_fields.keys()),
            "target_fields": list(target_fields.keys()),
            "num_initial_fields": len(initial_fields),
            "num_target_fields": len(target_fields),
            "target_timesteps": {
                name: len(fields) for name, fields in target_fields.items()
            },
        }

        # Add shape information from first field
        if initial_fields:
            first_field_name = list(initial_fields.keys())[0]
            first_field = initial_fields[first_field_name]
            info["field_type"] = type(first_field).__name__
            info["spatial_shape"] = first_field.shape.spatial

        return info

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return (
            f"FieldDataset(\n"
            f"  simulations={len(self.sim_indices)},\n"
            f"  samples={len(self)} (real={self.num_real}, aug={self.num_augmented}),\n"
            f"  fields={len(self.field_names)},\n"
            f"  frames={self.num_frames},\n"
            f"  predict_steps={self.num_predict_steps},\n"
            f"  sliding_window={self.use_sliding_window},\n"
            f"  move_to_gpu={self.move_to_gpu}\n"
            f")"
        )
