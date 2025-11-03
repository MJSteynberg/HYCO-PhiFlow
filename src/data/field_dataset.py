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
        
        # Call parent constructor (handles common initialization)
        super().__init__(
            data_manager=data_manager,
            sim_indices=sim_indices,
            field_names=field_names,
            num_frames=num_frames,
            num_predict_steps=num_predict_steps,
            use_sliding_window=use_sliding_window,
            augmentation_config=augmentation_config,
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
            sim_idx, 
            field_names=self.field_names, 
            num_frames=self.num_frames
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
    
    # ==================== Additional Utility Methods ====================
    
    def get_field_info(self) -> Dict[str, Any]:
        """
        Get information about field configuration.
        
        Returns:
            Dictionary with field names and counts
        """
        return {
            'field_names': self.field_names,
            'num_fields': len(self.field_names),
            'move_to_gpu': self.move_to_gpu,
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
            raise ValueError(f"Index {idx} is augmented, cannot inspect structure from real data")
        
        # Get a sample
        initial_fields, target_fields = self._get_real_sample(idx)
        
        # Extract information
        info = {
            'initial_fields': list(initial_fields.keys()),
            'target_fields': list(target_fields.keys()),
            'num_initial_fields': len(initial_fields),
            'num_target_fields': len(target_fields),
            'target_timesteps': {
                name: len(fields) for name, fields in target_fields.items()
            },
        }
        
        # Add shape information from first field
        if initial_fields:
            first_field_name = list(initial_fields.keys())[0]
            first_field = initial_fields[first_field_name]
            info['field_type'] = type(first_field).__name__
            info['spatial_shape'] = first_field.shape.spatial
        
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
