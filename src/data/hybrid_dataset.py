"""
Hybrid Dataset for PyTorch DataLoader

This module provides a PyTorch Dataset wrapper around DataManager
for efficient training with cached tensor data.
"""

from typing import Dict, List, Tuple, Any, Union
import torch
from torch.utils.data import Dataset
from phi.field import Field
from phiml.math import spatial

from .data_manager import DataManager
from src.utils.field_conversion import tensors_to_fields


class HybridDataset(Dataset):
    """
    PyTorch Dataset that wraps DataManager for autoregressive training.
    
    Provides (initial_state, rollout_targets) pairs where:
    - initial_state: tensor at t=0, shape [C_all, H, W] (all fields)
    - rollout_targets: tensors from t=1 to t=num_steps, shape [T, C_dynamic, H, W] (dynamic fields only)
    
    When return_fields=True, returns PhiFlow Fields instead of tensors for physical model training.
    
    Attributes:
        data_manager: DataManager instance for loading cached data
        sim_indices: List of simulation indices to include in dataset
        field_names: List of field names to load (all fields)
        dynamic_fields: List of fields that change over time (predicted by model)
        static_fields: List of fields that don't change (input-only)
        num_frames: Number of frames to load per simulation
        num_predict_steps: Number of rollout steps for training
        return_fields: If True, return PhiFlow Fields instead of tensors
    """
    
    def __init__(
        self,
        data_manager: DataManager,
        sim_indices: List[int],
        field_names: List[str],
        num_frames: int,
        num_predict_steps: int,
        dynamic_fields: List[str] = None,
        static_fields: List[str] = None,
        return_fields: bool = False
    ):
        """
        Initialize the HybridDataset.
        
        Args:
            data_manager: DataManager instance for data loading
            sim_indices: List of simulation indices to use
            field_names: List of field names to load (e.g., ['velocity', 'density', 'inflow'])
            num_frames: Total number of frames to load (must be >= num_predict_steps + 1)
            num_predict_steps: Number of autoregressive prediction steps
            dynamic_fields: List of fields that are predicted (default: all fields)
            static_fields: List of fields that are input-only (default: empty list)
            return_fields: If True, return PhiFlow Fields instead of tensors (for physical models)
        """
        self.data_manager = data_manager
        self.sim_indices = sim_indices
        self.field_names = field_names
        self.num_frames = num_frames
        self.num_predict_steps = num_predict_steps
        self.return_fields = return_fields
        
        # Handle static vs dynamic field distinction
        if dynamic_fields is None and static_fields is None:
            # Default: all fields are dynamic
            self.dynamic_fields = field_names
            self.static_fields = []
        elif dynamic_fields is not None:
            self.dynamic_fields = dynamic_fields
            self.static_fields = static_fields if static_fields is not None else []
        else:
            # Only static_fields provided
            self.static_fields = static_fields
            self.dynamic_fields = [f for f in field_names if f not in static_fields]
        
        # Validate
        if num_frames < num_predict_steps + 1:
            raise ValueError(
                f"num_frames ({num_frames}) must be >= num_predict_steps + 1 ({num_predict_steps + 1})"
            )
        
        # Pre-cache all simulations
        self._cache_all_simulations()
    
    def _cache_all_simulations(self):
        """Pre-load and cache all simulations with validation."""
        for sim_idx in self.sim_indices:
            # Check if cache exists and matches our requirements
            if not self.data_manager.is_cached(sim_idx, self.field_names, self.num_frames):
                self.data_manager.load_and_cache_simulation(
                    sim_idx, 
                    self.field_names, 
                    self.num_frames
                )
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sim_indices)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[Dict[str, Field], Dict[str, Field]]]:
        """
        Get a training sample.
        
        Args:
            idx: Index into the dataset
            
        Returns:
            If return_fields=False (default):
                Tuple of (initial_state, rollout_targets) where:
                - initial_state: [C_all, H, W] tensor for t=0 (all fields)
                - rollout_targets: [T, C_dynamic, H, W] tensor for t=1 to t=num_predict_steps (dynamic fields only)
            
            If return_fields=True:
                Tuple of (initial_fields, target_fields) where:
                - initial_fields: Dict[field_name, Field] for t=0 (all fields)
                - target_fields: Dict[field_name, List[Field]] for t=1 to t=num_predict_steps (dynamic fields only)
        """
        sim_idx = self.sim_indices[idx]
        
        # Load from cache
        data = self.data_manager.load_from_cache(sim_idx)
        
        if not self.return_fields:
            # Original tensor-based behavior
            # Concatenate ALL fields for initial state (in order of field_names)
            all_field_tensors = []
            for field_name in self.field_names:
                tensor = data['tensor_data'][field_name]  # [time, channels, x, y]
                all_field_tensors.append(tensor)
            
            # Stack all fields: [time, total_channels, x, y]
            all_data = torch.cat(all_field_tensors, dim=1)
            
            # Initial state contains ALL fields (static + dynamic)
            initial_state = all_data[0]  # [C_all, H, W]
            
            # Rollout targets contain ONLY dynamic fields
            dynamic_field_tensors = []
            for field_name in self.dynamic_fields:
                tensor = data['tensor_data'][field_name]  # [time, channels, x, y]
                dynamic_field_tensors.append(tensor)
            
            # Stack only dynamic fields: [time, dynamic_channels, x, y]
            dynamic_data = torch.cat(dynamic_field_tensors, dim=1)
            rollout_targets = dynamic_data[1:self.num_predict_steps + 1]  # [T, C_dynamic, H, W]
            
            return initial_state, rollout_targets
        
        else:
            # Field-based behavior for physical models
            field_metadata_dict = data['metadata']['field_metadata']
            
            # Convert metadata from cache format to FieldMetadata objects
            from src.utils.field_conversion import FieldMetadata
            from phi.geom import Box
            field_metadata = {}
            
            for name, meta in field_metadata_dict.items():
                # Parse bounds string to extract domain
                # Format: "Box[x:0.0 to 1.0, y:0.0 to 1.0]" or similar
                bounds_str = meta['bounds']
                # Simple parsing - extract domain from first field as they should all match
                # For now, use a default Box - physical models will provide their own domain
                # This is just for reconstruction purposes
                try:
                    exec(f"domain = {bounds_str}")
                except:
                    # Fallback: create default box
                    domain = Box(x=1, y=1)
                
                # Extract resolution from spatial_dims in the tensor
                # The tensor shape is [time, channels, x, y]
                tensor_shape = data['tensor_data'][name].shape
                spatial_dims = meta['spatial_dims']
                # Get spatial dimensions from tensor (skip time and channels)
                resolution_sizes = {dim: tensor_shape[i+2] for i, dim in enumerate(spatial_dims)}
                resolution = spatial(**resolution_sizes)
                
                field_metadata[name] = FieldMetadata.from_cache_metadata(meta, domain, resolution)
            
            # Convert initial state (t=0) for ALL fields
            initial_tensors = {
                name: data['tensor_data'][name][0:1]  # [1, channels, x, y]
                for name in self.field_names
            }
            initial_metadata = {
                name: field_metadata[name]
                for name in self.field_names
            }
            initial_fields = tensors_to_fields(initial_tensors, initial_metadata, time_slice=0)
            
            # Convert rollout targets (t=1 to t=num_predict_steps) for DYNAMIC fields only
            target_fields = {}
            for field_name in self.dynamic_fields:
                # Get tensors for timesteps 1 to num_predict_steps
                field_tensors = data['tensor_data'][field_name][1:self.num_predict_steps + 1]  # [T, channels, x, y]
                field_meta = field_metadata[field_name]
                
                # Convert each timestep to a Field
                fields_list = []
                for t in range(len(field_tensors)):
                    tensor_t = field_tensors[t:t+1]  # [1, channels, x, y]
                    field_t = tensors_to_fields({field_name: tensor_t}, {field_name: field_meta}, time_slice=0)[field_name]
                    fields_list.append(field_t)
                
                target_fields[field_name] = fields_list
            
            return initial_fields, target_fields
