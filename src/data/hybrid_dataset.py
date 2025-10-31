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
    
    Supports two modes:
    1. Single starting point (use_sliding_window=False): One sample per simulation
       - initial_state: tensor at t=0
       - rollout_targets: tensors from t=1 to t=num_predict_steps
    
    2. Sliding window (use_sliding_window=True): Multiple samples per simulation
       - Creates (num_frames - num_predict_steps) samples per simulation
       - Each timestep becomes a starting point for autoregressive rollout
       - Example: 50 frames, 3 predict steps → 47 samples per simulation
    
    When return_fields=True, returns PhiFlow Fields instead of tensors for physical model training.
    
    Attributes:
        data_manager: DataManager instance for loading cached data
        sim_indices: List of simulation indices to include in dataset
        field_names: List of field names to load (all fields)
        dynamic_fields: List of fields that change over time (predicted by model)
        static_fields: List of fields that don't change (input-only)
        num_frames: Number of frames to load per simulation
        num_predict_steps: Number of rollout steps for training
        use_sliding_window: If True, create multiple samples per simulation
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
        use_sliding_window: bool = False,
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
            use_sliding_window: If True, create multiple samples per simulation using sliding window
            return_fields: If True, return PhiFlow Fields instead of tensors (for physical models)
        """
        self.data_manager = data_manager
        self.sim_indices = sim_indices
        self.field_names = field_names
        self.num_frames = num_frames
        self.num_predict_steps = num_predict_steps
        self.use_sliding_window = use_sliding_window
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
        if num_frames is not None and num_frames < num_predict_steps + 1:
            raise ValueError(
                f"num_frames ({num_frames}) must be >= num_predict_steps + 1 ({num_predict_steps + 1})"
            )
        
        # Pre-cache all simulations
        self._cache_all_simulations()
        
        # Build sample index mapping for sliding window
        if self.use_sliding_window:
            self._build_sliding_window_index()
        
    def _build_sliding_window_index(self):
        """
        Build index mapping for sliding window samples.
        
        Creates a list of (sim_idx, start_frame) tuples representing each sample.
        Each sample needs:
        - 1 initial frame at start_frame
        - num_predict_steps target frames (start_frame+1 to start_frame+num_predict_steps)
        
        For example, with 10 frames (0-9) and 3 predict steps:
        - start_frame=0: initial=0, targets=[1,2,3] ✓
        - start_frame=1: initial=1, targets=[2,3,4] ✓
        - ...
        - start_frame=6: initial=6, targets=[7,8,9] ✓ (last valid)
        - start_frame=7: initial=7, targets=[8,9,?] ✗ (not enough frames)
        
        So: samples_per_sim = num_frames - num_predict_steps
        And valid start_frames: 0 to (num_frames - num_predict_steps - 1)
        Which is: range(num_frames - num_predict_steps)
        """
        self.sample_index = []
        
        # Calculate how many samples per simulation
        # Last valid start_frame needs num_predict_steps frames after it
        self.samples_per_sim = self.num_frames - self.num_predict_steps
        
        for sim_idx in self.sim_indices:
            for start_frame in range(self.samples_per_sim):
                self.sample_index.append((sim_idx, start_frame))
        
        print(f"  Sliding window: {self.samples_per_sim} samples per simulation")
        print(f"  Total samples: {len(self.sample_index)} (from {len(self.sim_indices)} simulations)")
        print(f"  Frame range per sample: start_frame to start_frame+{self.num_predict_steps}")
    
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
        
        # If num_frames was None (load all), determine actual number of frames from cached data
        # Also validate that all simulations have the same number of frames
        if self.num_frames is None:
            # Load first simulation to get actual frame count
            first_sim = self.sim_indices[0]
            data = self.data_manager.load_from_cache(first_sim)
            first_field = self.field_names[0]
            self.num_frames = data['tensor_data'][first_field].shape[0]
            print(f"  Loaded all available frames: {self.num_frames} frames")
            
            # Validate all simulations have the same number of frames
            for sim_idx in self.sim_indices[1:]:
                data = self.data_manager.load_from_cache(sim_idx)
                frames_in_sim = data['tensor_data'][first_field].shape[0]
                if frames_in_sim != self.num_frames:
                    raise ValueError(
                        f"Simulation {sim_idx} has {frames_in_sim} frames, "
                        f"but simulation {first_sim} has {self.num_frames} frames. "
                        f"All simulations must have the same number of frames for sliding window. "
                        f"Clear cache and regenerate data with consistent frame counts."
                    )
    
    def _convert_to_fields(self, data: Dict[str, Any]) -> Tuple[Dict[str, Field], Dict[str, Field]]:
        """
        Convert cached tensor data to PhiFlow Fields (starting from t=0).
        
        Args:
            data: Cached data dictionary from DataManager
            
        Returns:
            Tuple of (initial_fields, target_fields) where:
            - initial_fields: Dict[field_name, Field] for t=0
            - target_fields: Dict[field_name, List[Field]] for t=1 to t=num_predict_steps
        """
        return self._convert_to_fields_with_start(data, start_frame=0)
    
    def _convert_to_fields_with_start(
        self, 
        data: Dict[str, Any], 
        start_frame: int
    ) -> Tuple[Dict[str, Field], Dict[str, Field]]:
        """
        Convert cached tensor data to PhiFlow Fields with custom starting frame.
        
        Args:
            data: Cached data dictionary from DataManager
            start_frame: Starting frame index for the sample
            
        Returns:
            Tuple of (initial_fields, target_fields) where:
            - initial_fields: Dict[field_name, Field] for start_frame
            - target_fields: Dict[field_name, List[Field]] for subsequent frames
        """
        from src.utils.field_conversion import FieldMetadata
        from phi.geom import Box
        
        field_metadata_dict = data['metadata']['field_metadata']
        field_metadata = {}
        
        # Convert cached metadata to FieldMetadata objects
        for name, meta in field_metadata_dict.items():
            # Get domain from actual bounds values (not string representation)
            if 'bounds_lower' in meta and 'bounds_upper' in meta:
                # Use actual bounds values
                lower = meta['bounds_lower']
                upper = meta['bounds_upper']
                
                # Create Box with correct dimensions
                if len(lower) == 2:
                    domain = Box(x=(lower[0], upper[0]), y=(lower[1], upper[1]))
                elif len(lower) == 3:
                    domain = Box(x=(lower[0], upper[0]), y=(lower[1], upper[1]), z=(lower[2], upper[2]))
                else:
                    # Fallback for unexpected dimensions
                    domain = Box(x=1, y=1)
            else:
                # Fallback for old cache files without bounds_lower/upper
                try:
                    bounds_str = meta['bounds']
                    domain = eval(bounds_str, {"Box": Box})
                except:
                    domain = Box(x=1, y=1)
            
            # Extract resolution from tensor shape
            tensor_shape = data['tensor_data'][name].shape  # [time, channels, x, y]
            spatial_dims = meta['spatial_dims']
            resolution_sizes = {dim: tensor_shape[i+2] for i, dim in enumerate(spatial_dims)}
            resolution = spatial(**resolution_sizes)
            
            field_metadata[name] = FieldMetadata.from_cache_metadata(meta, domain, resolution)
        
        # Convert initial state (start_frame) for all fields
        # Move tensors to GPU if they're on CPU
        initial_tensors = {}
        for name in self.field_names:
            tensor = data['tensor_data'][name][start_frame:start_frame+1]
            if isinstance(tensor, torch.Tensor):
                initial_tensors[name] = tensor.cuda() if not tensor.is_cuda else tensor
            else:
                initial_tensors[name] = torch.from_numpy(tensor).cuda()
        
        initial_metadata = {name: field_metadata[name] for name in self.field_names}
        initial_fields = tensors_to_fields(initial_tensors, initial_metadata, time_slice=0)
        
        # Convert rollout targets (start_frame+1 to start_frame+num_predict_steps) for dynamic fields
        target_fields = {}
        target_start = start_frame + 1
        target_end = start_frame + 1 + self.num_predict_steps
        
        for field_name in self.dynamic_fields:
            field_tensors_raw = data['tensor_data'][field_name][target_start:target_end]
            if isinstance(field_tensors_raw, torch.Tensor):
                field_tensors = field_tensors_raw.cuda() if not field_tensors_raw.is_cuda else field_tensors_raw
            else:
                field_tensors = torch.from_numpy(field_tensors_raw).cuda()
            
            field_meta = field_metadata[field_name]
            
            # Convert each timestep to a Field
            fields_list = []
            for t in range(len(field_tensors)):
                tensor_t = field_tensors[t:t+1]
                field_t = tensors_to_fields(
                    {field_name: tensor_t}, 
                    {field_name: field_meta}, 
                    time_slice=0
                )[field_name]
                fields_list.append(field_t)
            
            target_fields[field_name] = fields_list
        
        return initial_fields, target_fields
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self.use_sliding_window:
            return len(self.sample_index)
        else:
            return len(self.sim_indices)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[Dict[str, Field], Dict[str, Field]]]:
        """
        Get a training sample.
        
        Args:
            idx: Index into the dataset
            
        Returns:
            If return_fields=False (default):
                Tuple of (initial_state, rollout_targets) where:
                - initial_state: [C_all, H, W] tensor for starting timestep (all fields)
                - rollout_targets: [T, C_dynamic, H, W] tensor for next num_predict_steps (dynamic fields only)
            
            If return_fields=True:
                Tuple of (initial_fields, target_fields) where:
                - initial_fields: Dict[field_name, Field] for starting timestep (all fields)
                - target_fields: Dict[field_name, List[Field]] for next num_predict_steps
        """
        # Determine simulation and starting frame
        if self.use_sliding_window:
            sim_idx, start_frame = self.sample_index[idx]
        else:
            sim_idx = self.sim_indices[idx]
            start_frame = 0
        
        # Load cached data
        data = self.data_manager.load_from_cache(sim_idx)
        
        if self.return_fields:
            return self._convert_to_fields_with_start(data, start_frame)
        
        # Tensor-based mode: concatenate fields for efficient training
        # Initial state contains ALL fields (static + dynamic)
        all_field_tensors = [data['tensor_data'][name] for name in self.field_names]
        all_data = torch.cat(all_field_tensors, dim=1)  # [time, total_channels, x, y]
        initial_state = all_data[start_frame]  # [C_all, H, W]
        
        # Rollout targets contain ONLY dynamic fields
        dynamic_field_tensors = [data['tensor_data'][name] for name in self.dynamic_fields]
        dynamic_data = torch.cat(dynamic_field_tensors, dim=1)  # [time, dynamic_channels, x, y]
        
        # Extract target frames starting from start_frame + 1
        target_start = start_frame + 1
        target_end = start_frame + 1 + self.num_predict_steps
        rollout_targets = dynamic_data[target_start:target_end]  # [T, C_dynamic, H, W]
        
        return initial_state, rollout_targets
