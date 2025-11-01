"""
Hybrid Dataset for PyTorch DataLoader

This module provides a PyTorch Dataset wrapper around DataManager
for efficient training with cached tensor data.

Features lazy loading with LRU cache to handle large datasets without
loading all simulations into memory at once.
"""

from typing import Dict, List, Tuple, Any, Union
from functools import lru_cache
import torch
from torch.utils.data import Dataset
from phi.field import Field
from phiml.math import spatial

from .data_manager import DataManager
from src.utils.field_conversion import tensors_to_fields


class HybridDataset(Dataset):
    """
    PyTorch Dataset that wraps DataManager for autoregressive training.
    
    Uses lazy loading with LRU caching for memory-efficient training with large datasets:
    - Simulations loaded on-demand, not all at once
    - LRU cache keeps N most recently used simulations in memory
    - Automatic memory management
    - Configurable cache size
    
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
        max_cached_sims: Maximum number of simulations to keep in memory (LRU cache size)
        pin_memory: If True, pin tensors in memory for faster GPU transfer
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
        return_fields: bool = False,
        max_cached_sims: int = 5,
        pin_memory: bool = True
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
            max_cached_sims: Maximum number of simulations to keep in memory (default: 5)
            pin_memory: If True, pin tensors for faster GPU transfer (default: True)
        """
        self.data_manager = data_manager
        self.sim_indices = sim_indices
        self.field_names = field_names
        self.num_frames = num_frames
        self.num_predict_steps = num_predict_steps
        self.use_sliding_window = use_sliding_window
        self.return_fields = return_fields
        self.max_cached_sims = max_cached_sims
        self.pin_memory = pin_memory and torch.cuda.is_available()
        
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
        
        # CHANGED: Don't pre-cache all simulations
        # Instead, verify cache exists and get metadata
        self._validate_cache_exists()
        
        # CHANGED: Create LRU cache for simulation data
        # Use a wrapper method to create the cached function
        self._create_cached_loader()
        
        # Build sample index mapping for sliding window
        if self.use_sliding_window:
            self._build_sliding_window_index()
    
    def _validate_cache_exists(self):
        """
        Validate that cache exists for all simulations without loading data.
        
        Only loads metadata (num_frames) from first simulation if needed.
        Does NOT pre-load all simulation data into memory.
        """
        if not self.sim_indices:
            raise ValueError("sim_indices cannot be empty")
        
        # Determine num_frames from first simulation if not provided
        if self.num_frames is None:
            first_sim_data = self.data_manager.get_or_load_simulation(
                self.sim_indices[0],
                field_names=self.field_names,
                num_frames=None
            )
            # Extract tensor_data and get num_frames from first field
            self.num_frames = first_sim_data['tensor_data'][self.field_names[0]].shape[0]
            del first_sim_data  # Free memory immediately
        
        # Verify all simulations are cached
        for sim_idx in self.sim_indices:
            if not self.data_manager.is_cached(sim_idx):
                raise ValueError(
                    f"Simulation {sim_idx} is not cached. "
                    f"Please run data generation first."
                )
    
    def _create_cached_loader(self):
        """
        Create LRU-cached simulation loader.
        
        The loader will keep at most max_cached_sims simulations in memory,
        automatically evicting least recently used simulations when cache is full.
        """
        # Create cached version of the uncached loader
        self._cached_load_simulation = lru_cache(maxsize=self.max_cached_sims)(
            self._load_simulation_uncached
        )
    
    def _load_simulation_uncached(self, sim_idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single simulation without caching (wrapped by LRU cache).
        
        Args:
            sim_idx: Simulation index to load
            
        Returns:
            Dictionary mapping field names to tensors with shape [T, ...]
            This is the 'tensor_data' part of the full cached data.
        """
        # Load full data structure from DataManager
        full_data = self.data_manager.get_or_load_simulation(
            sim_idx,
            field_names=self.field_names,
            num_frames=self.num_frames
        )
        
        # Extract just the tensor data
        sim_data = full_data['tensor_data']
        
        # Optionally pin memory for faster GPU transfer
        if self.pin_memory:
            sim_data = {
                field: tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor
                for field, tensor in sim_data.items()
            }
        
        return sim_data
    
    def clear_cache(self):
        """
        Manually clear the LRU cache of simulations.
        
        Useful for freeing memory or forcing reload of simulations.
        """
        self._cached_load_simulation.cache_clear()
    
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
                # Error - invalid cache format
                raise ValueError(
                    f"Invalid cache format for field '{name}'. "
                    f"Missing 'bounds_lower' or 'bounds_upper'. "
                    f"Please clear cache and regenerate data."
                )
            
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
        
        # CHANGED: Load data using LRU-cached loader instead of pre-cached data
        sim_data = self._cached_load_simulation(sim_idx)
        
        if self.return_fields:
            # For field mode, we need the full data structure with metadata
            # Convert sim_data back to the format expected by _convert_to_fields_with_start
            data = {'tensor_data': sim_data}
            # Load metadata from cache
            cache_data = self.data_manager.load_from_cache(sim_idx)
            data.update({k: v for k, v in cache_data.items() if k != 'tensor_data'})
            return self._convert_to_fields_with_start(data, start_frame)
        
        # Tensor-based mode: concatenate fields for efficient training
        # Initial state contains ALL fields (static + dynamic)
        all_field_tensors = [sim_data[name] for name in self.field_names]
        all_data = torch.cat(all_field_tensors, dim=1)  # [time, total_channels, x, y]
        initial_state = all_data[start_frame]  # [C_all, H, W]
        
        # Rollout targets contain ONLY dynamic fields
        dynamic_field_tensors = [sim_data[name] for name in self.dynamic_fields]
        dynamic_data = torch.cat(dynamic_field_tensors, dim=1)  # [time, dynamic_channels, x, y]
        
        # Extract target frames starting from start_frame + 1
        target_start = start_frame + 1
        target_end = start_frame + 1 + self.num_predict_steps
        rollout_targets = dynamic_data[target_start:target_end]  # [T, C_dynamic, H, W]
        
        return initial_state, rollout_targets
