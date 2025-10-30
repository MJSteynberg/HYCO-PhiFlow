"""
Hybrid Dataset for PyTorch DataLoader

This module provides a PyTorch Dataset wrapper around DataManager
for efficient training with cached tensor data.
"""

from typing import Dict, List, Tuple, Any
import torch
from torch.utils.data import Dataset

from .data_manager import DataManager


class HybridDataset(Dataset):
    """
    PyTorch Dataset that wraps DataManager for autoregressive training.
    
    Provides (initial_state, rollout_targets) pairs where:
    - initial_state: tensor at t=0, shape [C, H, W]
    - rollout_targets: tensors from t=1 to t=num_steps, shape [T, C, H, W]
    
    Attributes:
        data_manager: DataManager instance for loading cached data
        sim_indices: List of simulation indices to include in dataset
        field_names: List of field names to load
        num_frames: Number of frames to load per simulation
        num_predict_steps: Number of rollout steps for training
    """
    
    def __init__(
        self,
        data_manager: DataManager,
        sim_indices: List[int],
        field_names: List[str],
        num_frames: int,
        num_predict_steps: int
    ):
        """
        Initialize the HybridDataset.
        
        Args:
            data_manager: DataManager instance for data loading
            sim_indices: List of simulation indices to use
            field_names: List of field names to load (e.g., ['velocity', 'density'])
            num_frames: Total number of frames to load (must be >= num_predict_steps + 1)
            num_predict_steps: Number of autoregressive prediction steps
        """
        self.data_manager = data_manager
        self.sim_indices = sim_indices
        self.field_names = field_names
        self.num_frames = num_frames
        self.num_predict_steps = num_predict_steps
        
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
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample.
        
        Args:
            idx: Index into the dataset
            
        Returns:
            Tuple of (initial_state, rollout_targets) where:
            - initial_state: [C, H, W] tensor for t=0
            - rollout_targets: [T, C, H, W] tensor for t=1 to t=num_predict_steps
        """
        sim_idx = self.sim_indices[idx]
        
        # Load from cache
        data = self.data_manager.load_from_cache(sim_idx)
        
        # Concatenate all fields along channel dimension
        field_tensors = []
        for field_name in self.field_names:
            tensor = data['tensor_data'][field_name]  # [time, channels, x, y]
            field_tensors.append(tensor)
        
        # Stack fields: [time, total_channels, x, y]
        all_data = torch.cat(field_tensors, dim=1)
        
        # Split into initial state and rollout
        initial_state = all_data[0]  # [C, H, W]
        rollout_targets = all_data[1:self.num_predict_steps + 1]  # [T, C, H, W]
        
        return initial_state, rollout_targets
