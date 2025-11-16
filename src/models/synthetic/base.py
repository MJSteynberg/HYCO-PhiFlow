# In src/models/synthetic/base.py

from abc import ABC
from typing import Dict, Any, List

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging


class SyntheticModel(nn.Module, ABC):
    """
    Abstract base class for all synthetic models (neural networks).

    This class handles the boilerplate logic for:
    1.  Pre-processing: Converting a state dict of Phiflow Fields (including
        StaggeredGrids) into a single, multi-channel CenteredGrid tensor.
    2.  Post-processing: Converting the network's output CenteredGrid back
        into a state dict of individual Fields, restoring original
        StaggeredGrid types where appropriate.

    Subclasses are only required to implement the `_predict` method.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the synthetic model.

        Args:

        """
        super().__init__()
        print("Parsing config in SyntheticModel")
        self._parse_config(config)
        
        # Calculate channel counts for padding
        self.num_dynamic_channels = sum(self.output_specs.values())
        self.num_static_channels = sum(self.input_specs.values()) - self.num_dynamic_channels
        self._dynamic_slice = slice(0, self.num_dynamic_channels)

        
        
    def _parse_config(self, config: Dict[str, Any]):
        """
        Parse configuration dictionary to setup model.
        """
        self.input_specs = {field: config['data']['fields_scheme'].lower().count(field[0].lower())
            for field in config['data']['fields'] if field}
        
        self.output_specs = {field: config['data']['fields_scheme'].lower().count(field[0].lower())
            for i, field in enumerate(config['data']['fields'])
            if field and config['data']['fields_type'][i].upper() == 'D'}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the synthetic model using an additive residual.

        This implementation assumes that dynamic fields are ordered first in the
        channel dimension, followed by static fields.

        It predicts the residual for dynamic fields, pads it with zeros for the
        static fields, and adds the result to the input tensor `x`.

        Args:
            x: Input tensor. Expected shape:
               - BVTS: [B, V, T, H, W] where V==channels per-frame and T is time.

        Returns:
            Output tensor in BVTS layout [B, V, T, H, W] representing the
            next state.
        """
        # Guaranteed BVTS: [B, V, T, H, W]
        B, V, T, H, W = x.shape

        # Move time into batch: [B*T, V, H, W]
        reshaped_in = x.permute(0, 2, 1, 3, 4).reshape(B * T, V, H, W)

        # Run through network which expects [batch, channels, H, W]
        dynamic_out = self.net(reshaped_in)

        # dynamic_out: [B*T, num_dynamic_channels, H, W]
        out_frames = torch.empty_like(reshaped_in)
        out_frames[:, self._dynamic_slice] = dynamic_out
        out_frames[:, self.num_dynamic_channels:] = reshaped_in[:, self.num_dynamic_channels:]

        # Reshape back to BVTS: [B, V, T, H, W]
        out = out_frames.reshape(B, T, V, H, W).permute(0, 2, 1, 3, 4)
        return out


    # In src/models/synthetic/base.py

    @torch.no_grad()
    def generate_predictions(
        self,
        trajectories: List[Dict[str, torch.Tensor]],
        device: str = "cuda",
        batch_size: int = 1,
    ):
        """
        Generate one-step predictions from physical model trajectories.
        
        NEW BEHAVIOR:
        - Takes physical trajectories directly (not through dataset)
        - For each trajectory, generates one-step predictions autoregressively
        - Returns as trajectory format: [initial_real, pred_from_real, pred_from_pred, ...]
        - This allows the same windowing logic via indexing
        
        Args:
            trajectories: List of trajectory dicts in cache format
                        Format: [{'tensor_data': {field_name: tensor[C, T, H, W]}}]
            device: Device to run predictions on
            batch_size: Batch size for inference (currently unused, could batch multiple trajectories)
            
        Returns:
            List of prediction trajectories in BVTS format [1, V, T, H, W]
        """
        self.eval()
        self.to(device)
        
        prediction_trajectories = []
        
        for tensor_data in trajectories:
            # Concatenate all fields along channel dimension
            # Each field: [C, T, H, W]
            field_tensors = [tensor_data[field_name] for field_name in sorted(tensor_data.keys())]
            full_trajectory = torch.cat(field_tensors, dim=0)  # [C_all, T, H, W]
            
            # Move to device
            full_trajectory = full_trajectory.to(device, non_blocking=True)
            
            # Get trajectory length
            num_steps = full_trajectory.shape[1] 
            trajectory_frames = [full_trajectory[:, 0:1, :, :]] 
            
            with torch.amp.autocast(enabled=True, device_type=device):
                # Generate predictions for remaining timesteps
                for t in range(num_steps-1):
                    # One-step prediction
                    next_state = self(full_trajectory[:, t:t+1, :, :].unsqueeze(0)).squeeze(0)
                    # Store prediction (squeeze time dim)
                    trajectory_frames.append(next_state)  
            
            trajectory_tensor = torch.cat(trajectory_frames, dim=1)
            # Store as CPU tensor
            prediction_trajectories.append(trajectory_tensor.cpu())
        
        
        return prediction_trajectories

    @staticmethod
    def _select_proportional_indices(total_count: int, sample_count: int):
        """
        Select indices proportionally across the dataset.

        Ensures diverse sampling rather than just taking the first N samples.
        """
        if sample_count >= total_count:
            return list(range(total_count))

        # Calculate step size for proportional sampling
        step = total_count / sample_count

        # Select indices evenly distributed
        indices = [int(i * step) for i in range(sample_count)]

        # Ensure no duplicates and within bounds
        indices = sorted(list(set(indices)))[:sample_count]

        return indices
