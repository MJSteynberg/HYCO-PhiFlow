# In src/models/synthetic/base.py

from abc import ABC
from typing import Dict, Any, List

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging

from src.utils.field_conversion.validation import assert_bvts_format


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
            config: A dictionary containing model-specific configurations.
                    Expected to contain 'input_specs' and 'output_specs'
                    dictionaries, e.g., {'density': 1, 'velocity': 2}.
        """
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.input_specs = {
        field: config['physical']['fields_scheme'].lower().count(field[0].lower())
        for field in config['physical']['fields']
        if field
        }
        self.output_specs = {
        field: config['physical']['fields_scheme'].lower().count(field[0].lower())
        for i, field in enumerate(config['physical']['fields'])
        if field and config['physical']['fields_type'][i].upper() == 'D'
        }
        # Calculate channel counts for padding
        self.num_dynamic_channels = sum(self.output_specs.values())
        self.num_static_channels = sum(self.input_specs.values()) - self.num_dynamic_channels
        self._dynamic_slice = slice(0, self.num_dynamic_channels)
        


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
        # Enforce BVTS at entry
        assert_bvts_format(x, context=f"{self.__class__.__name__}.forward input")

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
        out = out_frames.reshape(B, T, V, H, W).permute(0, 2, 1, 3, 4).contiguous()

        # Enforce BVTS on output
        assert_bvts_format(out, context=f"{self.__class__.__name__}.forward output")
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
        
        if not trajectories:
            self.logger.warning("No trajectories provided for prediction generation")
            return []
        
        self.logger.debug(f"Generating predictions from {len(trajectories)} physical trajectories")
        
        prediction_trajectories = []
        
        for traj_idx, trajectory_dict in enumerate(trajectories):
            # Extract tensor_data from cache format
            if isinstance(trajectory_dict, dict) and 'tensor_data' in trajectory_dict:
                tensor_data = trajectory_dict['tensor_data']
            else:
                tensor_data = trajectory_dict
            
            # Concatenate all fields along channel dimension
            # Each field: [C, T, H, W]
            field_tensors = [tensor_data[field_name] for field_name in sorted(tensor_data.keys())]
            full_trajectory = torch.cat(field_tensors, dim=0)  # [C_all, T, H, W]
            
            # Move to device
            full_trajectory = full_trajectory.to(device, non_blocking=True)
            
            # Get trajectory length
            num_steps = full_trajectory.shape[1]  # T dimension
            
            # Extract real initial condition (first frame)
            initial_real = full_trajectory[:, 0:1, :, :]  # [C_all, 1, H, W]
            # Build prediction trajectory: [initial_real, pred1, pred2, ...]
            trajectory_frames = [initial_real]  # Start with real initial [C_all, H, W]
            current_state = initial_real # [1, C_all, 1, H, W]
            
            with torch.amp.autocast(enabled=True, device_type=device):
                # Generate predictions for remaining timesteps
                for t in range(1, num_steps):
                    # One-step prediction
                    next_state = self(current_state.unsqueeze(0)).squeeze(0) # [C_all, 1, H, W]
                    # Store prediction (squeeze time dim)
                    trajectory_frames.append(next_state)  # [C_all, H, W]
                    
                    # Use prediction as next input
                    current_state = full_trajectory[:, t:t+1, :, :]  # Teacher forcing with real data
            
            # Stack into trajectory: [T, C_all, H, W]
            trajectory_tensor = torch.cat(trajectory_frames, dim=1)
            # Store as CPU tensor
            prediction_trajectories.append(trajectory_tensor.cpu())
            
            if (traj_idx + 1) % 50 == 0:
                self.logger.debug(f"  Generated {traj_idx + 1}/{len(trajectories)} predictions")
        
        self.logger.debug(f"Generated {len(prediction_trajectories)} prediction trajectories")
        
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
