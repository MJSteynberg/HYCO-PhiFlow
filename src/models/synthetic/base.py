# In src/models/synthetic/base.py

from abc import ABC
from typing import Dict, Any

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


    @torch.no_grad()
    def generate_predictions(
        self,
        real_dataset,
        alpha: float,
        device: str = "cpu",
        batch_size: int = 32,
    ):
        """
        OPTIMIZED: Pre-allocate tensors, avoid list appends.
        """
        self.eval()
        self.to(device)

        # If dataset exposes `num_augmented` and it is > 0, prefer that.
        num_augmented = getattr(real_dataset, 'num_augmented', 0)
        num_generate = int(num_augmented)

        if num_generate == 0:
            self.logger.warning("Alpha too small, no samples will be generated")
            return torch.empty(0), torch.empty(0)

        # Get sample shape
        sample_input, _ = real_dataset[0]
        input_shape = sample_input.shape

        # PRE-ALLOCATE full tensors (MUCH FASTER!)
        all_inputs = torch.empty(
            num_generate, *input_shape, 
            dtype=sample_input.dtype, 
            device=device
        )
        all_predictions = torch.empty(
            num_generate, *input_shape,
            dtype=sample_input.dtype,
            device=device
        )

        # Select diverse indices. If augmented samples are present, we want to
        # select indices only from the augmented-region of the dataset so the
        # synthetic predictions correspond to physically-generated trajectories.
        total_aug = num_augmented
        aug_start = len(real_dataset) - total_aug
        indices = [aug_start + i for i in range(total_aug)]

        subset = torch.utils.data.Subset(real_dataset, indices)
        
        # Enable pin_memory for faster transfer
        loader = DataLoader(
            subset, 
            batch_size=batch_size, 
            shuffle=False,
            pin_memory=(device != 'cpu'),
            num_workers=0  # Can be increased if data loading is bottleneck
        )
        idx = 0
        for batch_inputs, _ in loader:
            batch_size_actual = batch_inputs.size(0)
            
            # Non-blocking transfer for async GPU copy
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_predictions = self(batch_inputs)

            # Direct copy to pre-allocated tensor
            all_inputs[idx:idx + batch_size_actual] = batch_inputs
            all_predictions[idx:idx + batch_size_actual] = batch_predictions
            
            idx += batch_size_actual

        # Return only the filled prefix in case num_generate was not perfectly
        # filled by the loader (due to rounding or selection).
        self.logger.debug(f"generate_predictions: actually generated {idx} samples (requested {num_generate})")

        # Normalize outputs to CPU, detached tensors. Returning CPU tensors
        # simplifies downstream code (HybridTrainer / FieldDataset) which
        # expects to receive CPU-side tensors that it can further process.
        return all_inputs[:idx].detach().cpu(), all_predictions[:idx].detach().cpu()

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
