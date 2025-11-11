# In src/models/synthetic/base.py

from abc import ABC
from typing import Dict, Any

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
            x: Input tensor of shape [B, C, H, W], where C is the total
               number of channels (dynamic and static).

        Returns:
            Output tensor of the same shape [B, C, H, W], representing the
            next state.
        """
        dynamic_out = self.net(x)
        out = torch.empty_like(x)
        out[:, self._dynamic_slice] = dynamic_out
        out[:, self.num_dynamic_channels:] = x[:, self.num_dynamic_channels:]
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

        num_real = len(real_dataset)
        num_generate = int(num_real * alpha)

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

        # Select diverse indices
        indices = self._select_proportional_indices(num_real, num_generate)
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

        return all_inputs, all_predictions

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
