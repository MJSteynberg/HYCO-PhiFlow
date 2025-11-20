"""
PhiML-native synthetic model base class.
Models work entirely with PhiML tensors (no PyTorch dependency).
"""

from phiml import math, Tensor
from phiml.math import channel
import logging
from typing import Optional, Dict, Any


class SyntheticModel:
    """
    Base class for synthetic models using pure PhiML.
    No torch.nn.Module inheritance.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Parse configuration
        self._parse_config(config)

        self.logger.info(
            f"Initializing {self.__class__.__name__}: "
            f"{self.num_dynamic_channels} dynamic, {self.num_static_channels} static channels"
        )

        # Network will be set by subclasses
        self.network = None

    def _parse_config(self, config: Dict[str, Any]):
        """
        Parse configuration dictionary to setup model.
        """
        # Calculate input/output specs from fields scheme
        self.input_specs = {
            field: config['data']['fields_scheme'].lower().count(field[0].lower())
            for field in config['data']['fields'] if field
        }

        self.output_specs = {
            field: config['data']['fields_scheme'].lower().count(field[0].lower())
            for i, field in enumerate(config['data']['fields'])
            if field and config['data']['fields_type'][i].upper() == 'D'
        }

        # Calculate channel counts
        self.num_dynamic_channels = sum(self.output_specs.values())
        self.num_static_channels = sum(self.input_specs.values()) - self.num_dynamic_channels
        self.total_channels = sum(self.input_specs.values())

    def __call__(self, x: Tensor) -> Tensor:
        """
        Forward pass with residual learning.

        Args:
            x: PhiML Tensor with shape (..., channels)
               Expected format: (batch?, time?, spatial..., channels)

        Returns:
            PhiML Tensor with same shape as input
        """
        # Find channel dimension name
        channel_dim = None
        for dim_name in x.shape.names:
            if dim_name in ['c', 'channels', 'channel', 'vector']:
                channel_dim = dim_name
                break

        if channel_dim is None:
            raise ValueError(f"Input must have channel dimension. Got shape: {x.shape}")

        # Split into dynamic and static channels
        dynamic = x[{channel_dim: slice(0, self.num_dynamic_channels)}]
        static = x[{channel_dim: slice(self.num_dynamic_channels, self.total_channels)}]

        # Predict residual for dynamic fields only
        residual = math.native_call(self.network, dynamic)

        # Residual learning: output = input + residual
        predicted_dynamic = dynamic + residual

        # Concatenate with unchanged static fields
        if self.num_static_channels > 0:
            output = math.concat([predicted_dynamic, static], channel_dim)
        else:
            output = predicted_dynamic

        return output

    def get_network(self):
        """Get the underlying network (for optimization)"""
        return self.network

    def save(self, path: str):
        """Save network parameters"""
        math.save(self.network, path)
        self.logger.info(f"Saved model to {path}")

    def load(self, path: str):
        """Load network parameters"""
        self.network = math.load(path)
        self.logger.info(f"Loaded model from {path}")
