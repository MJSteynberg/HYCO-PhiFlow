"""
PhiML-native synthetic model base class.
Models work entirely with PhiML tensors (no PyTorch dependency).
"""

from phiml import math, Tensor, nn
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

    def __call__(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Forward pass predicting next state directly.

        Args:
            x: Dictionary mapping field names to PhiML Tensors
               Example: {'velocity': Tensor(batch?, x, vector)}

        Returns:
            Dictionary mapping field names to predicted next state
        """
        field_order = list(self.input_specs.keys())
        field_tensors = [x[field_name] for field_name in field_order]
        concatenated = math.concat(field_tensors, 'vector')

        # Rename vector -> channel for network (U-Net expects 'channel' dimension)
        network_input = math.rename_dims(concatenated, 'vector', 'channel')

        # Case 1: All fields are dynamic (no static fields)
        if self.num_static_channels == 0:
            # Predict next state directly
            predicted = math.native_call(self.network, network_input)
        else:
            # Case 2: Some fields are static
            dynamic = network_input.channel[0:self.num_dynamic_channels]
            static = network_input.channel[self.num_dynamic_channels:self.total_channels]

            predicted_dynamic = math.native_call(self.network, dynamic)
            predicted = math.concat([predicted_dynamic, static], 'channel')

        # Ensure channel dimension exists (network might drop it if size=1)
        if 'channel' not in predicted.shape.names:
            # Network dropped the channel dimension - add it back
            # Get the original channel shape from input
            original_channel_shape = network_input.shape.only('channel')
            # Add the channel dimension back using merge_shapes
            predicted = math.expand(predicted, original_channel_shape)

        # Rename channel -> vector for output
        predicted = math.rename_dims(predicted, 'channel', 'vector')

        # Split back into separate fields
        output_dict = {}
        channel_idx = 0
        for field_name in field_order:
            num_channels = self.input_specs[field_name]
            output_dict[field_name] = predicted.vector[channel_idx:channel_idx + num_channels]
            channel_idx += num_channels

        return output_dict

    def get_network(self):
        """Get the underlying network (for optimization)"""
        return self.network

    def save(self, path: str):
        """Save network parameters"""
        nn.save_state(self.network, path)

    def load(self, path: str):
        """Load network parameters"""
        nn.load_state(self.network, path)
