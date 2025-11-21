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
        output_dict = {"velocity": math.native_call(self.network, x['velocity'])}

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
