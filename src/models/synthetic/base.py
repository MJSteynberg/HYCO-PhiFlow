"""PhiML synthetic model base class using field channel with static field separation."""

from phiml import math, nn
from phi.math import Tensor
from phiml.math import channel
import logging
from typing import Optional, Dict, Any, List


class SyntheticModel:
    """
    Base class for synthetic models using pure PhiML.

    Uses a single 'field' channel dimension with item names.
    Static fields (specified by name) are passed through unchanged.
    Dynamic fields are predicted by the network.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        num_channels: int,
        static_fields: Optional[List[str]] = None
    ):
        self.config = config
        self.num_channels = num_channels
        self.static_fields = static_fields or []
        self.logger = logging.getLogger(self.__class__.__name__)

        self.num_static = len(self.static_fields)
        self.num_dynamic = num_channels - self.num_static

        # Input noise configuration
        noise_config = config.get('model', {}).get('synthetic', {}).get('input_noise', {})
        self.input_noise_enabled = noise_config.get('enabled', False)
        self.input_noise_scale = noise_config.get('scale', 0.01)
        self.input_noise_type = noise_config.get('type', 'gaussian')

        # Training mode flag (set by trainer)
        self.training = False

        self.logger.info(
            f"Initializing {self.__class__.__name__}: "
            f"{self.num_dynamic} dynamic, {self.num_static} static channels"
        )

        if self.input_noise_enabled:
            self.logger.info(
                f"Input noise enabled: type={self.input_noise_type}, "
                f"scale={self.input_noise_scale} (training only)"
            )

        self._network = None

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, value):
        self._network = value

    def __call__(self, state: Tensor) -> Tensor:
        """
        Forward pass predicting next state.

        Extracts dynamic fields, passes through network, recombines with static.
        """
        # Apply input noise during training only
        if self.training and self.input_noise_enabled and self.input_noise_scale > 0:
            if self.input_noise_type == 'gaussian':
                noise = math.random_normal(state.shape) * self.input_noise_scale
                state = state + noise
            else:
                self.logger.warning(f"Unknown noise type: {self.input_noise_type}, skipping noise")

        if not self.static_fields:
            # No static fields - pass entire state through network
            # Extract field names to restore after network call
            field_names = state.shape['field'].item_names
            # Normalize field_names to list of strings
            if isinstance(field_names, tuple) and len(field_names) > 0 and isinstance(field_names[0], tuple):
                field_names = field_names[0]
            if isinstance(field_names, str):
                field_names = [field_names]
            elif not isinstance(field_names, (list, tuple)):
                field_names = list(field_names)

            predicted = math.native_call(self._network, state)

            # Restore field dimension name if needed
            if 'field' not in predicted.shape.names:
                channel_dim = predicted.shape.channel
                if channel_dim:
                    # Network has channel dim, rename it to field
                    predicted = math.rename_dims(
                        predicted,
                        channel_dim.name,
                        channel(field=','.join(field_names))
                    )
                else:
                    # Network output has no channel dim (single channel was squeezed)
                    # Explicitly add field dimension
                    predicted = math.expand(predicted, channel(field=','.join(field_names)))
            return predicted

        # Extract field names from tensor
        field_names = state.shape['field'].item_names
        # Normalize field_names to list of strings
        if isinstance(field_names, tuple) and len(field_names) > 0 and isinstance(field_names[0], tuple):
            field_names = field_names[0]
        if isinstance(field_names, str):
            field_names = [field_names]
        elif not isinstance(field_names, (list, tuple)):
            field_names = list(field_names)

        # Separate dynamic and static fields
        dynamic_names = [f for f in field_names if f not in self.static_fields]

        # Extract and stack dynamic fields
        dynamic_state = math.stack(
            [state.field[name] for name in dynamic_names],
            channel(field=','.join(dynamic_names))
        )

        # Predict dynamic fields
        predicted_dynamic = math.native_call(self._network, dynamic_state)

        # Restore field dimension name after network (network may output generic channel dim)
        if 'field' not in predicted_dynamic.shape.names:
            # Network output has a generic channel dimension - rename it to field
            channel_dim = predicted_dynamic.shape.channel
            if channel_dim:
                predicted_dynamic = math.rename_dims(
                    predicted_dynamic,
                    channel_dim.name,
                    channel(field=','.join(dynamic_names))
                )
            else:
                # No channel dim - expand with field
                predicted_dynamic = math.expand(
                    predicted_dynamic,
                    channel(field=','.join(dynamic_names))
                )

        # Extract and stack static fields
        static_state = math.stack(
            [state.field[name] for name in self.static_fields],
            channel(field=','.join(self.static_fields))
        )

        # Recombine: dynamic first, then static
        return math.concat([predicted_dynamic, static_state], 'field')

    def save(self, path: str):
        nn.save_state(self._network, path)

    def load(self, path: str):
        nn.load_state(self._network, path)
