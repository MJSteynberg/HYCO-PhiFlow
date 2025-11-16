# src/models/synthetic/convnet.py

from typing import Dict, Any
import torch
import torch.nn as nn
from phiml.nn import conv_net
from src.models import ModelRegistry
from src.models.synthetic.base import SyntheticModel


@ModelRegistry.register_synthetic("ConvNet")
class ConvNet(SyntheticModel):
    """
    Tensor-based ConvNet for efficient training.

    Works directly with PyTorch tensors in [batch, channels, height, width] format.
    All Field conversions are handled by DataManager before training.

    Handles static vs dynamic fields:
    - Input contains all fields (static + dynamic)
    - Model predicts only dynamic fields
    - Static fields are automatically preserved and re-attached to output
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the ConvNet model.

        Args:
            config: Model configuration containing:
                - input_specs: Dict[field_name, num_channels] - all input fields
                - output_specs: Dict[field_name, num_channels] - fields to predict
                - architecture: Dict with levels, filters, batch_norm
        """
        # Call parent constructor to set up base attributes
        super().__init__(config)

        # Get architecture params (with defaults for backwards compatibility)
        layers = config["synthetic"]['architecture']['layers']

        # Build the ConvNet using PhiML's conv_net
        self.net = conv_net(
            in_channels=sum(self.input_specs.values()),
            out_channels=sum(self.output_specs.values()),
            layers=layers,
            batch_norm=True,
        )
