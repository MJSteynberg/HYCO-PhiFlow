# src/models/synthetic/resnet.py

from typing import Dict, Any
import torch
import torch.nn as nn
from phiml.nn import res_net
from src.models import ModelRegistry
from src.models.synthetic.base import SyntheticModel


@ModelRegistry.register_synthetic("ResNet")
class ResNet(SyntheticModel):
    """
    Tensor-based ResNet for efficient training.

    Works directly with PyTorch tensors in [batch, channels, height, width] format.
    All Field conversions are handled by DataManager before training.

    Handles static vs dynamic fields:
    - Input contains all fields (static + dynamic)
    - Model predicts only dynamic fields
    - Static fields are automatically preserved and re-attached to output
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the ResNet model.

        Args:
            config: Model configuration containing:
                - input_specs: Dict[field_name, num_channels] - all input fields
                - output_specs: Dict[field_name, num_channels] - fields to predict
                - architecture: Dict with levels, filters, batch_norm
        """
        # Call parent constructor to set up base attributes
        super().__init__(config)

        # Get architecture params
        layers = config["synthetic"]['architecture']["layers"]

        # Build the ResNet using PhiML's res_net
        self.net = res_net(
            in_channels=sum(self.input_specs.values()),
            out_channels=sum(self.output_specs.values()),
            layers=layers,
            batch_norm=True,
        )

    