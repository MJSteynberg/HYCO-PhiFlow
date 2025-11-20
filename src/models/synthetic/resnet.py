"""
ResNet model using pure PhiML.
"""

from phiml import nn
from src.models.synthetic.base import SyntheticModel
from src.models import ModelRegistry
from typing import Dict, Any


@ModelRegistry.register_synthetic("ResNet")
class ResNet(SyntheticModel):
    """
    ResNet architecture for field-to-field mapping.
    Pure PhiML implementation.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Get architecture config
        arch_config = config["model"]["synthetic"]["architecture"]

        # Create ResNet using phiml.nn
        self.network = nn.res_net(
            in_channels=self.num_dynamic_channels,
            out_channels=self.num_dynamic_channels,
            layers=arch_config.get("layers", [16, 32, 64]),
            batch_norm=arch_config.get("batch_norm", True),
            activation=arch_config.get("activation", "ReLU")
        )

        self.logger.info(f"Created ResNet: layers={arch_config.get('layers', [16, 32, 64])}")
