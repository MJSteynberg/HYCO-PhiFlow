"""
UNet model using pure PhiML.
"""

from phiml import nn
from src.models.synthetic.base import SyntheticModel
from src.models import ModelRegistry
from typing import Dict, Any


@ModelRegistry.register_synthetic("UNet")
class UNet(SyntheticModel):
    """
    UNet architecture for field-to-field mapping.
    Pure PhiML implementation.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Get architecture config
        arch_config = config["model"]["synthetic"]["architecture"]

        # Create U-Net using phiml.nn
        self.network = nn.u_net(
            in_channels=self.num_dynamic_channels,
            out_channels=self.num_dynamic_channels,
            levels=arch_config.get("levels", 4),
            filters=arch_config.get("filters", 32),
            batch_norm=arch_config.get("batch_norm", True),
            activation=arch_config.get("activation", "ReLU"),
            in_spatial=2  # 2D spatial dimensions
        )

        self.logger.info(
            f"Created U-Net: levels={arch_config.get('levels', 4)}, "
            f"filters={arch_config.get('filters', 32)}"
        )
