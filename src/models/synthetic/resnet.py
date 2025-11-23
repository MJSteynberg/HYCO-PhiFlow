"""ResNet model using pure PhiML."""

from phiml import nn
from src.models.synthetic.base import SyntheticModel
from src.models import ModelRegistry
from typing import Dict, Any, Optional, List


@ModelRegistry.register_synthetic("ResNet")
class ResNet(SyntheticModel):
    """ResNet architecture for field-to-field mapping."""

    def __init__(
        self,
        config: Dict[str, Any],
        num_channels: int,
        static_fields: Optional[List[str]] = None
    ):
        super().__init__(config, num_channels, static_fields)

        arch_config = config["model"]["synthetic"]["architecture"]
        n_spatial_dims = len(config["model"]["physical"]["domain"]["dimensions"])

        self.network = nn.res_net(
            in_channels=self.num_dynamic,
            out_channels=self.num_dynamic,
            layers=arch_config.get("layers", [16, 32, 64]),
            batch_norm=arch_config.get("batch_norm", True),
            activation=arch_config.get("activation", "ReLU"),
            in_spatial=n_spatial_dims
        )

        self.logger.info(
            f"Created ResNet: layers={arch_config.get('layers', [16, 32, 64])}, "
            f"spatial_dims={n_spatial_dims}"
        )
