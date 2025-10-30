# src/models/synthetic/unet.py

from typing import Dict, Any, List
import torch
import phiml.math
import phiml.nn as pnn
from phi.torch.flow import *
from phi import field as phi_field
from phi.field import native_call

# Import the base class
from .base import SyntheticModel
# In src/models/synthetic/unet.py (EXAMPLE)
from typing import Dict, Any
from phi.field import CenteredGrid, native_call
from phiml.nn import u_net  # or from phi.torch.nets import u_net as pnn
from .base import SyntheticModel

class UNet(SyntheticModel):
    """
    A generic U-Net wrapper that plugs into the SyntheticModel base class.
    
    The base class handles all field (Staggered/Centered) conversions.
    This class just defines the network architecture and the call to it.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the U-Net model.
        """
        super().__init__(config)
        
        in_channels = sum(self.INPUT_SPECS.values())
        out_channels = sum(self.OUTPUT_SPECS.values())

        self.unet = u_net(  # Assuming this is your network constructor
            in_channels=in_channels,
            out_channels=out_channels,
            levels=config.get('levels', 4),
            filters=config.get('filters', 64),
            batch_norm=config.get('batch_norm', True),
        )

    def _predict(self, nn_input_grid: CenteredGrid, dt: float) -> CenteredGrid:
        """
        Performs the core prediction.
        
        The input grid's 'vector' dimension holds all stacked channels.
        The output grid must also stack all output channels on 'vector'.
        """
        # The base class's `forward` method calls this.
        # This is now the *only* logic this class needs.
        return native_call(
            self.unet, 
            nn_input_grid, 
            channel_dim='vector', 
            channels_last=False
        )