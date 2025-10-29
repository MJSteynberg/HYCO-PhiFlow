# src/models/synthetic/unet.py

from typing import Dict, Any, List
import torch
from phiml.math import Tensor
import phiml.nn as pnn

# Import the base class
from .base import SyntheticModel

class UNet(SyntheticModel):
    """
    A generic U-Net wrapper that conforms to the SyntheticModel interface.

    This model is "input agnostic". It assembles an input tensor from
    a dictionary of fields based on 'input_specs' and disassembles
    its output tensor into a dictionary based on 'output_specs'.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the U-Net model.

        Args:
            config: A dictionary, expected to contain:
                - 'input_specs' (Dict[str, int]): Map of {field_name: channels},
                  e.g., {'density': 1, 'velocity': 2, 'inflow': 1}
                - 'output_specs' (Dict[str, int]): Map of {field_name: channels},
                  e.g., {'density': 1, 'velocity': 2}
                - 'levels' (int): e.g., 4
                - 'filters' (int): e.g., 64
                - 'batch_norm' (bool): e.g., True
        """
        super().__init__(config)

        # --- NEW: Calculate in/out channels from specs ---
        in_channels = sum(self.INPUT_SPECS.values())
        out_channels = sum(self.OUTPUT_SPECS.values())

        # The actual PhiFlow U-Net model is held as a submodule
        self.unet = pnn.u_net(
            in_channels=in_channels,
            out_channels=out_channels,
            levels=config.get('levels', 4),
            filters=config.get('filters', 64),
            batch_norm=config.get('batch_norm', True),
        )

    def forward(self, state: Dict[str, Tensor], dt: float = 0.0) -> Dict[str, Tensor]:
        """
        Performs one forward pass.

        Args:
            state: A dictionary of Tensors. Must contain all keys
                   from 'self.input_specs'.
            dt: Time step (unused by this model).

        Returns:
            A dictionary of Tensors containing the predicted fields
            as defined in 'self.output_specs'.
        """
        # 1. --- NEW: Generic assembly of the input tensor ---
        tensors_to_cat = [state[key] for key in self.INPUT_FIELDS]
        
            
        model_input = torch.cat(tensors_to_cat, dim=1) # Concat on channel dim

        # 2. Forward pass
        pred_tensor = self.unet(model_input)

        # 3. --- NEW: Generic splitting of the output tensor ---
        output_dict = {}
        start_channel = 0
        for field_name, num_channels in self.OUTPUT_SPECS.items():
            end_channel = start_channel + num_channels
            output_dict[field_name] = pred_tensor[:, start_channel:end_channel, ...]
            start_channel = end_channel
            
        return output_dict