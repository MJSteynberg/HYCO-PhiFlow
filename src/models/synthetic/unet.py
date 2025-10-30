# src/models/synthetic/unet.py

from typing import Dict, Any
import torch
import torch.nn as nn
from phiml.nn import u_net


class UNet(nn.Module):
    """
    Tensor-based U-Net for efficient training.
    
    Works directly with PyTorch tensors in [batch, channels, height, width] format.
    All Field conversions are handled by DataManager before training.
    
    This replaces the old Field-based interface for better performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the U-Net model.
        
        Args:
            config: Model configuration containing:
                - input_specs: Dict[field_name, num_channels]
                - output_specs: Dict[field_name, num_channels]
                - architecture: Dict with levels, filters, batch_norm
        """
        super().__init__()
        
        self.config = config
        self.input_specs = config['input_specs']
        self.output_specs = config['output_specs']
        
        # Calculate total channels
        self.in_channels = sum(self.input_specs.values())
        self.out_channels = sum(self.output_specs.values())
        
        # Get architecture params
        arch_config = config.get('architecture', {})
        levels = arch_config.get('levels', 4)
        filters = arch_config.get('filters', 64)
        batch_norm = arch_config.get('batch_norm', True)
        
        # Build the U-Net using PhiML's u_net
        self.unet = u_net(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            levels=levels,
            filters=filters,
            batch_norm=batch_norm,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net.
        
        Args:
            x: Input tensor of shape [batch, channels, height, width]
            
        Returns:
            Output tensor of shape [batch, channels, height, width]
        """
        return self.unet(x)