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
    
    Handles static vs dynamic fields:
    - Input contains all fields (static + dynamic)
    - Model predicts only dynamic fields
    - Static fields are automatically preserved and re-attached to output
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the U-Net model.
        
        Args:
            config: Model configuration containing:
                - input_specs: Dict[field_name, num_channels] - all input fields
                - output_specs: Dict[field_name, num_channels] - fields to predict
                - architecture: Dict with levels, filters, batch_norm
        """
        super().__init__()
        
        self.config = config
        self.input_specs = config['input_specs']
        self.output_specs = config['output_specs']
        
        # Calculate total channels
        self.in_channels = sum(self.input_specs.values())
        self.out_channels = sum(self.output_specs.values())
        
        # Identify static fields (in input but not output)
        self.static_fields = [f for f in self.input_specs if f not in self.output_specs]
        self.dynamic_fields = list(self.output_specs.keys())
        
        # Build channel indices for slicing
        self._build_channel_indices()
        
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
    
    def _build_channel_indices(self):
        """Build channel start/end indices for all fields in input order."""
        self.input_channel_map = {}
        offset = 0
        
        for field_name, num_channels in self.input_specs.items():
            self.input_channel_map[field_name] = (offset, offset + num_channels)
            offset += num_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net.
        
        If there are static fields (in input but not output):
        - Extracts static field channels from input
        - Passes only relevant channels to U-Net
        - Re-attaches static fields to output in original order
        
        Args:
            x: Input tensor of shape [batch, in_channels, height, width]
               Contains all fields in input_specs order
            
        Returns:
            Output tensor of shape [batch, in_channels, height, width]
            Contains dynamic predictions + preserved static fields
        """
        if not self.static_fields:
            # No static fields - simple pass-through
            return self.unet(x)
        
        # Extract static field channels to preserve them
        static_channels = {}
        for field_name in self.static_fields:
            start, end = self.input_channel_map[field_name]
            static_channels[field_name] = x[:, start:end, :, :]
        
        # Run U-Net to predict dynamic fields
        dynamic_prediction = self.unet(x)  # [B, out_channels, H, W]
        
        # Reconstruct full output by interleaving static and dynamic fields
        # in the original input_specs order
        output_tensors = []
        dynamic_offset = 0
        
        for field_name in self.input_specs.keys():
            if field_name in self.static_fields:
                # Use preserved static field
                output_tensors.append(static_channels[field_name])
            else:
                # Use predicted dynamic field
                num_channels = self.output_specs[field_name]
                output_tensors.append(dynamic_prediction[:, dynamic_offset:dynamic_offset + num_channels, :, :])
                dynamic_offset += num_channels
        
        # Concatenate in input order
        return torch.cat(output_tensors, dim=1)  # [B, in_channels, H, W]