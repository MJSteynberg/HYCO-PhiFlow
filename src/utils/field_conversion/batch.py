"""
Simplified Batch Concatenation Converter

Key simplifications:
1. All single-field converters now produce consistent [B, C, H, W] format
2. Channel concatenation is straightforward: torch.cat along dim=-3
3. Channel slicing is simple: pre-computed offsets
4. No complex shape normalization needed
"""

from typing import Dict, Optional
import torch
from phi.field import Field
from .base import AbstractConverter
from .metadata import FieldMetadata
from .centered import CenteredConverter
from .staggered import StaggeredConverter


class BatchConcatenationConverter:
    """
    Simplified multi-field converter with channel concatenation.
    
    All single-field converters now guarantee [B, C, H, W] output,
    so concatenation and slicing are trivial operations.
    
    Channel layout:
    - Fields are concatenated in order of self.field_names
    - Each field contributes its channel count
    - Offsets are pre-computed for fast slicing
    """

    def __init__(self, field_metadata: Dict[str, FieldMetadata]):
        """
        Initialize batch converter.
        
        Args:
            field_metadata: Dict mapping field names to FieldMetadata
        """
        self.field_metadata = field_metadata
        self.field_names = list(field_metadata.keys())

        # Create individual converters
        self._converters: Dict[str, AbstractConverter] = {}
        for name, metadata in field_metadata.items():
            if metadata.field_type == "centered":
                self._converters[name] = CenteredConverter(metadata)
            elif metadata.field_type == "staggered":
                self._converters[name] = StaggeredConverter(metadata)
            else:
                raise ValueError(f"Unknown field type: {metadata.field_type}")

        # Pre-compute channel layout
        self._compute_channel_layout()

    def _compute_channel_layout(self):
        """
        Pre-compute channel counts and offsets for fast slicing.
        
        Channel count logic:
        - Vector field: len(spatial_dims) channels
        - Scalar field: 1 channel
        """
        self.channel_counts = {}
        self.channel_offsets = {}
        offset = 0

        for name in self.field_names:
            metadata = self.field_metadata[name]
            
            # Determine channel count
            if metadata.channel_dims and 'vector' in metadata.channel_dims:
                # Vector field: one channel per spatial dimension
                num_channels = len(metadata.spatial_dims)
            else:
                # Scalar field: single channel
                num_channels = 1

            self.channel_counts[name] = num_channels
            self.channel_offsets[name] = offset
            offset += num_channels

        self.total_channels = offset

    def fields_to_tensor_batch(
        self, 
        fields: Dict[str, Field], 
        ensure_cpu: bool = False
    ) -> torch.Tensor:
        """
        Convert fields dict to concatenated tensor.
        
        Simple process:
        1. Convert each field to tensor (guaranteed [B, C, H, W])
        2. Concatenate along channel dimension
        
        Args:
            fields: Dict mapping field names to Field objects
            ensure_cpu: If True, ensures output is on CPU
            
        Returns:
            Concatenated tensor [B, C_total, H, W]
        """
        if set(fields.keys()) != set(self.field_names):
            raise ValueError(
                f"Field names mismatch. Expected {self.field_names}, "
                f"got {list(fields.keys())}"
            )

        # Convert each field (guaranteed consistent format)
        tensors = []
        for name in self.field_names:  # Maintain order
            field = fields[name]
            converter = self._converters[name]
            tensor = converter.field_to_tensor(field, ensure_cpu=ensure_cpu)
            tensors.append(tensor)

        # Simple concatenation along channel dimension (dim=-3)
        return torch.cat(tensors, dim=-3)

    def tensor_to_fields_batch(
        self,
        tensor: torch.Tensor,
        time_slice: Optional[int] = None
    ) -> Dict[str, Field]:
        """
        Convert concatenated tensor back to fields dict.
        
        Simple process:
        1. Slice tensor using pre-computed offsets
        2. Convert each slice to Field
        
        Args:
            tensor: Concatenated tensor [B, C_total, H, W] or [C_total, H, W]
            time_slice: Optional time index (unused, kept for API compatibility)
            
        Returns:
            Dict mapping field names to Field objects
        """
        # Verify channel dimension
        channel_dim = -3  # Channel dimension in [B, C, H, W]
        actual_channels = tensor.shape[channel_dim]
        
        if actual_channels != self.total_channels:
            raise ValueError(
                f"Expected {self.total_channels} channels, got {actual_channels}. "
                f"Tensor shape: {tensor.shape}"
            )

        # Slice and convert each field
        fields = {}
        for name in self.field_names:
            start_idx = self.channel_offsets[name]
            end_idx = start_idx + self.channel_counts[name]

            # Slice field channels
            if tensor.dim() == 4:  # [B, C, H, W]
                field_tensor = tensor[:, start_idx:end_idx, :, :]
            else:  # [C, H, W]
                field_tensor = tensor[start_idx:end_idx, :, :]

            # Convert to Field
            converter = self._converters[name]
            metadata = self.field_metadata[name]
            field = converter.tensor_to_field(
                field_tensor, metadata, time_slice=time_slice
            )
            fields[name] = field

        return fields

    def get_channel_info(self) -> Dict[str, Dict[str, int]]:
        """
        Get channel layout information.
        
        Returns:
            Dict mapping field names to {'count': int, 'offset': int}
        """
        return {
            name: {
                'count': self.channel_counts[name],
                'offset': self.channel_offsets[name],
            }
            for name in self.field_names
        }

    def validate_fields(self, fields: Dict[str, Field]) -> bool:
        """Validate that fields dict is compatible."""
        if set(fields.keys()) != set(self.field_names):
            raise ValueError(
                f"Field names mismatch. Expected {self.field_names}, "
                f"got {list(fields.keys())}"
            )
        return True

    def validate_tensor(self, tensor: torch.Tensor) -> bool:
        """Validate that tensor is compatible."""
        if tensor.dim() not in [3, 4]:
            raise ValueError(
                f"Expected 3D or 4D tensor, got {tensor.dim()}D"
            )

        channel_dim = -3
        if tensor.shape[channel_dim] != self.total_channels:
            raise ValueError(
                f"Expected {self.total_channels} channels, "
                f"got {tensor.shape[channel_dim]}"
            )

        return True