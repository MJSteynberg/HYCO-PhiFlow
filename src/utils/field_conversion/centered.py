"""
Centered Grid Converter

This module implements conversion between CenteredGrid fields and tensors.
"""

from typing import Optional
import torch
from phi import math
from phi.math import spatial, channel, batch as batch_dim
from phi.field import Field, CenteredGrid
from .base import SingleFieldConverter
from .metadata import FieldMetadata


class CenteredConverter(SingleFieldConverter):
    """
    Converter for CenteredGrid fields.
    
    Handles conversion between PhiFlow CenteredGrid objects and PyTorch tensors.
    CenteredGrids have values sampled at cell centers, making the conversion
    straightforward.
    """
    
    @classmethod
    def can_handle(cls, metadata: FieldMetadata) -> bool:
        """Check if this converter can handle the field type."""
        return metadata.field_type == 'centered'
    
    def field_to_tensor(self, field: Field, *, ensure_cpu: bool = True) -> torch.Tensor:
        """
        Convert a CenteredGrid to a PyTorch tensor.
        
        Args:
            field: PhiFlow CenteredGrid to convert
            ensure_cpu: If True, ensures output tensor is on CPU
            
        Returns:
            PyTorch tensor with shape:
            - [H, W] for scalar fields without batch
            - [C, H, W] for vector fields without batch
            - [B, 1, H, W] for scalar fields with batch
            - [B, C, H, W] for vector fields with batch
        """
        self.validate_field(field)
        
        # Get the native tensor
        tensor = field.values._native
        
        # Determine field properties
        has_batch = field.shape.batch.rank > 0
        is_vector = field.shape.channel.rank > 0
        
        # PhiFlow native layout:
        # - Scalar no batch: [x, y]
        # - Scalar with batch: [batch, x, y]
        # - Vector no batch: [x, y, vector]
        # - Vector with batch: [batch, x, y, vector]
        
        # Target layout: [batch, channels, x, y] or [channels, x, y]
        
        if not has_batch and not is_vector:
            # Scalar field, no batch: [x, y] -> no change needed
            pass
        elif not has_batch and is_vector:
            # Vector field, no batch: [x, y, vector] -> [vector, x, y]
            tensor = tensor.permute(2, 0, 1)
        elif has_batch and not is_vector:
            # Scalar field with batch: [batch, x, y] -> [batch, 1, x, y]
            tensor = tensor.unsqueeze(1)
        elif has_batch and is_vector:
            # Vector field with batch: [batch, x, y, vector] -> [batch, vector, x, y]
            perm = [0, len(tensor.shape)-1] + list(range(1, len(tensor.shape)-1))
            tensor = tensor.permute(*perm)
        
        return self._ensure_device(tensor, ensure_cpu)
    
    def tensor_to_field(
        self,
        tensor: torch.Tensor,
        metadata: FieldMetadata,
        *,
        time_slice: Optional[int] = None
    ) -> Field:
        """
        Convert a PyTorch tensor to a CenteredGrid.
        
        Args:
            tensor: PyTorch tensor to convert
            metadata: FieldMetadata containing reconstruction information
            time_slice: Optional time index for batch tensors
            
        Returns:
            Reconstructed CenteredGrid
        """
        self.validate_tensor(tensor)
        
        # Handle time dimension
        if time_slice is not None:
            if len(tensor.shape) == 4:  # [time, channels, x, y]
                tensor = tensor[time_slice]  # -> [channels, x, y]
            elif len(tensor.shape) != 3:
                raise ValueError(
                    f"Expected tensor with 3 or 4 dimensions, got shape {tensor.shape}"
                )
        
        # Convert from [B, C, H, W] or [C, H, W] to PhiFlow layout
        # Target: [B, x, y, C] or [x, y, C]
        
        if len(tensor.shape) == 4:  # [B, C, H, W]
            tensor = tensor.permute(0, 2, 3, 1)  # -> [B, x, y, C]
        elif len(tensor.shape) == 3:  # [C, H, W]
            tensor = tensor.permute(1, 2, 0)  # -> [x, y, C]
        elif len(tensor.shape) == 2:  # [H, W] - scalar
            pass  # Already in correct layout
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
        
        # For scalar fields (1 channel), remove the channel dimension
        if tensor.shape[-1] == 1 and not metadata.channel_dims:
            tensor = tensor.squeeze(-1)
        
        # Determine spatial shape from tensor
        if len(tensor.shape) == 4:  # [B, x, y, C]
            tensor_spatial_shape = tensor.shape[1:1+len(metadata.spatial_dims)]
        elif len(tensor.shape) == 3:
            if metadata.channel_dims:  # [x, y, C]
                tensor_spatial_shape = tensor.shape[:len(metadata.spatial_dims)]
            else:  # [B, x, y] - scalar with batch
                tensor_spatial_shape = tensor.shape[1:]
        elif len(tensor.shape) == 2:  # [x, y]
            tensor_spatial_shape = tensor.shape
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
        
        # Create spatial shape
        spatial_sizes = {dim: size for dim, size in zip(metadata.spatial_dims, tensor_spatial_shape)}
        actual_resolution = spatial(**spatial_sizes)
        
        # Convert to PhiML tensor with proper dimensions
        if len(tensor.shape) == 2:  # [x, y] - scalar field
            phiml_tensor = math.tensor(tensor, actual_resolution)
        elif len(tensor.shape) == 3:
            if metadata.channel_dims:  # [x, y, C]
                # For vector fields, add labels matching spatial dimensions
                if 'vector' in metadata.channel_dims and len(metadata.channel_dims) == 1:
                    vector_labels = ','.join(metadata.spatial_dims)
                    phiml_tensor = math.tensor(
                        tensor, 
                        actual_resolution & channel(vector=vector_labels)
                    )
                else:
                    phiml_tensor = math.tensor(
                        tensor, 
                        actual_resolution & channel(*metadata.channel_dims)
                    )
            else:  # [B, x, y] - batch dimension
                phiml_tensor = math.tensor(tensor, batch_dim('time') & actual_resolution)
        elif len(tensor.shape) == 4:  # [B, x, y, C]
            if metadata.channel_dims:
                if 'vector' in metadata.channel_dims and len(metadata.channel_dims) == 1:
                    vector_labels = ','.join(metadata.spatial_dims)
                    phiml_tensor = math.tensor(
                        tensor,
                        batch_dim('time') & actual_resolution & channel(vector=vector_labels)
                    )
                else:
                    phiml_tensor = math.tensor(
                        tensor,
                        batch_dim('time') & actual_resolution & channel(*metadata.channel_dims)
                    )
            else:
                # Scalar field with time dimension
                phiml_tensor = math.tensor(tensor, batch_dim('time') & actual_resolution)
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
        
        # Create CenteredGrid
        centered_grid = CenteredGrid(
            phiml_tensor,
            metadata.extrapolation,
            bounds=metadata.domain
        )
        
        return centered_grid
