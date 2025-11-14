"""
Simplified Centered Grid Converter

Key simplifications:
1. Assume tensors were created by our field_to_tensor (known layout)
2. Use PhiFlow's math.tensor() with explicit dimensions
3. Remove complex shape inference logic
4. Leverage PhiFlow's automatic dimension handling
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
    Simplified converter for CenteredGrid fields.
    
    Assumptions (guaranteed by our pipeline):
    - field_to_tensor always produces [B, C, H, W] or [C, H, W]
    - C=1 for scalar fields, C=len(spatial_dims) for vector fields
    - Spatial order matches metadata.spatial_dims
    """

    @classmethod
    def can_handle(cls, metadata: FieldMetadata) -> bool:
        """Check if this converter can handle the field type."""
        return metadata.field_type == "centered"

    def field_to_tensor(self, field: Field, *, ensure_cpu: bool = True) -> torch.Tensor:
        """
        Convert CenteredGrid to PyTorch tensor.
        
        Output format: [B, C, H, W] where:
        - B: batch dimension (if present)
        - C: channels (1 for scalar, len(spatial_dims) for vector)
        - H, W: spatial dimensions in order of metadata.spatial_dims
        
        Args:
            field: PhiFlow CenteredGrid
            ensure_cpu: If True, ensures output is on CPU
            
        Returns:
            Tensor with shape [B, C, H, W] or [C, H, W]
        """
        # PhiFlow native tensor order: batch, spatial..., vector
        # We want: batch, vector, spatial...
        
        # Build dimension order for native()
        dims = []
   
        dims.append("time")
        
        # Add vector (channels) - always present even for scalars
        dims.append('vector')
        
        # Add spatial dimensions in order
        dims.extend(self.metadata.spatial_dims if self.metadata else field.shape.spatial.names)
        
        # Get native tensor with our desired layout
        tensor = field.values.native(dims)
        
        return self._ensure_device(tensor, ensure_cpu)

    def tensor_to_field(
        self,
        tensor: torch.Tensor,
        metadata: FieldMetadata,
        *,
        time_slice: Optional[int] = None,
    ) -> Field:
        """
        Convert PyTorch tensor to CenteredGrid.
        
        Assumes tensor was created by field_to_tensor, so layout is known:
        - [B, C, H, W] or [C, H, W]
        - Spatial order matches metadata.spatial_dims
        
        Args:
            tensor: Tensor from field_to_tensor
            metadata: Field reconstruction metadata
            time_slice: Optional time index (unused, kept for API compatibility)
            
        Returns:
            Reconstructed CenteredGrid
        """
        # Determine if we have batch dimension
        has_batch = tensor.dim() == 4
        
        # Get spatial dimensions
        spatial_dims = metadata.spatial_dims
        num_spatial = len(spatial_dims)
        
        # Build PhiML shape specification
        shape_spec = []
        
        if has_batch:
            shape_spec.append(batch_dim('time'))
        
        # Channel dimension
        num_channels = tensor.shape[-3]  # Channel dim is always -3 in our layout
        is_vector = num_channels > 1
        
        if is_vector:
            # Vector field: label channels with spatial dimension names
            vector_labels = ','.join(spatial_dims)
            shape_spec.append(channel(vector=vector_labels))
        else:
            # Scalar field: single channel (will be squeezed by PhiFlow)
            shape_spec.append(channel(vector='scalar'))
        
        # Spatial dimensions
        spatial_sizes = {
            dim: tensor.shape[-num_spatial + i] 
            for i, dim in enumerate(spatial_dims)
        }
        shape_spec.append(spatial(**spatial_sizes))
        
        # Create PhiML tensor with explicit shape
        from functools import reduce
        import operator
        combined_shape = reduce(operator.and_, shape_spec)
        phiml_tensor = math.tensor(tensor, combined_shape)
        
        # For scalar fields, remove the channel dimension
        if not is_vector:
            phiml_tensor = phiml_tensor.vector['scalar']
        
        # Create CenteredGrid
        return CenteredGrid(
            phiml_tensor,
            metadata.extrapolation,
            bounds=metadata.domain
        )


    def _get_spatial_layout(self, tensor: torch.Tensor, metadata: FieldMetadata) -> str:
        """
        Determine spatial dimension names from tensor shape and metadata.
        
        Returns string like 'x,y' or 'x,y,z'
        """
        spatial_dims = metadata.spatial_dims
        return ','.join(spatial_dims)