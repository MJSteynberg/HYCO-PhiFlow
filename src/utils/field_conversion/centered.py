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
from phi.math import spatial, channel, batch
from phi.field import Field, CenteredGrid
from .base import SingleFieldConverter
from .metadata import FieldMetadata
from src.utils.field_conversion.bvts import to_bvts
from src.utils.field_conversion.validation import assert_bvts_format


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
        # We want: time, vector, spatial...

        # Build dimension order for native(); request time and vector
        spatial_dims = self.metadata.spatial_dims if self.metadata else field.shape.spatial.names
        dims = ['time', 'vector', *spatial_dims]

        # Request native tensor in [T, V, *spatial] ordering where possible
        tensor = field.values.native(dims)

        # Convert to canonical BVTS: [B, V, T, H, W]
        tensor_bvts = to_bvts(tensor)

        # Validate that we produced BVTS internally
        assert_bvts_format(tensor_bvts, context="CenteredConverter.field_to_tensor intermediate BVTS")

        # For single-field converter API we return a single-timestep snapshot
        # with an explicit batch dimension: [B, V, H, W]. Do NOT drop the
        # batch dimension; callers should expect an explicit batch axis.
        snapshot = tensor_bvts[:, :, 0, :, :]

        return self._ensure_device(snapshot, ensure_cpu)

    def tensor_to_field(
        self,
        tensor: torch.Tensor,
        metadata: FieldMetadata,
        *,
        time_slice: Optional[int] = None,
    ) -> Field:
        """

        """
        if tensor.dim() == 4:
            phiml_tensor = math.tensor(tensor, batch('batch'), channel("vector"), spatial(*metadata.spatial_dims))

        elif tensor.dim() == 3:
            phiml_tensor = math.tensor(tensor, channel("vector"), spatial(*metadata.spatial_dims))

        
        ret = CenteredGrid(phiml_tensor, metadata.extrapolation, bounds=metadata.domain)
        return ret


    def _get_spatial_layout(self, tensor: torch.Tensor, metadata: FieldMetadata) -> str:
        """
        Determine spatial dimension names from tensor shape and metadata.
        
        Returns string like 'x,y' or 'x,y,z'
        """
        spatial_dims = metadata.spatial_dims
        return ','.join(spatial_dims)