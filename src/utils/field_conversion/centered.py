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
    # Normalize accepted input shapes to a values tensor ready for PhiFlow
    # Strict: accept only batched snapshots [B, V, *spatial] or BVTS
    # with a single timestep [B, V, 1, *spatial]. Reject 3D single-
    # snapshot tensors to avoid implicit coercions.
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor_to_field expects a torch.Tensor")

        if tensor.dim() == 5:
            # BVTS provided
            if time_slice is not None:
                tensor = tensor[:, :, time_slice:time_slice+1, :, :]
            B, V, T, *_spatial = tensor.shape
            if T != 1:
                raise ValueError("tensor_to_field received multi-timestep BVTS tensor; please provide single-timestep tensors")
            values_batch = tensor.squeeze(2)  # [B, V, *spatial]

        elif tensor.dim() == 4:
            # [B, V, *spatial]
            values_batch = tensor
        else:
            raise ValueError(
                f"tensor_to_field requires a batched snapshot [B,V,*spatial] or BVTS [B,V,1,*spatial]; got {tensor.dim()}D. "
                f"If you have a single [V,*spatial] snapshot, wrap it with an explicit batch dim first (tensor.unsqueeze(0))."
            )

        B = values_batch.shape[0]
        V = values_batch.shape[1]

        # Prepare PhiML shape spec
        spatial_dims = metadata.spatial_dims
        num_spatial = len(spatial_dims)

        # If singleton batch, remove batch from values to create CenteredGrid
        if B == 1:
            values = values_batch.squeeze(0)  # [V, *spatial]
        else:
            values = values_batch

        shape_spec = []
        if B > 1:
            shape_spec.append(batch_dim('time'))

        if V > 1:
            vector_labels = ','.join(spatial_dims)
            shape_spec.append(channel(vector=vector_labels))
        else:
            shape_spec.append(channel(vector='scalar'))

        spatial_sizes = {dim: values.shape[-num_spatial + i] for i, dim in enumerate(spatial_dims)}
        shape_spec.append(spatial(**spatial_sizes))

        from functools import reduce
        import operator
        combined_shape = reduce(operator.and_, shape_spec)
        phiml_tensor = math.tensor(values, combined_shape)

        if V == 1:
            phiml_tensor = phiml_tensor.vector['scalar']

        return CenteredGrid(phiml_tensor, metadata.extrapolation, bounds=metadata.domain)


    def _get_spatial_layout(self, tensor: torch.Tensor, metadata: FieldMetadata) -> str:
        """
        Determine spatial dimension names from tensor shape and metadata.
        
        Returns string like 'x,y' or 'x,y,z'
        """
        spatial_dims = metadata.spatial_dims
        return ','.join(spatial_dims)