"""
Staggered Grid Converter

This module implements conversion between StaggeredGrid fields and tensors.
"""

from typing import Optional
import torch
from phi.field import Field, StaggeredGrid
from .base import SingleFieldConverter
from .metadata import FieldMetadata
from .centered import CenteredConverter


class StaggeredConverter(SingleFieldConverter):
    """
    Converter for StaggeredGrid fields.

    Handles conversion between PhiFlow StaggeredGrid objects and PyTorch tensors.
    StaggeredGrids have values sampled at face centers, so conversion requires
    resampling through a centered grid intermediate.

    Strategy:
    - field_to_tensor: Convert staggered -> centered -> tensor
    - tensor_to_field: tensor -> centered -> staggered
    """

    def __init__(self, metadata: Optional[FieldMetadata] = None):
        """
        Initialize staggered converter.

        Args:
            metadata: Optional FieldMetadata for validation/reconstruction
        """
        super().__init__(metadata)
        # Use a centered converter for the actual tensor conversion
        self._centered_converter = CenteredConverter(metadata)

    @classmethod
    def can_handle(cls, metadata: FieldMetadata) -> bool:
        """Check if this converter can handle the field type."""
        return metadata.field_type == "staggered"

    def field_to_tensor(self, field: Field, *, ensure_cpu: bool = True) -> torch.Tensor:
        """
        Convert a StaggeredGrid to a PyTorch tensor.

        The staggered grid is first converted to a centered grid (resampling
        face-centered values to cell centers), then converted to a tensor.

        Args:
            field: PhiFlow StaggeredGrid to convert
            ensure_cpu: If True, ensures output tensor is on CPU

        Returns:
            PyTorch tensor with shape [C, H, W] or [B, C, H, W]
        """
        self.validate_field(field)

        # Convert staggered to centered (PhiFlow handles resampling)
        centered_field = field.at_centers()

        # Use centered converter for tensor conversion
        return self._centered_converter.field_to_tensor(
            centered_field, ensure_cpu=ensure_cpu
        )

    def tensor_to_field(
        self,
        tensor: torch.Tensor,
        metadata: FieldMetadata,
        *,
        time_slice: Optional[int] = None
    ) -> Field:
        """
        Convert a PyTorch tensor to a StaggeredGrid.

        The tensor is first converted to a centered grid, then PhiFlow
        automatically resamples it to staggered (face-centered) sample points.

        Args:
            tensor: PyTorch tensor to convert
            metadata: FieldMetadata containing reconstruction information
            time_slice: Optional time index for batch tensors

        Returns:
            Reconstructed StaggeredGrid
        """
        self.validate_tensor(tensor)

        # First convert to centered grid
        centered_grid = self._centered_converter.tensor_to_field(
            tensor, metadata, time_slice=time_slice
        )

        # Convert centered to staggered
        # PhiFlow's StaggeredGrid constructor resamples the centered field
        # to face-centered sample points
        resolution_dict = {
            dim: metadata.resolution.get_size(dim) for dim in metadata.spatial_dims
        }

        staggered_grid = StaggeredGrid(
            centered_grid,  # PhiFlow will resample this to face centers
            metadata.extrapolation,
            bounds=metadata.domain,
            **resolution_dict  # e.g., x=128, y=128
        )

        return staggered_grid
