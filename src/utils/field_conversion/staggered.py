"""
Simplified Staggered Grid Converter

Key insight: Staggered grids convert through centered intermediate.
The centered converter handles all the complex logic, so this just
needs to handle the staggered↔centered conversion using PhiFlow's
built-in methods.
"""

from typing import Optional
import torch
from phi.field import Field, StaggeredGrid, CenteredGrid
from .base import SingleFieldConverter
from .metadata import FieldMetadata
from .centered import CenteredConverter


class StaggeredConverter(SingleFieldConverter):
    """
    Simplified converter for StaggeredGrid fields.
    
    Strategy:
    - field_to_tensor: staggered.at_centers() → centered_converter
    - tensor_to_field: centered_converter → StaggeredGrid()
    
    All complexity is delegated to CenteredConverter and PhiFlow's
    built-in staggered↔centered conversion.
    """

    def __init__(self, metadata: Optional[FieldMetadata] = None):
        """Initialize with metadata and centered converter."""
        super().__init__(metadata)
        self._centered_converter = CenteredConverter(metadata)

    @classmethod
    def can_handle(cls, metadata: FieldMetadata) -> bool:
        """Check if this converter can handle the field type."""
        return metadata.field_type == "staggered"

    def field_to_tensor(self, field: Field, *, ensure_cpu: bool = True) -> torch.Tensor:
        """
        Convert StaggeredGrid to tensor via centered intermediate.
        
        PhiFlow's at_centers() handles the resampling automatically.
        
        Args:
            field: PhiFlow StaggeredGrid
            ensure_cpu: If True, ensures output is on CPU
            
        Returns:
            Tensor with same format as CenteredConverter
        """
        # Convert to centered (PhiFlow resamples automatically)
        centered = field.at_centers()
        
        # Use centered converter for tensor conversion
        return self._centered_converter.field_to_tensor(centered, ensure_cpu=ensure_cpu)

    def tensor_to_field(
        self,
        tensor: torch.Tensor,
        metadata: FieldMetadata,
        *,
        time_slice: Optional[int] = None
    ) -> Field:
        """
        Convert tensor to StaggeredGrid via centered intermediate.
        
        PhiFlow's StaggeredGrid constructor handles the resampling automatically.
        
        Args:
            tensor: Tensor from field_to_tensor
            metadata: Field reconstruction metadata
            time_slice: Optional time index (unused)
            
        Returns:
            Reconstructed StaggeredGrid
        """
        # First convert to centered grid
        centered = self._centered_converter.tensor_to_field(
            tensor, metadata, time_slice=time_slice
        )
        
        # Convert centered to staggered (PhiFlow resamples automatically)
        # Extract resolution from metadata
        resolution_dict = {
            dim: metadata.resolution.get_size(dim) 
            for dim in metadata.spatial_dims
        }
        
        # StaggeredGrid constructor automatically resamples to face centers
        return StaggeredGrid(
            centered,
            metadata.extrapolation,
            bounds=metadata.domain,
            **resolution_dict
        )