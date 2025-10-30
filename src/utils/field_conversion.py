"""
Field Conversion Utilities

This module provides utilities for converting between PyTorch tensors and PhiFlow Fields.
These conversions are essential for integrating the tensor-based DataLoader pipeline
with PhiFlow's Field-based physical models.

Key conversions:
- tensor_to_field: Convert PyTorch tensor to PhiFlow Field (CenteredGrid or StaggeredGrid)
- field_to_tensor: Convert PhiFlow Field to PyTorch tensor (already exists in DataManager)
- FieldMetadata: Dataclass to store Field reconstruction information
"""

from dataclasses import dataclass
from typing import Dict, Union, Optional, Tuple
import torch
from phi.torch.flow import *
from phi import math
from phi.math import Shape, spatial, channel, batch as batch_dim
from phi.field import Field, CenteredGrid, StaggeredGrid, stagger
from phi.geom import Box
from phi.field._field_math import Extrapolation
from phi import field as field_module


@dataclass
class FieldMetadata:
    """
    Metadata needed to reconstruct a PhiFlow Field from a tensor.
    
    This stores all the information required to convert a PyTorch tensor
    back into its original Field representation.
    
    Attributes:
        domain: The physical domain (Box) for the field
        resolution: The spatial resolution (Shape with x, y dimensions)
        extrapolation: Boundary condition (e.g., 'periodic', 'zero-gradient', etc.)
        field_type: 'centered' or 'staggered'
        spatial_dims: Names of spatial dimensions (e.g., ['x', 'y'])
        channel_dims: Names of channel dimensions (e.g., ['vector'])
    """
    domain: Box
    resolution: Shape
    extrapolation: Union[Extrapolation, str]
    field_type: str  # 'centered' or 'staggered'
    spatial_dims: Tuple[str, ...]
    channel_dims: Tuple[str, ...]
    
    @classmethod
    def from_field(cls, field: Field) -> 'FieldMetadata':
        """
        Extract metadata from an existing Field.
        
        Args:
            field: The Field to extract metadata from
            
        Returns:
            FieldMetadata object
        """
        # Determine field type using is_staggered property
        field_type = 'staggered' if field.is_staggered else 'centered'
        
        return cls(
            domain=field.bounds,
            resolution=field.resolution,
            extrapolation=field.extrapolation,
            field_type=field_type,
            spatial_dims=tuple(field.shape.spatial.names),
            channel_dims=tuple(field.shape.channel.names) if field.shape.channel else ()
        )
    
    @classmethod
    def from_cache_metadata(cls, cached_meta: Dict, domain: Box, resolution: Shape) -> 'FieldMetadata':
        """
        Reconstruct FieldMetadata from cached metadata dictionary.
        
        Args:
            cached_meta: Dictionary containing field metadata from cache
            domain: The physical domain (must be provided externally)
            resolution: The spatial resolution (must be provided externally)
            
        Returns:
            FieldMetadata object
        """
        # Parse extrapolation from string
        extrap_str = cached_meta.get('extrapolation', 'ZERO')
        
        # Map common extrapolation strings to PhiFlow objects
        from phi.math import extrapolation as extrap_module
        
        extrapolation_map = {
            'ZERO': extrap_module.ZERO,
            'BOUNDARY': extrap_module.BOUNDARY,
            'PERIODIC': extrap_module.PERIODIC,
            'zero-gradient': extrap_module.ZERO_GRADIENT,
            'ZERO_GRADIENT': extrap_module.ZERO_GRADIENT,
        }
        
        # Try to parse the extrapolation
        if extrap_str in extrapolation_map:
            extrapolation = extrapolation_map[extrap_str]
        else:
            # Try to extract the extrapolation name from a string like "<ZERO>"
            for key in extrapolation_map:
                if key in extrap_str.upper():
                    extrapolation = extrapolation_map[key]
                    break
            else:
                extrapolation = extrap_module.ZERO  # Default fallback
        
        # Determine field type (default to centered if not specified)
        field_type = cached_meta.get('field_type', 'centered')
        
        return cls(
            domain=domain,
            resolution=resolution,
            extrapolation=extrapolation,
            field_type=field_type,
            spatial_dims=tuple(cached_meta.get('spatial_dims', ['x', 'y'])),
            channel_dims=tuple(cached_meta.get('channel_dims', []))
        )


def tensor_to_field(
    tensor: torch.Tensor,
    metadata: FieldMetadata,
    time_slice: Optional[int] = None
) -> Field:
    """
    Convert a PyTorch tensor to a PhiFlow Field.
    
    This function reconstructs a Field object from a tensor that was previously
    extracted using DataManager. The tensor is expected to have shape:
    - [time, channels, x, y] if time_slice is None (returns stacked field)
    - When time_slice is provided, uses that timestep
    
    Args:
        tensor: PyTorch tensor with shape [time, channels, x, y] or [channels, x, y]
        metadata: FieldMetadata containing domain, resolution, extrapolation, etc.
        time_slice: Optional integer to select a specific timestep. If None, 
                   returns a field with batch(time) dimension.
        
    Returns:
        Field object (CenteredGrid or StaggeredGrid)
        
    Raises:
        ValueError: If tensor dimensions don't match expected format
    """
    # Handle time dimension
    if time_slice is not None:
        if len(tensor.shape) == 4:  # [time, channels, x, y]
            tensor = tensor[time_slice]  # -> [channels, x, y]
        elif len(tensor.shape) != 3:
            raise ValueError(
                f"Expected tensor with 3 or 4 dimensions, got shape {tensor.shape}"
            )
    
    # PhiML/PhiFlow expects tensors in format: [batch_dims..., spatial_dims..., channel_dims...]
    # Our tensor is [time/batch, channels, x, y] or [channels, x, y]
    # We need to permute to: [time/batch, x, y, channels] or [x, y, channels]
    
    if len(tensor.shape) == 4:  # [time, channels, x, y]
        tensor = tensor.permute(0, 2, 3, 1)  # -> [time, x, y, channels]
    elif len(tensor.shape) == 3:  # [channels, x, y]
        tensor = tensor.permute(1, 2, 0)  # -> [x, y, channels]
    else:
        raise ValueError(f"Expected tensor with 3 or 4 dimensions, got shape {tensor.shape}")
    
    # For scalar fields (1 channel), remove the channel dimension
    if tensor.shape[-1] == 1 and not metadata.channel_dims:
        tensor = tensor.squeeze(-1)
    
    # Determine the actual resolution from the tensor shape
    # For staggered fields converted to centered, the resolution may be different
    tensor_spatial_shape = tensor.shape[-len(metadata.spatial_dims)-1:-1] if metadata.channel_dims else tensor.shape[-len(metadata.spatial_dims):]
    if len(tensor.shape) == 4:  # Has batch dimension
        tensor_spatial_shape = tensor.shape[1:1+len(metadata.spatial_dims)]
    elif len(tensor.shape) == 3 and metadata.channel_dims:  # [x, y, channels]
        tensor_spatial_shape = tensor.shape[:len(metadata.spatial_dims)]
    elif len(tensor.shape) == 2:  # [x, y] scalar
        tensor_spatial_shape = tensor.shape
    
    # Create spatial shape with actual tensor dimensions
    spatial_sizes = {dim: size for dim, size in zip(metadata.spatial_dims, tensor_spatial_shape)}
    actual_resolution = spatial(**spatial_sizes)
    
    # Convert PyTorch tensor to PhiML math.Tensor with proper dimension names
    # Determine the shape based on tensor dimensions and metadata
    if len(tensor.shape) == 2:  # [x, y] - scalar field
        # Create math.Tensor with spatial dimensions
        phiml_tensor = math.tensor(tensor, actual_resolution)
    elif len(tensor.shape) == 3:
        if tensor.shape[0] > 1 and 'time' not in str(actual_resolution):
            # [time, x, y] or [x, y, channels]
            # Check if first dim could be time (from unsqueezed batch)
            # If channels expected, it's [x, y, channels]
            if metadata.channel_dims:
                # [x, y, channels]
                # For vector fields, add labels matching spatial dimensions
                if 'vector' in metadata.channel_dims and len(metadata.channel_dims) == 1:
                    # Create vector channel with labels matching spatial dims
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
            else:
                # [batch/time, x, y] - add batch dimension
                phiml_tensor = math.tensor(tensor, batch('time') & actual_resolution)
        else:
            # [x, y, channels]
            if metadata.channel_dims:
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
            else:
                # Likely [batch, x, y]
                phiml_tensor = math.tensor(tensor, batch('time') & actual_resolution)
    elif len(tensor.shape) == 4:  # [time, x, y, channels]
        if metadata.channel_dims:
            # For vector fields, add labels matching spatial dimensions
            if 'vector' in metadata.channel_dims and len(metadata.channel_dims) == 1:
                vector_labels = ','.join(metadata.spatial_dims)
                phiml_tensor = math.tensor(
                    tensor,
                    batch('time') & actual_resolution & channel(vector=vector_labels)
                )
            else:
                phiml_tensor = math.tensor(
                    tensor,
                    batch('time') & actual_resolution & channel(*metadata.channel_dims)
                )
        else:
            # Scalar field with time dimension
            phiml_tensor = math.tensor(tensor, batch('time') & actual_resolution)
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
    
    # Create CenteredGrid from the PhiML tensor
    centered_grid = CenteredGrid(
        phiml_tensor,
        metadata.extrapolation,
        bounds=metadata.domain
    )
    
    # If we need a StaggeredGrid, use the stagger utility to convert
    if metadata.field_type == 'staggered':
        # Use stagger to convert centered grid to staggered grid
        # The face_function takes (lower, upper) values and returns face value
        # For conversion, we average the neighboring cell values
        staggered_grid = stagger(
            centered_grid,
            face_function=lambda lower, upper: (lower + upper) / 2,
            boundary=metadata.extrapolation,
            at='face'
        )
        return staggered_grid
    else:
        return centered_grid


def field_to_tensor(field: Field, ensure_cpu: bool = True) -> torch.Tensor:
    """
    Convert a PhiFlow Field to a PyTorch tensor.
    
    This extracts the underlying native tensor from a Field object.
    The tensor will have shape [channels, x, y] for a single field (DataManager format),
    or may have additional batch/time dimensions.
    
    For StaggeredGrids, this function converts to centered grid first
    to ensure a uniform tensor representation.
    
    Args:
        field: PhiFlow Field (CenteredGrid or StaggeredGrid)
        ensure_cpu: If True, moves tensor to CPU (useful for caching)
        
    Returns:
        PyTorch tensor in DataManager format [channels, x, y]
        
    Note:
        This is compatible with the DataManager's tensor extraction approach,
        which converts staggered grids to centered grids before caching.
        The returned tensor is permuted to match DataManager's [channels, x, y] format.
    """
    # Convert staggered grids to centered grids (like DataManager does)
    if field.is_staggered:
        field = field.at_centers()
    
    tensor = field.values._native
    
    # PhiFlow native tensors are in format [batch..., x, y, channels]
    # We need to permute to DataManager format [batch..., channels, x, y]
    # Determine number of spatial dims and channel dims
    ndims = len(tensor.shape)
    spatial_rank = field.spatial_rank
    
    if ndims == spatial_rank:  # [x, y] - scalar field, no change needed
        pass
    elif ndims == spatial_rank + 1:  # [x, y, channels] or [batch, x, y]
        # Check if last dim is channels by checking if it's small
        if field.shape.channel.volume > 0:
            # It's [x, y, channels], permute to [channels, x, y]
            perm = list(range(ndims))
            perm = [ndims-1] + perm[:-1]  # Move last to first
            tensor = tensor.permute(*perm)
    elif ndims == spatial_rank + 2:  # [batch, x, y, channels]
        # Permute to [batch, channels, x, y]
        perm = [0, ndims-1] + list(range(1, ndims-1))
        tensor = tensor.permute(*perm)
    
    if ensure_cpu and tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    
    return tensor


def tensors_to_fields(
    tensor_dict: Dict[str, torch.Tensor],
    metadata_dict: Dict[str, FieldMetadata],
    time_slice: Optional[int] = None
) -> Dict[str, Field]:
    """
    Convert a dictionary of tensors to a dictionary of Fields.
    
    This is a convenience function for batch conversion, useful when
    working with multi-field data from the DataLoader.
    
    Args:
        tensor_dict: Dictionary mapping field names to tensors
        metadata_dict: Dictionary mapping field names to FieldMetadata
        time_slice: Optional timestep to extract from temporal data
        
    Returns:
        Dictionary mapping field names to Field objects
        
    Raises:
        ValueError: If keys don't match between dicts
    """
    if set(tensor_dict.keys()) != set(metadata_dict.keys()):
        raise ValueError(
            f"Mismatched keys: tensors have {set(tensor_dict.keys())}, "
            f"metadata has {set(metadata_dict.keys())}"
        )
    
    field_dict = {}
    for field_name, tensor in tensor_dict.items():
        metadata = metadata_dict[field_name]
        field_dict[field_name] = tensor_to_field(tensor, metadata, time_slice)
    
    return field_dict


def fields_to_tensors(
    field_dict: Dict[str, Field],
    ensure_cpu: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Convert a dictionary of Fields to a dictionary of tensors.
    
    Args:
        field_dict: Dictionary mapping field names to Fields
        ensure_cpu: If True, moves tensors to CPU
        
    Returns:
        Dictionary mapping field names to tensors
    """
    return {
        name: field_to_tensor(field, ensure_cpu)
        for name, field in field_dict.items()
    }


def create_field_metadata_from_model(
    model,
    field_names: list[str],
    field_types: Optional[Dict[str, str]] = None
) -> Dict[str, FieldMetadata]:
    """
    Create FieldMetadata for each field from a PhysicalModel instance.
    
    This is useful for physical trainers that need to convert tensors
    back to Fields for use with the model.
    
    Args:
        model: PhysicalModel instance with domain, resolution attributes
        field_names: List of field names (e.g., ['velocity', 'density'])
        field_types: Optional dict mapping field names to types ('centered' or 'staggered')
                    Defaults to 'centered' for all fields
        
    Returns:
        Dictionary mapping field names to FieldMetadata
        
    Example:
        >>> model = BurgersModel(domain=Box(...), resolution=spatial(x=128, y=128), ...)
        >>> metadata = create_field_metadata_from_model(model, ['velocity'], {'velocity': 'staggered'})
    """
    field_types = field_types or {}
    
    metadata_dict = {}
    for field_name in field_names:
        field_type = field_types.get(field_name, 'centered')
        
        # Determine channel dimensions based on common field types
        if 'velocity' in field_name.lower():
            channel_dims = ('vector',)  # Velocity is typically a vector field
        else:
            channel_dims = ()  # Scalar field
        
        metadata_dict[field_name] = FieldMetadata(
            domain=model.domain,
            resolution=model.resolution,
            extrapolation=extrapolation.PERIODIC,  # Default, may need to be configurable
            field_type=field_type,
            spatial_dims=tuple(model.resolution.names),
            channel_dims=channel_dims
        )
    
    return metadata_dict
