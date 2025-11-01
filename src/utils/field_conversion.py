"""
Field Conversion Utilities

This module provides utilities for converting between PyTorch tensors and PhiFlow Fields.
These conversions are essential for integrating the tensor-based DataLoader pipeline
with PhiFlow's Field-based physical models.

Architecture:
-----------
The module is designed with two levels of abstraction:

1. **FieldTensorConverter (Recommended for production use)**
   - Stateful converter with pre-computed channel mappings
   - Efficient for repeated conversions with the same field structure
   - Handles channel concatenation for neural network inputs
   - Provides validation and channel info methods
   
   Usage:
       >>> # Create converter once, reuse many times
       >>> metadata = {'velocity': vel_meta, 'density': dens_meta}
       >>> converter = FieldTensorConverter(metadata)
       >>> 
       >>> # Convert multiple times efficiently
       >>> tensor1 = converter.fields_to_tensors_batch(fields1)
       >>> tensor2 = converter.fields_to_tensors_batch(fields2)

2. **Standalone functions (Convenience wrappers)**
   - Simple API for one-off conversions
   - Useful for testing and prototyping
   - Delegate to FieldTensorConverter internally
   
   Usage:
       >>> # Single field conversion
       >>> tensor = field_to_tensor(field)
       >>> field = tensor_to_field(tensor, metadata)
       >>> 
       >>> # Multiple fields (no concatenation)
       >>> tensors = fields_to_tensors(field_dict)
       >>> fields = tensors_to_fields(tensor_dict, metadata_dict)

Key Conversions:
---------------
- tensor_to_field: PyTorch tensor → PhiFlow Field
- field_to_tensor: PhiFlow Field → PyTorch tensor  
- tensors_to_fields: Dict of tensors → Dict of Fields
- fields_to_tensors: Dict of Fields → Dict of tensors
- FieldTensorConverter.fields_to_tensors_batch: Dict of Fields → Concatenated tensor [B, C, H, W]
- FieldTensorConverter.tensors_to_fields_batch: Concatenated tensor → Dict of Fields

When to use each:
----------------
- Use **FieldTensorConverter** when:
  * Converting data for neural network input/output
  * Need channel concatenation across multiple fields
  * Performing repeated conversions with same structure
  * Need validation or channel layout information

- Use **standalone functions** when:
  * Quick one-off conversions
  * Testing or prototyping
  * Converting single fields
  * Don't need channel concatenation
"""

from dataclasses import dataclass
from typing import Dict, Union, Optional, Tuple
import torch
from phi.torch.flow import *
from phi import math
from phi.math import Shape, spatial, channel, batch as batch_dim
from phi.field import Field, CenteredGrid, StaggeredGrid
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


def _tensor_to_field(
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
    
    # If we need a StaggeredGrid, use the CORRECT approach:
    # Pass the centered field to StaggeredGrid constructor which automatically
    # resamples it to face-centered sample points (as per PhiFlow documentation)
    if metadata.field_type == 'staggered':
        # CORRECT: StaggeredGrid constructor automatically resamples the Field
        # to staggered sample points (face centers)
        # Create resolution kwargs from metadata.resolution
        resolution_dict = {dim: metadata.resolution.get_size(dim) 
                          for dim in metadata.spatial_dims}
        
        staggered_grid = StaggeredGrid(
            centered_grid,  # PhiFlow will resample this to face centers
            metadata.extrapolation,
            bounds=metadata.domain,
            **resolution_dict  # e.g., x=128, y=128
        )
        return staggered_grid
    else:
        return centered_grid


def _field_to_tensor(field: Field, ensure_cpu: bool = True) -> torch.Tensor:
    """
    Convert a PhiFlow Field to a PyTorch tensor.
    
    This extracts the underlying native tensor from a Field object.
    The tensor will have shape [channels, x, y] for a single field (DataManager format),
    or [batch, channels, x, y] if the field has batch dimensions.
    
    For StaggeredGrids, this function converts to centered grid first
    to ensure a uniform tensor representation.
    
    Args:
        field: PhiFlow Field (CenteredGrid or StaggeredGrid)
        ensure_cpu: If True, moves tensor to CPU (useful for caching)
        
    Returns:
        PyTorch tensor in DataManager format:
        - [channels, x, y] for single field
        - [batch, channels, x, y] for batched field
        
    Note:
        This is compatible with the DataManager's tensor extraction approach,
        which converts staggered grids to centered grids before caching.
        The returned tensor is permuted to match DataManager's format.
    """
    # Convert staggered grids to centered grids (like DataManager does)
    if field.is_staggered:
        field = field.at_centers()
    
    tensor = field.values._native
    
    # Determine field properties
    has_batch = field.shape.batch.rank > 0
    is_vector = field.shape.channel.rank > 0
    spatial_rank = field.spatial_rank
    
    # PhiFlow native layout depends on dimension types:
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
        # Need to add channel dimension
        tensor = tensor.unsqueeze(1)
    elif has_batch and is_vector:
        # Vector field with batch: [batch, x, y, vector] -> [batch, vector, x, y]
        # Permute: move vector (last) to position 1
        perm = [0, len(tensor.shape)-1] + list(range(1, len(tensor.shape)-1))
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
    
    This is a convenience function that creates a temporary FieldTensorConverter
    and delegates to it. For repeated conversions, create a FieldTensorConverter
    instance directly for better performance.
    
    Args:
        tensor_dict: Dictionary mapping field names to tensors
        metadata_dict: Dictionary mapping field names to FieldMetadata
        time_slice: Optional timestep to extract from temporal data
        
    Returns:
        Dictionary mapping field names to Field objects
        
    Raises:
        ValueError: If keys don't match between dicts
    
    Example:
        >>> # For single-use conversion (simple but less efficient)
        >>> fields = tensors_to_fields(tensor_dict, metadata_dict)
        >>> 
        >>> # For repeated conversions (recommended)
        >>> converter = FieldTensorConverter(metadata_dict)
        >>> fields1 = converter.tensors_to_fields_batch(tensor1)
        >>> fields2 = converter.tensors_to_fields_batch(tensor2)
    """
    if set(tensor_dict.keys()) != set(metadata_dict.keys()):
        raise ValueError(
            f"Mismatched keys: tensors have {set(tensor_dict.keys())}, "
            f"metadata has {set(metadata_dict.keys())}"
        )
    
    field_dict = {}
    for field_name, tensor in tensor_dict.items():
        metadata = metadata_dict[field_name]
        field_dict[field_name] = _tensor_to_field(tensor, metadata, time_slice)
    
    return field_dict


def fields_to_tensors(
    field_dict: Dict[str, Field],
    ensure_cpu: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Convert a dictionary of Fields to a dictionary of tensors.
    
    This is a convenience function for simple conversions. For conversions
    that require channel concatenation or repeated use, consider using
    FieldTensorConverter directly.
    
    Args:
        field_dict: Dictionary mapping field names to Fields
        ensure_cpu: If True, moves tensors to CPU
        
    Returns:
        Dictionary mapping field names to tensors
    
    Example:
        >>> # For simple conversion (independent tensors)
        >>> tensors = fields_to_tensors(field_dict)
        >>> 
        >>> # For concatenated tensors (e.g., UNet input)
        >>> converter = FieldTensorConverter(metadata_dict)
        >>> concatenated_tensor = converter.fields_to_tensors_batch(field_dict)
    """
    return {
        name: _field_to_tensor(field, ensure_cpu)
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


class FieldTensorConverter:
    """
    Bidirectional converter between PhiFlow Fields and PyTorch tensors.
    
    This is the bridge between physical (Field-based) and synthetic (tensor-based) models.
    It handles batched conversion of multiple fields to/from concatenated tensors suitable
    for neural network input/output.
    
    Key Features:
    - Batch conversion: Multiple fields → Single concatenated tensor
    - Preserves field order for reconstruction
    - Handles both scalar and vector fields
    - Supports time-series data with batch dimensions
    
    Static Methods (for single fields):
        - field_to_tensor(): Convert a single Field to a tensor
        - tensor_to_field(): Convert a tensor to a Field using metadata
    
    Instance Methods (for batches):
        - fields_to_tensors_batch(): Concatenate multiple Fields into single tensor
        - tensors_to_fields_batch(): Split concatenated tensor back to Fields
    
    Example Usage:
        >>> # Single field conversion (static)
        >>> tensor = FieldTensorConverter.field_to_tensor(field)
        >>> field = FieldTensorConverter.tensor_to_field(tensor, metadata)
        >>> 
        >>> # Batch conversion (instance)
        >>> converter = FieldTensorConverter(field_metadata_dict)
        >>> 
        >>> # Physical model output (Fields) → Synthetic model input (Tensor)
        >>> fields = {'velocity': velocity_field, 'density': density_field}
        >>> tensor = converter.fields_to_tensors_batch(fields)  # Shape: [B, C, H, W]
        >>> 
        >>> # Synthetic model output (Tensor) → Physical model input (Fields)
        >>> pred_tensor = synthetic_model(tensor)  # Shape: [B, C, H, W]
        >>> pred_fields = converter.tensors_to_fields_batch(pred_tensor)
    """
    
    def __init__(self, field_metadata: Dict[str, FieldMetadata]):
        """
        Initialize converter with field metadata.
        
        Args:
            field_metadata: Dictionary mapping field names to FieldMetadata objects.
                           The order of fields in this dict determines the order of
                           channels in concatenated tensors.
        """
        self.field_metadata = field_metadata
        self.field_names = list(field_metadata.keys())
        
        # Pre-compute channel counts and offsets for efficient slicing
        self.channel_counts = {}
        self.channel_offsets = {}
        offset = 0
        
        for name in self.field_names:
            metadata = field_metadata[name]
            # Determine number of channels for this field
            if metadata.channel_dims:
                # Vector field: number of channels = number of spatial dimensions
                num_channels = len(metadata.spatial_dims)
            else:
                # Scalar field: 1 channel
                num_channels = 1
            
            self.channel_counts[name] = num_channels
            self.channel_offsets[name] = offset
            offset += num_channels
        
        self.total_channels = offset
    
    @staticmethod
    def field_to_tensor(field: CenteredGrid) -> torch.Tensor:
        """
        Convert a single PhiFlow Field to a PyTorch tensor (static method).
        
        Args:
            field: PhiFlow CenteredGrid to convert
            
        Returns:
            PyTorch tensor with shape [B,C,H,W] or [C,H,W]
        """
        return _field_to_tensor(field)
    
    @staticmethod
    def tensor_to_field(tensor: torch.Tensor, metadata: FieldMetadata, time_slice: int = 0) -> CenteredGrid:
        """
        Convert a PyTorch tensor to a PhiFlow Field using metadata (static method).
        
        Args:
            tensor: PyTorch tensor to convert
            metadata: FieldMetadata containing reconstruction information
            time_slice: Time index for batch tensors (default: 0)
            
        Returns:
            Reconstructed PhiFlow CenteredGrid
        """
        return _tensor_to_field(tensor, metadata, time_slice)
    
    def fields_to_tensors_batch(
        self,
        fields: Dict[str, Field],
        ensure_cpu: bool = False
    ) -> torch.Tensor:
        """
        Convert dict of Fields to concatenated tensor for synthetic model.
        
        This method is used to convert physical model predictions (Fields) into
        a format suitable for synthetic model training or input.
        
        Args:
            fields: Dictionary mapping field names to Field objects.
                   Each Field may have shape [B, x, y, channels] or [x, y, channels]
            ensure_cpu: If True, ensures output tensor is on CPU
            
        Returns:
            Tensor of shape [B, C, H, W] where:
            - B: batch size (if present in fields)
            - C: sum of channels across all fields
            - H, W: spatial dimensions
            
        Raises:
            ValueError: If field names don't match metadata or shapes are incompatible
            
        Example:
            >>> fields = {
            ...     'velocity': velocity_field,  # [B, x, y, 2] or [x, y, 2]
            ...     'density': density_field     # [B, x, y] or [x, y]
            ... }
            >>> tensor = converter.fields_to_tensors_batch(fields)  # [B, 3, H, W]
        """
        if set(fields.keys()) != set(self.field_names):
            raise ValueError(
                f"Field names mismatch. Expected {self.field_names}, "
                f"got {list(fields.keys())}"
            )
        
        # Convert each field to tensor
        tensors = []
        for name in self.field_names:  # Maintain consistent order
            field = fields[name]
            tensor = _field_to_tensor(field, ensure_cpu=ensure_cpu)
            
            # Handle scalar fields (may need to add channel dimension)
            # Tensor shape should be [B, C, H, W] or [C, H, W]
            if tensor.dim() == 2:  # [H, W] - scalar without batch
                tensor = tensor.unsqueeze(0)  # [1, H, W]
            elif tensor.dim() == 3:  # Could be [B, H, W] or [C, H, W]
                # If it's a scalar field (channel count = 1), it's [B, H, W]
                if self.channel_counts[name] == 1 and tensor.shape[0] > 1:
                    # It's likely [B, H, W], add channel dim
                    tensor = tensor.unsqueeze(1)  # [B, 1, H, W]
                elif tensor.shape[0] != self.channel_counts[name]:
                    # It's [C, H, W], verify channel count matches
                    if tensor.shape[0] == self.channel_counts[name]:
                        pass  # Already correct
                    else:
                        # Must be [B, H, W] with wrong interpretation
                        tensor = tensor.unsqueeze(1)  # [B, 1, H, W]
            # If it's 4D [B, C, H, W], it's already in correct format
            
            tensors.append(tensor)
        
        # Concatenate along channel dimension
        # All tensors should now be [B, C, H, W] or [C, H, W]
        concatenated = torch.cat(tensors, dim=-3)  # Concatenate on channel dim
        
        return concatenated
    
    def tensors_to_fields_batch(
        self,
        tensor: torch.Tensor,
        time_slice: Optional[int] = None
    ) -> Dict[str, Field]:
        """
        Convert concatenated tensor back to dict of Fields.
        
        This method is used to convert synthetic model predictions (concatenated tensor)
        back into individual Fields for physical model use or evaluation.
        
        Args:
            tensor: Tensor of shape [B, C, H, W] or [C, H, W] from UNet output
                   where C is the sum of all field channels
            time_slice: Optional timestep to extract for temporal data
            
        Returns:
            Dict mapping field names to Field objects
            
        Raises:
            ValueError: If tensor channel dimension doesn't match expected total
            
        Example:
            >>> pred_tensor = synthetic_model(input_tensor)  # [B, 3, 128, 128]
            >>> pred_fields = converter.tensors_to_fields_batch(pred_tensor)
            >>> # pred_fields = {'velocity': Field[B,x,y,2], 'density': Field[B,x,y]}
        """
        # Verify channel dimension
        channel_dim = -3  # Channel dimension in [B, C, H, W]
        if tensor.shape[channel_dim] != self.total_channels:
            raise ValueError(
                f"Expected {self.total_channels} channels in tensor, "
                f"got {tensor.shape[channel_dim]}. Tensor shape: {tensor.shape}"
            )
        
        fields = {}
        for name in self.field_names:
            # Extract channels for this field
            start_idx = self.channel_offsets[name]
            end_idx = start_idx + self.channel_counts[name]
            field_tensor = tensor[:, start_idx:end_idx, :, :] if tensor.dim() == 4 else tensor[start_idx:end_idx, :, :]
            
            # Convert to Field
            metadata = self.field_metadata[name]
            field = _tensor_to_field(field_tensor, metadata, time_slice=time_slice)
            fields[name] = field
        
        return fields
    
    def get_channel_info(self) -> Dict[str, Dict[str, int]]:
        """
        Get information about channel layout in concatenated tensors.
        
        Returns:
            Dictionary mapping field names to {'count': int, 'offset': int}
            
        Example:
            >>> info = converter.get_channel_info()
            >>> # {'velocity': {'count': 2, 'offset': 0},
            >>> #  'density': {'count': 1, 'offset': 2}}
        """
        return {
            name: {
                'count': self.channel_counts[name],
                'offset': self.channel_offsets[name]
            }
            for name in self.field_names
        }
    
    def validate_fields(self, fields: Dict[str, Field]) -> bool:
        """
        Validate that fields dict is compatible with this converter.
        
        Args:
            fields: Dictionary of fields to validate
            
        Returns:
            True if valid, raises ValueError otherwise
            
        Raises:
            ValueError: If validation fails with descriptive message
        """
        # Check field names match
        if set(fields.keys()) != set(self.field_names):
            raise ValueError(
                f"Field names mismatch. Expected {self.field_names}, "
                f"got {list(fields.keys())}"
            )
        
        # Check each field's metadata is compatible
        for name, field in fields.items():
            metadata = self.field_metadata[name]
            
            # Check spatial dimensions
            field_spatial_dims = tuple(field.shape.spatial.names)
            if field_spatial_dims != metadata.spatial_dims:
                raise ValueError(
                    f"Field '{name}' has spatial dims {field_spatial_dims}, "
                    f"expected {metadata.spatial_dims}"
                )
            
            # Check field type
            field_type = 'staggered' if field.is_staggered else 'centered'
            if field_type != metadata.field_type:
                raise ValueError(
                    f"Field '{name}' is {field_type}, expected {metadata.field_type}"
                )
        
        return True
    
    def validate_tensor(self, tensor: torch.Tensor) -> bool:
        """
        Validate that tensor is compatible with this converter.
        
        Args:
            tensor: Tensor to validate (should be [B, C, H, W] or [C, H, W])
            
        Returns:
            True if valid, raises ValueError otherwise
            
        Raises:
            ValueError: If validation fails with descriptive message
        """
        # Check dimensions
        if tensor.dim() not in [3, 4]:
            raise ValueError(
                f"Expected tensor with 3 or 4 dimensions, got {tensor.dim()}"
            )
        
        # Check channel count
        channel_dim = -3
        if tensor.shape[channel_dim] != self.total_channels:
            raise ValueError(
                f"Expected {self.total_channels} channels, "
                f"got {tensor.shape[channel_dim]}"
            )
        
        return True
