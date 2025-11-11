"""
Batch Concatenation Converter

This module implements the BatchConcatenationConverter that handles
conversion of multiple fields to/from concatenated tensors suitable
for neural network input/output.
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
    Converter that handles multiple fields with channel concatenation.

    This is the primary converter used for training pipelines where multiple
    fields (e.g., velocity + density) need to be concatenated into a single
    tensor for neural network processing.

    Features:
    - Pre-computes channel offsets for efficient slicing
    - Maintains field order consistency
    - Composes single-field converters based on metadata
    - Validates field compatibility

    Example:
        >>> metadata = {
        ...     'velocity': FieldMetadata(..., field_type='staggered'),
        ...     'density': FieldMetadata(..., field_type='centered')
        ... }
        >>> converter = BatchConcatenationConverter(metadata)
        >>>
        >>> # Convert fields to concatenated tensor
        >>> tensor = converter.fields_to_tensor_batch(fields)  # [B, 3, H, W]
        >>>
        >>> # Convert back to fields
        >>> fields = converter.tensor_to_fields_batch(tensor)
    """

    def __init__(self, field_metadata: Dict[str, FieldMetadata]):
        """
        Initialize batch converter with field metadata.

        Args:
            field_metadata: Dictionary mapping field names to FieldMetadata.
                           Field order in this dict determines channel order.
        """
        self.field_metadata = field_metadata
        self.field_names = list(field_metadata.keys())

        # Create individual converters for each field
        self._converters: Dict[str, AbstractConverter] = {}
        for name, metadata in field_metadata.items():
            if metadata.field_type == "centered":
                self._converters[name] = CenteredConverter(metadata)
            elif metadata.field_type == "staggered":
                self._converters[name] = StaggeredConverter(metadata)
            else:
                raise ValueError(f"Unknown field type: {metadata.field_type}")

        # Pre-compute channel counts and offsets
        self.channel_counts = {}
        self.channel_offsets = {}
        offset = 0

        for name in self.field_names:
            metadata = field_metadata[name]
            # Determine number of channels
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

    def fields_to_tensor_batch(
        self, fields: Dict[str, Field], ensure_cpu: bool = False
    ) -> torch.Tensor:
        """
        Convert dictionary of Fields to concatenated tensor.

        This is used to convert physical model output (Fields) into
        format suitable for synthetic model input.

        Args:
            fields: Dictionary mapping field names to Field objects
            ensure_cpu: If True, ensures output tensor is on CPU

        Returns:
            Tensor of shape [B, C, H, W] where:
            - B: batch size (if present in fields)
            - C: sum of channels across all fields
            - H, W: spatial dimensions

        Raises:
            ValueError: If field names don't match or shapes incompatible
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
            converter = self._converters[name]
            tensor = converter.field_to_tensor(field, ensure_cpu=ensure_cpu)

            # Normalize tensor shape to [B, C, H, W] or [C, H, W]
            tensor = self._normalize_tensor_shape(tensor, name)
            tensors.append(tensor)

        # Concatenate along channel dimension
        concatenated = torch.cat(tensors, dim=-3)  # Channel dim

        return concatenated

    def tensor_to_fields_batch(
        self, tensor: torch.Tensor, time_slice: Optional[int] = None
    ) -> Dict[str, Field]:
        """
        Convert concatenated tensor back to dictionary of Fields.

        This is used to convert synthetic model output (tensor) back into
        individual Fields for physical model use or evaluation.

        Args:
            tensor: Tensor of shape [B, C, H, W] or [C, H, W]
            time_slice: Optional time index for temporal data

        Returns:
            Dictionary mapping field names to Field objects

        Raises:
            ValueError: If tensor channel dimension doesn't match expected
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

            if tensor.dim() == 4:
                field_tensor = tensor[:, start_idx:end_idx, :, :]
            else:
                field_tensor = tensor[start_idx:end_idx, :, :]

            # Convert to Field using appropriate converter
            converter = self._converters[name]
            metadata = self.field_metadata[name]
            field = converter.tensor_to_field(
                field_tensor, metadata, time_slice=time_slice
            )
            fields[name] = field

        return fields

    def _normalize_tensor_shape(
        self, tensor: torch.Tensor, field_name: str
    ) -> torch.Tensor:
        """
        Normalize tensor to [B, C, H, W] or [C, H, W] format.

        Args:
            tensor: Tensor to normalize
            field_name: Name of field for error messages

        Returns:
            Normalized tensor
        """
        expected_channels = self.channel_counts[field_name]

        if tensor.dim() == 2:  # [H, W] - scalar without batch
            tensor = tensor.unsqueeze(0)  # [1, H, W]
        elif tensor.dim() == 3:  # [B, H, W] or [C, H, W]
            # Check if it's a scalar field with batch dimension
            if expected_channels == 1 and tensor.shape[0] > 1:
                # Likely [B, H, W], add channel dim
                tensor = tensor.unsqueeze(1)  # [B, 1, H, W]
            elif tensor.shape[0] != expected_channels:
                # Must be [B, H, W] misinterpreted
                tensor = tensor.unsqueeze(1)  # [B, 1, H, W]
        # If 4D [B, C, H, W], already in correct format

        return tensor

    def get_channel_info(self) -> Dict[str, Dict[str, int]]:
        """
        Get information about channel layout in concatenated tensors.

        Returns:
            Dictionary mapping field names to {'count': int, 'offset': int}
        """
        return {
            name: {
                "count": self.channel_counts[name],
                "offset": self.channel_offsets[name],
            }
            for name in self.field_names
        }

    def validate_fields(self, fields: Dict[str, Field]) -> bool:
        """
        Validate that fields dictionary is compatible with this converter.

        Args:
            fields: Dictionary of fields to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        if set(fields.keys()) != set(self.field_names):
            raise ValueError(
                f"Field names mismatch. Expected {self.field_names}, "
                f"got {list(fields.keys())}"
            )

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
            field_type = "staggered" if field.is_staggered else "centered"
            if field_type != metadata.field_type:
                raise ValueError(
                    f"Field '{name}' is {field_type}, expected {metadata.field_type}"
                )

        return True

    def validate_tensor(self, tensor: torch.Tensor) -> bool:
        """
        Validate that tensor is compatible with this converter.

        Args:
            tensor: Tensor to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        if tensor.dim() not in [3, 4]:
            raise ValueError(
                f"Expected tensor with 3 or 4 dimensions, got {tensor.dim()}"
            )

        channel_dim = -3
        if tensor.shape[channel_dim] != self.total_channels:
            raise ValueError(
                f"Expected {self.total_channels} channels, "
                f"got {tensor.shape[channel_dim]}"
            )

        return True
