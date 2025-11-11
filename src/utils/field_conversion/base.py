"""
Base Converter Classes

This module defines the abstract base classes for field-tensor conversions.
All concrete converters inherit from these classes and implement specific
conversion strategies for different field types.
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
from phi.field import Field
from .metadata import FieldMetadata


class AbstractConverter(ABC):
    """
    Abstract base class defining the conversion contract.

    All field-tensor conversions must go through converter classes - no
    standalone functions are exposed in the public API. This ensures
    conversion logic is properly encapsulated and extensible.

    Subclasses must implement:
    - can_handle: Determine if converter can handle given metadata
    - field_to_tensor: Convert Field to tensor
    - tensor_to_field: Convert tensor to Field
    """

    @classmethod
    @abstractmethod
    def can_handle(cls, metadata: FieldMetadata) -> bool:
        """
        Determine if this converter can handle the given field metadata.

        Args:
            metadata: FieldMetadata describing the field

        Returns:
            True if this converter can handle the field type
        """
        pass

    @abstractmethod
    def field_to_tensor(self, field: Field, *, ensure_cpu: bool = True) -> torch.Tensor:
        """
        Convert a PhiFlow Field to a PyTorch tensor.

        Args:
            field: PhiFlow Field to convert
            ensure_cpu: If True, ensures output tensor is on CPU

        Returns:
            PyTorch tensor with shape [C, H, W] or [B, C, H, W]
            where C is the number of channels
        """
        pass

    @abstractmethod
    def tensor_to_field(
        self,
        tensor: torch.Tensor,
        metadata: FieldMetadata,
        *,
        time_slice: Optional[int] = None,
    ) -> Field:
        """
        Convert a PyTorch tensor back to a PhiFlow Field.

        Args:
            tensor: PyTorch tensor to convert
            metadata: FieldMetadata containing reconstruction information
            time_slice: Optional time index for batch tensors

        Returns:
            Reconstructed PhiFlow Field
        """
        pass

    def _ensure_device(self, tensor: torch.Tensor, ensure_cpu: bool) -> torch.Tensor:
        """
        Helper method to move tensor to CPU if requested.

        Args:
            tensor: Tensor to check
            ensure_cpu: If True, move to CPU

        Returns:
            Tensor on appropriate device
        """
        if ensure_cpu and tensor.device.type != "cpu":
            return tensor.cpu()
        return tensor


class SingleFieldConverter(AbstractConverter):
    """
    Base class for converters that handle single field types.

    This provides common functionality for converters that work with
    individual fields (centered or staggered grids).
    """

    def __init__(self, metadata: Optional[FieldMetadata] = None):
        """
        Initialize converter with optional metadata.

        Args:
            metadata: Optional FieldMetadata for validation/reconstruction
        """
        self.metadata = metadata

    def validate_field(self, field: Field) -> None:
        """
        Validate that field is compatible with this converter.

        Args:
            field: Field to validate

        Raises:
            ValueError: If field is incompatible
        """
        if self.metadata is None:
            return  # No metadata to validate against

        # Check spatial dimensions
        field_spatial_dims = tuple(field.shape.spatial.names)
        if field_spatial_dims != self.metadata.spatial_dims:
            raise ValueError(
                f"Field has spatial dims {field_spatial_dims}, "
                f"expected {self.metadata.spatial_dims}"
            )

    def validate_tensor(self, tensor: torch.Tensor) -> None:
        """
        Validate that tensor is compatible with this converter.

        Args:
            tensor: Tensor to validate

        Raises:
            ValueError: If tensor is incompatible
        """
        # Check dimensions
        if tensor.dim() not in [2, 3, 4]:
            raise ValueError(f"Expected tensor with 2-4 dimensions, got {tensor.dim()}")
