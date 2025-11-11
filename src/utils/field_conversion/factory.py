"""
Converter Factory

This module provides factory functions to create appropriate converters
based on field metadata or Field objects.
"""

from typing import Dict, Union
from phi.field import Field
from .base import AbstractConverter
from .metadata import FieldMetadata
from .centered import CenteredConverter
from .staggered import StaggeredConverter
from .batch import BatchConcatenationConverter


def make_converter(
    field_or_metadata: Union[Field, FieldMetadata, Dict[str, FieldMetadata]]
) -> Union[AbstractConverter, BatchConcatenationConverter]:
    """
    Create appropriate converter based on input.

    This is the main entry point for obtaining converters. It automatically
    selects the right converter class based on the input type.

    Args:
        field_or_metadata: Can be one of:
            - Field: Single PhiFlow Field (creates single-field converter)
            - FieldMetadata: Metadata for single field
            - Dict[str, FieldMetadata]: Multiple fields (creates batch converter)

    Returns:
        Appropriate converter instance:
        - CenteredConverter for centered fields
        - StaggeredConverter for staggered fields
        - BatchConcatenationConverter for multiple fields

    Raises:
        ValueError: If input type is invalid or field type is unknown

    Examples:
        >>> # Single field from Field object
        >>> converter = make_converter(velocity_field)
        >>> tensor = converter.field_to_tensor(velocity_field)
        >>>
        >>> # Single field from metadata
        >>> converter = make_converter(velocity_metadata)
        >>> field = converter.tensor_to_field(tensor, velocity_metadata)
        >>>
        >>> # Multiple fields (batch converter)
        >>> converter = make_converter({
        ...     'velocity': velocity_metadata,
        ...     'density': density_metadata
        ... })
        >>> batch_tensor = converter.fields_to_tensor_batch(fields)
    """
    # Case 1: Dictionary of metadata -> BatchConcatenationConverter
    if isinstance(field_or_metadata, dict):
        return BatchConcatenationConverter(field_or_metadata)

    # Case 2: Single Field -> extract metadata and create single converter
    if isinstance(field_or_metadata, Field):
        metadata = FieldMetadata.from_field(field_or_metadata)
        return _create_single_converter(metadata)

    # Case 3: Single FieldMetadata -> create single converter
    if isinstance(field_or_metadata, FieldMetadata):
        return _create_single_converter(field_or_metadata)

    raise ValueError(
        f"Invalid input type: {type(field_or_metadata)}. "
        f"Expected Field, FieldMetadata, or Dict[str, FieldMetadata]"
    )


def _create_single_converter(metadata: FieldMetadata) -> AbstractConverter:
    """
    Create a single-field converter based on metadata.

    Args:
        metadata: FieldMetadata describing the field

    Returns:
        Appropriate single-field converter

    Raises:
        ValueError: If field type is unknown
    """
    if metadata.field_type == "centered":
        return CenteredConverter(metadata)
    elif metadata.field_type == "staggered":
        return StaggeredConverter(metadata)
    else:
        raise ValueError(f"Unknown field type: {metadata.field_type}")


def make_batch_converter(
    field_metadata: Dict[str, FieldMetadata]
) -> BatchConcatenationConverter:
    """
    Explicitly create a batch concatenation converter.

    This is a convenience function when you specifically want a batch converter
    and want to be explicit about it.

    Args:
        field_metadata: Dictionary mapping field names to FieldMetadata

    Returns:
        BatchConcatenationConverter instance

    Example:
        >>> converter = make_batch_converter({
        ...     'velocity': velocity_metadata,
        ...     'density': density_metadata
        ... })
    """
    return BatchConcatenationConverter(field_metadata)


def make_centered_converter(metadata: FieldMetadata = None) -> CenteredConverter:
    """
    Explicitly create a centered grid converter.

    Args:
        metadata: Optional FieldMetadata for the converter

    Returns:
        CenteredConverter instance
    """
    return CenteredConverter(metadata)


def make_staggered_converter(metadata: FieldMetadata = None) -> StaggeredConverter:
    """
    Explicitly create a staggered grid converter.

    Args:
        metadata: Optional FieldMetadata for the converter

    Returns:
        StaggeredConverter instance
    """
    return StaggeredConverter(metadata)
