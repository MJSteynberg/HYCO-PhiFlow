"""
Field Conversion Package

This package provides class-based converters for bidirectional conversion
between PhiFlow Fields and PyTorch tensors.

Public API:
-----------
Classes:
    - FieldMetadata: Metadata for field reconstruction
    - AbstractConverter: Base class for all converters
    - CenteredConverter: Converter for centered grids
    - StaggeredConverter: Converter for staggered grids
    - BatchConcatenationConverter: Multi-field batch converter

Factory Functions:
    - make_converter: Create appropriate converter from Field/metadata
    - make_batch_converter: Create batch concatenation converter
    - make_centered_converter: Create centered grid converter
    - make_staggered_converter: Create staggered grid converter

Helper Functions:
    - create_field_metadata_from_model: Create metadata from physical model

Design Philosophy:
-----------------
All conversion functionality is exposed through classes, not standalone functions.
This ensures:
- Clear encapsulation of conversion logic
- Easy extension for new field types
- Type-safe API with proper validation
- Efficient reuse through stateful converters

Usage Examples:
--------------
Single field conversion:
    >>> from src.utils.field_conversion import make_converter
    >>> converter = make_converter(my_field)
    >>> tensor = converter.field_to_tensor(my_field)
    >>> reconstructed = converter.tensor_to_field(tensor, metadata)

Batch conversion (multiple fields):
    >>> converter = make_converter(metadata_dict)
    >>> batch_tensor = converter.fields_to_tensor_batch(fields_dict)
    >>> fields = converter.tensor_to_fields_batch(batch_tensor)

For detailed documentation, see docs/FIELD_CONVERSION.md
"""

# Core classes
from .metadata import FieldMetadata, create_field_metadata_from_model
from .base import AbstractConverter, SingleFieldConverter
from .centered import CenteredConverter
from .staggered import StaggeredConverter
from .batch import BatchConcatenationConverter

# Factory functions
from .factory import (
    make_converter,
    make_batch_converter,
    make_centered_converter,
    make_staggered_converter,
)

# Backward compatibility: provide FieldTensorConverter as alias
# This maintains compatibility with existing code that uses:
# from src.utils.field_conversion import FieldTensorConverter
FieldTensorConverter = BatchConcatenationConverter

__all__ = [
    # Core classes
    "FieldMetadata",
    "AbstractConverter",
    "SingleFieldConverter",
    "CenteredConverter",
    "StaggeredConverter",
    "BatchConcatenationConverter",
    "FieldTensorConverter",  # Backward compatibility alias
    # Factory functions
    "make_converter",
    "make_batch_converter",
    "make_centered_converter",
    "make_staggered_converter",
    # Helper functions
    "create_field_metadata_from_model",
]
