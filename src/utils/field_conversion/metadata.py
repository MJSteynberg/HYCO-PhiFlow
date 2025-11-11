"""
Field Metadata Module

This module defines the FieldMetadata dataclass that stores all information
needed to reconstruct a PhiFlow Field from a tensor representation.
"""

from dataclasses import dataclass
from typing import Dict, Union, Tuple
from phi.field import Field
from phi.geom import Box
from phi.math import Shape
from phi.field._field_math import Extrapolation


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
    def from_field(cls, field: Field) -> "FieldMetadata":
        """
        Extract metadata from an existing Field.

        Args:
            field: The Field to extract metadata from

        Returns:
            FieldMetadata object
        """
        # Determine field type using is_staggered property
        field_type = "staggered" if field.is_staggered else "centered"

        return cls(
            domain=field.bounds,
            resolution=field.resolution,
            extrapolation=field.extrapolation,
            field_type=field_type,
            spatial_dims=tuple(field.shape.spatial.names),
            channel_dims=(
                tuple(field.shape.channel.names) if field.shape.channel else ()
            ),
        )

    @classmethod
    def from_cache_metadata(
        cls, cached_meta: Dict, domain: Box, resolution: Shape
    ) -> "FieldMetadata":
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
        extrap_str = cached_meta.get("extrapolation", "ZERO")

        # Map common extrapolation strings to PhiFlow objects
        from phi.math import extrapolation as extrap_module

        extrapolation_map = {
            "ZERO": extrap_module.ZERO,
            "BOUNDARY": extrap_module.BOUNDARY,
            "PERIODIC": extrap_module.PERIODIC,
            "zero-gradient": extrap_module.ZERO_GRADIENT,
            "ZERO_GRADIENT": extrap_module.ZERO_GRADIENT,
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
        field_type = cached_meta.get("field_type", "centered")

        return cls(
            domain=domain,
            resolution=resolution,
            extrapolation=extrapolation,
            field_type=field_type,
            spatial_dims=tuple(cached_meta.get("spatial_dims", ["x", "y"])),
            channel_dims=tuple(cached_meta.get("channel_dims", [])),
        )


def create_field_metadata_from_model(
    model, field_names: list[str], field_types: Dict[str, str] = None
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
    from phi.math import extrapolation

    field_types = field_types or {}

    metadata_dict = {}
    for field_name in field_names:
        field_type = field_types.get(field_name, "centered")

        # Determine channel dimensions based on common field types
        if "velocity" in field_name.lower():
            channel_dims = ("vector",)  # Velocity is typically a vector field
        else:
            channel_dims = ()  # Scalar field

        metadata_dict[field_name] = FieldMetadata(
            domain=model.domain,
            resolution=model.resolution,
            extrapolation=extrapolation.PERIODIC,  # Default, may need to be configurable
            field_type=field_type,
            spatial_dims=tuple(model.resolution.names),
            channel_dims=channel_dims,
        )

    return metadata_dict
