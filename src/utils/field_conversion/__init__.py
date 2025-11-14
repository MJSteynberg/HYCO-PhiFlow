"""
Field conversion package (strict canonical design).

Exports:
 - CenteredConverter
 - StaggeredConverter
 - minimal layout helper access (optional)
"""
from .centered import CenteredConverter
from .staggered import StaggeredConverter
from .layout import canonical_to_conv_input  # convenience export

__all__ = ["CenteredConverter", "StaggeredConverter", "canonical_to_conv_input"]
