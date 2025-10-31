# src/models/__init__.py

"""
Models package initialization.

This file imports all models to trigger their registration with the ModelRegistry.
It also exports the registry for use by trainers and evaluators.
"""

# Import registry first
from .registry import ModelRegistry

# Import physical models (triggers registration via decorators)
from . import physical

# Import synthetic models (triggers registration via decorators)
from . import synthetic

# Export the registry and model classes
__all__ = ['ModelRegistry', 'physical', 'synthetic']
