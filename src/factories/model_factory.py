"""Factory for creating models."""

from typing import Dict, Any, List, Optional
from src.models import ModelRegistry


class ModelFactory:
    """Factory for creating model instances."""

    @staticmethod
    def create_physical_model(config: Dict[str, Any], downsample_factor: int = 0):
        """Create physical model from config."""
        return ModelRegistry.get_physical_model(config, downsample_factor=downsample_factor)

    @staticmethod
    def create_synthetic_model(
        config: Dict[str, Any],
        num_channels: int,
        static_fields: Optional[List[str]] = None,
        physical_model=None
    ):
        """
        Create synthetic model from config.

        Args:
            config: Configuration dictionary
            num_channels: Number of input/output channels
            static_fields: Explicit list of static field names. If None, reads from config.
            physical_model: If provided and static_fields is 'auto', infers from physical model.
        """
        # Get static_fields from config if not provided
        if static_fields is None:
            static_fields = config.get('model', {}).get('synthetic', {}).get('static_fields', None)

        # If 'auto', infer from physical model
        if static_fields == 'auto' and physical_model is not None:
            if hasattr(physical_model, 'static_field_names'):
                static_fields = list(physical_model.static_field_names)
            else:
                static_fields = None

        # Convert 'auto' to None if no physical model provided
        if static_fields == 'auto':
            static_fields = None

        return ModelRegistry.get_synthetic_model(
            config, num_channels=num_channels, static_fields=static_fields
        )

    @staticmethod
    def list_available_models():
        """List all available models."""
        return {
            "physical": ModelRegistry.list_physical_models(),
            "synthetic": ModelRegistry.list_synthetic_models(),
        }
