"""Factory for creating models."""

from typing import Dict, Any
import torch.nn as nn
from src.models import ModelRegistry
import torch


class ModelFactory:
    """Factory for creating model instances."""

    @staticmethod
    def create_physical_model(config: Dict[str, Any]):
        """
        Create physical model from config.

        Args:
            config: Full configuration dictionary

        Returns:
            Physical model instance
        """
        return ModelRegistry.get_physical_model(config)

    @staticmethod
    def create_synthetic_model(config: Dict[str, Any]) -> nn.Module:
        """
        Create synthetic model from config.

        Args:
            config: Full configuration dictionary

        Returns:
            Synthetic model instance
        """
        print( "Creating synthetic model..." )
        return ModelRegistry.get_synthetic_model(config)

    @staticmethod
    def list_available_models():
        """
        List all available models.

        Returns:
            Dictionary with 'physical' and 'synthetic' model lists
        """
        return {
            "physical": ModelRegistry.list_physical_models(),
            "synthetic": ModelRegistry.list_synthetic_models(),
        }
