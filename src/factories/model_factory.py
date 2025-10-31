"""Factory for creating models."""

from typing import Dict, Any
import torch.nn as nn
from src.models.registry import ModelRegistry


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
        model_config = config['model']['physical']
        model_name = model_config['name']
        return ModelRegistry.get_physical_model(model_name, model_config)
    
    @staticmethod
    def create_synthetic_model(config: Dict[str, Any]) -> nn.Module:
        """
        Create synthetic model from config.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            Synthetic model instance
        """
        model_config = config['model']['synthetic']
        model_name = model_config['name']
        return ModelRegistry.get_synthetic_model(model_name, model_config)
    
    @staticmethod
    def list_available_models():
        """
        List all available models.
        
        Returns:
            Dictionary with 'physical' and 'synthetic' model lists
        """
        return {
            'physical': ModelRegistry.list_physical_models(),
            'synthetic': ModelRegistry.list_synthetic_models(),
        }
