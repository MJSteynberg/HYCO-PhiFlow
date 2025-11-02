"""
Model Registry for Automatic Model Discovery

This module provides a registry pattern for dynamically registering and
instantiating models. This eliminates hard-coded model instantiation and
makes it easy to add new models without modifying existing code.

Features:
- Automatic model discovery via decorators
- Separate registries for physical and synthetic models
- Clear error messages for missing models
- Easy listing of available models

Usage:
    # Register a model
    @ModelRegistry.register_physical('BurgersModel')
    class BurgersModel(PhysicalModel):
        pass
    
    # Get a model instance
    model = ModelRegistry.get_physical_model('BurgersModel', config)
    
    # List available models
    available = ModelRegistry.list_physical_models()
"""

from typing import Dict, Type, Any, Callable, List
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """
    Registry for automatic model discovery and instantiation.

    This class maintains separate registries for physical and synthetic models,
    allowing dynamic model creation based on configuration without hard-coding
    model names throughout the codebase.

    Attributes:
        _physical_models: Dictionary mapping model names to physical model classes
        _synthetic_models: Dictionary mapping model names to synthetic model classes
    """

    _physical_models: Dict[str, Type] = {}
    _synthetic_models: Dict[str, Type] = {}

    @classmethod
    def register_physical(cls, name: str) -> Callable:
        """
        Decorator to register a physical model.

        Args:
            name: Name to register the model under (e.g., 'BurgersModel')

        Returns:
            Decorator function that registers the model class

        Example:
            @ModelRegistry.register_physical('BurgersModel')
            class BurgersModel(PhysicalModel):
                pass
        """

        def decorator(model_class: Type) -> Type:
            if name in cls._physical_models:
                logger.warning(f"Overwriting physical model '{name}'")
            cls._physical_models[name] = model_class
            logger.debug(f"Registered physical model: {name}")
            return model_class

        return decorator

    @classmethod
    def register_synthetic(cls, name: str) -> Callable:
        """
        Decorator to register a synthetic model.

        Args:
            name: Name to register the model under (e.g., 'UNet')

        Returns:
            Decorator function that registers the model class

        Example:
            @ModelRegistry.register_synthetic('UNet')
            class UNet(SyntheticModel):
                pass
        """

        def decorator(model_class: Type) -> Type:
            if name in cls._synthetic_models:
                logger.warning(f"Overwriting synthetic model '{name}'")
            cls._synthetic_models[name] = model_class
            logger.debug(f"Registered synthetic model: {name}")
            return model_class

        return decorator

    @classmethod
    def get_physical_model(cls, name: str, config: Dict[str, Any]):
        """
        Get an instance of a physical model.

        Args:
            name: Name of the model to instantiate
            config: Configuration dictionary for the model

        Returns:
            Instance of the requested physical model

        Raises:
            ValueError: If the model name is not registered

        Example:
            config = {'domain': {...}, 'resolution': {...}}
            model = ModelRegistry.get_physical_model('BurgersModel', config)
        """
        if name not in cls._physical_models:
            available = ", ".join(cls._physical_models.keys()) or "none"
            raise ValueError(
                f"Physical model '{name}' not found in registry. "
                f"Available models: {available}"
            )

        model_class = cls._physical_models[name]
        logger.debug(f"Creating physical model: {name}")
        return model_class(config)

    @classmethod
    def get_synthetic_model(cls, name: str, config: Dict[str, Any]):
        """
        Get an instance of a synthetic model.

        Args:
            name: Name of the model to instantiate
            config: Configuration dictionary for the model

        Returns:
            Instance of the requested synthetic model

        Raises:
            ValueError: If the model name is not registered

        Example:
            config = {'input_specs': {...}, 'output_specs': {...}}
            model = ModelRegistry.get_synthetic_model('UNet', config)
        """
        if name not in cls._synthetic_models:
            available = ", ".join(cls._synthetic_models.keys()) or "none"
            raise ValueError(
                f"Synthetic model '{name}' not found in registry. "
                f"Available models: {available}"
            )

        model_class = cls._synthetic_models[name]
        logger.debug(f"Creating synthetic model: {name}")
        return model_class(config)

    @classmethod
    def list_physical_models(cls) -> List[str]:
        """
        List all registered physical models.

        Returns:
            List of physical model names

        Example:
            >>> ModelRegistry.list_physical_models()
            ['BurgersModel', 'SmokeModel', 'HeatModel']
        """
        return sorted(cls._physical_models.keys())

    @classmethod
    def list_synthetic_models(cls) -> List[str]:
        """
        List all registered synthetic models.

        Returns:
            List of synthetic model names

        Example:
            >>> ModelRegistry.list_synthetic_models()
            ['UNet', 'ResNet', 'FNO']
        """
        return sorted(cls._synthetic_models.keys())

    @classmethod
    def is_physical_model_registered(cls, name: str) -> bool:
        """
        Check if a physical model is registered.

        Args:
            name: Model name to check

        Returns:
            True if model is registered, False otherwise
        """
        return name in cls._physical_models

    @classmethod
    def is_synthetic_model_registered(cls, name: str) -> bool:
        """
        Check if a synthetic model is registered.

        Args:
            name: Model name to check

        Returns:
            True if model is registered, False otherwise
        """
        return name in cls._synthetic_models

    @classmethod
    def clear_registry(cls):
        """
        Clear all registered models.

        This is mainly useful for testing purposes.
        """
        cls._physical_models.clear()
        cls._synthetic_models.clear()
        logger.debug("Cleared model registry")
