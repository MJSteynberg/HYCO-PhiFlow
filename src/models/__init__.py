"""Models package - exposes ModelRegistry for model instantiation."""

from typing import Dict, Type, Any, Callable, List
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Registry for automatic model discovery and instantiation."""

    _physical_models: Dict[str, Type] = {}
    _synthetic_models: Dict[str, Type] = {}

    @classmethod
    def register_physical(cls, name: str) -> Callable:
        def decorator(model_class: Type) -> Type:
            if name in cls._physical_models:
                logger.warning(f"Overwriting physical model '{name}'")
            cls._physical_models[name] = model_class
            logger.debug(f"Registered physical model: {name}")
            return model_class
        return decorator

    @classmethod
    def register_synthetic(cls, name: str) -> Callable:
        def decorator(model_class: Type) -> Type:
            if name in cls._synthetic_models:
                logger.warning(f"Overwriting synthetic model '{name}'")
            cls._synthetic_models[name] = model_class
            logger.debug(f"Registered synthetic model: {name}")
            return model_class
        return decorator

    @classmethod
    def get_physical_model(cls, config: Dict[str, Any], downsample_factor: int = 0):
        name = config["model"]["physical"]["name"]
        if name not in cls._physical_models:
            available = ", ".join(cls._physical_models.keys()) or "none"
            raise ValueError(f"Physical model '{name}' not found. Available: {available}")
        return cls._physical_models[name](config, downsample_factor=downsample_factor)

    @classmethod
    def get_synthetic_model(cls, config: Dict[str, Any], num_channels: int, static_fields: List[str] = None):
        name = config["model"]["synthetic"]["name"]
        if name not in cls._synthetic_models:
            available = ", ".join(cls._synthetic_models.keys()) or "none"
            raise ValueError(f"Synthetic model '{name}' not found. Available: {available}")
        return cls._synthetic_models[name](config, num_channels=num_channels, static_fields=static_fields)

    @classmethod
    def list_physical_models(cls) -> List[str]:
        return sorted(cls._physical_models.keys())

    @classmethod
    def list_synthetic_models(cls) -> List[str]:
        return sorted(cls._synthetic_models.keys())

    @classmethod
    def is_physical_model_registered(cls, name: str) -> bool:
        return name in cls._physical_models

    @classmethod
    def is_synthetic_model_registered(cls, name: str) -> bool:
        return name in cls._synthetic_models

    @classmethod
    def clear_registry(cls):
        cls._physical_models.clear()
        cls._synthetic_models.clear()


# Import models to trigger registration via decorators
from . import physical  # noqa: E402
from . import synthetic  # noqa: E402

__all__ = ["ModelRegistry", "physical", "synthetic"]
