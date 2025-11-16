# src/models/__init__.py

"""
Models package initialization.

This file imports all models to trigger their registration with the ModelRegistry.
It also exports the registry for use by trainers and evaluators.
"""

"""
Models package initialization.

This file exposes the ModelRegistry and imports models to trigger
decorator-based registration.
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
	"""

	_physical_models: Dict[str, Type] = {}
	_synthetic_models: Dict[str, Type] = {}

	@classmethod
	def register_physical(self, name: str) -> Callable:
		def decorator(model_class: Type) -> Type:
			if name in self._physical_models:
				logger.warning(f"Overwriting physical model '{name}'")
			self._physical_models[name] = model_class
			logger.debug(f"Registered physical model: {name}")
			return model_class

		return decorator

	@classmethod
	def register_synthetic(self, name: str) -> Callable:
		def decorator(model_class: Type) -> Type:
			if name in self._synthetic_models:
				logger.warning(f"Overwriting synthetic model '{name}'")
			self._synthetic_models[name] = model_class
			logger.debug(f"Registered synthetic model: {name}")
			return model_class

		return decorator

	@classmethod
	def get_physical_model(self, config: Dict[str, Any]):
		name = config["model"]["physical"]["name"]
		if name not in self._physical_models:
			available = ", ".join(self._physical_models.keys()) or "none"
			raise ValueError(
				f"Physical model '{name}' not found in registry. "
				f"Available models: {available}"
			)

		model_class = self._physical_models[name]
		logger.debug(f"Creating physical model: {name}")
		return model_class(config)

	@classmethod
	def get_synthetic_model(self, config: Dict[str, Any]):
		name = config["model"]["synthetic"]["name"]
		if name not in self._synthetic_models:
			available = ", ".join(self._synthetic_models.keys()) or "none"
			raise ValueError(
				f"Synthetic model '{name}' not found in registry. "
				f"Available models: {available}"
			)

		model_class = self._synthetic_models[name]
		logger.debug(f"Creating synthetic model: {name}")
		return model_class(config)

	@classmethod
	def list_physical_models(self) -> List[str]:
		return sorted(self._physical_models.keys())

	@classmethod
	def list_synthetic_models(self) -> List[str]:
		return sorted(self._synthetic_models.keys())

	@classmethod
	def is_physical_model_registered(self, name: str) -> bool:
		return name in self._physical_models

	@classmethod
	def is_synthetic_model_registered(self, name: str) -> bool:
		return name in self._synthetic_models

	@classmethod
	def clear_registry(self):
		self._physical_models.clear()
		self._synthetic_models.clear()
		logger.debug("Cleared model registry")


# Import physical models (triggers registration via decorators)
from . import physical  # noqa: E402

# Import synthetic models (triggers registration via decorators)
from . import synthetic  # noqa: E402

# Export the registry and model classes
__all__ = ["ModelRegistry", "physical", "synthetic"]
