"""Structured configuration using Hydra dataclasses."""

from .config_helper import ConfigHelper
from .data_config import DataConfig
from .model_config import PhysicalModelConfig, SyntheticModelConfig
from .trainer_config import SyntheticTrainerConfig, PhysicalTrainerConfig
from .generation_config import GenerationConfig
from .evaluation_config import EvaluationConfig
from .experiment_config import ExperimentConfig
from .augmentation_config import (
    AugmentationConfig,
    validate_cache_config,
    create_cache_path,
    get_augmentation_summary,
)

__all__ = [
    "ConfigHelper",
    "DataConfig",
    "PhysicalModelConfig",
    "SyntheticModelConfig",
    "SyntheticTrainerConfig",
    "PhysicalTrainerConfig",
    "GenerationConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "AugmentationConfig",
    "validate_cache_config",
    "create_cache_path",
    "get_augmentation_summary",
]
