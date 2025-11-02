"""Structured configuration using Hydra dataclasses."""

from .data_config import DataConfig
from .model_config import PhysicalModelConfig, SyntheticModelConfig
from .trainer_config import SyntheticTrainerConfig, PhysicalTrainerConfig
from .generation_config import GenerationConfig
from .evaluation_config import EvaluationConfig
from .experiment_config import ExperimentConfig

__all__ = [
    "DataConfig",
    "PhysicalModelConfig",
    "SyntheticModelConfig",
    "SyntheticTrainerConfig",
    "PhysicalTrainerConfig",
    "GenerationConfig",
    "EvaluationConfig",
    "ExperimentConfig",
]
