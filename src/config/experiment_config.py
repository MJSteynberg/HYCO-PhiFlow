from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from omegaconf import MISSING

from .data_config import DataConfig
from .model_config import PhysicalModelConfig, SyntheticModelConfig
from .trainer_config import SyntheticTrainerConfig, PhysicalTrainerConfig
from .generation_config import GenerationConfig
from .evaluation_config import EvaluationConfig


@dataclass
class RunConfig:
    """Top-level run configuration."""
    experiment_name: str = MISSING
    notes: str = ""
    mode: List[str] = field(default_factory=list)  # ['generate', 'train', 'evaluate']
    model_type: str = 'synthetic'  # 'synthetic' or 'physical'


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    # Hydra settings
    defaults: List[Any] = field(default_factory=lambda: [
        '_self_',
        {'data': 'burgers_128'},
        {'model/physical': 'burgers'},
        {'model/synthetic': 'unet'},
        {'trainer': 'synthetic'},
    ])
    
    # Main config sections
    run_params: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Model configs (one will be used based on model_type)
    model: Dict[str, Any] = field(default_factory=dict)
    
    # Task-specific configs
    generation_params: Optional[GenerationConfig] = None
    trainer_params: Optional[Any] = None  # Can be Synthetic or Physical
    evaluation_params: Optional[EvaluationConfig] = None
    
    # Runtime
    project_root: str = '.'