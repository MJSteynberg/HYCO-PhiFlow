from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from omegaconf import MISSING


@dataclass
class DomainConfig:
    """Physical domain configuration."""
    size_x: float = 100.0
    size_y: float = 100.0
    size_z: Optional[float] = None


@dataclass
class ResolutionConfig:
    """Grid resolution configuration."""
    x: int = MISSING
    y: int = MISSING
    z: Optional[int] = None


@dataclass
class PhysicalModelConfig:
    """Configuration for physical PDE models."""
    
    name: str = MISSING  # e.g., 'BurgersModel', 'SmokeModel'
    domain: DomainConfig = field(default_factory=DomainConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    dt: float = 0.8
    pde_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchitectureConfig:
    """Neural network architecture parameters."""
    levels: int = 4
    filters: int = 64
    batch_norm: bool = True


@dataclass
class SyntheticModelConfig:
    """Configuration for synthetic (neural network) models."""
    
    name: str = MISSING  # e.g., 'UNet'
    model_path: str = 'results/models'
    model_save_name: str = MISSING
    
    input_specs: Dict[str, int] = field(default_factory=dict)
    output_specs: Dict[str, int] = field(default_factory=dict)
    
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)