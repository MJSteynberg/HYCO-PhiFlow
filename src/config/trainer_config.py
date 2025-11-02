from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SyntheticTrainerConfig:
    """Configuration for synthetic model training."""

    learning_rate: float = 1e-4
    batch_size: int = 16
    epochs: int = 100
    num_predict_steps: int = 4

    train_sim: List[int] = field(default_factory=list)
    val_sim: Optional[List[int]] = None

    use_sliding_window: bool = False

    # Optimizer settings
    optimizer: str = "adam"
    scheduler: str = "cosine"
    weight_decay: float = 0.0

    # Checkpoint settings
    save_interval: int = 10
    save_best_only: bool = True


@dataclass
class LearnableParameter:
    """Definition of a learnable parameter for inverse problems."""

    name: str
    initial_guess: float
    bounds: Optional[tuple] = None


@dataclass
class PhysicalTrainerConfig:
    """Configuration for physical model inverse problem training."""

    epochs: int = 100
    num_predict_steps: int = 10
    train_sim: List[int] = field(default_factory=list)

    learnable_parameters: List[LearnableParameter] = field(default_factory=list)

    # Optimizer settings
    method: str = "L-BFGS-B"
    abs_tol: float = 1e-6
    max_iterations: Optional[int] = None
