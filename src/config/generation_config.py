from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationConfig:
    """Configuration for data generation."""

    num_simulations: int = 10
    total_steps: int = 50
    save_interval: int = 1

    # Optional: random seed for reproducibility
    seed: Optional[int] = None
