from dataclasses import dataclass, field
from typing import List


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    test_sim: List[int] = field(default_factory=list)
    num_frames: int = 51
    metrics: List[str] = field(default_factory=lambda: ['mse', 'mae', 'rmse'])
    
    keyframe_count: int = 5
    animation_fps: int = 10
    save_animations: bool = True
    save_plots: bool = True
    
    output_dir: str = 'results/evaluation'