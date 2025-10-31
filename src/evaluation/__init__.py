"""
Evaluation module for model performance assessment.

This module provides tools for evaluating trained models, including:
- Inference on test data
- Metric computation (MSE, RMSE, MAE, etc.)
- Visualization generation (animations, error plots, keyframe comparisons)
- Main Evaluator class for orchestrating complete evaluation workflows
"""

from .visualizations import (
    create_comparison_gif,
    create_comparison_gif_from_specs,
    plot_side_by_side_frame,
    plot_error_vs_time,
    plot_error_vs_time_multi_field,
    plot_error_comparison,
    plot_error_heatmap,
    plot_keyframe_comparison,
    plot_keyframe_comparison_multi_field,
    create_evaluation_summary,
)

from .metrics import (
    compute_mse_per_timestep,
    compute_rmse_per_timestep,
    compute_mae_per_timestep,
    compute_relative_error_per_timestep,
    compute_normalized_error_per_timestep,
    compute_all_metrics,
    compute_metrics_per_field,
    aggregate_metrics,
)

from .evaluator import Evaluator

__all__ = [
    # Main Evaluator
    'Evaluator',
    # Visualizations
    'create_comparison_gif',
    'create_comparison_gif_from_specs',
    'plot_side_by_side_frame',
    'plot_error_vs_time',
    'plot_error_vs_time_multi_field',
    'plot_error_comparison',
    'plot_error_heatmap',
    'plot_keyframe_comparison',
    'plot_keyframe_comparison_multi_field',
    'create_evaluation_summary',
    # Metrics
    'compute_mse_per_timestep',
    'compute_rmse_per_timestep',
    'compute_mae_per_timestep',
    'compute_relative_error_per_timestep',
    'compute_normalized_error_per_timestep',
    'compute_all_metrics',
    'compute_metrics_per_field',
    'aggregate_metrics',
]
