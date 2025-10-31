"""
Metric calculation functions for model evaluation.

This module provides functions to compute various error metrics comparing
model predictions against ground truth data, including per-timestep errors
and aggregate statistics.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np


def compute_mse_per_timestep(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute Mean Squared Error at each time step.
    
    Args:
        prediction: Model prediction tensor, shape [T, C, H, W]
        ground_truth: Ground truth tensor, shape [T, C, H, W]
        reduction: How to reduce spatial dimensions:
                  'mean' - average over spatial dims
                  'sum' - sum over spatial dims
                  'none' - keep spatial dimensions
    
    Returns:
        MSE values, shape depends on reduction:
        - 'mean' or 'sum': [T, C] (per timestep, per channel)
        - 'none': [T, C, H, W] (full spatial error map)
        
    Example:
        >>> mse = compute_mse_per_timestep(pred, gt, reduction='mean')
        >>> # mse.shape = [50, 2] for 50 timesteps, 2 channels
    """
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: prediction {prediction.shape} != ground_truth {ground_truth.shape}"
        )
    
    # Compute squared error
    squared_error = (prediction - ground_truth) ** 2
    
    if reduction == 'mean':
        # Average over spatial dimensions [H, W]
        return squared_error.mean(dim=(-2, -1))  # [T, C]
    elif reduction == 'sum':
        # Sum over spatial dimensions
        return squared_error.sum(dim=(-2, -1))  # [T, C]
    elif reduction == 'none':
        # Keep all dimensions
        return squared_error  # [T, C, H, W]
    else:
        raise ValueError(f"Unknown reduction: {reduction}. Use 'mean', 'sum', or 'none'")


def compute_rmse_per_timestep(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute Root Mean Squared Error at each time step.
    
    Args:
        prediction: Model prediction tensor, shape [T, C, H, W]
        ground_truth: Ground truth tensor, shape [T, C, H, W]
        reduction: How to reduce spatial dimensions ('mean', 'sum', 'none')
    
    Returns:
        RMSE values, shape [T, C] for 'mean'/'sum', [T, C, H, W] for 'none'
    """
    mse = compute_mse_per_timestep(prediction, ground_truth, reduction)
    return torch.sqrt(mse)


def compute_mae_per_timestep(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute Mean Absolute Error at each time step.
    
    Args:
        prediction: Model prediction tensor, shape [T, C, H, W]
        ground_truth: Ground truth tensor, shape [T, C, H, W]
        reduction: How to reduce spatial dimensions ('mean', 'sum', 'none')
    
    Returns:
        MAE values, shape [T, C] for 'mean'/'sum', [T, C, H, W] for 'none'
    """
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: prediction {prediction.shape} != ground_truth {ground_truth.shape}"
        )
    
    absolute_error = torch.abs(prediction - ground_truth)
    
    if reduction == 'mean':
        return absolute_error.mean(dim=(-2, -1))  # [T, C]
    elif reduction == 'sum':
        return absolute_error.sum(dim=(-2, -1))  # [T, C]
    elif reduction == 'none':
        return absolute_error  # [T, C, H, W]
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def compute_relative_error_per_timestep(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    epsilon: float = 1e-8,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute relative error at each time step.
    
    Relative error = |pred - gt| / (|gt| + epsilon)
    
    Args:
        prediction: Model prediction tensor, shape [T, C, H, W]
        ground_truth: Ground truth tensor, shape [T, C, H, W]
        epsilon: Small constant to avoid division by zero
        reduction: How to reduce spatial dimensions ('mean', 'sum', 'none')
    
    Returns:
        Relative error values, shape [T, C] for 'mean'/'sum', [T, C, H, W] for 'none'
    """
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: prediction {prediction.shape} != ground_truth {ground_truth.shape}"
        )
    
    absolute_error = torch.abs(prediction - ground_truth)
    gt_magnitude = torch.abs(ground_truth) + epsilon
    relative_error = absolute_error / gt_magnitude
    
    if reduction == 'mean':
        return relative_error.mean(dim=(-2, -1))  # [T, C]
    elif reduction == 'sum':
        return relative_error.sum(dim=(-2, -1))  # [T, C]
    elif reduction == 'none':
        return relative_error  # [T, C, H, W]
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def compute_normalized_error_per_timestep(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute normalized error at each time step.
    
    Normalized error = |pred - gt| / (max(|gt|) - min(|gt|))
    
    This normalizes by the range of the ground truth, making errors
    comparable across different scales.
    
    Args:
        prediction: Model prediction tensor, shape [T, C, H, W]
        ground_truth: Ground truth tensor, shape [T, C, H, W]
        reduction: How to reduce spatial dimensions ('mean', 'sum', 'none')
    
    Returns:
        Normalized error values, shape [T, C] for 'mean'/'sum', [T, C, H, W] for 'none'
    """
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: prediction {prediction.shape} != ground_truth {ground_truth.shape}"
        )
    
    absolute_error = torch.abs(prediction - ground_truth)
    
    # Compute range per channel across all timesteps and spatial locations
    gt_flat = ground_truth.reshape(ground_truth.shape[0], ground_truth.shape[1], -1)
    gt_range = gt_flat.max(dim=-1).values - gt_flat.min(dim=-1).values  # [T, C]
    gt_range = gt_range.unsqueeze(-1).unsqueeze(-1)  # [T, C, 1, 1] for broadcasting
    
    # Avoid division by zero
    gt_range = torch.clamp(gt_range, min=1e-8)
    
    normalized_error = absolute_error / gt_range
    
    if reduction == 'mean':
        return normalized_error.mean(dim=(-2, -1))  # [T, C]
    elif reduction == 'sum':
        return normalized_error.sum(dim=(-2, -1))  # [T, C]
    elif reduction == 'none':
        return normalized_error  # [T, C, H, W]
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def aggregate_metrics(
    errors: torch.Tensor,
    dim: int = 0
) -> Dict[str, float]:
    """
    Compute aggregate statistics over error values.
    
    Args:
        errors: Error tensor, typically shape [T, C]
        dim: Dimension to aggregate over (default: 0 for time)
    
    Returns:
        Dictionary with aggregate statistics:
        - 'mean': Mean error
        - 'std': Standard deviation
        - 'min': Minimum error
        - 'max': Maximum error
        - 'median': Median error
        - 'q25': 25th percentile
        - 'q75': 75th percentile
    """
    errors_np = errors.detach().cpu().numpy()
    
    return {
        'mean': float(np.mean(errors_np, axis=dim).mean()),
        'std': float(np.std(errors_np, axis=dim).mean()),
        'min': float(np.min(errors_np)),
        'max': float(np.max(errors_np)),
        'median': float(np.median(errors_np, axis=dim).mean()),
        'q25': float(np.percentile(errors_np, 25, axis=dim).mean()),
        'q75': float(np.percentile(errors_np, 75, axis=dim).mean()),
    }


def compute_all_metrics(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    metrics: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute multiple metrics at once.
    
    Args:
        prediction: Model prediction tensor, shape [T, C, H, W]
        ground_truth: Ground truth tensor, shape [T, C, H, W]
        metrics: List of metric names to compute. If None, computes all.
                Available: 'mse', 'rmse', 'mae', 'relative', 'normalized'
    
    Returns:
        Dictionary mapping metric names to error tensors [T, C]
        
    Example:
        >>> metrics = compute_all_metrics(pred, gt, metrics=['mse', 'mae'])
        >>> mse_over_time = metrics['mse'][:, 0]  # First channel
    """
    if metrics is None:
        metrics = ['mse', 'rmse', 'mae', 'relative', 'normalized']
    
    results = {}
    
    if 'mse' in metrics:
        results['mse'] = compute_mse_per_timestep(prediction, ground_truth)
    
    if 'rmse' in metrics:
        results['rmse'] = compute_rmse_per_timestep(prediction, ground_truth)
    
    if 'mae' in metrics:
        results['mae'] = compute_mae_per_timestep(prediction, ground_truth)
    
    if 'relative' in metrics:
        results['relative'] = compute_relative_error_per_timestep(prediction, ground_truth)
    
    if 'normalized' in metrics:
        results['normalized'] = compute_normalized_error_per_timestep(prediction, ground_truth)
    
    return results


def compute_metrics_per_field(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    field_specs: Dict[str, int],
    metrics: Optional[List[str]] = None
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute metrics for each field separately in multi-field data.
    
    Args:
        prediction: Model prediction tensor, shape [T, C_total, H, W]
        ground_truth: Ground truth tensor, shape [T, C_total, H, W]
        field_specs: Dictionary mapping field names to channel counts
                    e.g., {'velocity': 2, 'density': 1}
        metrics: List of metric names to compute
    
    Returns:
        Nested dictionary: field_name -> metric_name -> error_tensor
        
    Example:
        >>> specs = {'velocity': 2, 'density': 1}
        >>> results = compute_metrics_per_field(pred, gt, specs, ['mse', 'mae'])
        >>> velocity_mse = results['velocity']['mse']  # Shape [T, 2]
        >>> density_mae = results['density']['mae']    # Shape [T, 1]
    """
    results = {}
    channel_idx = 0
    
    for field_name, num_channels in field_specs.items():
        # Extract field data
        pred_field = prediction[:, channel_idx:channel_idx+num_channels, :, :]
        gt_field = ground_truth[:, channel_idx:channel_idx+num_channels, :, :]
        
        # Compute metrics for this field
        results[field_name] = compute_all_metrics(pred_field, gt_field, metrics)
        
        channel_idx += num_channels
    
    return results
