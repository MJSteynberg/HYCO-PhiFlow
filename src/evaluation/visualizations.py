"""
Visualization functions for model evaluation.

This module provides functions to create various visualizations comparing
model predictions against ground truth data, including animations, error plots,
and keyframe comparisons.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

from .metrics import compute_all_metrics, compute_metrics_per_field


def create_comparison_gif(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    field_name: str,
    save_path: Union[str, Path],
    fps: int = 10,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    titles: Optional[Tuple[str, str]] = None,
    show_difference: bool = True
) -> None:
    """
    Create a side-by-side animated GIF comparing prediction and ground truth.
    
    This function creates an animation with up to 3 panels:
    1. Ground Truth
    2. Prediction
    3. Absolute Difference (optional)
    
    Args:
        prediction: Model prediction tensor, shape [T, C, H, W] or [T, H, W]
        ground_truth: Ground truth tensor, shape [T, C, H, W] or [T, H, W]
        field_name: Name of the field being visualized (e.g., 'velocity', 'density')
        save_path: Path where the GIF will be saved
        fps: Frames per second for the animation
        vmin: Minimum value for color scale (auto-computed if None)
        vmax: Maximum value for color scale (auto-computed if None)
        titles: Custom titles for (ground_truth, prediction) panels
        show_difference: Whether to show the difference panel
        
    Raises:
        ValueError: If tensor shapes don't match or are invalid
    """
    # Validate inputs
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: prediction {prediction.shape} != ground_truth {ground_truth.shape}"
        )
    
    # Convert to numpy and move to CPU
    pred_np = prediction.detach().cpu().numpy()
    gt_np = ground_truth.detach().cpu().numpy()
    
    # Handle different tensor shapes
    if pred_np.ndim == 4:  # [T, C, H, W]
        # For multi-channel data (e.g., 2D velocity), compute magnitude
        if pred_np.shape[1] == 2:
            pred_np = np.sqrt(pred_np[:, 0]**2 + pred_np[:, 1]**2)
            gt_np = np.sqrt(gt_np[:, 0]**2 + gt_np[:, 1]**2)
        elif pred_np.shape[1] == 1:
            pred_np = pred_np[:, 0]
            gt_np = gt_np[:, 0]
        else:
            raise ValueError(f"Unsupported channel count: {pred_np.shape[1]}")
    elif pred_np.ndim != 3:  # Should be [T, H, W]
        raise ValueError(f"Expected 3 or 4 dimensions, got {pred_np.ndim}")
    
    # Transpose spatial dimensions: PhiFlow uses [x, y] but matplotlib expects [y, x] (row, col)
    pred_np = np.transpose(pred_np, (0, 2, 1))  # [T, H, W] -> [T, W, H] (swaps last two axes)
    gt_np = np.transpose(gt_np, (0, 2, 1))
    
    num_frames = pred_np.shape[0]
    
    # Compute global min/max for consistent color scale
    if vmin is None:
        vmin = min(gt_np.min(), pred_np.min())
    if vmax is None:
        vmax = max(gt_np.max(), pred_np.max())
    
    # Compute difference
    diff_np = np.abs(gt_np - pred_np)
    diff_vmax = diff_np.max()
    
    # Set up titles
    if titles is None:
        titles = ('Ground Truth', 'Prediction')
    
    # Create figure
    num_cols = 3 if show_difference else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(6*num_cols, 5))
    
    if not show_difference:
        axes = list(axes)
    else:
        axes = list(axes)
    
    # Initialize images
    im1 = axes[0].imshow(gt_np[0], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title(titles[0])
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(pred_np[0], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title(titles[1])
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    if show_difference:
        im3 = axes[2].imshow(diff_np[0], cmap='hot', vmin=0, vmax=diff_vmax, origin='lower')
        axes[2].set_title('Absolute Difference')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Add title with frame counter
    title_text = fig.suptitle(f'{field_name.capitalize()} - Frame 0/{num_frames-1}', 
                               fontsize=14, fontweight='bold')
    
    def update(frame):
        """Update function for animation."""
        im1.set_array(gt_np[frame])
        im2.set_array(pred_np[frame])
        if show_difference:
            im3.set_array(diff_np[frame])
        title_text.set_text(f'{field_name.capitalize()} - Frame {frame}/{num_frames-1}')
        return [im1, im2, im3, title_text] if show_difference else [im1, im2, title_text]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=1000/fps, blit=True
    )
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving animation to {save_path}...")
    anim.save(str(save_path), writer='pillow', fps=fps)
    plt.close(fig)
    print(f"Animation saved successfully!")


def create_comparison_gif_from_specs(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    field_specs: Dict[str, int],
    save_dir: Union[str, Path],
    fps: int = 10,
    show_difference: bool = True
) -> Dict[str, Path]:
    """
    Create comparison GIFs for all fields based on specs.
    
    This function handles multi-field tensors by splitting them according
    to the provided specs and creating separate animations for each field.
    
    Args:
        prediction: Model prediction tensor, shape [T, C_total, H, W]
        ground_truth: Ground truth tensor, shape [T, C_total, H, W]
        field_specs: Dictionary mapping field names to channel counts
                    e.g., {'velocity': 2, 'density': 1}
        save_dir: Directory where GIFs will be saved
        fps: Frames per second for animations
        show_difference: Whether to show difference panels
        
    Returns:
        Dictionary mapping field names to saved file paths
        
    Example:
        >>> specs = {'velocity': 2, 'density': 1}
        >>> paths = create_comparison_gif_from_specs(pred, gt, specs, 'results/')
        >>> # Creates: results/velocity_comparison.gif, results/density_comparison.gif
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    channel_idx = 0
    
    for field_name, num_channels in field_specs.items():
        print(f"\nCreating animation for '{field_name}' ({num_channels} channels)...")
        
        # Extract field data
        pred_field = prediction[:, channel_idx:channel_idx+num_channels, :, :]
        gt_field = ground_truth[:, channel_idx:channel_idx+num_channels, :, :]
        
        # Create animation
        save_path = save_dir / f"{field_name}_comparison.gif"
        create_comparison_gif(
            pred_field,
            gt_field,
            field_name,
            save_path,
            fps=fps,
            show_difference=show_difference
        )
        
        saved_paths[field_name] = save_path
        channel_idx += num_channels
    
    return saved_paths


def plot_side_by_side_frame(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    frame_idx: int,
    field_name: str,
    save_path: Optional[Union[str, Path]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_difference: bool = True
) -> plt.Figure:
    """
    Plot a single frame side-by-side comparison.
    
    Useful for debugging or creating static frame comparisons.
    
    Args:
        prediction: Model prediction tensor, shape [T, C, H, W] or [T, H, W]
        ground_truth: Ground truth tensor, shape [T, C, H, W] or [T, H, W]
        frame_idx: Which time step to visualize
        field_name: Name of the field
        save_path: If provided, save the figure to this path
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        show_difference: Whether to show difference panel
        
    Returns:
        Matplotlib figure object
    """
    # Extract frame
    pred_frame = prediction[frame_idx].detach().cpu().numpy()
    gt_frame = ground_truth[frame_idx].detach().cpu().numpy()
    
    # Handle multi-channel (compute magnitude)
    if pred_frame.ndim == 3 and pred_frame.shape[0] == 2:
        pred_frame = np.sqrt(pred_frame[0]**2 + pred_frame[1]**2)
        gt_frame = np.sqrt(gt_frame[0]**2 + gt_frame[1]**2)
    elif pred_frame.ndim == 3 and pred_frame.shape[0] == 1:
        pred_frame = pred_frame[0]
        gt_frame = gt_frame[0]
    
    # Transpose spatial dimensions: PhiFlow uses [x, y] but matplotlib expects [y, x]
    pred_frame = pred_frame.T
    gt_frame = gt_frame.T
    
    # Compute color scale
    if vmin is None:
        vmin = min(gt_frame.min(), pred_frame.min())
    if vmax is None:
        vmax = max(gt_frame.max(), pred_frame.max())
    
    # Create figure
    num_cols = 3 if show_difference else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(6*num_cols, 5))
    
    if num_cols == 2:
        axes = list(axes)
    
    # Plot ground truth
    im1 = axes[0].imshow(gt_frame, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot prediction
    im2 = axes[1].imshow(pred_frame, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot difference
    if show_difference:
        diff = np.abs(gt_frame - pred_frame)
        im3 = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=diff.max(), origin='lower')
        axes[2].set_title('Absolute Difference')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    fig.suptitle(f'{field_name.capitalize()} - Frame {frame_idx}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Frame saved to {save_path}")
    
    return fig


def plot_error_vs_time(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    field_name: str,
    save_path: Union[str, Path],
    metrics: Optional[List[str]] = None,
    channel_names: Optional[List[str]] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot error metrics as a function of time.
    
    Creates a line plot showing how different error metrics evolve over time.
    Useful for understanding error accumulation in autoregressive predictions.
    
    Args:
        prediction: Model prediction tensor, shape [T, C, H, W]
        ground_truth: Ground truth tensor, shape [T, C, H, W]
        field_name: Name of the field being visualized
        save_path: Path where the plot will be saved
        metrics: List of metrics to plot. If None, plots ['mse', 'mae']
        channel_names: Optional names for each channel (e.g., ['x', 'y'] for velocity)
        title: Custom title for the plot
        
    Returns:
        Matplotlib figure object
        
    Example:
        >>> plot_error_vs_time(pred, gt, 'velocity', 'error.png', metrics=['mse', 'rmse'])
    """
    if metrics is None:
        metrics = ['mse', 'mae']
    
    # Compute all requested metrics
    error_dict = compute_all_metrics(prediction, ground_truth, metrics)
    
    num_timesteps = prediction.shape[0]
    num_channels = prediction.shape[1]
    time_steps = np.arange(num_timesteps)
    
    # Set up channel names
    if channel_names is None:
        if num_channels == 2:
            channel_names = ['x-component', 'y-component']
        elif num_channels == 1:
            channel_names = ['value']
        else:
            channel_names = [f'channel_{i}' for i in range(num_channels)]
    
    # Create subplots - one per metric
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4*num_metrics))
    
    if num_metrics == 1:
        axes = [axes]
    
    # Color palette for channels
    colors = plt.cm.tab10(np.linspace(0, 1, num_channels))
    
    for ax, metric_name in zip(axes, metrics):
        error_values = error_dict[metric_name].detach().cpu().numpy()  # [T, C]
        
        # Plot each channel
        for c in range(num_channels):
            ax.plot(time_steps, error_values[:, c], 
                   label=channel_names[c], 
                   color=colors[c],
                   linewidth=2,
                   marker='o' if num_timesteps < 20 else None,
                   markersize=4)
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel(f'{metric_name.upper()}', fontsize=12)
        ax.set_title(f'{metric_name.upper()} vs Time', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Add mean line for multi-channel
        if num_channels > 1:
            mean_error = error_values.mean(axis=1)
            ax.plot(time_steps, mean_error, 
                   label='Mean', 
                   color='black',
                   linewidth=2.5,
                   linestyle='--',
                   alpha=0.7)
    
    # Overall title
    if title is None:
        title = f'{field_name.capitalize()} - Error Evolution Over Time'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Error plot saved to {save_path}")
    
    return fig


def plot_error_vs_time_multi_field(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    field_specs: Dict[str, int],
    save_dir: Union[str, Path],
    metrics: Optional[List[str]] = None
) -> Dict[str, Path]:
    """
    Plot error vs time for multiple fields.
    
    Creates separate error plots for each field in multi-field data.
    
    Args:
        prediction: Model prediction tensor, shape [T, C_total, H, W]
        ground_truth: Ground truth tensor, shape [T, C_total, H, W]
        field_specs: Dictionary mapping field names to channel counts
        save_dir: Directory where plots will be saved
        metrics: List of metrics to plot
        
    Returns:
        Dictionary mapping field names to saved file paths
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    channel_idx = 0
    
    for field_name, num_channels in field_specs.items():
        print(f"\nCreating error plot for '{field_name}'...")
        
        # Extract field data
        pred_field = prediction[:, channel_idx:channel_idx+num_channels, :, :]
        gt_field = ground_truth[:, channel_idx:channel_idx+num_channels, :, :]
        
        # Determine channel names
        if num_channels == 2 and field_name == 'velocity':
            channel_names = ['x-component', 'y-component']
        elif num_channels == 1:
            channel_names = [field_name]
        else:
            channel_names = [f'{field_name}_{i}' for i in range(num_channels)]
        
        # Create plot
        save_path = save_dir / f"{field_name}_error_vs_time.png"
        plot_error_vs_time(
            pred_field,
            gt_field,
            field_name,
            save_path,
            metrics=metrics,
            channel_names=channel_names
        )
        
        saved_paths[field_name] = save_path
        channel_idx += num_channels
    
    return saved_paths


def plot_error_comparison(
    errors_dict: Dict[str, torch.Tensor],
    field_name: str,
    save_path: Union[str, Path],
    channel_idx: int = 0,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot multiple error metrics on the same graph for comparison.
    
    Useful for comparing different error metrics (MSE vs MAE vs RMSE) on
    the same scale to understand their relative behavior.
    
    Args:
        errors_dict: Dictionary mapping metric names to error tensors [T, C]
        field_name: Name of the field
        save_path: Path where the plot will be saved
        channel_idx: Which channel to plot (for multi-channel fields)
        title: Custom title for the plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    num_timesteps = None
    colors = plt.cm.Set2(np.linspace(0, 1, len(errors_dict)))
    
    for (metric_name, error_values), color in zip(errors_dict.items(), colors):
        error_np = error_values.detach().cpu().numpy()  # [T, C]
        if num_timesteps is None:
            num_timesteps = error_np.shape[0]
        
        time_steps = np.arange(num_timesteps)
        
        # Plot the specified channel
        ax.plot(time_steps, error_np[:, channel_idx], 
               label=metric_name.upper(),
               color=color,
               linewidth=2.5,
               marker='o' if num_timesteps < 20 else None,
               markersize=5)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    
    if title is None:
        title = f'{field_name.capitalize()} - Error Metric Comparison'
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    
    return fig


def plot_error_heatmap(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    field_name: str,
    save_path: Union[str, Path],
    metric: str = 'mse',
    max_frames: int = 50
) -> plt.Figure:
    """
    Create a heatmap showing error evolution across time and channels.
    
    Useful for multi-channel fields to see which channels accumulate
    more error over time.
    
    Args:
        prediction: Model prediction tensor, shape [T, C, H, W]
        ground_truth: Ground truth tensor, shape [T, C, H, W]
        field_name: Name of the field
        save_path: Path where the plot will be saved
        metric: Which metric to visualize ('mse', 'mae', 'rmse')
        max_frames: Maximum number of frames to show (subsamples if exceeded)
        
    Returns:
        Matplotlib figure object
    """
    # Compute metric
    error_dict = compute_all_metrics(prediction, ground_truth, metrics=[metric])
    error_values = error_dict[metric].detach().cpu().numpy()  # [T, C]
    
    num_timesteps, num_channels = error_values.shape
    
    # Subsample if too many frames
    if num_timesteps > max_frames:
        indices = np.linspace(0, num_timesteps-1, max_frames, dtype=int)
        error_values = error_values[indices, :]
        time_labels = indices
    else:
        time_labels = np.arange(num_timesteps)
    
    # Create heatmap
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    im = ax.imshow(error_values.T, aspect='auto', cmap='hot', origin='lower')
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Channel', fontsize=12)
    ax.set_title(f'{field_name.capitalize()} - {metric.upper()} Heatmap', 
                fontsize=13, fontweight='bold')
    
    # Set ticks
    ax.set_yticks(np.arange(num_channels))
    if num_channels == 2:
        ax.set_yticklabels(['x', 'y'])
    else:
        ax.set_yticklabels([f'ch{i}' for i in range(num_channels)])
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric.upper(), fontsize=11)
    
    plt.tight_layout()
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved to {save_path}")
    
    return fig


def plot_keyframe_comparison(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    field_name: str,
    save_path: Union[str, Path],
    num_keyframes: int = 5,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_difference: bool = True,
    show_metrics: bool = True
) -> plt.Figure:
    """
    Plot evenly-spaced keyframes comparing prediction and ground truth.
    
    Creates a grid layout with keyframes at t=0, T/4, T/2, 3T/4, T showing
    side-by-side comparison of model prediction vs ground truth.
    
    Args:
        prediction: Model prediction tensor, shape [T, C, H, W]
        ground_truth: Ground truth tensor, shape [T, C, H, W]
        field_name: Name of the field being visualized
        save_path: Path where the plot will be saved
        num_keyframes: Number of evenly-spaced frames to show (default: 5)
        vmin: Minimum value for color scale (auto-computed if None)
        vmax: Maximum value for color scale (auto-computed if None)
        show_difference: Whether to show difference column
        show_metrics: Whether to show error metrics for each frame
        
    Returns:
        Matplotlib figure object
        
    Example:
        >>> plot_keyframe_comparison(pred, gt, 'velocity', 'keyframes.png')
        # Creates 5×3 grid: [Ground Truth | Prediction | Difference] × 5 frames
    """
    # Validate inputs
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: prediction {prediction.shape} != ground_truth {ground_truth.shape}"
        )
    
    # Convert to numpy and move to CPU
    pred_np = prediction.detach().cpu().numpy()
    gt_np = ground_truth.detach().cpu().numpy()
    
    # Handle different tensor shapes
    if pred_np.ndim == 4:  # [T, C, H, W]
        # For multi-channel data (e.g., 2D velocity), compute magnitude
        if pred_np.shape[1] == 2:
            pred_np = np.sqrt(pred_np[:, 0]**2 + pred_np[:, 1]**2)
            gt_np = np.sqrt(gt_np[:, 0]**2 + gt_np[:, 1]**2)
            field_label = f'{field_name.capitalize()} Magnitude'
        elif pred_np.shape[1] == 1:
            pred_np = pred_np[:, 0]
            gt_np = gt_np[:, 0]
            field_label = field_name.capitalize()
        else:
            raise ValueError(f"Unsupported channel count: {pred_np.shape[1]}")
    elif pred_np.ndim != 3:  # Should be [T, H, W]
        raise ValueError(f"Expected 3 or 4 dimensions, got {pred_np.ndim}")
    else:
        field_label = field_name.capitalize()
    
    # Transpose spatial dimensions: PhiFlow uses [x, y] but matplotlib expects [y, x]
    pred_np = np.transpose(pred_np, (0, 2, 1))  # [T, H, W] -> [T, W, H]
    gt_np = np.transpose(gt_np, (0, 2, 1))
    
    num_frames = pred_np.shape[0]
    
    # Select evenly-spaced keyframe indices
    if num_frames < num_keyframes:
        # If we have fewer frames than requested, use all of them
        keyframe_indices = list(range(num_frames))
        num_keyframes = num_frames
    else:
        # Evenly space keyframes: 0, T/4, T/2, 3T/4, T
        keyframe_indices = np.linspace(0, num_frames - 1, num_keyframes, dtype=int)
    
    # Compute global min/max for consistent color scale
    if vmin is None:
        vmin = min(gt_np.min(), pred_np.min())
    if vmax is None:
        vmax = max(gt_np.max(), pred_np.max())
    
    # Compute difference
    diff_np = np.abs(gt_np - pred_np)
    diff_vmax = diff_np.max()
    
    # Compute metrics if requested
    metrics_text = []
    if show_metrics:
        pred_tensor = prediction
        gt_tensor = ground_truth
        
        # Compute MSE and MAE for each keyframe
        for idx in keyframe_indices:
            frame_pred = pred_tensor[idx:idx+1]
            frame_gt = gt_tensor[idx:idx+1]
            
            mse = torch.mean((frame_pred - frame_gt) ** 2).item()
            mae = torch.mean(torch.abs(frame_pred - frame_gt)).item()
            
            metrics_text.append(f"MSE: {mse:.2e}\nMAE: {mae:.2e}")
    
    # Create figure
    num_cols = 3 if show_difference else 2
    fig = plt.figure(figsize=(6*num_cols, 4*num_keyframes))
    gs = GridSpec(num_keyframes, num_cols, figure=fig, hspace=0.3, wspace=0.3)
    
    # Column titles
    col_titles = ['Ground Truth', 'Prediction']
    if show_difference:
        col_titles.append('Absolute Difference')
    
    # Plot each keyframe
    for row_idx, frame_idx in enumerate(keyframe_indices):
        # Ground Truth
        ax_gt = fig.add_subplot(gs[row_idx, 0])
        im_gt = ax_gt.imshow(gt_np[frame_idx], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        
        if row_idx == 0:
            ax_gt.set_title(col_titles[0], fontsize=13, fontweight='bold')
        
        # Frame label on the left
        time_label = f't = {frame_idx}/{num_frames-1}'
        if num_keyframes == 5:
            # Add fraction labels for standard 5-frame layout
            fractions = ['t = 0', 't = T/4', 't = T/2', 't = 3T/4', 't = T']
            if row_idx < len(fractions):
                time_label = fractions[row_idx]
        
        ax_gt.set_ylabel(time_label, fontsize=11, fontweight='bold')
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        
        # Add colorbar to first row
        if row_idx == 0:
            plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
        
        # Prediction
        ax_pred = fig.add_subplot(gs[row_idx, 1])
        im_pred = ax_pred.imshow(pred_np[frame_idx], cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        
        if row_idx == 0:
            ax_pred.set_title(col_titles[1], fontsize=13, fontweight='bold')
        
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        
        # Add metrics text if requested
        if show_metrics and metrics_text:
            ax_pred.text(0.02, 0.98, metrics_text[row_idx],
                        transform=ax_pred.transAxes,
                        fontsize=9,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add colorbar to first row
        if row_idx == 0:
            plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
        
        # Difference
        if show_difference:
            ax_diff = fig.add_subplot(gs[row_idx, 2])
            im_diff = ax_diff.imshow(diff_np[frame_idx], cmap='hot', vmin=0, vmax=diff_vmax, origin='lower')
            
            if row_idx == 0:
                ax_diff.set_title(col_titles[2], fontsize=13, fontweight='bold')
            
            ax_diff.set_xticks([])
            ax_diff.set_yticks([])
            
            # Add colorbar to first row
            if row_idx == 0:
                plt.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)
    
    # Overall title
    fig.suptitle(f'{field_label} - Keyframe Comparison', 
                 fontsize=15, fontweight='bold', y=0.995)
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Keyframe comparison saved to {save_path}")
    plt.close(fig)
    
    return fig


def plot_keyframe_comparison_multi_field(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    field_specs: Dict[str, int],
    save_dir: Union[str, Path],
    num_keyframes: int = 5,
    show_difference: bool = True,
    show_metrics: bool = True
) -> Dict[str, Path]:
    """
    Create keyframe comparisons for multiple fields.
    
    Args:
        prediction: Model prediction tensor, shape [T, C_total, H, W]
        ground_truth: Ground truth tensor, shape [T, C_total, H, W]
        field_specs: Dictionary mapping field names to channel counts
        save_dir: Directory where plots will be saved
        num_keyframes: Number of evenly-spaced frames to show
        show_difference: Whether to show difference column
        show_metrics: Whether to show error metrics
        
    Returns:
        Dictionary mapping field names to saved file paths
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    channel_idx = 0
    
    for field_name, num_channels in field_specs.items():
        print(f"\nCreating keyframe comparison for '{field_name}'...")
        
        # Extract field data
        pred_field = prediction[:, channel_idx:channel_idx+num_channels, :, :]
        gt_field = ground_truth[:, channel_idx:channel_idx+num_channels, :, :]
        
        # Create plot
        save_path = save_dir / f"{field_name}_keyframes.png"
        plot_keyframe_comparison(
            pred_field,
            gt_field,
            field_name,
            save_path,
            num_keyframes=num_keyframes,
            show_difference=show_difference,
            show_metrics=show_metrics
        )
        
        saved_paths[field_name] = save_path
        channel_idx += num_channels
    
    return saved_paths


def create_evaluation_summary(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    field_name: str,
    save_dir: Union[str, Path],
    metrics_to_compute: Optional[List[str]] = None,
    num_keyframes: int = 5,
    animation_fps: int = 10
) -> Dict[str, Path]:
    """
    Create a complete evaluation summary with all visualizations.
    
    This is a convenience function that generates:
    1. Side-by-side comparison animation (GIF)
    2. Error vs time plots
    3. Keyframe comparison
    4. Error heatmap (if multi-channel)
    
    Args:
        prediction: Model prediction tensor, shape [T, C, H, W]
        ground_truth: Ground truth tensor, shape [T, C, H, W]
        field_name: Name of the field
        save_dir: Directory where all outputs will be saved
        metrics_to_compute: List of metrics for error plots
        num_keyframes: Number of keyframes for comparison
        animation_fps: FPS for animation
        
    Returns:
        Dictionary mapping output types to file paths
        
    Example:
        >>> paths = create_evaluation_summary(pred, gt, 'velocity', 'results/')
        >>> # Creates: animation.gif, error_plot.png, keyframes.png, heatmap.png
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if metrics_to_compute is None:
        metrics_to_compute = ['mse', 'mae']
    
    saved_paths = {}
    
    print(f"\n{'='*60}")
    print(f"Creating evaluation summary for '{field_name}'")
    print(f"{'='*60}")
    
    # 1. Create animation
    print("\n[1/4] Creating comparison animation...")
    anim_path = save_dir / f"{field_name}_animation.gif"
    create_comparison_gif(
        prediction, ground_truth, field_name, anim_path, 
        fps=animation_fps, show_difference=True
    )
    saved_paths['animation'] = anim_path
    
    # 2. Create error plot
    print("\n[2/4] Creating error vs time plot...")
    error_path = save_dir / f"{field_name}_error_vs_time.png"
    plot_error_vs_time(
        prediction, ground_truth, field_name, error_path,
        metrics=metrics_to_compute
    )
    saved_paths['error_plot'] = error_path
    
    # 3. Create keyframe comparison
    print("\n[3/4] Creating keyframe comparison...")
    keyframe_path = save_dir / f"{field_name}_keyframes.png"
    plot_keyframe_comparison(
        prediction, ground_truth, field_name, keyframe_path,
        num_keyframes=num_keyframes, show_difference=True, show_metrics=True
    )
    saved_paths['keyframes'] = keyframe_path
    
    # 4. Create heatmap (if multi-channel)
    if prediction.shape[1] > 1:
        print("\n[4/4] Creating error heatmap...")
        heatmap_path = save_dir / f"{field_name}_error_heatmap.png"
        plot_error_heatmap(
            prediction, ground_truth, field_name, heatmap_path
        )
        saved_paths['heatmap'] = heatmap_path
    else:
        print("\n[4/4] Skipping heatmap (single channel field)")
    
    print(f"\n{'='*60}")
    print(f"Evaluation summary complete!")
    print(f"{'='*60}")
    print(f"\nGenerated files:")
    for output_type, path in saved_paths.items():
        print(f"  - {output_type}: {path}")
    
    return saved_paths
