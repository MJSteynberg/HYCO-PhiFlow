"""Visualization utilities for observation sparsity."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Optional, List, Tuple
from phi.flow import *

from src.data.sparsity import TemporalMask, SpatialMask, SparsityConfig, ObservationMask


def plot_temporal_mask(
    temporal_mask: TemporalMask,
    trajectory: Optional[Tensor] = None,
    field_idx: int = 0,
    spatial_idx: int = None,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize temporal sparsity mask.

    Args:
        temporal_mask: TemporalMask instance
        trajectory: Optional trajectory tensor to show actual values
        field_idx: Which field to plot if trajectory provided
        spatial_idx: Which spatial point to plot (middle if None)
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 3])

    T = temporal_mask.trajectory_length
    visible = temporal_mask.visible_indices

    # Top plot: Binary mask
    ax1 = axes[0]
    mask_array = np.zeros(T)
    mask_array[visible] = 1.0

    ax1.bar(range(T), mask_array, color='steelblue', alpha=0.7, width=1.0)
    ax1.set_xlim(-0.5, T - 0.5)
    ax1.set_ylim(0, 1.2)
    ax1.set_ylabel('Visible')
    ax1.set_title(f'Temporal Mask ({len(visible)}/{T} timesteps visible)')
    ax1.set_xticks([])

    # Bottom plot: Trajectory with mask overlay
    ax2 = axes[1]

    if trajectory is not None:
        # Extract 1D signal from trajectory
        traj_np = trajectory.numpy(trajectory.shape.names)

        # Handle different shapes
        if 'field' in trajectory.shape.names:
            traj_np = traj_np[..., field_idx]

        # If spatial dimensions exist, pick a point
        spatial_dims = [d for d in trajectory.shape.names if d in ['x', 'y', 'z']]
        for dim in spatial_dims:
            dim_size = trajectory.shape[dim].size
            idx = spatial_idx if spatial_idx is not None else dim_size // 2
            # This is a simplification - proper indexing depends on dim order
            traj_np = np.take(traj_np, idx, axis=trajectory.shape.names.index(dim))

        # Ensure we have time dimension
        if len(traj_np.shape) == 0:
            traj_np = np.array([float(traj_np)])

        times = np.arange(len(traj_np))

        # Plot full trajectory (faded)
        ax2.plot(times, traj_np, 'k-', alpha=0.3, label='Hidden')

        # Plot visible points
        visible_times = [t for t in visible if t < len(traj_np)]
        visible_values = [traj_np[t] for t in visible_times]
        ax2.scatter(visible_times, visible_values, c='steelblue', s=50,
                   zorder=5, label='Visible')
        ax2.plot(visible_times, visible_values, 'b-', alpha=0.7)

        ax2.legend()
    else:
        # Just show the mask pattern
        for t in visible:
            ax2.axvline(t, color='steelblue', alpha=0.3)
        ax2.set_ylabel('(No trajectory data)')

    ax2.set_xlabel('Time')
    ax2.set_xlim(-0.5, T - 0.5)
    ax2.set_title('Trajectory with Visible Timesteps')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_spatial_mask(
    spatial_mask: SpatialMask,
    field: Optional[Tensor] = None,
    field_idx: int = 0,
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize spatial sparsity mask.

    Args:
        spatial_mask: SpatialMask instance
        field: Optional field tensor to show with mask overlay
        field_idx: Which field channel to plot
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure
    """
    mask = spatial_mask.mask
    spatial_names = mask.shape.names

    if len(spatial_names) == 1:
        # 1D case
        return _plot_spatial_mask_1d(spatial_mask, field, field_idx, figsize, save_path)
    else:
        # 2D case
        return _plot_spatial_mask_2d(spatial_mask, field, field_idx, figsize, save_path)


def _plot_spatial_mask_1d(
    spatial_mask: SpatialMask,
    field: Optional[Tensor],
    field_idx: int,
    figsize: Tuple[int, int],
    save_path: Optional[str]
) -> plt.Figure:
    """Plot 1D spatial mask."""
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    mask_np = spatial_mask.mask.numpy(spatial_mask.mask.shape.names)
    x = np.arange(len(mask_np))

    # Top: Mask
    ax1 = axes[0]
    ax1.fill_between(x, 0, mask_np, alpha=0.3, color='steelblue', label='Visible region')
    ax1.set_ylabel('Mask')
    ax1.set_ylim(0, 1.2)
    ax1.set_title(f'Spatial Mask ({spatial_mask.visible_fraction:.1%} visible)')
    ax1.legend()

    # Bottom: Field with mask
    ax2 = axes[1]
    if field is not None:
        field_np = field.numpy(field.shape.names)
        if 'field' in field.shape.names:
            field_np = field_np[..., field_idx]

        # Plot hidden (faded)
        ax2.plot(x, field_np, 'k-', alpha=0.3, label='Hidden')

        # Plot visible
        visible_field = field_np * mask_np
        visible_x = x[mask_np > 0.5]
        visible_y = field_np[mask_np > 0.5]
        ax2.scatter(visible_x, visible_y, c='steelblue', s=20, zorder=5, label='Visible')

        ax2.legend()

    ax2.set_xlabel('x')
    ax2.set_ylabel('Field value')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def _plot_spatial_mask_2d(
    spatial_mask: SpatialMask,
    field: Optional[Tensor],
    field_idx: int,
    figsize: Tuple[int, int],
    save_path: Optional[str]
) -> plt.Figure:
    """Plot 2D spatial mask."""
    fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 1.5, figsize[1]))

    mask_np = spatial_mask.mask.numpy(['y', 'x'])

    # Left: Mask alone
    ax1 = axes[0]
    im1 = ax1.imshow(mask_np, cmap='Blues', aspect='equal', origin='lower')
    ax1.set_title(f'Spatial Mask\n({spatial_mask.visible_fraction:.1%} visible)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1, label='Visible')

    if field is not None:
        field_np = field.numpy(['y', 'x', 'field'] if 'field' in field.shape.names else ['y', 'x'])
        if len(field_np.shape) == 3:
            field_np = field_np[..., field_idx]

        # Middle: Original field
        ax2 = axes[1]
        im2 = ax2.imshow(field_np, cmap='viridis', aspect='equal', origin='lower')
        ax2.set_title('Original Field')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(im2, ax=ax2, label='Value')

        # Right: Masked field
        ax3 = axes[2]
        masked_field = field_np * mask_np
        masked_field[mask_np < 0.5] = np.nan  # Make hidden region transparent
        im3 = ax3.imshow(masked_field, cmap='viridis', aspect='equal', origin='lower')
        ax3.set_title('Visible Region Only')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        plt.colorbar(im3, ax=ax3, label='Value')
    else:
        axes[1].set_visible(False)
        axes[2].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_observation_summary(
    observation_mask: ObservationMask,
    trajectory: Optional[Tensor] = None,
    time_idx: int = 0,
    field_idx: int = 0,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive visualization of both temporal and spatial sparsity.

    Args:
        observation_mask: Combined ObservationMask instance
        trajectory: Optional full trajectory tensor (time, x, [y], field)
        time_idx: Which timestep to visualize for spatial mask
        field_idx: Which field to visualize
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)

    # Create grid for subplots
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])

    # Top-left: Temporal mask summary
    ax_time = fig.add_subplot(gs[0, 0])
    T = observation_mask.temporal_mask.trajectory_length
    visible = observation_mask.temporal_mask.visible_indices

    mask_array = np.zeros(T)
    mask_array[visible] = 1.0
    ax_time.bar(range(T), mask_array, color='steelblue', alpha=0.7, width=1.0)
    ax_time.set_xlim(-0.5, T - 0.5)
    ax_time.set_ylim(0, 1.2)
    ax_time.set_xlabel('Time')
    ax_time.set_ylabel('Visible')
    ax_time.set_title(f'Temporal: {len(visible)}/{T} timesteps')

    # Top-right: Spatial mask summary (show mask only)
    ax_space = fig.add_subplot(gs[0, 1])
    mask = observation_mask.spatial_mask.mask

    if len(mask.shape.names) == 1:
        mask_np = mask.numpy(mask.shape.names)
        ax_space.fill_between(range(len(mask_np)), 0, mask_np, alpha=0.5, color='steelblue')
        ax_space.set_ylim(0, 1.2)
    else:
        mask_np = mask.numpy(['y', 'x'])
        ax_space.imshow(mask_np, cmap='Blues', aspect='equal', origin='lower')

    ax_space.set_title(f'Spatial: {observation_mask.spatial_mask.visible_fraction:.1%} visible')

    # Bottom: Combined visualization with trajectory
    ax_combined = fig.add_subplot(gs[1, :])

    if trajectory is not None:
        # Create a heatmap-style visualization
        # Extract field data
        traj_np = trajectory.numpy(trajectory.shape.names)

        # Handle field dimension
        if 'field' in trajectory.shape.names:
            field_axis = trajectory.shape.names.index('field')
            traj_np = np.take(traj_np, field_idx, axis=field_axis)

        # For 1D spatial, create time-space heatmap
        spatial_names = [n for n in trajectory.shape.names if n in ['x', 'y', 'z']]

        if len(spatial_names) == 1:
            # Transpose to (time, space)
            time_axis = trajectory.shape.names.index('time')
            space_axis = trajectory.shape.names.index(spatial_names[0])

            if time_axis > space_axis:
                traj_np = traj_np.T

            # Create mask overlay
            full_mask = np.outer(mask_array, mask.numpy(mask.shape.names))

            im = ax_combined.imshow(traj_np, aspect='auto', cmap='viridis', origin='lower')
            ax_combined.contour(full_mask, levels=[0.5], colors='red', linewidths=2)

            ax_combined.set_xlabel('Space (x)')
            ax_combined.set_ylabel('Time')
            ax_combined.set_title('Trajectory with Observation Mask (red contour = boundary)')
            plt.colorbar(im, ax=ax_combined, label='Field value')

        else:
            # For 2D spatial, show a single timestep
            ax_combined.text(0.5, 0.5, f'2D spatial: showing timestep {time_idx}',
                           transform=ax_combined.transAxes, ha='center')
    else:
        ax_combined.text(0.5, 0.5, 'No trajectory data provided',
                        transform=ax_combined.transAxes, ha='center', fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_sparsity_report(
    config: SparsityConfig,
    trajectory_length: int,
    spatial_shape,
    output_dir: str,
    sample_trajectory: Optional[Tensor] = None
):
    """
    Generate a complete sparsity visualization report.

    Args:
        config: Sparsity configuration
        trajectory_length: Number of timesteps
        spatial_shape: Spatial dimensions shape
        output_dir: Directory to save visualizations
        sample_trajectory: Optional sample trajectory for context
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    observation_mask = ObservationMask(config, trajectory_length, spatial_shape)

    # Print text summary
    print(observation_mask.summary)

    # Save summary to file
    with open(os.path.join(output_dir, 'sparsity_summary.txt'), 'w') as f:
        f.write(observation_mask.summary)

    # Generate visualizations
    if config.temporal.enabled:
        fig = plot_temporal_mask(
            observation_mask.temporal_mask,
            trajectory=sample_trajectory,
            save_path=os.path.join(output_dir, 'temporal_mask.png')
        )
        plt.close(fig)

    if config.spatial.enabled:
        field = sample_trajectory.time[0] if sample_trajectory is not None else None
        fig = plot_spatial_mask(
            observation_mask.spatial_mask,
            field=field,
            save_path=os.path.join(output_dir, 'spatial_mask.png')
        )
        plt.close(fig)

    # Combined summary
    fig = plot_observation_summary(
        observation_mask,
        trajectory=sample_trajectory,
        save_path=os.path.join(output_dir, 'observation_summary.png')
    )
    plt.close(fig)

    print(f"Sparsity report saved to {output_dir}")
