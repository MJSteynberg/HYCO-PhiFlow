"""
Visualize the potential field and its gradient (forcing) for the inviscid Burgers model.
"""

import numpy as np
import matplotlib.pyplot as plt
from phi.torch.flow import *
from phi.math import math
import yaml
import sys


def visualize_potential_and_forcing(config_path: str):
    """
    Load config and visualize the potential field φ(x,y) and its gradient -∇φ.

    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    pde_params = config['model']['physical']['pde_params']
    domain_config = config['model']['physical']['domain']['dimensions']
    potential_expr = pde_params['value']

    # Get domain information
    if 'x' in domain_config and 'y' in domain_config:
        # 2D case
        size_x = domain_config['x']['size']
        size_y = domain_config['y']['size']
        res_x = domain_config['x']['resolution']
        res_y = domain_config['y']['resolution']

        # Create coordinate grids
        x_coords = np.linspace(0, size_x, res_x)
        y_coords = np.linspace(0, size_y, res_y)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

        # Evaluate potential using PhiFlow
        def potential_fn(location):
            local_vars = {
                'math': math,
                'size_x': float(size_x),
                'size_y': float(size_y),
                'x': location.vector['x'],
                'y': location.vector['y']
            }
            result = eval(potential_expr, local_vars)
            return result

        # Create potential grid
        potential_grid = CenteredGrid(
            potential_fn,
            extrapolation.PERIODIC,
            bounds=Box(x=size_x, y=size_y),
            x=res_x, y=res_y
        )

        # Compute gradient
        grad_potential = potential_grid.gradient(boundary=extrapolation.PERIODIC)

        # Extract numpy arrays
        potential_values = potential_grid.values.numpy('y,x')
        grad_x = grad_potential.values.vector['x'].numpy('y,x')
        grad_y = grad_potential.values.vector['y'].numpy('y,x')

        # Forcing is negative gradient
        forcing_x = -grad_x
        forcing_y = -grad_y

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Potential field as heatmap
        ax = axes[0, 0]
        im1 = ax.contourf(X, Y, potential_values, levels=20, cmap='viridis')
        ax.contour(X, Y, potential_values, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        plt.colorbar(im1, ax=ax, label='φ(x,y)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Potential Field φ(x,y)')
        ax.set_aspect('equal')

        # 2. Gradient magnitude
        ax = axes[0, 1]
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        im2 = ax.contourf(X, Y, grad_magnitude, levels=20, cmap='plasma')
        plt.colorbar(im2, ax=ax, label='|∇φ|')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Gradient Magnitude |∇φ|')
        ax.set_aspect('equal')

        # 3. Forcing vector field (quiver plot)
        ax = axes[1, 0]
        # Downsample for clearer visualization
        skip = max(res_x // 20, 1)
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                  forcing_x[::skip, ::skip], forcing_y[::skip, ::skip],
                  color='blue', alpha=0.7, scale=None, scale_units='xy')
        ax.contourf(X, Y, potential_values, levels=20, cmap='viridis', alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Forcing Field f = -∇φ (arrows show direction/magnitude)')
        ax.set_aspect('equal')

        # 4. Forcing magnitude with streamlines
        ax = axes[1, 1]
        forcing_magnitude = np.sqrt(forcing_x**2 + forcing_y**2)
        im4 = ax.contourf(X, Y, forcing_magnitude, levels=20, cmap='inferno')
        # Add streamlines to show flow pattern
        ax.streamplot(x_coords, y_coords, forcing_x.T, forcing_y.T,
                     color='white', density=1.5, linewidth=1, arrowsize=1.5)
        plt.colorbar(im4, ax=ax, label='|f|')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Forcing Magnitude with Streamlines')
        ax.set_aspect('equal')

        plt.tight_layout()

        # Save figure
        output_path = config_path.replace('.yaml', '_potential_visualization.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")

        plt.show()

        # Print statistics
        print("\n=== Potential and Forcing Statistics ===")
        print(f"Potential φ(x,y):")
        print(f"  Min: {potential_values.min():.4f}")
        print(f"  Max: {potential_values.max():.4f}")
        print(f"  Mean: {potential_values.mean():.4f}")
        print(f"\nGradient ∇φ:")
        print(f"  Magnitude range: [{grad_magnitude.min():.4f}, {grad_magnitude.max():.4f}]")
        print(f"\nForcing f = -∇φ:")
        print(f"  Magnitude range: [{forcing_magnitude.min():.4f}, {forcing_magnitude.max():.4f}]")
        print(f"  Mean magnitude: {forcing_magnitude.mean():.4f}")

    elif 'x' in domain_config and 'y' not in domain_config:
        # 1D case
        size_x = domain_config['x']['size']
        res_x = domain_config['x']['resolution']

        # Create coordinate array
        x_coords = np.linspace(0, size_x, res_x)

        # Evaluate potential using PhiFlow
        def potential_fn(location):
            local_vars = {
                'math': math,
                'size_x': float(size_x),
                'x': location
            }
            result = eval(potential_expr, local_vars)
            return result

        # Create potential grid
        potential_grid = CenteredGrid(
            potential_fn,
            extrapolation.PERIODIC,
            bounds=Box(x=size_x),
            x=res_x
        )

        # Compute gradient
        grad_potential = potential_grid.gradient(boundary=extrapolation.PERIODIC)

        # Extract numpy arrays
        potential_values = potential_grid.values.numpy('x')
        grad_x = grad_potential.values.vector['x'].numpy('x')

        # Forcing is negative gradient
        forcing_x = -grad_x

        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # 1. Potential field
        ax = axes[0]
        ax.plot(x_coords, potential_values, 'b-', linewidth=2, label='φ(x)')
        ax.fill_between(x_coords, potential_values, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('φ(x)')
        ax.set_title('Potential Field φ(x)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 2. Gradient
        ax = axes[1]
        ax.plot(x_coords, grad_x, 'r-', linewidth=2, label='∇φ')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('∇φ')
        ax.set_title('Gradient ∇φ(x)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 3. Forcing
        ax = axes[2]
        ax.plot(x_coords, forcing_x, 'g-', linewidth=2, label='f = -∇φ')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        # Add arrows to show forcing direction
        arrow_skip = max(res_x // 20, 1)
        for i in range(0, res_x, arrow_skip):
            ax.arrow(x_coords[i], 0, 0, forcing_x[i] * 0.8,
                    head_width=size_x/50, head_length=abs(forcing_x[i])*0.15,
                    fc='green', ec='green', alpha=0.6)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Forcing Field f(x) = -∇φ(x)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        # Save figure
        output_path = config_path.replace('.yaml', '_potential_visualization.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")

        plt.show()

        # Print statistics
        print("\n=== Potential and Forcing Statistics ===")
        print(f"Potential φ(x):")
        print(f"  Min: {potential_values.min():.4f}")
        print(f"  Max: {potential_values.max():.4f}")
        print(f"  Mean: {potential_values.mean():.4f}")
        print(f"\nGradient ∇φ:")
        print(f"  Range: [{grad_x.min():.4f}, {grad_x.max():.4f}]")
        print(f"\nForcing f = -∇φ:")
        print(f"  Range: [{forcing_x.min():.4f}, {forcing_x.max():.4f}]")
        print(f"  Mean: {forcing_x.mean():.4f}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_potential.py <config_path>")
        print("\nExamples:")
        print("  python visualize_potential.py conf/inviscid_burgers_1d.yaml")
        print("  python visualize_potential.py conf/inviscid_burgers_2d.yaml")
        sys.exit(1)

    config_path = sys.argv[1]
    visualize_potential_and_forcing(config_path)
