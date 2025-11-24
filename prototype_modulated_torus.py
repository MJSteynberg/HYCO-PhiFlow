
import matplotlib.pyplot as plt
import numpy as np
import torch
from phi.torch.flow import *
from phi.math import math, channel

def create_modulated_torus():
    # Domain setup (matching advection.yaml)
    resolution = dict(x=128, y=128)
    domain_size = dict(x=100, y=100)
    bounds = Box(x=100, y=100)
    
    # Ring parameters
    center_pos = math.tensor([50.0, 50.0], channel(vector='x,y'))
    ring_radius = 30.0
    ring_width = 15.0
    
    # Modulation parameters
    num_lobes = 6
    modulation_amplitude = 0.5
    
    def modulated_velocity(location):
        # Calculate vector from center
        diff = location - center_pos
        
        # Calculate distance (r) and angle (theta)
        r = math.vec_length(diff)
        theta = math.arctan(diff.vector['y'], diff.vector['x'])
        
        # Ring falloff (Gaussian)
        falloff = math.exp(-((r - ring_radius) / ring_width) ** 2)
        
        # Sinusoidal modulation around the ring
        # 1.0 + amp * sin(n * theta)
        modulation = 1.0 + modulation_amplitude * math.sin(num_lobes * theta)
        
        # Total magnitude
        magnitude = falloff * modulation
        
        # Direction: Tangent to the circle (-y, x)
        # Normalized tangent vector
        tangent_x = -diff.vector['y'] / (r + 1e-6)
        tangent_y = diff.vector['x'] / (r + 1e-6)
        
        # Combine
        vel_x = magnitude * tangent_x
        vel_y = magnitude * tangent_y
        
        return math.stack([vel_x, vel_y], channel(vector='x,y'))

    # Create grid
    velocity_grid = CenteredGrid(
        modulated_velocity,
        extrapolation.PERIODIC,
        bounds=bounds,
        **resolution
    )
    
    return velocity_grid

def plot_velocity_field(grid):
    # Extract magnitude for heatmap
    velocity = grid.values
    magnitude = math.vec_length(velocity).numpy('y,x')
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(magnitude, origin='lower', extent=[0, 100, 0, 100], cmap='jet')
    plt.colorbar(label='Velocity Magnitude')
    plt.title('Modulated Torus Velocity Field')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Save
    output_path = 'modulated_torus_prototype.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    grid = create_modulated_torus()
    plot_velocity_field(grid)
