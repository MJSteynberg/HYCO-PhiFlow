# src/models/physical/advection.py

from typing import Dict
import numpy as np

# --- PhiFlow Imports ---
from phi.torch.flow import *
from phi.math import Shape, Tensor, batch, math

# --- Repo Imports ---
from .base import PhysicalModel
from src.models import ModelRegistry


@jit_compile
def _advection_step(
    density: CenteredGrid, velocity: CenteredGrid, advection_coeff: Tensor, dt: float
) -> CenteredGrid:
    """
    Performs one step of pure advection using semi-Lagrangian method.

    Args:
        density (CenteredGrid): The current density field.
        velocity (CenteredGrid): The prescribed velocity field (vector-valued).
        advection_coeff (Tensor): Coefficient to scale the velocity field.
        dt (float): The time step.

    Returns:
        CenteredGrid: The density field at the next time step.
    """
    # Scale velocity by advection coefficient
    velocity = velocity * advection_coeff
    return advect.semi_lagrangian(density, velocity, dt=dt), velocity, advection_coeff


@ModelRegistry.register_physical("AdvectionModel")
class AdvectionModel(PhysicalModel):
    """
    Physical model for pure advection with a donut shaped velocity field.
    The density is transported by a donut shaped velocity field,
    scaled by a learnable advection coefficient.
    """

    def __init__(self, config: dict):
        """Initialize the advection model."""
        super().__init__(config)

    def get_initial_state(self, batch_size: int = 1) -> Dict[str, Field]:
        """
        Returns a batched initial state with density and static velocity field.

        The velocity field is created once here and will be passed through
        the state dictionary to each step.
        """
        # Create a batch shape
        b = batch(batch=batch_size)

        # Create a nice swirling/rotating velocity field
        def velocity_fn(x, y):
            # Vortex-like pattern: velocity vector depends on position
            center_x = self.domain.size[0] / 2
            center_y = self.domain.size[1] / 2
            dy = y - center_y
            dx = x - center_x
            r = math.sqrt(dx**2 + dy**2 + 1e-6)

            # Circular flow with some variation
            vx = -dy * math.exp(
                -(r**2) / (0.2 * self.domain.size[0]) ** 2
            ) + 0.2 * math.sin(2 * math.pi * y / self.domain.size[1])
            vy = dx * math.exp(
                -(r**2) / (0.2 * self.domain.size[0]) ** 2
            ) + 0.2 * math.cos(2 * math.pi * x / self.domain.size[0])

            return math.stack([vx, vy], channel("vector"))

        # Random line through two random points
        x1 = np.random.uniform(0.2 * self.domain.size[0], 0.8 * self.domain.size[0])
        y1 = np.random.uniform(0.2 * self.domain.size[1], 0.8 * self.domain.size[1])
        x2 = np.random.uniform(0.2 * self.domain.size[0], 0.8 * self.domain.size[0])
        y2 = np.random.uniform(0.2 * self.domain.size[1], 0.8 * self.domain.size[1])

        # Line direction vector
        dx_line = x2 - x1
        dy_line = y2 - y1
        line_length = np.sqrt(dx_line**2 + dy_line**2)

        # Normal vector to the line (perpendicular)
        nx = -dy_line / line_length
        ny = dx_line / line_length

        def density_fn(x, y):
            # Signed distance from point (x,y) to the line
            signed_distance = (x - x1) * nx + (y - y1) * ny
            steepness = 10.0 / max(self.domain.size[0], self.domain.size[1])
            return math.tanh(steepness * signed_distance)

        # Create CenteredGrid for velocity with vector values
        velocity_0 = CenteredGrid(
            velocity_fn,
            extrapolation=extrapolation.PERIODIC,
            x=self.resolution.get_size("x"),
            y=self.resolution.get_size("y"),
            bounds=self.domain,
        )
        velocity_0 = math.expand(velocity_0, b)

        # Create density field with smooth tanh transition
        density_0 = CenteredGrid(
            density_fn,
            extrapolation=extrapolation.ZERO_GRADIENT,
            x=self.resolution.get_size("x"),
            y=self.resolution.get_size("y"),
            bounds=self.domain,
        )
        density_0 = math.expand(density_0, b)

        return {"density": density_0, "velocity": velocity_0}

    def get_random_state(self, batch_size: int = 1) -> Dict[str, Field]:
        """
        Returns a batched initial state with density and static velocity field.

        The velocity field is created once here and will be passed through
        the state dictionary to each step (like smoke's inflow pattern).
        """
        b = batch(batch=batch_size)
        # Create a nice swirling/rotating velocity field
        def velocity_fn(x, y):
            # Vortex-like pattern: velocity vector depends on position
            center_x = self.domain.size[0] / 2
            center_y = self.domain.size[1] / 2
            dy = y - center_y
            dx = x - center_x
            r = math.sqrt(dx**2 + dy**2 + 1e-6)

            # Circular flow with some variation
            vx = -dy * math.exp(
                -(r**2) / (0.2 * self.domain.size[0]) ** 2
            ) + 0.2 * math.sin(2 * math.pi * y / self.domain.size[1])
            vy = dx * math.exp(
                -(r**2) / (0.2 * self.domain.size[0]) ** 2
            ) + 0.2 * math.cos(2 * math.pi * x / self.domain.size[0])

            return math.stack([vx, vy], channel("vector"))

        # Create CenteredGrid for velocity with vector values
        velocity_0 = CenteredGrid(
            velocity_fn,
            extrapolation=extrapolation.PERIODIC,
            x=self.resolution.get_size("x"),
            y=self.resolution.get_size("y"),
            bounds=self.domain,
        )
        velocity_0 = math.expand(velocity_0, b)
        # Create density field with smooth tanh transition
        scale = math.random_uniform(low=1, high=10)
        smoothness = math.random_uniform(low=1.0, high=5.0)
        density_0 = CenteredGrid(
            Noise(scale=scale, smoothness=smoothness),
            extrapolation=extrapolation.ZERO_GRADIENT,
            x=self.resolution.get_size("x"),
            y=self.resolution.get_size("y"),
            bounds=self.domain,
        )
        density_0 = math.tanh(2.0 * density_0)
        density_0 = math.expand(density_0, b)

        return {"density": density_0, "velocity": velocity_0}
    
    def rollout(self, initial_state, num_steps):
        """
        Perform multiple simulation steps starting from the initial state.

        Args:
            initial_state: Dictionary containing initial 'density' and 'velocity' fields.
            num_steps: Number of simulation steps to perform.
        Returns:
            List of states at each timestep.
        """
        density_trj, velocity_trj, _ = iterate(_advection_step, batch(time=num_steps), initial_state["density"], initial_state["velocity"], self.advection_coeff, dt = self.dt)
        return {"density": density_trj, "velocity": velocity_trj}

    def forward(self, current_state: Dict[str, Field]) -> Dict[str, Field]:
        """
        Performs a single simulation step using pure advection.

        Args:
            current_state: Dictionary containing 'density' and 'velocity' fields.

        Returns:
            Dictionary with updated 'density' and unchanged 'velocity'.
        """
        new_density, new_velocity, _ = _advection_step(
            density=current_state["density"], 
            velocity=current_state["velocity"],
            advection_coeff=self.advection_coeff,
            dt=self.dt,
        )
        # Velocity field remains static (like smoke's inflow)
        return {"density": new_density, "velocity": new_velocity}
