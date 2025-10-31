import torch
from phi.torch.flow import *
from phi.math import jit_compile, batch

from .base import PhysicalModel  # <-- Assuming this base class exists
from typing import Dict

# --- JIT-Compiled Physics Function ---
@jit_compile
def _burgers_physics_step(velocity: StaggeredGrid, dt: float, nu: Tensor) -> StaggeredGrid:
    """
    Performs one physics-based Burgers' equation step.

    Args:
        velocity (StaggeredGrid): Current velocity field.
        dt (float): Time step.
        nu (float): Viscosity.

    Returns:
        StaggeredGrid: new_velocity
    """

    velocity = diffuse.explicit(u=velocity, diffusivity=nu, dt=dt)
    # Advect velocity (self-advection: u * grad(u))
    velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)
    
    # Diffuse velocity (viscosity: nu * laplace(u))
    
    
    return velocity

# --- Model Class Implementation ---

class BurgersModel(PhysicalModel):
    """
    Physical model for the Burgers' equation.
    Implements the PhysicalModel interface.
    """
    
    # Declare PDE-specific parameters
    PDE_PARAMETERS = {
        'nu': {
            'type': float,
            'default': 0.01,
        }
    }

    def get_initial_state(self) -> Dict[str, Field]:
        """
        Returns an initial state of (noisy velocity).
        We use periodic boundaries as they are common for Burgers.
        """
        b = batch(batch=self.batch_size)

        velocity_0 = StaggeredGrid(
            Noise(scale=20), # Initialize with noise
            extrapolation.PERIODIC,    # Use periodic boundaries
            x=self.resolution.get_size('x'),
            y=self.resolution.get_size('y'),
            bounds=self.domain,
        )
        velocity_0 = math.expand(velocity_0, b)

        return {"velocity": velocity_0}

    def step(self, current_state: Dict[str, Field]) -> Dict[str, Field]:
        """
        Performs a single simulation step.
        """
        new_velocity = _burgers_physics_step(
            velocity=current_state["velocity"],
            dt=self.dt,
            nu=self.nu
        )
        return {"velocity": new_velocity}