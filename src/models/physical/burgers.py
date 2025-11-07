# src/models/physical/burgers.py

from typing import Dict
import numpy as np

# --- PhiFlow Imports ---
from phi.torch.flow import *
from phi.math import Shape, Tensor, batch, math

# --- Repo Imports ---
from .base import PhysicalModel
from src.models.registry import ModelRegistry


@jit_compile
def _burgers_physics_step(
    velocity: CenteredGrid, dt: float, nu: Tensor
) -> CenteredGrid:
    """
    Performs one physics-based Burgers' equation step.

    Args:
        velocity (CenteredGrid): Current velocity field.
        dt (float): Time step.
        nu (Tensor): Viscosity parameter.

    Returns:
        CenteredGrid: new_velocity
    """
    # Advect velocity (self-advection: u * grad(u))
    velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)

    # Diffuse velocity (viscosity: nu * laplace(u))
    velocity = diffuse.explicit(velocity, nu, dt=dt)

    return velocity


# --- Model Class Implementation ---


@ModelRegistry.register_physical("BurgersModel")
class BurgersModel(PhysicalModel):
    """
    Physical model for the Burgers' equation.
    Implements the PhysicalModel interface.
    """

    # Declare PDE-specific parameters
    PDE_PARAMETERS = {
        "nu": {
            "type": float,
            "default": 0.01,
        }
    }

    def __init__(self, config: dict):
        """Initialize the Burgers model."""
        super().__init__(config)

    def get_initial_state(self) -> Dict[str, Field]:
        """
        Returns an initial state of (noisy velocity).
        We use periodic boundaries as they are common for Burgers.
        """
        b = batch(batch=self.batch_size)

        temp = StaggeredGrid(
            Noise(scale=20),  # Initialize with noise
            extrapolation.PERIODIC,  # Use periodic boundaries
            x=self.resolution.get_size("x"),
            y=self.resolution.get_size("y"),
            bounds=self.domain,
        )

        velocity_0 = CenteredGrid(
            temp,
            extrapolation.PERIODIC,  # Use periodic boundaries
            x=self.resolution.get_size("x"),
            y=self.resolution.get_size("y"),
            bounds=self.domain,
        )
        velocity_0 = math.expand(velocity_0, b)
        return {"velocity": velocity_0}

    def get_random_state(self) -> Dict[str, Field]:
        """
        Returns a random initial state of (noisy velocity).
        We use periodic boundaries as they are common for Burgers.
        """
        temp = StaggeredGrid(
            Noise(scale=20),  # Initialize with noise
            extrapolation.PERIODIC,  # Use periodic boundaries
            x=self.resolution.get_size("x"),
            y=self.resolution.get_size("y"),
            bounds=self.domain,
        )

        velocity_0 = CenteredGrid(
            temp,
            extrapolation.PERIODIC,  # Use periodic boundaries
            x=self.resolution.get_size("x"),
            y=self.resolution.get_size("y"),
            bounds=self.domain,
        )
        return {"velocity": velocity_0}

    def forward(self, current_state: Dict[str, Field]) -> Dict[str, Field]:
        """
        Performs a single simulation step.
        """
        new_velocity = _burgers_physics_step(
            velocity=current_state["velocity"], dt=self.dt, nu=self.nu
        )
        return {"velocity": new_velocity}
