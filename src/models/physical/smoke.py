# src/models/physical/smoke_model.py

from typing import Dict, Any
import numpy as np
import random

# --- PhiFlow Imports ---
from phi.torch.flow import *
from phi.math import Shape, Tensor, batch, math

# --- Repo Imports ---
from .base import PhysicalModel
from src.models.registry import ModelRegistry


# --- JIT-Compiled Physics Function ---
@jit_compile
def _smoke_physics_step(velocity: CenteredGrid, density: CenteredGrid, inflow: CenteredGrid, domain: Box, dt: float, buoyancy_factor: float, nu: float) -> tuple[Field, Field]:
    """
    Performs one physics-based smoke simulation step.

    Args:
        velocity (CenteredGrid): Current velocity field.
        density (CenteredGrid): Current density field.
        inflow (CenteredGrid): Inflow mask.
        domain (Box): Simulation domain.
        dt (float): Time step.
        buoyancy_factor (float): Strength of buoyancy.
        nu (float): Viscosity.

    Returns:
        tuple: (new_velocity, new_density)
    """
    # Advect density and add inflow
    density = advect.mac_cormack(density, velocity, dt=dt) + dt * inflow

    # Apply forces
    buoyancy_force = (density * (0, buoyancy_factor)).at(velocity)
    velocity = velocity + dt * buoyancy_force

    # Advect velocity
    velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)

    velocity = diffuse.explicit(velocity, nu, dt=dt)

    # Make incompressible
    velocity, pressure = fluid.make_incompressible(
        velocity,
        solve=Solve("CG", 1e-3, rank_deficiency=0, suppress=[phi.math.NotConverged]),
    )

    return velocity, density


# --- Model Class Implementation ---


@ModelRegistry.register_physical("SmokeModel")
class SmokeModel(PhysicalModel):
    """
    Physical model for the smoke simulation.
    Implements the PhysicalModel interface.
    """

    # Declare PDE-specific parameters
    PDE_PARAMETERS = {
        "nu": {
            "type": float,
            "default": 0.0,
        },
        "buoyancy": {
            "type": float,
            "default": 1.0,
        },
        "inflow_radius": {
            "type": float,
            "default": 10.0,
        },
        "inflow_rate": {
            "type": float,
            "default": 0.1,
        },
        "inflow_rand_x_range": {
            "type": list,
            "default": [0.2, 0.8],
        },
        "inflow_rand_y_range": {
            "type": list,
            "default": [0.15, 0.25],
        },
    }

    def __init__(self, config: dict):
        """
        Initializes the smoke model.

        Handles special inflow center logic after base initialization.
        """
        # Call parent init to handle standard parameters
        super().__init__(config)

    def get_initial_state(self) -> Dict[str, Field]:
        """
        Returns an initial state of (zero velocity, zero density).
        """
        # Generate random inflow position within specified ranges
        inflow_center = self._get_inflow_center()

        b = batch(batch=self.batch_size)

        velocity_0 = CenteredGrid(
            (0, 0),
            extrapolation.ZERO,
            x=self.resolution.get_size("x"),
            y=self.resolution.get_size("y"),
            bounds=self.domain,
        )
        velocity_0 = math.expand(velocity_0, b)

        density_0 = CenteredGrid(
            0,
            extrapolation.BOUNDARY,
            x=self.resolution.get_size("x"),
            y=self.resolution.get_size("y"),
            bounds=self.domain,
        )
        density_0 = math.expand(density_0, b)

        inflow_shape = Sphere(center=inflow_center, radius=self.inflow_radius)
        inflow_0 = self.inflow_rate * CenteredGrid(
            inflow_shape,
            extrapolation.BOUNDARY,
            x=self.resolution.get_size("x"),
            y=self.resolution.get_size("y"),
            bounds=self.domain,
        )
        inflow_0 = math.expand(inflow_0, b)

        return {"velocity": velocity_0, "density": density_0, "inflow": inflow_0}

    def get_random_state(self) -> Dict[str, Field]:
        """
        Returns an initial state of (zero velocity, zero density).
        """
        # Generate random inflow position within specified ranges
        inflow_center = self._get_inflow_center()

        velocity_0 = CenteredGrid(
            (0, 0),
            extrapolation.ZERO,
            x=self.resolution.get_size("x"),
            y=self.resolution.get_size("y"),
            bounds=self.domain,
        )


        density_0 = CenteredGrid(
            0,
            extrapolation.BOUNDARY,
            x=self.resolution.get_size("x"),
            y=self.resolution.get_size("y"),
            bounds=self.domain,
        )

        inflow_shape = Sphere(center=inflow_center, radius=self.inflow_radius)
        inflow_0 = self.inflow_rate * CenteredGrid(
            inflow_shape,
            extrapolation.BOUNDARY,
            x=self.resolution.get_size("x"),
            y=self.resolution.get_size("y"),
            bounds=self.domain,
        )

        return {"velocity": velocity_0, "density": density_0, "inflow": inflow_0}

    def forward(self, current_state: Dict[str, Field]) -> Dict[str, Field]:
        """
        Performs a single simulation step.
        """
        # This is unchanged and will now use the internally created self.inflow
        new_velocity, new_density = _smoke_physics_step(
            velocity=current_state["velocity"],
            density=current_state["density"],
            inflow=current_state["inflow"],
            domain=self.domain,
            dt=self.dt,
            buoyancy_factor=self.buoyancy,
            nu=self.nu,
        )
        return {
            "velocity": new_velocity,
            "density": new_density,
            "inflow": current_state["inflow"],
        }
    
    def _get_inflow_center(self) -> Tensor:
        """
        Computes a random inflow center within the specified ranges.
        """
        rand_x = self.domain.size[0] * (
            self._inflow_rand_x_range[0] + self._inflow_rand_x_range[1] * random.random()
        )
        rand_y = self.domain.size[1] * (
            self._inflow_rand_y_range[0] + self._inflow_rand_y_range[1] * random.random()
        )
        inflow_center = (rand_x, rand_y)
        return math.tensor(inflow_center, channel(vector="x,y"))
