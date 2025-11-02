# src/models/physical/smoke_model.py

import torch
from phi.torch.flow import *
from phi.math import jit_compile, batch
from .base import PhysicalModel  # <-- Import our new base class
from src.models.registry import ModelRegistry
import random
from typing import Dict, Any


# --- JIT-Compiled Physics Function ---
# (This is kept separate for performance and clarity)
@jit_compile
def _smoke_physics_step(velocity, density, inflow, domain, dt, buoyancy_factor, nu):
    """
    Performs one physics-based smoke simulation step.

    Args:
        velocity (StaggeredGrid): Current velocity field.
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
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the smoke model.

        Handles special inflow center logic after base initialization.
        """
        # Call parent init to handle standard parameters
        super().__init__(config)

        # Handle inflow center (special logic not in PDE_PARAMETERS)
        pde_params = config.get("pde_params", {})
        inflow_center = pde_params.get("inflow_center", None)
        inflow_rand_x_range = pde_params.get("inflow_rand_x_range", [0.2, 0.8])
        inflow_rand_y_range = pde_params.get("inflow_rand_y_range", [0.15, 0.25])

        if inflow_center is None:
            rand_x = self.domain.size[0] * (
                inflow_rand_x_range[0] + inflow_rand_x_range[1] * random.random()
            )
            rand_y = self.domain.size[1] * (
                inflow_rand_y_range[0] + inflow_rand_y_range[1] * random.random()
            )
            inflow_center = (rand_x, rand_y)
            print(f"Generated new inflow position: ({rand_x:.1f}, {rand_y:.1f})")

        self.inflow_center = math.tensor(inflow_center, channel(vector="x,y"))

    def get_initial_state(self) -> Dict[str, Field]:
        """
        Returns an initial state of (zero velocity, zero density).
        """
        b = batch(batch=self.batch_size)

        velocity_0 = StaggeredGrid(
            0,
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

        INFLOW_SHAPE = Sphere(center=self.inflow_center, radius=self.inflow_radius)
        inflow_0 = self.inflow_rate * CenteredGrid(
            INFLOW_SHAPE,
            extrapolation.BOUNDARY,
            x=self.resolution.get_size("x"),
            y=self.resolution.get_size("y"),
            bounds=self.domain,
        )

        # Add a batch dimension for broadcasting
        inflow_0 = math.expand(inflow_0, b)

        # No more batch size check against inflow,
        # as inflow_0 (batch=1) will broadcast to batch=N

        return {"velocity": velocity_0, "density": density_0, "inflow": inflow_0}

    def step(self, current_state: Dict[str, Field]) -> Dict[str, Field]:
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
