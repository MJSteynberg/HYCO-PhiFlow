# src/models/physical/kolmogorov.py

from typing import Dict, Any
import numpy as np

# --- PhiFlow Imports ---
from phi.torch.flow import *
from phi.math import Shape, Tensor, batch, math

# --- Repo Imports ---
from .base import PhysicalModel
from src.models import ModelRegistry

@jit_compile
def _kolmogorov_physics_step(
    velocity: CenteredGrid, forcing_field: CenteredGrid, nu: float, dt: float
) -> CenteredGrid:
    """
    Performs one physics-based Kolmogorov flow step.

    Args:
        velocity (CenteredGrid): Current velocity field.
        forcing_field (CenteredGrid): Forcing term field.
        nu (float): Kinematic viscosity.
        dt (float): Time step.

    Returns:
        CenteredGrid: new_velocity
    """
    # Advect velocity (self-advection: u * grad(u))
    velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)
    
    # Apply diffusion
    velocity = diffuse.explicit(velocity, nu, dt=dt, substeps=5)
    
    # Add forcing term
    velocity = velocity + forcing_field * dt
    
    # Make divergence-free (project onto incompressible space)
    velocity, _ = fluid.make_incompressible(velocity,
                                            solve=Solve("CG", 1e-3, rank_deficiency=0, suppress=[phi.math.NotConverged]))
    
    return velocity, forcing_field, nu

# --- Model Class Implementation ---


@ModelRegistry.register_physical("KolmogorovModel")
class KolmogorovModel(PhysicalModel):
    """
    Physical model for Kolmogorov flow.
    Implements the PhysicalModel interface.
    """

    def __init__(self, config: dict):
        """Initialize the Kolmogorov flow model."""
        super().__init__(config)
        self.pde_params = config["model"]["physical"]["pde_params"]
        self.nu = self.pde_params.get("nu", 0.1)  # Kinematic viscosity
        self._initialize_fields(self.pde_params)

    def _initialize_fields(self, pde_params: Dict[str, Any]):
        """Initialize model fields from PDE parameters."""
        def f(x, y):
            evaluation = eval(pde_params['value'], {'x':x, 'y':y, 'math': math, 'channel': channel, 'size_x': self.domain.size[0], 'size_y': self.domain.size[1]})
            return math.stack(evaluation, channel("vector"))
        
        self._initialize_forcing_field(f)

    def _initialize_forcing_field(self, value):
        """Initialize forcing field as a CenteredGrid field."""
        value = lambda x, y: math.stack([50.0 * math.sin(4 * math.pi * y / self.domain.size[1]), 0 * x], channel("vector"))
        self._forcing_field = CenteredGrid(
            value,
            extrapolation.PERIODIC,
            x=self.resolution.get_size("x"),
            y=self.resolution.get_size("y"),
            bounds=self.domain,
        )

    @property
    def forcing_field(self) -> CenteredGrid:
        """Get the forcing field."""
        return self._forcing_field
    
    @forcing_field.setter
    def forcing_field(self, value: Any):
        """Set the forcing field."""
        if isinstance(value, Field):
            self._forcing_field = value
        else:
            self._initialize_forcing_field(value)

    def get_initial_state(self, batch_size: int = 1) -> Dict[str, Field]:
        """
        Returns an initial state of (noisy velocity).
        We use periodic boundaries as they are common for Kolmogorov flow.
        """
        b = batch(batch=batch_size)

        velocity_0 = StaggeredGrid(
            Noise(scale=5, smoothness=5),  # Initialize with noise
            extrapolation.PERIODIC,  # Use periodic boundaries
            x=self.resolution.get_size("x"),
            y=self.resolution.get_size("y"),
            bounds=self.domain,
        )
        velocity_0 = CenteredGrid(0.1 * velocity_0, 
                                    extrapolation.PERIODIC,
                                    x=self.resolution.get_size("x"),
                                    y=self.resolution.get_size("y"),
                                    bounds=self.domain)
        velocity_0 = math.expand(velocity_0, b)
        return {"velocity": velocity_0}
    
    def rollout(self, initial_state, num_steps: int) -> Dict[str, Tensor]:
        """
        Perform a rollout of Kolmogorov flow from the initial state.

        Args:
            initial_state (Dict[str, Field]): Initial state containing 'velocity'.
            num_steps (int): Number of time steps to simulate.          
        Returns:
            Dict[str, Tensor]: Dictionary containing the velocity trajectory
                               with shape [batch, time, y, x].
        """
        velocity_trj, _, _ = iterate(_kolmogorov_physics_step, batch(time=num_steps), 
                               initial_state["velocity"], self.forcing_field, 
                               self.nu, dt=self.dt)
        return {"velocity": velocity_trj}

    def forward(self, current_state: Dict[str, Field]) -> Dict[str, Field]:
        """
        Performs a single simulation step.
        """
        batch_size = current_state["velocity"].shape.get_size("batch")
        forcing_field_batched = math.expand(
            self.forcing_field,
            batch(batch=batch_size)
        )
        new_velocity, _, _ = _kolmogorov_physics_step(
            velocity=current_state["velocity"], 
            forcing_field=forcing_field_batched, 
            nu=self.nu, 
            dt=self.dt
        )
        return {"velocity": new_velocity}