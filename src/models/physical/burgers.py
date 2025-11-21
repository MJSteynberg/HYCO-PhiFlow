# src/models/physical/burgers.py

from typing import Dict, Any
import numpy as np

# --- PhiFlow Imports ---
from phi.torch.flow import *
from phi.math import Shape, Tensor, batch, math
import numexpr

# --- Repo Imports ---
from .base import PhysicalModel
from src.models import ModelRegistry

@jit_compile
def _burgers_physics_step(
    velocity: CenteredGrid, diffusion_coeff: CenteredGrid, dt: float
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

    # Split velocity into its components
    components = list(unstack(velocity, channel('vector')))

    # Diffuse each component separately
    for i, u_comp in enumerate(components):
        components[i] = diffuse.explicit(u_comp, diffusion_coeff, dt=dt, substeps=5)

    # Reassemble the velocity field
    velocity = stack(components, channel('vector'))
    return velocity, diffusion_coeff

# --- Model Class Implementation ---


@ModelRegistry.register_physical("BurgersModel")
class BurgersModel(PhysicalModel):
    """
    Physical model for the Burgers' equation.
    Implements the PhysicalModel interface.
    """

    def __init__(self, config: dict, downsample_factor: int = 1):
        """Initialize the Burgers model."""
        super().__init__(config, downsample_factor)
        self.pde_params = config["model"]["physical"]["pde_params"]
        # Calculate the maximum value that the diffusion coefficient can be whilst being stable
        self.max_diffusion = 0.5 * min(
            self.domain.size[0] / (self.resolution.get_size("x")),
            self.domain.size[1] / (self.resolution.get_size("y")),
        ) ** 2 / self.dt
        self._initialize_fields(self.pde_params)

        # Create field template for velocity (will be updated from tensors)
        self.velocity = self._create_field_template('velocity', vector_dim=2)


    def _initialize_fields(self, pde_params: Dict[str, Any]):
        """Initialize model fields from PDE parameters."""
        def f(x, y):
            evaluation = eval(pde_params['value'], {'x':x, 'y':y, 'math': math, 'size_x': self.domain.size[0], 'size_y': self.domain.size[1]})
            return evaluation
        
        self._initialize_diffusion_field(f)

    def _initialize_diffusion_field(self, value):
        """Initialize diffusion_coeff as a CenteredGrid field."""
        self._diffusion_coeff = CenteredGrid(
            value,
            extrapolation.PERIODIC,
            x=self.resolution.get_size("x"),
            y=self.resolution.get_size("y"),
            bounds=self.domain,
        )
        
        self._diffusion_coeff = CenteredGrid(
            math.clip(self._diffusion_coeff.values, 0, self.max_diffusion),
            extrapolation.PERIODIC,
            x=self.resolution.get_size("x"),
            y=self.resolution.get_size("y"),
            bounds=self.domain,
        )

    def _create_field_template(self, field_name: str, vector_dim: int = 1) -> CenteredGrid:
        """
        Create a field template that will be updated from tensors.

        Args:
            field_name: Name of the field
            vector_dim: Vector dimension (1 for scalar, 2 for 2D vector)

        Returns:
            CenteredGrid template
        """
        # Create a zero-initialized field with correct shape
        if vector_dim == 1:
            values = 0.0
        else:
            values = (0.0,) * vector_dim

        return CenteredGrid(
            values,
            extrapolation.PERIODIC,
            x=self.resolution.get_size("x"),
            y=self.resolution.get_size("y"),
            bounds=self.domain,
        )

    def update_from_tensors(self, tensors: Dict[str, Tensor]):
        """
        Update internal field values from PhiML tensors.

        Args:
            tensors: Dict of PhiML tensors {'velocity': Tensor(batch, x, y, vector)}
        """
        if 'velocity' in tensors:
            # Update velocity field with new tensor values
            self.velocity = CenteredGrid(
                tensors['velocity'],
                extrapolation.PERIODIC,
                bounds=self.domain,
            )

    @property
    def diffusion_coeff(self) -> CenteredGrid:
        """Get the diffusion coefficient field."""
        return self._diffusion_coeff

    @diffusion_coeff.setter
    def diffusion_coeff(self, value: Any):
        """Set the diffusion coefficient field."""
        if isinstance(value, Field):
            self._diffusion_coeff = CenteredGrid(math.clip(value.values, 0, self.max_diffusion),
                                                extrapolation.PERIODIC,
                                                x=self.resolution.get_size("x"),
                                                y=self.resolution.get_size("y"),
                                                bounds=self.domain)
        else:
            self._initialize_diffusion_field(value)

    def get_initial_state(self, batch_size: int = 1) -> Dict[str, Field]:
        """
        Returns an initial state of (noisy velocity).
        We use periodic boundaries as they are common for Burgers.
        """
        b = batch(batch=batch_size)

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
    
    def rollout(self, initial_state, num_steps: int) -> Dict[str, Tensor]:
        """
        Perform a rollout of the Burgers' equation from the initial state.

        Args:
            initial_state (Dict[str, Field]): Initial state containing 'velocity'.
            num_steps (int): Number of time steps to simulate.          
        Returns:
            Dict[str, Tensor]: Dictionary containing the velocity trajectory
                               with shape [batch, time, y, x].
        """
        velocity_trj, _ = iterate(_burgers_physics_step, batch(time=num_steps), initial_state["velocity"], self.diffusion_coeff, dt = self.dt)
        return {"velocity": velocity_trj}

    def forward(self, current_state: Dict[str, Field] = None) -> Dict[str, Field]:
        """
        Performs a single simulation step using internal fields.

        Args:
            current_state: Optional dict of fields (for backward compatibility).
                          If None, uses internal self.velocity

        Returns:
            Dict containing updated fields
        """
        # 
        # Use internal velocity if no state provided
        if current_state is None:
            velocity = self.velocity
        else:
            velocity = current_state["velocity"]

        for _ in range(self.downsample_factor):
            velocity = field.downsample2x(velocity)

        batch_size = velocity.shape.get_size("batch")
        diffusion_coeff_batched = math.expand(
            self.diffusion_coeff,
            batch(batch=batch_size)
        )
        new_velocity, _ = _burgers_physics_step(
            velocity=velocity, diffusion_coeff=diffusion_coeff_batched, dt=self.dt
        )

        # Update internal velocity
        self.velocity = new_velocity

        for _ in range(self.downsample_factor):
            new_velocity = field.upsample2x(new_velocity)

        return {"velocity": new_velocity}
