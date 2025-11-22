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

def _custom_diffusion(u: CenteredGrid, diffusivity: CenteredGrid, dt: float, substeps: int = 5) -> CenteredGrid:
    """
    Dimension-agnostic diffusion with spatially-varying diffusivity.
    Implements: ∂u/∂t = ∇·(D(x) ∇u) using finite differences.

    This works in 1D, 2D, and 3D without modification by processing each spatial dimension independently.

    Args:
        u: Field to diffuse
        diffusivity: Spatially-varying diffusion coefficient
        dt: Time step
        substeps: Number of substeps for numerical stability

    Returns:
        Diffused field
    """
    dt_sub = dt / substeps
    result = u

    for _ in range(substeps):
        # Accumulate diffusion term for each spatial dimension
        diffusion_term = math.tensor(0.0)

        for dim in result.shape.spatial.names:
            # Shift values in this dimension to get neighbors
            # stack_dim=None returns list, [0] extracts the shifted tensor
            left = math.shift(result.values, (-1,), dims=dim, padding=result.extrapolation, stack_dim=None)[0]
            right = math.shift(result.values, (1,), dims=dim, padding=result.extrapolation, stack_dim=None)[0]

            # Shift diffusivity to get neighbor values
            left_D = math.shift(diffusivity.values, (-1,), dims=dim, padding=diffusivity.extrapolation, stack_dim=None)[0]
            right_D = math.shift(diffusivity.values, (1,), dims=dim, padding=diffusivity.extrapolation, stack_dim=None)[0]

            # Diffusivity at cell faces (arithmetic average)
            D_left_face = (diffusivity.values + left_D) / 2.0
            D_right_face = (diffusivity.values + right_D) / 2.0

            # Gradient at cell faces: (u[i] - u[i-1]) / dx
            dx = result.dx.vector[dim]
            grad_left_face = (result.values - left) / dx
            grad_right_face = (right - result.values) / dx

            # Flux at faces: D * ∇u
            flux_left = D_left_face * grad_left_face
            flux_right = D_right_face * grad_right_face

            # Divergence: (flux_right - flux_left) / dx
            diffusion_term += (flux_right - flux_left) / dx

        # Explicit Euler step
        result = result.with_values(result.values + dt_sub * diffusion_term)

    return result


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

    # Check if diffusivity is spatially-varying
    is_varying = math.spatial(diffusion_coeff.values)

    # Split velocity into its components
    components = list(unstack(velocity, channel('vector')))
    # Diffuse each component separately
    for i, u_comp in enumerate(components):
        if is_varying:
            # Use custom diffusion for spatially-varying diffusivity (works in all dimensions)
            components[i] = _custom_diffusion(u_comp, diffusion_coeff, dt, substeps=5)
        else:
            # Use standard PhiFlow diffusion for constant diffusivity
            components[i] = diffuse.explicit(u_comp, diffusion_coeff, dt=dt, substeps=5)

    # Reassemble the velocity field
    velocity = stack(components, channel('vector'))
    return velocity, diffusion_coeff


def _create_jit_step_with_resampling(downsample_factor: int):
    """
    Factory function to create a jit-compiled step function with specific downsample factor.

    The downsample_factor is baked into the function at creation time, allowing the JIT
    compiler to fully trace and optimize the up/downsampling operations.

    Args:
        downsample_factor: Number of 2x downsampling/upsampling steps (2^factor reduction)

    Returns:
        A jit-compiled step function that handles downsampling, physics, and upsampling
    """
    @jit_compile
    def _step_with_resampling(velocity: CenteredGrid, diffusion_coeff: CenteredGrid, dt: float):
        """Jit-compiled step with integrated up/downsampling."""
        # Downsample velocity to match diffusion coefficient resolution
        v = velocity
        for _ in range(downsample_factor):
            v = field.downsample2x(v)

        # Physics step at reduced resolution
        v, _ = _burgers_physics_step(v, diffusion_coeff, dt)

        # Upsample back to original resolution
        for _ in range(downsample_factor):
            v = field.upsample2x(v)

        return v

    return _step_with_resampling

# --- Model Class Implementation ---


@ModelRegistry.register_physical("BurgersModel")
class BurgersModel(PhysicalModel):
    """
    Physical model for the Burgers' equation.
    Implements the PhysicalModel interface.
    """

    def __init__(self, config: dict, downsample_factor: int = 0):
        """Initialize the Burgers model."""
        super().__init__(config, downsample_factor)
        self.pde_params = config["model"]["physical"]["pde_params"]
        # Calculate the maximum value that the diffusion coefficient can be whilst being stable
        self._initialize_fields(self.pde_params)

        # Create field template for velocity (will be updated from tensors)
        self.velocity = self._create_field_template('velocity', vector_dim=2)

        # Create jit-compiled step function with resampling baked in
        # This provides significant speedup by fusing the downsampling, physics, and upsampling
        self._jit_step = _create_jit_step_with_resampling(downsample_factor)


    def _initialize_fields(self, pde_params: Dict[str, Any]):
        """Initialize model fields from PDE parameters."""
        if self.n_spatial_dims == 1:
            def f(x):
                evaluation = eval(pde_params['value'], {'x':x, 'math': math, 'size_x': self.domain.size[0]})
                return evaluation
        elif self.n_spatial_dims == 2:
            def f(x, y):
                evaluation = eval(pde_params['value'], {'x':x, 'y':y, 'math': math, 'size_x': self.domain.size[0], 'size_y': self.domain.size[1]})
                return evaluation
        else:  # 3D
            def f(x, y, z):
                evaluation = eval(pde_params['value'], {'x':x, 'y':y, 'z':z, 'math': math, 'size_x': self.domain.size[0], 'size_y': self.domain.size[1], 'size_z': self.domain.size[2]})
                return evaluation
        
        self._initialize_diffusion_field(f)

    def _initialize_diffusion_field(self, value):
        # Build kwargs dynamically from resolution Shape
        grid_kwargs = {
            name: self.resolution.get_size(name)
            for name in self.resolution.names
        }

        self._diffusion_coeff = CenteredGrid(
            value,
            extrapolation.PERIODIC,
            bounds=self.domain,
            **grid_kwargs 
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
        # Build kwargs dynamically from resolution Shape
        grid_kwargs = {
            name: self.resolution.get_size(name)
            for name in self.resolution.names
        }

        # Create a zero-initialized field with correct shape
        if vector_dim == 1:
            values = 0.0
        else:
            values = (0.0,) * vector_dim

        return CenteredGrid(
            values,
            extrapolation.PERIODIC,
            bounds=self.domain,
            **grid_kwargs
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
            self._diffusion_coeff = value
        else:
            self._initialize_diffusion_field(value)

    def get_initial_state(self, batch_size: int = 1) -> Dict[str, Field]:
        """
        Returns an initial state of (noisy velocity).
        We use periodic boundaries as they are common for Burgers.
        Dimension-agnostic: works for 1D, 2D, and 3D.
        """
        b = batch(batch=batch_size)

        # Build kwargs dynamically from resolution Shape
        grid_kwargs = {
            name: self.resolution.get_size(name)
            for name in self.resolution.names
        }
        noise = StaggeredGrid(
            Noise(scale=10, smoothness=10),
            extrapolation.PERIODIC,  # Use periodic boundaries
            bounds=self.domain,
            **grid_kwargs
        )
        noise = CenteredGrid(noise, extrapolation.PERIODIC, bounds=self.domain, **grid_kwargs)

        # Create dimension-agnostic initial velocity field
        # Initialize each component with a sinusoidal wave based on its coordinate
        dim_names = self.resolution.names

        def initial_velocity(position):
            """Create initial velocity components for each spatial dimension."""
            components = {}
            for i, dim_name in enumerate(dim_names):
                # Get the coordinate for this dimension
                coord = position.vector[dim_name]
                # Use different wave numbers for different dimensions
                wave_number = 4 * math.pi * (i + 1) / self.domain[i].size
                components[dim_name] = sin(wave_number * coord)
            return vec(**components)

        velocity_0 = CenteredGrid(
            initial_velocity,
            extrapolation.PERIODIC,  # Use periodic boundaries
            bounds=self.domain,
            **grid_kwargs
        ) + 0.5*noise
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

        Uses jit-compiled step function with integrated up/downsampling for speedup.

        Args:
            current_state: Optional dict of fields (for backward compatibility).
                          If None, uses internal self.velocity

        Returns:
            Dict containing updated fields
        """
        # Use internal velocity if no state provided
        if current_state is None:
            velocity = self.velocity
        else:
            velocity = current_state["velocity"]

        # Use jit-compiled step with integrated up/downsampling
        # The _jit_step function handles downsampling, physics, and upsampling in one traced call
        new_velocity = self._jit_step(velocity, self.diffusion_coeff, self.dt)

        # Update internal velocity (store at low resolution for efficiency)
        # Note: new_velocity is at high resolution, we store downsampled version
        internal_v = new_velocity
        for _ in range(self.downsample_factor):
            internal_v = field.downsample2x(internal_v)
        self.velocity = internal_v

        return {"velocity": new_velocity}
