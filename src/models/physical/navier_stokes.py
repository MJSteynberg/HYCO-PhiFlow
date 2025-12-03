"""Incompressible Navier-Stokes model with trainable body force field.

This implements the hidden force field reconstruction problem:
Given velocity observations, recover the unknown body force f(x,y).

PDE: ∂u/∂t + (u·∇)u = -∇p + ν∇²u + f(x,y)

where:
- u is the velocity field (2D vector)
- p is the pressure (solved via projection)
- ν is the kinematic viscosity (known)
- f(x,y) is the unknown body force field to identify
"""

from typing import Dict, Any, Tuple

from phi.torch.flow import *
from phi.math import Shape, Tensor, batch, math
from phi.field import downsample2x, upsample2x
from phiml.math import channel

from .base import PhysicalModel
from src.models import ModelRegistry


@ModelRegistry.register_physical("NavierStokesForceModel")
class NavierStokesForceModel(PhysicalModel):
    """
    Incompressible Navier-Stokes model with trainable body force field.

    PDE: ∂u/∂t + (u·∇)u = -∇p + ν∇²u + f(x,y)

    The trainable parameter is the body force field f(x,y), which can be:
    - A scalar potential (f = -∇φ) for conservative forces
    - A direct vector field (f_x, f_y) for general forces

    Supports optional downsampling for efficient training:
    - Input states at full resolution are downsampled during physics step
    - Physics computed at reduced resolution
    - Output stays at reduced resolution (upsample via forward() if needed)
    - Parameters stay at reduced resolution
    """

    def __init__(self, config: dict, downsample_factor: int = 0):
        super().__init__(config, downsample_factor)

        pde_params = config["model"]["physical"]["pde_params"]
        self._force_type = pde_params.get('type', 'vector_field')  # 'potential' or 'vector_field'
        self._force_value_str = pde_params.get('value', '0')
        self._viscosity = float(pde_params.get('viscosity', 0.01))

        # Pressure solve settings
        self._pressure_solve = Solve('CG', 1e-5, 1e-5, max_iterations=1000)

        self._jit_step = self._create_jit_step()
        self._params = self.get_initial_params()

    @property
    def dynamic_field_names(self) -> Tuple[str, ...]:
        """Velocity components are the dynamic fields."""
        return tuple(f'vel_{d}' for d in self.spatial_dims)

    @property
    def static_field_names(self) -> Tuple[str, ...]:
        return ()

    @property
    def field_names(self) -> Tuple[str, ...]:
        return self.dynamic_field_names

    @property
    def scalar_param_names(self) -> Tuple[str, ...]:
        return ()

    @property
    def field_param_names(self) -> Tuple[str, ...]:
        if self._force_type == 'potential':
            return ('force_potential',)
        else:  # vector_field
            return tuple(f'force_{d}' for d in self.spatial_dims)

    @property
    def params(self) -> Tensor:
        return self._params

    @params.setter
    def params(self, params: Tensor):
        # No stability constraints needed for force field
        self._params = params

    def _create_force_field(self) -> Tensor:
        """Create spatial force field from config at REDUCED resolution."""
        grid_kwargs = {name: self.resolution.get_size(name) for name in self.spatial_dims}
        
        # Build coordinate tensors at reduced resolution
        coords = {}
        for dim_name in self.spatial_dims:
            size = self.domain.size[dim_name]
            res = grid_kwargs[dim_name]
            # Cell-centered coordinates
            coords[dim_name] = math.linspace(
                size / (2 * res), 
                size - size / (2 * res), 
                spatial(**{dim_name: res})
            )
            # Also provide size for expressions
            coords[f'size_{dim_name}'] = size

        # Create meshgrid for 2D
        if self.n_spatial_dims == 2:
            x, y = coords['x'], coords['y']
            size_x, size_y = coords['size_x'], coords['size_y']
            
            if self._force_type == 'potential':
                # Evaluate potential field expression
                potential = eval(self._force_value_str)
                return potential
            else:
                # Evaluate vector field expression (expects [fx, fy] list or similar)
                force_expr = eval(self._force_value_str)
                if isinstance(force_expr, (list, tuple)):
                    # Stack components into vector field
                    return math.stack(force_expr, channel(vector=','.join(self.spatial_dims)))
                else:
                    # Assume it's already a properly formatted tensor
                    return force_expr
        else:
            # 1D case
            x = coords['x']
            size_x = coords['size_x']
            force = eval(self._force_value_str)
            return force

    def get_real_params(self) -> Tensor:
        """Get ground truth parameters at REDUCED resolution."""
        if self._force_type == 'potential':
            force_field = self._create_force_field()
            field_names = ','.join(self.field_param_names)
            force_field = math.expand(force_field, channel(field=field_names))
        else:
            force_field = self._create_force_field()
            # Rename vector dimension to field dimension
            if 'vector' in force_field.shape.names:
                field_names = ','.join(self.field_param_names)
                force_field = math.rename_dims(force_field, 'vector', channel(field=field_names))
            else:
                field_names = ','.join(self.field_param_names)
                force_field = math.expand(force_field, channel(field=field_names))
        
        return force_field

    def get_initial_params(self) -> Tensor:
        """Initialize parameters to zero at REDUCED resolution."""
        real_params = self.get_real_params()
        return math.zeros_like(real_params)

    def _create_jit_step(self):
        """Create JIT-compiled Navier-Stokes step with force field."""
        # Capture configuration
        domain = self.domain
        resolution = self.resolution
        full_resolution = self.full_resolution
        spatial_dims = self.spatial_dims
        field_names = self.field_names
        dt = self.dt
        downsample_factor = self.downsample_factor
        force_type = self._force_type
        viscosity = self._viscosity
        pressure_solve = self._pressure_solve

        grid_kwargs = {name: resolution.get_size(name) for name in spatial_dims}

        @jit_compile
        def navier_stokes_step(state: Tensor, params: Tensor) -> Tuple[Tensor, Tensor]:
            """
            Single Navier-Stokes timestep with body force.
            
            Args:
                state: Velocity field tensor (batch?, x, y, field='vel_x,vel_y')
                params: Force field parameters (x, y, field='force_x,force_y')
            
            Returns:
                next_state: Updated velocity field
                params: Unchanged parameters
            """
            # ============================================================
            # STEP 1: Downsample input if needed
            # ============================================================
            if downsample_factor > 0:
                # Convert to grid for downsampling
                velocity_tensor = math.rename_dims(
                    state, 'field', channel(vector=','.join(spatial_dims))
                )
                state_grid = CenteredGrid(velocity_tensor, PERIODIC, bounds=domain, **{
                    name: full_resolution.get_size(name) for name in spatial_dims
                })

                for _ in range(downsample_factor):
                    state_grid = downsample2x(state_grid)

                working_state = math.rename_dims(
                    state_grid.values, 'vector', channel(field=','.join(field_names))
                )
            else:
                working_state = state

            # ============================================================
            # STEP 2: Run Navier-Stokes physics at reduced resolution
            # ============================================================

            # Convert field -> vector for physics
            velocity_tensor = math.rename_dims(
                working_state, 'field', channel(vector=','.join(spatial_dims))
            )

            # Create velocity grid
            velocity_grid = CenteredGrid(velocity_tensor, PERIODIC, bounds=domain, **grid_kwargs)

            # --- Advection: (u·∇)u ---
            velocity_grid = advect.semi_lagrangian(velocity_grid, velocity_grid, dt=dt)

            # --- Diffusion: ν∇²u ---
            velocity_grid = diffuse.explicit(velocity_grid, viscosity, dt=dt)

            # --- Body force: f(x,y) ---
            if force_type == 'potential':
                # Force from potential: f = -∇φ
                potential_tensor = params.field['force_potential']
                potential_grid = CenteredGrid(potential_tensor, PERIODIC, bounds=domain, **grid_kwargs)
                force_grid = -potential_grid.gradient(boundary=PERIODIC)
            else:
                # Direct vector force field
                force_components = []
                for d in spatial_dims:
                    force_components.append(params.field[f'force_{d}'])
                force_tensor = math.stack(force_components, channel(vector=','.join(spatial_dims)))
                force_grid = CenteredGrid(force_tensor, PERIODIC, bounds=domain, **grid_kwargs)

            velocity_grid = velocity_grid + dt * force_grid

            # --- Pressure projection: make incompressible ---
            velocity_grid, _ = fluid.make_incompressible(velocity_grid, solve=pressure_solve)

            # ============================================================
            # STEP 3: Convert back to tensor format
            # ============================================================
            next_state = math.rename_dims(
                velocity_grid.values, 'vector', channel(field=','.join(field_names))
            )

            return next_state, params

        return navier_stokes_step

    def get_initial_state(self, batch_size: int = 1) -> Tensor:
        """Generate initial velocity field at FULL resolution."""
        full_grid_kwargs = {name: self.full_resolution.get_size(name) for name in self.spatial_dims}

        # Build coordinate tensors
        coords = {}
        for dim_name in self.spatial_dims:
            size = self.domain.size[dim_name]
            res = full_grid_kwargs[dim_name]
            coords[dim_name] = math.linspace(
                size / (2 * res),
                size - size / (2 * res),
                spatial(**{dim_name: res})
            )
            coords[f'size_{dim_name}'] = size

        if self.n_spatial_dims == 2:
            x, y = coords['x'], coords['y']
            size_x, size_y = coords['size_x'], coords['size_y']
            
            # Initial condition: perturbed base flow or vortices
            # Taylor-Green vortex-like initial condition
            vel_x = math.sin(2 * math.pi * x / size_x) * math.cos(2 * math.pi * y / size_y)
            vel_y = -math.cos(2 * math.pi * x / size_x) * math.sin(2 * math.pi * y / size_y)
            
            velocity = math.stack([vel_x, vel_y], channel(field=','.join(self.field_names)))
        else:
            x = coords['x']
            size_x = coords['size_x']
            vel_x = math.sin(2 * math.pi * x / size_x)
            velocity = math.expand(vel_x + noise, channel(field=self.field_names[0]))

        # Add batch dimension
        if batch_size > 1:
            velocity = math.stack([
                velocity + math.random_normal(velocity.shape) * 0.05 
                for _ in range(batch_size)
            ], batch('batch'))
        else:
            velocity = math.expand(velocity, batch(batch=1))

        return velocity

    def forward(self, state: Tensor, params: Tensor, upsample_output: bool = True) -> Tensor:
        """
        Forward pass through the model.

        Args:
            state: Input state (at full resolution if downsample_factor > 0)
            params: Model parameters (at reduced resolution)
            upsample_output: If True, upsample output to full resolution.

        Returns:
            Prediction at full resolution (if upsample_output=True) or
            reduced resolution (if upsample_output=False)
        """
        pred, _ = self._jit_step(state, params)

        if upsample_output and self.downsample_factor > 0:
            pred = self._upsample_state(pred, self.field_names)

        return pred

    def rollout(self, initial_state: Tensor, params: Tensor, num_steps: int,
                upsample_output: bool = True) -> Tensor:
        """
        Rollout simulation for multiple steps.

        Args:
            initial_state: Initial state (at full resolution)
            params: Model parameters (at reduced resolution)
            num_steps: Number of steps to simulate
            upsample_output: If True, upsample trajectory to full resolution.

        Returns:
            Trajectory tensor with time dimension
        """
        trajectory, _ = iterate(self._jit_step, batch(time=num_steps), initial_state, params)
        if isinstance(trajectory, tuple):
            trajectory = trajectory[0]

        if upsample_output and self.downsample_factor > 0:
            upsampled_steps = []
            for t in range(trajectory.shape.get_size('time')):
                step = trajectory.time[t]
                upsampled_step = self._upsample_state(step, self.field_names)
                upsampled_steps.append(upsampled_step)
            trajectory = math.stack(upsampled_steps, batch('time'))

        return trajectory

    def save(self, path):
        """Save model parameters."""
        math.save(str(path), self.params)

    def load(self, path):
        """Load model parameters."""
        self.params = math.load(str(path))





