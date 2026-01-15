"""Buoyancy-driven smoke model with trainable buoyancy field.

Inverse problem: Given smoke observations, recover buoyancy field b(x,y).

Physics:
- Smoke advected by velocity: ∂s/∂t + (u·∇)s = 0
- Velocity driven by buoyancy: ∂u/∂t + (u·∇)u = -∇p + b(x,y) * s * ĵ
- Incompressibility: ∇·u = 0

Starts with initial smoke blob - buoyancy drives all dynamics.
"""

from typing import Tuple

from phi.torch.flow import *
from phi.math import Shape, Tensor, batch, math, channel
from phi.field import downsample2x, upsample2x
from .base import PhysicalModel
from src.models import ModelRegistry


@ModelRegistry.register_physical("NavierStokesModel")
class NavierStokesModel(PhysicalModel):
    """Buoyancy-driven smoke with trainable buoyancy field."""

    def __init__(self, config: dict, downsample_factor: int = 0):
        super().__init__(config, downsample_factor)

        pde_params = config["model"]["physical"]["pde_params"]
        self._buoyancy_type = pde_params.get('type', 'field')
        self._buoyancy_value_str = pde_params.get('value', '0.1')
        
        # Initial smoke blob config
        initial = pde_params.get('initial_smoke', {})

        self._jit_step = self._create_jit_step()
        self._params = self.get_initial_params()

    @property
    def dynamic_field_names(self) -> Tuple[str, ...]:
        return ('vel_x', 'vel_y', 'smoke')

    @property
    def static_field_names(self) -> Tuple[str, ...]:
        return ()

    @property
    def field_names(self) -> Tuple[str, ...]:
        return self.dynamic_field_names

    @property
    def scalar_param_names(self) -> Tuple[str, ...]:
        return ('buoyancy_coeff',) if self._buoyancy_type == 'scalar' else ()

    @property
    def field_param_names(self) -> Tuple[str, ...]:
        return ('buoyancy_field',) if self._buoyancy_type == 'field' else ()

    @property
    def params(self) -> Tensor:
        return self._params

    @params.setter
    def params(self, params: Tensor):
        self._params = math.maximum(params, 0.0)

    def _create_buoyancy_field(self) -> Tensor:
        """Create ground truth buoyancy field at REDUCED resolution."""
        grid_kwargs = {name: self.resolution.get_size(name) for name in self.spatial_dims}
        
        coords = {}
        for dim_name in self.spatial_dims:
            size = self.domain.size[dim_name]
            res = grid_kwargs[dim_name]
            coords[dim_name] = math.linspace(size / (2 * res), size - size / (2 * res), 
                                             spatial(**{dim_name: res}))
            coords[f'size_{dim_name}'] = size

        x, y = coords['x'], coords['y']
        size_x, size_y = coords['size_x'], coords['size_y']
        
        buoyancy = eval(self._buoyancy_value_str)
        return math.expand(buoyancy, x.shape & y.shape)

    def get_real_params(self) -> Tensor:
        if self._buoyancy_type == 'scalar':
            return math.expand(eval(self._buoyancy_value_str), channel(field='buoyancy_coeff'))
        else:
            return math.expand(self._create_buoyancy_field(), channel(field='buoyancy_field'))

    def get_initial_params(self) -> Tensor:
        return math.ones_like(self.get_real_params())

    def _create_jit_step(self):
        """Create physics step function with downsampling support."""
        domain = self.domain
        resolution = self.resolution  # REDUCED resolution
        full_resolution = self.full_resolution
        spatial_dims = self.spatial_dims
        dt = self.dt
        buoyancy_type = self._buoyancy_type
        downsample_factor = self.downsample_factor
        field_names = self.field_names
        
        # Grid kwargs at REDUCED resolution
        grid_kwargs = {name: resolution.get_size(name) for name in spatial_dims}

        @jit_compile
        def smoke_step(state: Tensor, params: Tensor) -> Tuple[Tensor, Tensor]:
            # ============================================================
            # STEP 1: Get working state (downsample only if at full resolution)
            # ============================================================
            # Detect if input is at full or reduced resolution
            # This is needed because iterate() feeds output back as input
            first_spatial_dim = spatial_dims[0]
            input_res = state.shape.get_size(first_spatial_dim)
            full_res = full_resolution.get_size(first_spatial_dim)
            
            # Only downsample if input is at full resolution
            needs_downsample = downsample_factor > 0 and input_res == full_res
            
            if needs_downsample:
                # Convert tensor to grid for downsampling
                full_grid_kwargs = {name: full_resolution.get_size(name) for name in spatial_dims}
                
                # Stack velocity components for vector field
                vel_tensor = math.stack([state.field['vel_x'], state.field['vel_y']], 
                                       channel(vector='x,y'))
                vel_grid = CenteredGrid(vel_tensor, 0, bounds=domain, **full_grid_kwargs)
                smoke_grid = CenteredGrid(state.field['smoke'], ZERO_GRADIENT, 
                                         bounds=domain, **full_grid_kwargs)
                
                # Downsample
                for _ in range(downsample_factor):
                    vel_grid = downsample2x(vel_grid)
                    smoke_grid = downsample2x(smoke_grid)
                
                # Extract working tensors at reduced resolution
                vel_x = vel_grid.values.vector['x']
                vel_y = vel_grid.values.vector['y']
                smoke_data = smoke_grid.values
            else:
                # Input already at working resolution (or no downsampling needed)
                vel_x = state.field['vel_x']
                vel_y = state.field['vel_y']
                smoke_data = state.field['smoke']
            
            # ============================================================
            # STEP 2: Run physics at reduced resolution
            # ============================================================
            
            # Create grids at reduced resolution
            smoke = CenteredGrid(smoke_data, ZERO_GRADIENT, bounds=domain, **grid_kwargs)
            vel_centered = CenteredGrid(
                math.stack([vel_x, vel_y], channel(vector='x,y')),
                0, bounds=domain, **grid_kwargs
            )
            velocity = resample(vel_centered, to=StaggeredGrid(0, 0, bounds=domain, **grid_kwargs))

            # Physics
            smoke = advect.mac_cormack(smoke, velocity, dt=dt)
            
            # Buoyancy force
            if buoyancy_type == 'scalar':
                b = params.field['buoyancy_coeff']
            else:
                b = params.field['buoyancy_field']
            buoyancy = resample(
                CenteredGrid(b, 0, bounds=domain, **grid_kwargs) * smoke * vec(x=0, y=1), 
                to=velocity
            )
            
            velocity = advect.semi_lagrangian(velocity, velocity, dt=dt) + buoyancy * dt
            velocity, _ = fluid.make_incompressible(
                velocity, 
                solve=Solve('CG', 1e-3, rank_deficiency=0, suppress=[Diverged, NotConverged])
            )

            # ============================================================
            # STEP 3: Pack output (stays at reduced resolution)
            # ============================================================
            vel_out = velocity.at_centers()
            next_state = math.stack(
                [vel_out.values.vector['x'], vel_out.values.vector['y'], smoke.values],
                channel(field='vel_x,vel_y,smoke')
            )

            return next_state, params

        return smoke_step

    def get_initial_state(self, batch_size: int = 1) -> Tensor:
        """Initial state at FULL resolution: zero velocity, random smooth smoke."""
        grid_kwargs = {name: self.full_resolution.get_size(name) for name in self.spatial_dims}
        
        # Random smooth smoke field using Noise
        smoke = CenteredGrid(
            Noise(batch(batch=batch_size), scale=self.domain.size['x'] / 10, smoothness=1.5),
            extrapolation.PERIODIC,
            bounds=self.domain,
            **grid_kwargs
        ).values
        
        vel_x = math.zeros(smoke.shape)
        vel_y = math.zeros(smoke.shape)
        
        state = math.stack([vel_x, vel_y, smoke], channel(field='vel_x,vel_y,smoke'))
        return state

    def forward(self, state: Tensor, params: Tensor, upsample_output: bool = True) -> Tensor:
        """Single step forward. Optionally upsample output to full resolution."""
        pred, _ = self._jit_step(state, params)
        if upsample_output and self.downsample_factor > 0:
            pred = self._upsample_state(pred, self.field_names)
        return pred

    def rollout(self, initial_state: Tensor, params: Tensor, num_steps: int,
                upsample_output: bool = True) -> Tensor:
        """
        Roll out simulation for multiple steps.
        
        Args:
            initial_state: Initial state (at full resolution)
            params: Model parameters (at reduced resolution)
            num_steps: Number of steps to simulate
            upsample_output: If True, upsample trajectory to full resolution.
                           If False, return at reduced resolution.

        Returns:
            Trajectory at full resolution (if upsample_output=True) or
            reduced resolution (if upsample_output=False)
        """
        trajectory, _ = iterate(self._jit_step, batch(time=num_steps), initial_state, params)
        if isinstance(trajectory, tuple):
            trajectory = trajectory[0]

        # Upsample each timestep if requested
        # Note: iterate() returns [initial_state, step1, step2, ...]
        # The initial state is at FULL resolution, while subsequent steps are at REDUCED resolution
        # So we only upsample states after the first one (which is already at full resolution)
        if upsample_output and self.downsample_factor > 0:
            upsampled_steps = [trajectory.time[0]]  # Keep initial state as-is (already full res)
            for t in range(1, trajectory.shape.get_size('time')):
                step = trajectory.time[t]
                upsampled_step = self._upsample_state(step, self.field_names)
                upsampled_steps.append(upsampled_step)
            trajectory = math.stack(upsampled_steps, batch('time'))

        return trajectory

    def save(self, path):
        math.save(str(path), self.params)

    def load(self, path):
        self.params = math.load(str(path))