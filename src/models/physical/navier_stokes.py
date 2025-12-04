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

from .base import PhysicalModel
from src.models import ModelRegistry


@ModelRegistry.register_physical("SmokePlumeModel")
class SmokePlumeModel(PhysicalModel):
    """Buoyancy-driven smoke with trainable buoyancy field."""

    def __init__(self, config: dict, downsample_factor: int = 0):
        super().__init__(config, downsample_factor)

        pde_params = config["model"]["physical"]["pde_params"]
        self._buoyancy_type = pde_params.get('type', 'field')
        self._buoyancy_value_str = pde_params.get('value', '0.1')
        
        # Initial smoke blob config
        initial = pde_params.get('initial_smoke', {})
        self._smoke_x = float(initial.get('x', self.domain.size['x'] / 2))
        self._smoke_y = float(initial.get('y', self.domain.size['y'] * 0.25))
        self._smoke_radius = float(initial.get('radius', 10.0))
        self._smoke_amplitude = float(initial.get('amplitude', 1.0))

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
        """Create ground truth buoyancy field."""
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
        return math.ones_like(self.get_real_params()) * 0.05

    def _create_jit_step(self):
        """Create physics step function."""
        domain = self.domain
        resolution = self.resolution
        spatial_dims = self.spatial_dims
        dt = self.dt
        buoyancy_type = self._buoyancy_type
        grid_kwargs = {name: resolution.get_size(name) for name in spatial_dims}

        @jit_compile
        def smoke_step(state: Tensor, params: Tensor) -> Tuple[Tensor, Tensor]:
            # Unpack state
            vel_x = state.field['vel_x']
            vel_y = state.field['vel_y']
            smoke_data = state.field['smoke']
            
            # Create grids
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
            buoyancy = resample(CenteredGrid(b, 0, bounds=domain, **grid_kwargs) * smoke * vec(x=0, y=1), to=velocity)
            
            velocity = advect.semi_lagrangian(velocity, velocity, dt=dt) + buoyancy * dt
            velocity, _ = fluid.make_incompressible(velocity, solve=Solve('CG', 1e-3, 1e-3,  suppress=[Diverged, NotConverged]))

            # Pack output
            vel_out = velocity.at_centers()
            return math.stack([vel_out.values.vector['x'], vel_out.values.vector['y'], smoke.values],
                            channel(field='vel_x,vel_y,smoke')), params

        return smoke_step

    def get_initial_state(self, batch_size: int = 1) -> Tensor:
        """Initial state: zero velocity, Gaussian smoke blob."""
        res_x = self.full_resolution.get_size('x')
        res_y = self.full_resolution.get_size('y')
        size_x = self.domain.size['x']
        size_y = self.domain.size['y']
        
        x = math.linspace(size_x / (2 * res_x), size_x - size_x / (2 * res_x), spatial(x=res_x))
        y = math.linspace(size_y / (2 * res_y), size_y - size_y / (2 * res_y), spatial(y=res_y))
        grid_kwargs = {name: self.full_resolution.get_size(name) for name in self.spatial_dims}
        vector_str = ','.join(self.spatial_dims)
        # Gaussian smoke blob
        smoke = CenteredGrid(
            Noise(batch(batch=batch_size), scale=self.domain.size/10, smoothness=1.5),
            extrapolation.PERIODIC,
            bounds=self.domain,
            **grid_kwargs
        ).values
        
        vel_x = math.zeros(smoke.shape)
        vel_y = math.zeros(smoke.shape)
        
        state = math.stack([vel_x, vel_y, smoke], channel(field='vel_x,vel_y,smoke'))
        
        
        return state

    def forward(self, state: Tensor, params: Tensor, upsample_output: bool = True) -> Tensor:
        pred, _ = self._jit_step(state, params)
        return pred

    def rollout(self, initial_state: Tensor, params: Tensor, num_steps: int,
                upsample_output: bool = True) -> Tensor:
        trajectory, _ = iterate(self._jit_step, batch(time=num_steps), initial_state, params)
        if isinstance(trajectory, tuple):
            trajectory = trajectory[0]
        return trajectory

    def save(self, path):
        math.save(str(path), self.params)

    def load(self, path):
        self.params = math.load(str(path))