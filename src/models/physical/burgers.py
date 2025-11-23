"""Burgers model using unified field channel dimension."""

from typing import Dict, Any, Tuple

from phi.torch.flow import *
from phi.math import Shape, Tensor, batch, math
from phiml.math import channel

from .base import PhysicalModel
from src.models import ModelRegistry


@ModelRegistry.register_physical("BurgersModel")
class BurgersModel(PhysicalModel):
    """Burgers equation model with all fields dynamic (no static fields)."""

    def __init__(self, config: dict, downsample_factor: int = 0):
        super().__init__(config, downsample_factor)

        pde_params = config["model"]["physical"]["pde_params"]
        self._diffusion_coeff = self._initialize_diffusion(pde_params)
        self._jit_step = self._create_jit_step()

    @property
    def dynamic_field_names(self) -> Tuple[str, ...]:
        return tuple(f'vel_{d}' for d in self.spatial_dims)

    @property
    def static_field_names(self) -> Tuple[str, ...]:
        return ()

    @property
    def field_names(self) -> Tuple[str, ...]:
        return self.dynamic_field_names

    @property
    def diffusion_coeff(self):
        return self._diffusion_coeff

    @diffusion_coeff.setter
    def diffusion_coeff(self, value):
        self._diffusion_coeff = value
        self._jit_step = self._create_jit_step()

    def _initialize_diffusion(self, pde_params: Dict[str, Any]):
        """Initialize diffusion coefficient from config (scalar or field)."""
        value_str = pde_params['value']
        grid_kwargs = {name: self.resolution.get_size(name) for name in self.spatial_dims}

        try:
            return float(eval(value_str))
        except:
            pass

        coord_names = self.spatial_dims
        sizes = {f'size_{d}': float(self.domain[i].size) for i, d in enumerate(coord_names)}

        def diffusion_fn(**coords):
            local_vars = {**coords, 'math': math, **sizes}
            return eval(value_str, local_vars)

        return CenteredGrid(
            diffusion_fn,
            extrapolation.PERIODIC,
            bounds=self.domain,
            **grid_kwargs
        )

    def _create_jit_step(self):
        """Create JIT-compiled physics step."""
        domain = self.domain
        dt = self.dt
        field_names = self.field_names
        spatial_dims = self.spatial_dims
        diffusion_coeff = self._diffusion_coeff
        grid_kwargs = {name: self.resolution.get_size(name) for name in spatial_dims}

        @jit_compile
        def burgers_step(state: Tensor) -> Tensor:
            # Convert field -> vector for physics
            velocity_tensor = math.rename_dims(
                state, 'field', channel(vector=','.join(spatial_dims))
            )

            # Advection
            velocity_grid = CenteredGrid(velocity_tensor, PERIODIC, bounds=domain, **grid_kwargs)
            velocity_grid = advect.semi_lagrangian(velocity_grid, velocity_grid, dt=dt)

            # Diffusion
            velocity_grid = diffuse.explicit(velocity_grid, diffusion_coeff, dt=dt, substeps=5)

            # Convert vector -> field
            return math.rename_dims(
                velocity_grid.values, 'vector', channel(field=','.join(field_names))
            )

        return burgers_step

    def get_initial_state(self, batch_size: int = 1) -> Tensor:
        """Generate noisy initial velocity using Noise."""
        grid_kwargs = {name: self.resolution.get_size(name) for name in self.spatial_dims}
        vector_str = ','.join(self.spatial_dims)

        velocity_grid = CenteredGrid(
            Noise(scale=1.0, smoothness=2.0, vector=vector_str),
            extrapolation.PERIODIC,
            bounds=self.domain,
            **grid_kwargs
        )

        velocity = math.expand(velocity_grid.values, batch(batch=batch_size))
        return math.rename_dims(
            velocity, 'vector', channel(field=','.join(self.field_names))
        )

    def forward(self, state: Tensor) -> Tensor:
        return self._jit_step(state)

    def rollout(self, initial_state: Tensor, num_steps: int) -> Tensor:
        """Rollout simulation for multiple steps."""
        trajectory = iterate(self._jit_step, batch(time=num_steps), initial_state)
        if isinstance(trajectory, tuple):
            trajectory = trajectory[0]
        return trajectory
