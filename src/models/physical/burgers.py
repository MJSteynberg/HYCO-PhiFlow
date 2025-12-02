"""Burgers model using unified field channel dimension with trainable diffusion and down/upsampling."""

from typing import Dict, Any, Tuple

from phi.torch.flow import *
from phi.math import Shape, Tensor, batch, math
from phi.field import downsample2x, upsample2x
from phiml.math import channel

from .base import PhysicalModel
from src.models import ModelRegistry


@ModelRegistry.register_physical("BurgersModel")
class BurgersModel(PhysicalModel):
    """
    Burgers equation model with trainable diffusion coefficient.
    
    Supports optional downsampling for efficient training:
    - Input states at full resolution are downsampled
    - Physics computed at reduced resolution
    - Output upsampled back to full resolution
    - Parameters stay at reduced resolution
    """

    def __init__(self, config: dict, downsample_factor: int = 0):
        super().__init__(config, downsample_factor)

        pde_params = config["model"]["physical"]["pde_params"]
        self._diffusion_type = pde_params['type']  # 'scalar' or 'field'
        self._diffusion_value_str = pde_params['value']

        # Calculate max stable diffusion at REDUCED resolution: D <= 0.5 * dx^2 / dt
        grid_kwargs = {name: self.resolution.get_size(name) for name in self.spatial_dims}
        dx_vals = [self.domain.size[d] / grid_kwargs[d] for d in self.spatial_dims]
        min_dx = min(dx_vals)
        self._max_diffusion = 0.5 * (min_dx ** 2) / self.dt

        self._jit_step = self._create_jit_step()
        self._params = self.get_initial_params()

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
    def scalar_param_names(self) -> Tuple[str, ...]:
        if self._diffusion_type == 'scalar':
            return ('diffusion_coeff',)
        return ()

    @property
    def field_param_names(self) -> Tuple[str, ...]:
        if self._diffusion_type == 'field':
            return ('diffusion_field',)
        return ()

    @property
    def params(self) -> Tensor:
        return self._params

    @params.setter
    def params(self, params: Tensor):
        # Enforce stability clipping when updating parameters
        self._params = math.clip(params, 0.0, self._max_diffusion)

    def _create_diffusion_field(self) -> Tensor:
        """Create spatial diffusion field from config at REDUCED resolution."""
        # Use reduced resolution for params
        grid_kwargs = {name: self.resolution.get_size(name) for name in self.spatial_dims}
        sizes = {f'size_{d}': float(self.domain[i].size) for i, d in enumerate(self.spatial_dims)}

        def diffusion_fn(location):
            local_vars = {'math': math, **sizes}

            if self.n_spatial_dims == 1:
                local_vars['x'] = location
            elif self.n_spatial_dims == 2:
                local_vars['x'] = location.vector['x']
                local_vars['y'] = location.vector['y']
            elif self.n_spatial_dims == 3:
                local_vars['x'] = location.vector['x']
                local_vars['y'] = location.vector['y']
                local_vars['z'] = location.vector['z']

            result = eval(self._diffusion_value_str, local_vars)
            if not isinstance(result, math.Tensor):
                result = math.ones_like(location if self.n_spatial_dims == 1 else location.vector['x']) * result
            return result

        diffusion_grid = CenteredGrid(
            diffusion_fn,
            extrapolation.PERIODIC,
            bounds=self.domain,
            **grid_kwargs
        )
        return diffusion_grid.values

    def _create_jit_step(self):
        """
        Create JIT-compiled physics step with optional down/upsampling.
        
        If downsample_factor > 0:
        1. Downsample input state from full resolution
        2. Run physics at reduced resolution
        3. Upsample output back to full resolution
        """
        domain = self.domain
        dt = self.dt
        field_names = self.field_names
        spatial_dims = self.spatial_dims
        diffusion_type = self._diffusion_type
        downsample_factor = self.downsample_factor
        max_diffusion = self._max_diffusion
        
        # Reduced resolution grid kwargs (for physics computation)
        grid_kwargs = {name: self.resolution.get_size(name) for name in spatial_dims}
        
        # Full resolution grid kwargs (for input/output)
        full_grid_kwargs = {name: self.full_resolution.get_size(name) for name in spatial_dims}

        @jit_compile
        def burgers_step(state: Tensor, params: Tensor) -> Tuple[Tensor, Tensor]:
            # ============================================================
            # STEP 1: Downsample input state if needed
            # ============================================================
            if downsample_factor > 0:
                # Convert state tensor to grid at full resolution
                velocity_tensor_full = math.rename_dims(
                    state, 'field', channel(vector=','.join(spatial_dims))
                )
                state_grid = CenteredGrid(velocity_tensor_full, PERIODIC, bounds=domain, **full_grid_kwargs)
                
                # Downsample repeatedly
                for _ in range(downsample_factor):
                    state_grid = downsample2x(state_grid)
                
                # Back to tensor format
                working_state = math.rename_dims(
                    state_grid.values, 'vector', channel(field=','.join(field_names))
                )
            else:
                working_state = state
            
            # ============================================================
            # STEP 2: Run physics at reduced resolution
            # ============================================================
            
            # Extract diffusion coefficient from params (already at reduced resolution)
            if diffusion_type == 'scalar':
                diffusion_coeff = params.field['diffusion_coeff']
                diffusion_coeff = math.clip(diffusion_coeff, 0.0, max_diffusion)
            else:
                diffusion_coeff_tensor = params.field['diffusion_field']
                diffusion_coeff_tensor = math.clip(diffusion_coeff_tensor, 0.0, max_diffusion)

            # Convert field -> vector for physics
            velocity_tensor = math.rename_dims(
                working_state, 'field', channel(vector=','.join(spatial_dims))
            )

            # Advection at reduced resolution
            velocity_grid = CenteredGrid(velocity_tensor, PERIODIC, bounds=domain, **grid_kwargs)
            velocity_grid = advect.semi_lagrangian(velocity_grid, velocity_grid, dt=dt)

            # Diffusion
            if diffusion_type == 'scalar':
                velocity_grid = diffuse.explicit(velocity_grid, diffusion_coeff, dt=dt, substeps=5)
            else:
                # Spatially-varying diffusion: ∇·(D∇u) = D∇²u + ∇D·∇u
                diffusion_coeff_grid = CenteredGrid(diffusion_coeff_tensor, PERIODIC, bounds=domain, **grid_kwargs)
                grad_D = diffusion_coeff_grid.gradient(boundary=PERIODIC)

                components = list(math.unstack(velocity_grid, channel('vector')))

                for i, u_comp in enumerate(components):
                    laplacian_u = u_comp.laplace()
                    grad_u = u_comp.gradient()

                    diffusion_term = diffusion_coeff_grid.values * laplacian_u
                    advection_term = math.sum(grad_D.values * grad_u.values, 'vector')

                    u_new = u_comp.values + dt * (diffusion_term + advection_term)
                    u_comp = CenteredGrid(u_new, PERIODIC, bounds=domain, **grid_kwargs)
                    components[i] = u_comp

                velocity_grid = math.stack(components, channel('vector'))

            # ============================================================
            # STEP 3: Upsample output if needed
            # ============================================================
            if downsample_factor > 0:
                # Upsample back to full resolution
                output_grid = CenteredGrid(velocity_grid.values, PERIODIC, bounds=domain, **grid_kwargs)
                for _ in range(downsample_factor):
                    output_grid = upsample2x(output_grid)
                
                next_state = math.rename_dims(
                    output_grid.values, 'vector', channel(field=','.join(field_names))
                )
            else:
                next_state = math.rename_dims(
                    velocity_grid.values, 'vector', channel(field=','.join(field_names))
                )

            return next_state, params

        return burgers_step

    def get_initial_state(self, batch_size: int = 1) -> Tensor:
        """Generate noisy initial velocity at FULL resolution."""
        # Initial state is always at full resolution
        grid_kwargs = {name: self.full_resolution.get_size(name) for name in self.spatial_dims}
        vector_str = ','.join(self.spatial_dims)
        velocity_grid = CenteredGrid(
            Noise(batch(batch=batch_size), scale=self.domain.size/4, smoothness=2.0, vector=vector_str),
            extrapolation.PERIODIC,
            bounds=self.domain,
            **grid_kwargs
        )

        velocity = velocity_grid.values
        return math.rename_dims(
            velocity, 'vector', channel(field=','.join(self.field_names))
        )

    def get_real_params(self) -> Tensor:
        """Get real/target parameters at REDUCED resolution."""
        if self._diffusion_type == 'scalar':
            diffusion_value = float(eval(self._diffusion_value_str))
            diffusion_tensor = math.wrap(diffusion_value)
            return math.expand(diffusion_tensor, channel(field='diffusion_coeff'))
        else:
            diffusion_field = self._create_diffusion_field()
            return math.expand(diffusion_field, channel(field='diffusion_field'))

    def get_initial_params(self) -> Tensor:
        """Initialize parameters at REDUCED resolution."""
        if self._diffusion_type == 'scalar':
            return math.expand(math.wrap(1.0), channel(field='diffusion_coeff'))
        else:
            return math.ones_like(self.get_real_params())

    def forward(self, state: Tensor, params: Tensor) -> Tensor:
        """
        Forward pass through the model.
        
        Input state is at full resolution, output is at full resolution.
        Physics computation happens at reduced resolution internally.
        """
        pred, _ = self._jit_step(state, params)
        return pred

    def rollout(self, initial_state: Tensor, params: Tensor, num_steps: int) -> Tensor:
        """Rollout simulation for multiple steps."""
        trajectory, _ = iterate(self._jit_step, batch(time=num_steps), initial_state, params)
        if isinstance(trajectory, tuple):
            trajectory = trajectory[0]
        return trajectory

    def save(self, path):
        math.save(str(path), self.params)

    def load(self, path):
        self.params = math.load(str(path))