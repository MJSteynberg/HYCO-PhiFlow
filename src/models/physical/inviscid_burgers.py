"""Inviscid Burgers model with trainable potential field (forcing via gradient)."""

from typing import Dict, Any, Tuple

from phi.torch.flow import *
from phi.math import Shape, Tensor, batch, math
from phi.field import downsample2x, upsample2x
from phiml.math import channel

from .base import PhysicalModel
from src.models import ModelRegistry


@ModelRegistry.register_physical("InviscidBurgersModel")
class InviscidBurgersModel(PhysicalModel):
    """
    Inviscid Burgers equation model with trainable potential field.

    PDE: ∂u/∂t + u·∇u = -∇φ(x)

    where φ(x) is the unknown potential field to identify.
    The forcing is computed as the negative gradient of the potential,
    allowing the forcing to "push around" the solution in any direction.

    Supports optional downsampling for efficient training:
    - Input states at full resolution are downsampled during physics step
    - Physics computed at reduced resolution
    - Output stays at reduced resolution (upsample via forward() if needed)
    - Parameters stay at reduced resolution
    """

    def __init__(self, config: dict, downsample_factor: int = 0):
        super().__init__(config, downsample_factor)

        pde_params = config["model"]["physical"]["pde_params"]
        self._potential_type = pde_params['type']  # 'scalar' or 'field'
        self._potential_value_str = pde_params['value']

        # No stability constraints needed for potential-based forcing

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
        if self._potential_type == 'scalar':
            return ('potential_coeff',)
        return ()

    @property
    def field_param_names(self) -> Tuple[str, ...]:
        if self._potential_type == 'field':
            return ('potential_field',)
        return ()

    @property
    def params(self) -> Tensor:
        return self._params

    @params.setter
    def params(self, params: Tensor):
        # No clipping needed for potential field (can be any value)
        self._params = params

    def _create_potential_field(self) -> Tensor:
        """Create spatial potential field from config at REDUCED resolution."""
        # Use reduced resolution for params
        grid_kwargs = {name: self.resolution.get_size(name) for name in self.spatial_dims}
        sizes = {f'size_{d}': float(self.domain[i].size) for i, d in enumerate(self.spatial_dims)}

        def potential_fn(location):
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

            result = eval(self._potential_value_str, local_vars)
            if not isinstance(result, math.Tensor):
                result = math.ones_like(location if self.n_spatial_dims == 1 else location.vector['x']) * result
            return result

        potential_grid = CenteredGrid(
            potential_fn,
            extrapolation.PERIODIC,
            bounds=self.domain,
            **grid_kwargs
        )
        return potential_grid.values

    def _create_jit_step(self):
        """
        Create JIT-compiled physics step with optional downsampling.

        If downsample_factor > 0:
        1. Downsample input state from full resolution
        2. Run physics at reduced resolution
        3. Return output at reduced resolution (upsampling done externally)
        """
        domain = self.domain
        dt = self.dt
        field_names = self.field_names
        spatial_dims = self.spatial_dims
        potential_type = self._potential_type
        downsample_factor = self.downsample_factor
        n_spatial_dims = self.n_spatial_dims

        # Reduced resolution grid kwargs (for physics computation)
        grid_kwargs = {name: self.resolution.get_size(name) for name in spatial_dims}

        # Full resolution grid kwargs (for input/output)
        full_grid_kwargs = {name: self.full_resolution.get_size(name) for name in spatial_dims}

        @jit_compile
        def inviscid_burgers_step(state: Tensor, params: Tensor) -> Tuple[Tensor, Tensor]:
            # ============================================================
            # STEP 1: Get working state (downsample only if at full resolution)
            # ============================================================
            # Detect if input is at full or reduced resolution
            # This is needed because iterate() feeds output back as input
            first_spatial_dim = spatial_dims[0]
            input_res = state.shape.get_size(first_spatial_dim)
            full_res = full_grid_kwargs[first_spatial_dim]
            
            # Only downsample if input is at full resolution
            needs_downsample = downsample_factor > 0 and input_res == full_res
            
            if needs_downsample:
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
                # Input already at working resolution (or no downsampling needed)
                working_state = state

            # ============================================================
            # STEP 2: Run physics at reduced resolution
            # ============================================================

            # Extract potential from params (already at reduced resolution)
            if potential_type == 'scalar':
                potential_coeff = params.field['potential_coeff']
            else:
                potential_field_tensor = params.field['potential_field']

            # Convert field -> vector for physics
            velocity_tensor = math.rename_dims(
                working_state, 'field', channel(vector=','.join(spatial_dims))
            )

            # Advection at reduced resolution (inviscid: u·∇u term)
            velocity_grid = CenteredGrid(velocity_tensor, PERIODIC, bounds=domain, **grid_kwargs)
            velocity_grid = advect.semi_lagrangian(velocity_grid, velocity_grid, dt=dt)

            # Compute forcing as f = -∇φ
            if potential_type == 'scalar':
                # Scalar potential: uniform, gradient is zero (no forcing)
                # This case doesn't make much physical sense, but we handle it
                pass  # No forcing applied
            else:
                # Field potential: compute gradient to get forcing
                potential_grid = CenteredGrid(potential_field_tensor, PERIODIC, bounds=domain, **grid_kwargs)

                # Compute gradient: ∇φ
                grad_potential = potential_grid.gradient(boundary=PERIODIC)

                # Apply forcing: u_new = u_advected - dt * ∇φ
                # The negative sign makes high potential push flow away (like pressure)
                velocity_grid = velocity_grid - dt * grad_potential

            # ============================================================
            # STEP 3: Convert back to tensor format (NO upsampling)
            # ============================================================
            # Output stays at reduced resolution for training efficiency
            # Upsampling happens externally in forward() when needed
            next_state = math.rename_dims(
                velocity_grid.values, 'vector', channel(field=','.join(field_names))
            )

            return next_state, params

        return inviscid_burgers_step

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
        if self._potential_type == 'scalar':
            potential_value = float(eval(self._potential_value_str))
            potential_tensor = math.wrap(potential_value)
            return math.expand(potential_tensor, channel(field='potential_coeff'))
        else:
            potential_field = self._create_potential_field()
            return math.expand(potential_field, channel(field='potential_field'))

    def get_initial_params(self) -> Tensor:
        """Initialize parameters at REDUCED resolution (start from zero potential)."""
        if self._potential_type == 'scalar':
            return math.expand(math.wrap(0.0), channel(field='potential_coeff'))
        else:
            return math.zeros_like(self.get_real_params())

    def forward(self, state: Tensor, params: Tensor, upsample_output: bool = True) -> Tensor:
        """
        Forward pass through the model.

        Args:
            state: Input state (at full resolution if downsample_factor > 0)
            params: Model parameters (at reduced resolution)
            upsample_output: If True, upsample output to full resolution.
                           If False, return at reduced resolution (for training).

        Returns:
            Prediction at full resolution (if upsample_output=True) or
            reduced resolution (if upsample_output=False)
        """
        pred, _ = self._jit_step(state, params)

        # Upsample if requested and downsampling is enabled
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
                           If False, return at reduced resolution.

        Returns:
            Trajectory at full resolution (if upsample_output=True) or
            reduced resolution (if upsample_output=False)
        """
        trajectory, _ = iterate(self._jit_step, batch(time=num_steps), initial_state, params)
        if isinstance(trajectory, tuple):
            trajectory = trajectory[0]

        # Upsample each timestep if requested
        if upsample_output and self.downsample_factor > 0:
            upsampled_steps = []
            for t in range(trajectory.shape.get_size('time')):
                step = trajectory.time[t]
                upsampled_step = self._upsample_state(step, self.field_names)
                upsampled_steps.append(upsampled_step)
            trajectory = math.stack(upsampled_steps, batch('time'))

        return trajectory

    def save(self, path):
        math.save(str(path), self.params)

    def load(self, path):
        self.params = math.load(str(path))
