"""Advection model using unified field channel dimension with down/upsampling support."""

from typing import Dict, Any, Tuple

from phi.torch.flow import *
from phi.math import Shape, Tensor, batch, math
from phi.field import AngularVelocity, downsample2x, upsample2x
from phiml.math import channel

from .base import PhysicalModel
from src.models import ModelRegistry


@ModelRegistry.register_physical("AdvectionModel")
class AdvectionModel(PhysicalModel):
    """
    Advection model with static velocity and dynamic density.

    Supports optional downsampling for efficient training:
    - Input states are downsampled during physics step
    - Physics computed at reduced resolution
    - Output stays at reduced resolution (upsample via forward() if needed)
    - Parameters stay at reduced resolution
    """

    def __init__(self, config: dict, downsample_factor: int = 0):
        super().__init__(config, downsample_factor)

        self._advection_coeff = float(eval(config["model"]["physical"]["pde_params"]["value"]))
        
        # Parse modulation configuration
        modulation_config = config["model"]["physical"].get("modulation", {})
        self._modulation_enabled = modulation_config.get("enabled", False)
        self._modulation_amplitude = float(modulation_config.get("amplitude", 0.0))
        self._modulation_lobes = int(modulation_config.get("lobes", 6))
        
        self._jit_step = self._create_jit_step()
        self._params = self.get_initial_params()

    @property
    def static_field_names(self) -> Tuple[str, ...]:
        if self.n_spatial_dims == 2:
            return ('vel_x', 'vel_y')
        return ('vel_x',)

    @property
    def dynamic_field_names(self) -> Tuple[str, ...]:
        return ('density',)

    @property
    def field_names(self) -> Tuple[str, ...]:
        return self.dynamic_field_names + self.static_field_names

    @property
    def scalar_param_names(self) -> Tuple[str, ...]:
        return ()

    @property
    def field_param_names(self) -> Tuple[str, ...]:
        if self.n_spatial_dims == 2:
            return ('mod_field_x', 'mod_field_y')
        return ('mod_field_x',)

    @property
    def params(self) -> Tensor:
        return self._params 
    
    @params.setter
    def params(self, params: Tensor):
        self._params = params
    
    def _create_velocity_field(self, use_full_resolution: bool = False) -> Tensor:
        """
        Create velocity field using AngularVelocity with ring falloff.
        
        Args:
            use_full_resolution: If True, create at full resolution. Otherwise use working resolution.
        """
        if use_full_resolution:
            grid_kwargs = {name: self.full_resolution.get_size(name) for name in self.spatial_dims}
        else:
            grid_kwargs = {name: self.resolution.get_size(name) for name in self.spatial_dims}

        if self.n_spatial_dims == 2:
            center_values = [float(self.domain[i].size) / 2 for i in range(2)]
            vector_names = ','.join(self.spatial_dims)
            center = math.tensor(center_values, channel(vector=vector_names))

            domain_size = float(self.domain[0].size)
            ring_radius = domain_size * 0.3
            ring_width = domain_size * 0.15

            def ring_falloff(distances):
                r = math.vec_length(distances)
                return math.exp(-((r - ring_radius) / ring_width) ** 2)

            angular_vel = AngularVelocity(location=center, strength=0.5, falloff=ring_falloff)
            velocity_grid = CenteredGrid(
                angular_vel,
                extrapolation.PERIODIC,
                bounds=self.domain,
                **grid_kwargs
            )
                
            return velocity_grid.values
        else:
            def velocity_fn(x):
                size_x = float(self.domain[0].size)
                vx = 0.5 + 0.1 * math.sin(2 * math.pi * x / size_x)
                return math.stack([vx], channel("vector"))
            
            velocity_grid = CenteredGrid(velocity_fn, PERIODIC, bounds=self.domain, **grid_kwargs)
            return velocity_grid.values

    def _create_modulation_field(self) -> Tensor:
        """Create the sinusoidal modulation field at REDUCED resolution."""
        center_values = [float(self.domain[i].size) / 2 for i in range(2)]
        vector_names = ','.join(self.spatial_dims)
        center = math.tensor(center_values, channel(vector=vector_names))

        domain_size = float(self.domain[0].size)
        ring_radius = domain_size * 0.3
        ring_width = domain_size * 0.05
        
        # Use reduced resolution for params
        grid_kwargs = {name: self.resolution.get_size(name) for name in self.spatial_dims}
        
        def modulated_velocity(location):
            diff = location - center
            r = math.vec_length(diff)
            theta = math.arctan(diff.vector['y'], diff.vector['x'])
            
            falloff = math.exp(-((r - ring_radius) / ring_width) ** 2)
            modulation = self._modulation_amplitude * math.sin(self._modulation_lobes * theta)
            magnitude = falloff * modulation
            
            tangent_x = -diff.vector['y'] / (r + 1e-6)
            tangent_y = diff.vector['x'] / (r + 1e-6)
            
            vel_x = magnitude * tangent_x
            vel_y = magnitude * tangent_y
            
            return 5 * math.stack([vel_x, vel_y], channel(vector='x,y'))

        modulation_grid = CenteredGrid(
            modulated_velocity,
            extrapolation.PERIODIC,
            bounds=self.domain,
            **grid_kwargs
        )
        return modulation_grid.values

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
        downsample_factor = self.downsample_factor
        
        # Reduced resolution grid kwargs (for physics computation)
        grid_kwargs = {name: self.resolution.get_size(name) for name in self.spatial_dims}
        
        # Full resolution grid kwargs (for input/output)
        full_grid_kwargs = {name: self.full_resolution.get_size(name) for name in self.spatial_dims}
        
        static_names = self.static_field_names
        field_params_names = self.field_param_names
        all_field_names = self.field_names
        spatial_dims = self.spatial_dims

        @jit_compile
        def advection_step(state: Tensor, params: Tensor) -> Tuple[Tensor, Tensor]:
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
                # Downsample density from full resolution
                density_full = state.field['density']
                density_grid = CenteredGrid(density_full, PERIODIC, bounds=domain, **full_grid_kwargs)
                for _ in range(downsample_factor):
                    density_grid = downsample2x(density_grid)
                working_density = density_grid.values
                
                # Downsample static velocity fields
                static_full = math.stack(
                    [state.field[n] for n in static_names],
                    channel(field=','.join(static_names))
                )
                velocity_full = rename_dims(static_full, channel(field=','.join(static_names)), 
                                           channel(vector=','.join(spatial_dims)))
                velocity_grid = CenteredGrid(velocity_full, PERIODIC, bounds=domain, **full_grid_kwargs)
                for _ in range(downsample_factor):
                    velocity_grid = downsample2x(velocity_grid)
                working_velocity = velocity_grid.values
            else:
                # Input already at working resolution (or no downsampling needed)
                working_density = state.field['density']
                static = math.stack(
                    [state.field[n] for n in static_names],
                    channel(field=','.join(static_names))
                )
                working_velocity = rename_dims(static, channel(field=','.join(static_names)), 
                                              channel(vector=','.join(spatial_dims)))

            # ============================================================
            # STEP 2: Run physics at reduced resolution
            # ============================================================
            
            # Create grids from working state
            density_grid = CenteredGrid(working_density, PERIODIC, bounds=domain, **grid_kwargs)
            velocity_grid = CenteredGrid(working_velocity, PERIODIC, bounds=domain, **grid_kwargs)

            # Create grids from params (already at reduced resolution)
            field_params = math.stack(
                [params.field[n] for n in field_params_names],
                channel(field=','.join(field_params_names))
            )
            modulation_field = rename_dims(field_params, 
                                          channel(field=','.join(field_params_names)), 
                                          channel(vector=','.join(spatial_dims)))

            # Velocity added to modulation field
            velocity_field = velocity_grid + modulation_field 
            
            # Advect density using velocity
            density_grid = advect.semi_lagrangian(density_grid, velocity_field, dt=dt)

            # ============================================================
            # STEP 3: Convert back to tensor format (NO upsampling)
            # ============================================================
            # Output stays at reduced resolution for training efficiency
            # Upsampling happens externally in forward() when needed
            density_out = math.expand(density_grid.values, channel(field='density'))
            static_out = math.rename_dims(velocity_grid.values, 'vector',
                                         channel(field=','.join(static_names)))

            return math.concat([density_out, static_out], 'field'), params

        return advection_step

    def get_initial_state(self, batch_size: int = 1) -> Tensor:
        """Generate initial state with density and velocity fields at FULL resolution."""
        # Use full resolution for initial state
        grid_kwargs = {name: self.full_resolution.get_size(name) for name in self.spatial_dims}

        # Random density
        density_grid = CenteredGrid(
            Noise(scale=30.0, smoothness=5.0),
            extrapolation.PERIODIC,
            bounds=self.domain,
            **grid_kwargs
        )
        density_grid = math.tanh(2 * density_grid)
        density = math.expand(density_grid.values, batch(batch=batch_size))
        density = math.expand(density, channel(field='density'))

        # Velocity from pre-computed field at full resolution
        vel_values = self._create_velocity_field(use_full_resolution=True)
        static_names = ','.join(self.static_field_names)
        velocity = math.rename_dims(vel_values, 'vector', channel(field=static_names))
        velocity = math.expand(velocity, batch(batch=batch_size))

        return math.concat([density, velocity], 'field')

    def get_real_params(self) -> Tensor:
        """Get real/target parameters at REDUCED resolution."""
        modulation_field = self._create_modulation_field()
        field_names = ','.join(self.field_param_names)
        modulation_field = math.rename_dims(modulation_field, 'vector', channel(field=field_names))
        return math.concat([modulation_field], 'field')

    def get_initial_params(self) -> Tensor:
        """Initialize parameters at REDUCED resolution."""
        return math.ones_like(self.get_real_params())

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