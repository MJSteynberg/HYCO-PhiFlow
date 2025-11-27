"""Advection model using unified field channel dimension."""

from typing import Dict, Any, Tuple

from phi.torch.flow import *
from phi.math import Shape, Tensor, batch, math
from phi.field import AngularVelocity
from phiml.math import channel

from .base import PhysicalModel
from src.models import ModelRegistry


@ModelRegistry.register_physical("AdvectionModel")
class AdvectionModel(PhysicalModel):
    """Advection model with static velocity and dynamic density."""

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
    
    def _create_velocity_field(self) -> CenteredGrid:
        """Create velocity field using AngularVelocity with ring falloff."""
        grid_kwargs = {name: self.resolution.get_size(name) for name in self.spatial_dims}

        if self.n_spatial_dims == 2:
            # Create location tensor with proper spatial dimension names
            center_values = [float(self.domain[i].size) / 2 for i in range(2)]
            vector_names = ','.join(self.spatial_dims)
            center = math.tensor(center_values, channel(vector=vector_names))

            # Ring parameters based on domain size
            domain_size = float(self.domain[0].size)
            ring_radius = domain_size * 0.3  # Ring at 30% of domain size
            ring_width = domain_size * 0.15  # Width of the ring

            def ring_falloff(distances):
                """Ring-like falloff: velocity peaks at ring_radius."""
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

    def _create_modulation_field(self):
        """Create the sinusoidal modulation field."""
        center_values = [float(self.domain[i].size) / 2 for i in range(2)]
        vector_names = ','.join(self.spatial_dims)
        center = math.tensor(center_values, channel(vector=vector_names))

        # Ring parameters based on domain size
        domain_size = float(self.domain[0].size)
        ring_radius = domain_size * 0.3  # Ring at 30% of domain size
        ring_width = domain_size * 0.05  # Width of the ring
        grid_kwargs = {name: self.resolution.get_size(name) for name in self.spatial_dims}
        
        def modulated_velocity(location):
            # Calculate vector from center
            diff = location - center
            
            # Calculate distance (r) and angle (theta)
            r = math.vec_length(diff)
            theta = math.arctan(diff.vector['y'], diff.vector['x'])
            
            # Ring falloff (Gaussian) - same as base field
            falloff = math.exp(-((r - ring_radius) / ring_width) ** 2)
            
            # Sinusoidal modulation: additive field with amplitude * sin(n * theta)
            modulation = self._modulation_amplitude * math.sin(self._modulation_lobes * theta)
            
            # Total magnitude of the ADDITIVE field
            magnitude = falloff * modulation
            
            # Direction: Tangent to the circle (-y, x)
            tangent_x = -diff.vector['y'] / (r + 1e-6)
            tangent_y = diff.vector['x'] / (r + 1e-6)
            
            # Combine
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
        """Create JIT-compiled physics step."""
        domain = self.domain
        dt = self.dt
        
        grid_kwargs = {name: self.resolution.get_size(name) for name in self.spatial_dims}
        static_names = self.static_field_names
        field_params_names = self.field_param_names
        scalar_params_names = self.scalar_param_names

        @jit_compile
        def advection_step(state: Tensor, params: Tensor) -> Tensor:
            
            # Extract and preserve static fields from STATE (which contains original velocity)
            static = math.stack(
                [state.field[n] for n in static_names],
                channel(field=','.join(static_names))
            )
            velocity = rename_dims(static, channel(field=','.join(static_names)), channel(vector=','.join(grid_kwargs.keys())))

            # Create grids from state
            density_grid = CenteredGrid(state.field['density'], PERIODIC, bounds=domain, **grid_kwargs)
            velocity_grid = CenteredGrid(velocity, PERIODIC, bounds=domain, **grid_kwargs)

            # Create grids from params
            field_params = math.stack(
                [params.field[n] for n in field_params_names],
                channel(field=','.join(field_params_names))
            )
            modulation_field = rename_dims(field_params, channel(field=','.join(field_params_names)), channel(vector=','.join(grid_kwargs.keys())))   

            # Velocity added to modulation field
            velocity_field = velocity_grid + modulation_field 
            # Advect density using velocity
            density_grid = advect.semi_lagrangian(density_grid, velocity_field, dt=dt)

            # Create output with field channel
            density_out = math.expand(density_grid.values, channel(field='density'))

            return math.concat([density_out, static], 'field'), params

        return advection_step

    def get_initial_state(self, batch_size: int = 1) -> Tensor:
        """Generate initial state with density and velocity fields."""
        grid_kwargs = {name: self.resolution.get_size(name) for name in self.spatial_dims}

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

        # Velocity from pre-computed field (Base velocity only)
        vel_values = self._create_velocity_field()
        static_names = ','.join(self.static_field_names)
        velocity = math.rename_dims(vel_values, 'vector', channel(field=static_names))
        velocity = math.expand(velocity, batch(batch=batch_size))

        return math.concat([density, velocity], 'field')

    def get_real_params(self) -> Tensor:
        modulation_field = self._create_modulation_field()
        field_names = ','.join(self.field_param_names)
        modulation_field = math.rename_dims(modulation_field, 'vector', channel(field=field_names))
        return math.concat([modulation_field], 'field')

    def get_initial_params(self) -> Tensor:
        return math.ones_like(self.get_real_params())

    def forward(self, state: Tensor, params: Tensor) -> Tensor:
        """
        Forward pass through the model.
        
        Args:
            state: Current state tensor
        """
            
        # Call JIT step with resolved arguments
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

