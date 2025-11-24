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
        
        self._velocity_field = self._create_velocity_field()
        self._jit_step = self._create_jit_step()

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
    def advection_coeff(self) -> float:
        return self._advection_coeff

    @advection_coeff.setter
    def advection_coeff(self, value: float):
        self._advection_coeff = float(value)
        self._jit_step = self._create_jit_step()

    @property
    def modulation_amplitude(self) -> float:
        return self._modulation_amplitude

    @modulation_amplitude.setter
    def modulation_amplitude(self, value: float):
        self._modulation_amplitude = float(value)
        # Re-create velocity field when amplitude changes
        self._velocity_field = self._create_velocity_field()
        self._jit_step = self._create_jit_step()

    @property
    def velocity_field(self) -> CenteredGrid:
        return self._velocity_field

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

            angular_vel = AngularVelocity(location=center, strength=1.0, falloff=ring_falloff)
            velocity_grid = CenteredGrid(
                angular_vel,
                extrapolation.PERIODIC,
                bounds=self.domain,
                **grid_kwargs
            )
            
            # Add modulation if enabled
            if self._modulation_enabled:
                modulation_field = self._create_modulation_field(center, ring_radius, ring_width, grid_kwargs)
                velocity_grid = velocity_grid + modulation_field
                
            return velocity_grid
        else:
            def velocity_fn(x):
                size_x = float(self.domain[0].size)
                vx = 0.5 + 0.1 * math.sin(2 * math.pi * x / size_x)
                return math.stack([vx], channel("vector"))
            return CenteredGrid(velocity_fn, PERIODIC, bounds=self.domain, **grid_kwargs)

    def _create_modulation_field(self, center, ring_radius, ring_width, grid_kwargs):
        """Create the sinusoidal modulation field."""
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
            
            return math.stack([vel_x, vel_y], channel(vector='x,y'))

        return CenteredGrid(
            modulated_velocity,
            extrapolation.PERIODIC,
            bounds=self.domain,
            **grid_kwargs
        )

    def _create_jit_step(self):
        """Create JIT-compiled physics step."""
        domain = self.domain
        dt = self.dt
        advection_coeff = self._advection_coeff
        velocity_field = self._velocity_field
        grid_kwargs = {name: self.resolution.get_size(name) for name in self.spatial_dims}
        static_names = self.static_field_names

        @jit_compile
        def advection_step(state: Tensor) -> Tensor:
            # Extract density from field channel
            density_tensor = state.field['density']
            density_grid = CenteredGrid(density_tensor, PERIODIC, bounds=domain, **grid_kwargs)

            # Advect density
            scaled_velocity = velocity_field * advection_coeff
            density_grid = advect.semi_lagrangian(density_grid, scaled_velocity, dt=dt)

            # Create output with field channel
            density_out = math.expand(density_grid.values, channel(field='density'))

            # Extract and preserve static fields
            static = math.stack(
                [state.field[n] for n in static_names],
                channel(field=','.join(static_names))
            )

            return math.concat([density_out, static], 'field')

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

        # Velocity from pre-computed field
        vel_values = self._velocity_field.values
        static_names = ','.join(self.static_field_names)
        velocity = math.rename_dims(vel_values, 'vector', channel(field=static_names))
        velocity = math.expand(velocity, batch(batch=batch_size))

        return math.concat([density, velocity], 'field')

    def forward(self, state: Tensor, advection_coeff: Tensor = None, modulation_amplitude: Tensor = None) -> Tensor:
        """
        Forward pass through the model.
        
        Args:
            state: Current state tensor
            advection_coeff: Optional override for advection coefficient (for physical training)
            modulation_amplitude: Optional override for modulation amplitude (for physical training, currently unused)
        
        Note: This simplified version ignores modulation_amplitude since the base model doesn't have modulation.
        """
        # If parameters are provided (during physical training), use non-JIT path
        if advection_coeff is not None or modulation_amplitude is not None:
            # Extract density from field channel
            density_tensor = state.field['density']
            grid_kwargs = {name: self.resolution.get_size(name) for name in self.spatial_dims}
            density_grid = CenteredGrid(density_tensor, PERIODIC, bounds=self.domain, **grid_kwargs)

            # Use provided coefficient or default
            coeff = advection_coeff if advection_coeff is not None else math.tensor(self._advection_coeff)
            
            # Rebuild velocity field if modulation_amplitude is provided
            if self._modulation_enabled and modulation_amplitude is not None:
                # Create base angular velocity field
                center_values = [float(self.domain[i].size) / 2 for i in range(2)]
                vector_names = ','.join(self.spatial_dims)
                center = math.tensor(center_values, channel(vector=vector_names))
                ring_radius = float(self.domain[0].size) * 0.3
                ring_width = float(self.domain[0].size) * 0.15
                
                def ring_falloff(distances):
                    r = math.vec_length(distances)
                    return math.exp(-((r - ring_radius) / ring_width) ** 2)
                
                from phi.field import AngularVelocity
                angular_vel = AngularVelocity(location=center, strength=1.0, falloff=ring_falloff)
                velocity_grid = CenteredGrid(angular_vel, extrapolation.PERIODIC, bounds=self.domain, **grid_kwargs)
                
                # Create modulation field using a location function (avoids nested function issues)
                def modulation_fn(location):
                    diff = location - center
                    r = math.vec_length(diff)
                    theta = math.arctan(diff.vector['y'], diff.vector['x'])
                    falloff = math.exp(-((r - ring_radius) / ring_width) ** 2)
                    modulation = modulation_amplitude * math.sin(self._modulation_lobes * theta)
                    magnitude = falloff * modulation
                    tangent_x = -diff.vector['y'] / (r + 1e-6)
                    tangent_y = diff.vector['x'] / (r + 1e-6)
                    vel_x = magnitude * tangent_x
                    vel_y = magnitude * tangent_y
                    return math.stack([vel_x, vel_y], channel(vector='x,y'))
                
                modulation_grid = CenteredGrid(modulation_fn, extrapolation.PERIODIC, bounds=self.domain, **grid_kwargs)
                
                # Add modulation to base field
                velocity_grid = velocity_grid + modulation_grid
            else:
                velocity_grid = self._velocity_field

            # Advect with combined velocity field
            scaled_velocity = velocity_grid * coeff
            density_grid = advect.semi_lagrangian(density_grid, scaled_velocity, dt=self.dt)

            # Create output with field channel
            density_out = math.expand(density_grid.values, channel(field='density'))

            # Extract and preserve static fields
            static_names = self.static_field_names
            static = math.stack(
                [state.field[n] for n in static_names],
                channel(field=','.join(static_names))
            )

            return math.concat([density_out, static], 'field')
        else:
            # Use pre-compiled JIT step for inference
            return self._jit_step(state)

    def rollout(self, initial_state: Tensor, num_steps: int) -> Tensor:
        """Rollout simulation for multiple steps."""
        trajectory = iterate(self._jit_step, batch(time=num_steps), initial_state)
        if isinstance(trajectory, tuple):
            trajectory = trajectory[0]
        return trajectory
