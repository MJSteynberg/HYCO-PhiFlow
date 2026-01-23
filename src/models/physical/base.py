"""Abstract base class for physical PDE models with optional down/upsampling."""

from abc import ABC, abstractmethod
from phi.flow import *
from phi.math import Shape, spatial, Tensor
from phi.field import downsample2x, upsample2x
from phiml.math import channel
from typing import Dict, Any, Tuple


class PhysicalModel(ABC):
    """
    Base class for physical PDE models with tensor-first architecture.
    
    Supports optional downsampling for efficient training:
    - Input states are downsampled before physics computation
    - Physics runs at reduced resolution
    - Outputs are upsampled back to original resolution
    - Parameters (learnable fields) stay at reduced resolution

    State format: Tensor(batch?, x, y?, field='vel_x,vel_y,...')
    """

    def __init__(self, config: Dict[str, Any], downsample_factor: int = 0):
        self._parse_config(config, downsample_factor)
        self._jit_step = None

    def _parse_config(self, config: Dict[str, Any], downsample_factor: int):
        """Parse configuration to setup domain and resolution."""
        dim_config = config["model"]["physical"]["domain"]["dimensions"]

        box_kwargs = {name: dim['size'] for name, dim in dim_config.items()}
        self.domain = Box(**box_kwargs)

        # Store FULL resolution (original data resolution)
        full_res_kwargs = {name: dim['resolution'] for name, dim in dim_config.items()}
        self.full_resolution = spatial(**full_res_kwargs)

        # Store REDUCED resolution for physics computation
        reduced_res_kwargs = {
            name: dim['resolution'] // (2**downsample_factor)
            for name, dim in dim_config.items()
        }
        self.resolution = spatial(**reduced_res_kwargs)  # This is the working resolution

        self.spatial_dims = list(dim_config.keys())
        self.n_spatial_dims = len(self.spatial_dims)
        self.downsample_factor = downsample_factor
        self.dt = float(config["model"]["physical"]["dt"])

    @property
    @abstractmethod
    def field_names(self) -> Tuple[str, ...]:
        """Field names for output tensor (e.g., ('vel_x', 'vel_y'))."""
        pass

    @property
    def num_channels(self) -> int:
        return len(self.field_names)

    @property
    def static_field_names(self) -> Tuple[str, ...]:
        """Static field names (fields that don't change during simulation). Override in subclass."""
        return ()

    @property
    def dynamic_field_names(self) -> Tuple[str, ...]:
        """Dynamic field names (fields that change during simulation)."""
        return tuple(f for f in self.field_names if f not in self.static_field_names)

    @abstractmethod
    def _create_jit_step(self):
        """Create JIT-compiled physics step: Tensor -> Tensor."""
        pass

    @abstractmethod
    def get_initial_state(self, batch_size: int = 1) -> Tensor:
        """Generate random initial state as unified tensor."""
        pass

    def _downsample_state(self, state: Tensor, field_names: Tuple[str, ...]) -> Tensor:
        """
        Downsample state tensor from full resolution to working resolution.
        
        Args:
            state: Tensor at full resolution with 'field' channel dimension
            field_names: Names of the fields for grid creation
            
        Returns:
            Downsampled tensor at self.resolution
        """
        if self.downsample_factor == 0:
            return state
        
        # Get full resolution grid kwargs
        full_grid_kwargs = {name: self.full_resolution.get_size(name) for name in self.spatial_dims}
        
        # Create grid from state values
        grid = CenteredGrid(
            state,
            extrapolation.PERIODIC,
            bounds=self.domain,
            **full_grid_kwargs
        )
        
        # Apply downsample2x repeatedly
        for _ in range(self.downsample_factor):
            grid = downsample2x(grid)
        
        return grid.values

    def _upsample_state(self, state: Tensor, field_names: Tuple[str, ...]) -> Tensor:
        """
        Upsample state tensor from working resolution back to full resolution.

        Args:
            state: Tensor at working resolution with 'field' channel dimension
            field_names: Names of the fields for grid creation

        Returns:
            Upsampled tensor at self.full_resolution
        """
        if self.downsample_factor == 0:
            return state

        # Get working resolution grid kwargs
        grid_kwargs = {name: self.resolution.get_size(name) for name in self.spatial_dims}

        # Create grid from state values
        grid = CenteredGrid(
            state,
            extrapolation.PERIODIC,
            bounds=self.domain,
            **grid_kwargs
        )

        # Apply upsample2x repeatedly
        for _ in range(self.downsample_factor):
            grid = upsample2x(grid)

        return grid.values

    def _downsample_targets(self, targets: Tensor) -> Tensor:
        """
        Downsample target trajectory from data resolution to working resolution.

        Used during training to compare predictions at reduced resolution.

        Args:
            targets: Tensor at data resolution with 'time' and 'field' dimensions

        Returns:
            Downsampled tensor at self.resolution
        """
        if self.downsample_factor == 0:
            return targets

        # Get actual resolution from target tensor's spatial dimensions
        # This handles cases where data resolution differs from model's full_resolution
        target_grid_kwargs = {name: targets.shape.get_size(name) for name in self.spatial_dims}

        # Downsample each timestep
        downsampled_steps = []
        for t in range(targets.shape.get_size('time')):
            target_step = targets.time[t]

            # Create grid from target values using actual data resolution
            grid = CenteredGrid(
                target_step,
                extrapolation.PERIODIC,
                bounds=self.domain,
                **target_grid_kwargs
            )

            # Calculate how many times to downsample to reach working resolution
            # We need to match the working resolution, not just apply downsample_factor blindly
            target_res = target_grid_kwargs[self.spatial_dims[0]]
            working_res = self.resolution.get_size(self.spatial_dims[0])

            # Downsample until we reach working resolution
            while grid.shape.get_size(self.spatial_dims[0]) > working_res:
                grid = downsample2x(grid)

            downsampled_steps.append(grid.values)

        # Stack back into time dimension
        return math.stack(downsampled_steps, batch('time'))

    def forward(self, state: Tensor) -> Tensor:
        """Single physics step."""
        if self._jit_step is None:
            raise RuntimeError("_jit_step not initialized")
        return self._jit_step(state)

    def rollout(self, initial_state: Tensor, num_steps: int) -> Tensor:
        """Rollout simulation for multiple steps (includes initial state)."""
        if self._jit_step is None:
            raise RuntimeError("_jit_step not initialized")
        trajectory, = iterate(self._jit_step, batch(time=num_steps), initial_state)
        return trajectory

    def __call__(self, state: Tensor) -> Tensor:
        return self.forward(state)

    @staticmethod
    def _select_proportional_indices(total_count: int, sample_count: int):
        """Select indices proportionally across the dataset."""
        if sample_count >= total_count:
            return list(range(total_count))
        elif sample_count <= 0:
            return []
        step = total_count / sample_count
        indices = [int(i * step) for i in range(sample_count)]
        return sorted(list(set(indices)))[:sample_count]