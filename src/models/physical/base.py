"""Abstract base class for physical PDE models."""

from abc import ABC, abstractmethod
from phi.flow import Field, Box, math, batch, iterate
from phi.math import Shape, spatial, Tensor
from phiml.math import channel
from typing import Dict, Any, Tuple


class PhysicalModel(ABC):
    """
    Base class for physical PDE models with tensor-first architecture.

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

        res_kwargs = {
            name: dim['resolution'] // (2**downsample_factor)
            for name, dim in dim_config.items()
        }
        self.resolution = spatial(**res_kwargs)

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
