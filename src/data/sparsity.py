"""Sparsity configuration and observation masking for sparse data scenarios."""

from dataclasses import dataclass, field as dataclass_field
from typing import Optional, Tuple, List, Union
from phi.flow import *


@dataclass
class TemporalSparsityConfig:
    """Configuration for temporal (time) sparsity."""
    enabled: bool = False
    mode: str = 'full'  # 'full', 'endpoints', 'uniform', 'custom'

    # For 'endpoints' mode: fraction of trajectory at start/end to keep
    start_fraction: float = 0.1
    end_fraction: float = 0.1

    # For 'uniform' mode: keep every nth timestep
    uniform_stride: int = 1

    # For 'custom' mode: explicit list of time indices (as fractions 0-1)
    custom_fractions: List[float] = dataclass_field(default_factory=list)


@dataclass
class SpatialSparsityConfig:
    """Configuration for spatial sparsity."""
    enabled: bool = False
    mode: str = 'full'  # 'full', 'range', 'center', 'random_points'

    # For 'range' mode: specify bounds as fractions (0-1) of domain
    x_range: Optional[Tuple[float, float]] = None  # e.g., (0.0, 0.5) for first half
    y_range: Optional[Tuple[float, float]] = None

    # For 'center' mode: observe a centered region
    center_fraction: float = 0.5  # Fraction of domain to observe (centered)

    # For 'random_points' mode: number of random observation points
    num_random_points: int = 100
    random_seed: int = 42


@dataclass
class SparsityConfig:
    """Combined sparsity configuration."""
    temporal: TemporalSparsityConfig = dataclass_field(default_factory=TemporalSparsityConfig)
    spatial: SpatialSparsityConfig = dataclass_field(default_factory=SpatialSparsityConfig)


class TemporalMask:
    """
    Computes which time indices are visible given a temporal sparsity config.

    This class is used by the Dataset to filter accessible timesteps.
    """

    def __init__(self, config: TemporalSparsityConfig, trajectory_length: int):
        """
        Args:
            config: Temporal sparsity configuration
            trajectory_length: Total number of timesteps in trajectory
        """
        self.config = config
        self.trajectory_length = trajectory_length
        self._visible_indices = self._compute_visible_indices()

    def _compute_visible_indices(self) -> List[int]:
        """Compute list of visible time indices based on config."""
        T = self.trajectory_length

        if not self.config.enabled or self.config.mode == 'full':
            return list(range(T))

        elif self.config.mode == 'endpoints':
            start_count = max(1, int(T * self.config.start_fraction))
            end_count = max(1, int(T * self.config.end_fraction))
            start_indices = list(range(start_count))
            end_indices = list(range(T - end_count, T))
            # Remove duplicates and sort
            return sorted(list(set(start_indices + end_indices)))

        elif self.config.mode == 'uniform':
            return list(range(0, T, self.config.uniform_stride))

        elif self.config.mode == 'custom':
            indices = [int(f * (T - 1)) for f in self.config.custom_fractions]
            return sorted(list(set(indices)))

        return list(range(T))

    @property
    def visible_indices(self) -> List[int]:
        """List of visible time indices."""
        return self._visible_indices

    @property
    def num_visible(self) -> int:
        """Number of visible timesteps."""
        return len(self._visible_indices)

    def is_visible(self, t: int) -> bool:
        """Check if timestep t is visible."""
        return t in self._visible_indices

    def get_visible_mask_tensor(self) -> Tensor:
        """Return a PhiML tensor mask (1.0 for visible, 0.0 for hidden)."""
        mask_values = [1.0 if i in self._visible_indices else 0.0
                       for i in range(self.trajectory_length)]
        return math.tensor(mask_values, batch('time'))


class SpatialMask:
    """
    Creates spatial observation masks for loss computation.

    The model sees the full domain, but loss is only computed on visible regions.
    """

    def __init__(self, config: SpatialSparsityConfig, spatial_shape: Shape):
        """
        Args:
            config: Spatial sparsity configuration
            spatial_shape: Shape object containing spatial dimensions (e.g., spatial(x=100, y=100))
        """
        self.config = config
        self.spatial_shape = spatial_shape
        self._mask = self._build_mask()

    def _build_mask(self) -> Tensor:
        """Build the spatial mask tensor."""
        if not self.config.enabled or self.config.mode == 'full':
            return math.ones(self.spatial_shape)

        elif self.config.mode == 'range':
            return self._build_range_mask()

        elif self.config.mode == 'center':
            return self._build_center_mask()

        elif self.config.mode == 'random_points':
            return self._build_random_points_mask()

        return math.ones(self.spatial_shape)

    def _build_range_mask(self) -> Tensor:
        """Build mask for rectangular range mode."""
        mask = math.ones(self.spatial_shape)

        # Handle x dimension
        if self.config.x_range is not None and 'x' in self.spatial_shape.names:
            x_size = self.spatial_shape['x'].size
            x_min = int(self.config.x_range[0] * x_size)
            x_max = int(self.config.x_range[1] * x_size)

            x_coords = math.arange(spatial(x=x_size))
            x_mask = (x_coords >= x_min) & (x_coords < x_max)
            mask = mask * math.cast(x_mask, float)

        # Handle y dimension
        if self.config.y_range is not None and 'y' in self.spatial_shape.names:
            y_size = self.spatial_shape['y'].size
            y_min = int(self.config.y_range[0] * y_size)
            y_max = int(self.config.y_range[1] * y_size)

            y_coords = math.arange(spatial(y=y_size))
            y_mask = (y_coords >= y_min) & (y_coords < y_max)
            mask = mask * math.cast(y_mask, float)

        return mask

    def _build_center_mask(self) -> Tensor:
        """Build mask for centered observation region."""
        mask = math.ones(self.spatial_shape)
        frac = self.config.center_fraction

        for dim_name in self.spatial_shape.names:
            dim_size = self.spatial_shape[dim_name].size
            margin = int(dim_size * (1 - frac) / 2)
            start = margin
            end = dim_size - margin

            coords = math.arange(spatial(**{dim_name: dim_size}))
            dim_mask = (coords >= start) & (coords < end)
            mask = mask * math.cast(dim_mask, float)

        return mask

    def _build_random_points_mask(self) -> Tensor:
        """Build mask with random observation points."""
        import numpy as np
        np.random.seed(self.config.random_seed)

        # Start with zeros
        mask = math.zeros(self.spatial_shape)

        # Get total number of points and sample
        total_points = self.spatial_shape.volume
        num_points = min(self.config.num_random_points, total_points)

        # For 1D case
        if len(self.spatial_shape.names) == 1:
            dim_name = self.spatial_shape.names[0]
            indices = np.random.choice(self.spatial_shape[dim_name].size, num_points, replace=False)
            mask_np = np.zeros(self.spatial_shape[dim_name].size)
            mask_np[indices] = 1.0
            mask = math.tensor(mask_np, spatial(**{dim_name: len(mask_np)}))

        # For 2D case
        elif len(self.spatial_shape.names) == 2:
            x_size = self.spatial_shape['x'].size
            y_size = self.spatial_shape['y'].size
            flat_indices = np.random.choice(x_size * y_size, num_points, replace=False)
            mask_np = np.zeros((y_size, x_size))
            for idx in flat_indices:
                y_idx, x_idx = divmod(idx, x_size)
                mask_np[y_idx, x_idx] = 1.0
            mask = math.tensor(mask_np, spatial(y=y_size, x=x_size))

        return mask

    @property
    def mask(self) -> Tensor:
        """The spatial mask tensor."""
        return self._mask

    @property
    def visible_fraction(self) -> float:
        """Fraction of spatial domain that is visible."""
        return float(math.mean(self._mask))

    @property
    def visible_count(self) -> int:
        """Number of visible spatial points."""
        return int(math.sum(self._mask))

    def apply_to_difference(self, prediction: Tensor, target: Tensor) -> Tensor:
        """
        Apply mask to prediction-target difference.

        Returns masked difference tensor (zeros where not visible).
        """
        diff = prediction - target
        return diff * self._mask

    def compute_masked_mse(self, prediction: Tensor, target: Tensor) -> Tensor:
        """
        Compute MSE loss only over visible spatial region.

        Normalizes by the number of visible points.
        """
        masked_diff = self.apply_to_difference(prediction, target)
        # Sum squared differences and normalize by visible count
        mse = math.sum(masked_diff ** 2) / math.sum(self._mask)
        return mse


class ObservationMask:
    """
    Combined observation mask handling both temporal and spatial sparsity.

    This is the main interface class used by trainers.
    """

    def __init__(self, config: SparsityConfig, trajectory_length: int, spatial_shape: Shape):
        """
        Args:
            config: Combined sparsity configuration
            trajectory_length: Total timesteps in trajectory
            spatial_shape: Spatial dimensions shape
        """
        self.config = config
        self.temporal_mask = TemporalMask(config.temporal, trajectory_length)
        self.spatial_mask = SpatialMask(config.spatial, spatial_shape)

    def compute_masked_loss(self, prediction: Tensor, target: Tensor, t: int) -> Tensor:
        """
        Compute loss for a single timestep with both masks applied.

        Args:
            prediction: Model prediction at timestep t
            target: Ground truth at timestep t
            t: Current timestep index

        Returns:
            Masked MSE loss (0 if timestep is not visible)
        """
        # Check temporal visibility (handled at data level, but useful for interaction loss)
        if not self.temporal_mask.is_visible(t):
            return math.tensor(0.0)

        # Apply spatial mask
        return self.spatial_mask.compute_masked_mse(prediction, target)

    @property
    def summary(self) -> str:
        """Human-readable summary of sparsity settings."""
        lines = ["Observation Sparsity Configuration:"]

        # Temporal
        if self.config.temporal.enabled:
            lines.append(f"  Temporal: {self.config.temporal.mode} mode")
            lines.append(f"    Visible timesteps: {self.temporal_mask.num_visible}/{self.temporal_mask.trajectory_length}")
        else:
            lines.append("  Temporal: Full (all timesteps visible)")

        # Spatial
        if self.config.spatial.enabled:
            lines.append(f"  Spatial: {self.config.spatial.mode} mode")
            lines.append(f"    Visible fraction: {self.spatial_mask.visible_fraction:.1%}")
        else:
            lines.append("  Spatial: Full (entire domain visible)")

        return "\n".join(lines)
