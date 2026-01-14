"""
Evaluation metrics for comparing predicted vs ground truth trajectories.

Implements common metrics for physics-informed machine learning:
- Spatial L2, L1, L∞ errors
- Relative errors
- Temporal evolution metrics
- Parameter recovery metrics
"""

from typing import Dict, Any, Optional, List
from phi.math import math, Tensor
import numpy as np
from dataclasses import dataclass, field

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrajectoryMetrics:
    """Container for trajectory-level evaluation metrics."""

    # Spatial L2 error at each timestep
    l2_error_spatial: np.ndarray = field(default_factory=lambda: np.array([]))

    # Relative L2 error at each timestep (normalized by ground truth)
    relative_l2_error: np.ndarray = field(default_factory=lambda: np.array([]))

    # Spatial L1 error (MAE) at each timestep
    l1_error_spatial: np.ndarray = field(default_factory=lambda: np.array([]))

    # Maximum error (L∞) at each timestep
    linf_error_spatial: np.ndarray = field(default_factory=lambda: np.array([]))

    # Time-averaged metrics
    l2_error_mean: float = 0.0
    relative_l2_error_mean: float = 0.0
    l1_error_mean: float = 0.0
    linf_error_mean: float = 0.0

    # Final time errors
    l2_error_final: float = 0.0
    l1_error_final: float = 0.0
    linf_error_final: float = 0.0

    # Per-field metrics (if available)
    field_l2_errors: Dict[str, float] = field(default_factory=dict)
    field_relative_l2_errors: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for saving/logging."""
        return {
            'l2_error_spatial': self.l2_error_spatial.tolist() if len(self.l2_error_spatial) > 0 else [],
            'relative_l2_error': self.relative_l2_error.tolist() if len(self.relative_l2_error) > 0 else [],
            'l1_error_spatial': self.l1_error_spatial.tolist() if len(self.l1_error_spatial) > 0 else [],
            'linf_error_spatial': self.linf_error_spatial.tolist() if len(self.linf_error_spatial) > 0 else [],
            'l2_error_mean': float(self.l2_error_mean),
            'relative_l2_error_mean': float(self.relative_l2_error_mean),
            'l1_error_mean': float(self.l1_error_mean),
            'linf_error_mean': float(self.linf_error_mean),
            'l2_error_final': float(self.l2_error_final),
            'l1_error_final': float(self.l1_error_final),
            'linf_error_final': float(self.linf_error_final),
            'field_l2_errors': {k: float(v) for k, v in self.field_l2_errors.items()},
            'field_relative_l2_errors': {k: float(v) for k, v in self.field_relative_l2_errors.items()},
        }


@dataclass
class ParameterMetrics:
    """Container for parameter recovery metrics."""

    # Scalar parameter errors
    scalar_l2_error: Dict[str, float] = field(default_factory=dict)
    scalar_relative_error: Dict[str, float] = field(default_factory=dict)

    # Field parameter errors (spatially-varying)
    field_l2_error: Dict[str, float] = field(default_factory=dict)
    field_relative_l2_error: Dict[str, float] = field(default_factory=dict)
    field_l1_error: Dict[str, float] = field(default_factory=dict)
    field_linf_error: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for saving/logging."""
        return {
            'scalar_l2_error': {k: float(v) for k, v in self.scalar_l2_error.items()},
            'scalar_relative_error': {k: float(v) for k, v in self.scalar_relative_error.items()},
            'field_l2_error': {k: float(v) for k, v in self.field_l2_error.items()},
            'field_relative_l2_error': {k: float(v) for k, v in self.field_relative_l2_error.items()},
            'field_l1_error': {k: float(v) for k, v in self.field_l1_error.items()},
            'field_linf_error': {k: float(v) for k, v in self.field_linf_error.items()},
        }


class MetricsComputer:
    """Computes evaluation metrics for trajectory predictions and parameter recovery."""

    def __init__(self, field_names: Optional[List[str]] = None):
        """
        Initialize metrics computer.

        Args:
            field_names: List of field names for per-field metrics
        """
        self.field_names = field_names or []

    @staticmethod
    def _to_scalar(tensor: Tensor) -> float:
        """
        Safely convert a PhiML tensor to a Python scalar.

        Handles gradients and ensures proper conversion.

        Args:
            tensor: PhiML tensor to convert

        Returns:
            Python float
        """
        value = tensor.native()
        if hasattr(value, 'detach'):
            value = value.detach()
        if hasattr(value, 'item'):
            return value.item()
        return float(value)

    def _resample_field(self, source_field: Tensor, target_shape) -> Tensor:
        """
        Resample a field to match target spatial resolution.

        Uses PhiML's grid-based resampling for accurate interpolation.

        Args:
            source_field: Field to resample
            target_shape: Target shape to resample to

        Returns:
            Resampled field matching target_shape
        """
        from phi.field import CenteredGrid, resample
        from phi.geom import Box
        from phi.torch.flow import PERIODIC

        # Get spatial dimensions using PhiML's shape.spatial
        source_spatial = source_field.shape.spatial
        target_spatial = target_shape.spatial

        if set(source_spatial.names) != set(target_spatial.names):
            raise ValueError(
                f"Cannot resample: spatial dimensions don't match. "
                f"Source: {source_spatial.names}, Target: {target_spatial.names}"
            )

        # Create a domain box (assuming periodic domain from 0 to 1 in each dimension)
        # This is a common assumption for physics problems
        bounds = Box(**{dim: 1.0 for dim in source_spatial.names})

        # Create grid kwargs for source and target
        source_kwargs = {dim: source_field.shape.get_size(dim) for dim in source_spatial.names}
        target_kwargs = {dim: target_shape.get_size(dim) for dim in source_spatial.names}

        # Create CenteredGrid from source field
        source_grid = CenteredGrid(
            source_field,
            boundary=PERIODIC,
            bounds=bounds,
            **source_kwargs
        )

        # Resample to target resolution
        target_grid = resample(source_grid, to=CenteredGrid(
            0,  # dummy value
            boundary=PERIODIC,
            bounds=bounds,
            **target_kwargs
        ))

        # Extract values
        return target_grid.values

    def compute_trajectory_metrics(
        self,
        prediction: Tensor,
        ground_truth: Tensor,
        compute_per_field: bool = True
    ) -> TrajectoryMetrics:
        """
        Compute comprehensive trajectory metrics.

        Args:
            prediction: Predicted trajectory tensor (must have 'time' dimension)
            ground_truth: Ground truth trajectory tensor (must have 'time' dimension)
            compute_per_field: Whether to compute per-field metrics

        Returns:
            TrajectoryMetrics object with all computed metrics
        """
        metrics = TrajectoryMetrics()

        # Ensure both tensors have time dimension
        if 'time' not in prediction.shape or 'time' not in ground_truth.shape:
            raise ValueError("Both prediction and ground_truth must have 'time' dimension")

        num_timesteps = prediction.shape.get_size('time')

        # Initialize arrays for temporal evolution
        l2_errors = []
        relative_l2_errors = []
        l1_errors = []
        linf_errors = []

        # Compute metrics at each timestep
        for t in range(num_timesteps):
            pred_t = prediction.time[t]
            true_t = ground_truth.time[t]

            # L2 error (MSE in space)
            squared_error = (pred_t - true_t) ** 2
            l2_error = math.sqrt(math.mean(squared_error))
            l2_errors.append(self._to_scalar(l2_error))

            # Relative L2 error
            true_norm = math.sqrt(math.mean(true_t ** 2))
            if self._to_scalar(true_norm) > 1e-10:
                relative_error = l2_error / true_norm
                relative_l2_errors.append(self._to_scalar(relative_error))
            else:
                relative_l2_errors.append(0.0)

            # L1 error (MAE)
            abs_error = math.abs(pred_t - true_t)
            l1_error = math.mean(abs_error)
            l1_errors.append(self._to_scalar(l1_error))

            # L∞ error (max absolute error)
            linf_error = math.max(abs_error)
            linf_errors.append(self._to_scalar(linf_error))

        # Store temporal evolution
        metrics.l2_error_spatial = np.array(l2_errors)
        metrics.relative_l2_error = np.array(relative_l2_errors)
        metrics.l1_error_spatial = np.array(l1_errors)
        metrics.linf_error_spatial = np.array(linf_errors)

        # Compute time-averaged metrics
        metrics.l2_error_mean = float(np.mean(l2_errors))
        metrics.relative_l2_error_mean = float(np.mean(relative_l2_errors))
        metrics.l1_error_mean = float(np.mean(l1_errors))
        metrics.linf_error_mean = float(np.mean(linf_errors))

        # Final time errors
        metrics.l2_error_final = l2_errors[-1]
        metrics.l1_error_final = l1_errors[-1]
        metrics.linf_error_final = linf_errors[-1]

        # Per-field metrics (time-averaged)
        if compute_per_field and 'field' in prediction.shape:
            metrics.field_l2_errors = self._compute_per_field_l2(prediction, ground_truth)
            metrics.field_relative_l2_errors = self._compute_per_field_relative_l2(prediction, ground_truth)

        return metrics

    def _compute_per_field_l2(self, prediction: Tensor, ground_truth: Tensor) -> Dict[str, float]:
        """Compute time-averaged L2 error for each field separately."""
        field_errors = {}

        field_names = prediction.shape['field'].item_names
        if field_names and isinstance(field_names[0], tuple):
            field_names = field_names[0]

        for field_name in field_names:
            pred_field = prediction.field[field_name]
            true_field = ground_truth.field[field_name]

            # Time-averaged L2 error - reduce all dimensions
            squared_error = (pred_field - true_field) ** 2
            # Use finalize() to ensure all dimensions are reduced to scalar
            l2_error = math.sqrt(math.mean(squared_error, pred_field.shape))
            field_errors[field_name] = self._to_scalar(l2_error)

        return field_errors

    def _compute_per_field_relative_l2(self, prediction: Tensor, ground_truth: Tensor) -> Dict[str, float]:
        """Compute time-averaged relative L2 error for each field separately."""
        field_errors = {}

        field_names = prediction.shape['field'].item_names
        if field_names and isinstance(field_names[0], tuple):
            field_names = field_names[0]

        for field_name in field_names:
            pred_field = prediction.field[field_name]
            true_field = ground_truth.field[field_name]

            # Time-averaged relative L2 error
            squared_error = (pred_field - true_field) ** 2
            l2_error = math.sqrt(math.mean(squared_error, pred_field.shape))

            true_norm = math.sqrt(math.mean(true_field ** 2, true_field.shape))
            true_norm_val = self._to_scalar(true_norm)

            if true_norm_val > 1e-10:
                relative_error = l2_error / true_norm
                field_errors[field_name] = self._to_scalar(relative_error)
            else:
                field_errors[field_name] = 0.0

        return field_errors

    def compute_parameter_metrics(
        self,
        learned_params: Tensor,
        ground_truth_params: Tensor,
        scalar_param_names: List[str] = None,
        field_param_names: List[str] = None
    ) -> ParameterMetrics:
        """
        Compute parameter recovery metrics.

        Args:
            learned_params: Learned parameters tensor
            ground_truth_params: Ground truth parameters tensor
            scalar_param_names: Names of scalar parameters
            field_param_names: Names of field (spatially-varying) parameters

        Returns:
            ParameterMetrics object with all computed metrics
        """
        metrics = ParameterMetrics()

        scalar_param_names = scalar_param_names or []
        field_param_names = field_param_names or []

        # Scalar parameter metrics
        for param_name in scalar_param_names:
            learned_val = self._to_scalar(learned_params.field[param_name])
            true_val = self._to_scalar(ground_truth_params.field[param_name])

            # L2 (absolute) error
            error = abs(learned_val - true_val)
            metrics.scalar_l2_error[param_name] = error

            # Relative error
            if abs(true_val) > 1e-10:
                rel_error = error / abs(true_val)
                metrics.scalar_relative_error[param_name] = rel_error
            else:
                metrics.scalar_relative_error[param_name] = 0.0

        # Field parameter metrics
        for param_name in field_param_names:
            learned_field = learned_params.field[param_name]
            true_field = ground_truth_params.field[param_name]

            # Check if resolutions match
            if learned_field.shape != true_field.shape:
                logger.warning(
                    f"Resolution mismatch for parameter '{param_name}': "
                    f"learned={learned_field.shape}, ground_truth={true_field.shape}. "
                    f"Resampling learned field to match ground truth resolution."
                )

                # Resample learned field to match ground truth resolution using interpolation
                learned_field = self._resample_field(learned_field, true_field.shape)

            # L2 error (spatial RMS)
            squared_error = (learned_field - true_field) ** 2
            l2_error = math.sqrt(math.mean(squared_error))
            metrics.field_l2_error[param_name] = self._to_scalar(l2_error)

            # Relative L2 error
            true_norm = math.sqrt(math.mean(true_field ** 2))
            true_norm_val = self._to_scalar(true_norm)
            if true_norm_val > 1e-10:
                rel_error = l2_error / true_norm
                metrics.field_relative_l2_error[param_name] = self._to_scalar(rel_error)
            else:
                metrics.field_relative_l2_error[param_name] = 0.0

            # L1 error (spatial MAE)
            abs_error = math.abs(learned_field - true_field)
            l1_error = math.mean(abs_error)
            metrics.field_l1_error[param_name] = self._to_scalar(l1_error)

            # L∞ error (max spatial error)
            linf_error = math.max(abs_error)
            metrics.field_linf_error[param_name] = self._to_scalar(linf_error)

        return metrics

    def compute_energy_error(
        self,
        prediction: Tensor,
        ground_truth: Tensor
    ) -> Dict[str, float]:
        """
        Compute energy conservation error (useful for conservative systems).

        Measures how well total energy is preserved over time.

        Args:
            prediction: Predicted trajectory
            ground_truth: Ground truth trajectory

        Returns:
            Dictionary with energy metrics
        """
        # Compute total energy at each timestep (sum of squared values)
        pred_energy = []
        true_energy = []

        num_timesteps = prediction.shape.get_size('time')

        for t in range(num_timesteps):
            pred_t = prediction.time[t]
            true_t = ground_truth.time[t]

            pred_e = math.sum(pred_t ** 2)
            true_e = math.sum(true_t ** 2)

            pred_energy.append(float(pred_e.native()))
            true_energy.append(float(true_e.native()))

        pred_energy = np.array(pred_energy)
        true_energy = np.array(true_energy)

        # Energy drift (change from initial)
        pred_drift = np.abs(pred_energy - pred_energy[0])
        true_drift = np.abs(true_energy - true_energy[0])

        # Relative energy error
        energy_error = np.abs(pred_energy - true_energy)
        if true_energy[0] > 1e-10:
            relative_energy_error = energy_error / true_energy[0]
        else:
            relative_energy_error = energy_error

        return {
            'prediction_energy': pred_energy.tolist(),
            'ground_truth_energy': true_energy.tolist(),
            'prediction_energy_drift': pred_drift.tolist(),
            'ground_truth_energy_drift': true_drift.tolist(),
            'energy_error': energy_error.tolist(),
            'relative_energy_error': relative_energy_error.tolist(),
            'mean_relative_energy_error': float(np.mean(relative_energy_error)),
            'final_relative_energy_error': float(relative_energy_error[-1]),
        }
