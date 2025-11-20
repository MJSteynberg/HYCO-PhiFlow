"""
Dataset - Pure PhiML Data Pipeline

Key Features:
- No PyTorch dependency - pure PhiML throughout
- Generator-based iteration (no __getitem__/__len__)
- PhiML tensor batching with named dimensions
- Sliding window extraction from trajectories
- Support for augmented (physically-generated) data

Data Flow:
  PhiML Cache (.npz) → PhiML Tensors → PhiML Batches → Model
  (NO torch conversion!)
"""

from typing import Dict, Any, List, Optional
import random
from pathlib import Path

from phiml import math as phimath
from phiml.math import batch as batch_dim, stack
from phi.field import Field

from .data_manager import DataManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Dataset:
    """
    Pure PhiML dataset that yields PhiML tensor batches.

    No PyTorch dependency - uses PhiML's native tensor operations.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        data_manager: DataManager,
        sim_indices: List[int],
        field_names: List[str],
        num_frames: Optional[int] = None,
        rollout_steps: int = 1,
        percentage_real_data: float = 1.0,
        enable_augmentation: bool = False,
    ):
        """
        Initialize Dataset.

        Args:
            config: Configuration dictionary
            data_manager: PhiML DataManager instance
            sim_indices: List of simulation indices to use
            field_names: List of field names to load
            num_frames: Number of frames per simulation (None = all)
            rollout_steps: Number of prediction steps
            percentage_real_data: Fraction of real data to use (0.0-1.0)
            enable_augmentation: Whether to enable augmented data
        """
        self.config = config
        self.data_manager = data_manager
        self.sim_indices = sim_indices
        self.field_names = field_names
        self.rollout_steps = rollout_steps
        self.percentage_real_data = percentage_real_data
        self.enable_augmentation = enable_augmentation

        # Cache simulation data on initialization
        logger.debug(f"Caching {len(sim_indices)} simulations...")
        self.num_frames = self._cache_all_simulations(num_frames)

        # Compute samples per simulation (sliding window)
        self.samples_per_sim = self.num_frames - self.rollout_steps
        logger.debug(
            f"  Each simulation: {self.num_frames} frames → "
            f"{self.samples_per_sim} samples (window size {rollout_steps})"
        )

        # Total real samples
        total_real_samples = len(sim_indices) * self.samples_per_sim

        # Apply filtering if percentage_real_data < 1.0
        if percentage_real_data < 1.0:
            self.num_real = int(total_real_samples * percentage_real_data)
            self.filtered_indices = sorted(
                random.sample(range(total_real_samples), self.num_real)
            )
            logger.debug(
                f"  Filtering enabled: using {self.num_real}/{total_real_samples} "
                f"samples ({percentage_real_data*100:.1f}%)"
            )
        else:
            self.num_real = total_real_samples
            self.filtered_indices = None

        # Augmentation (to be set later via set_augmented_trajectories)
        self.augmented_samples = []
        self.num_augmented = 0
        self._num_augmented_trajectories = 0
        self._samples_per_trajectory = self.samples_per_sim

        # Access policy
        self.access_policy = "both" if enable_augmentation else "real_only"

        logger.info(
            f"Dataset initialized: {self.num_real} real samples, "
            f"{self.num_augmented} augmented samples"
        )

    def _cache_all_simulations(self, num_frames: Optional[int]) -> int:
        """
        Cache all simulations and return actual num_frames.

        Args:
            num_frames: Requested number of frames (None = all available)

        Returns:
            Actual number of frames available
        """
        for sim_idx in self.sim_indices:
            # This will cache if not already cached
            cached_data = self.data_manager.load_simulation(
                sim_idx,
                field_names=self.field_names,
                num_frames=num_frames
            )

            # Get actual num_frames from first simulation
            if num_frames is None:
                first_field = cached_data[self.field_names[0]]
                num_frames = first_field.shape.get_size('time')

        return num_frames

    def __len__(self) -> int:
        """Return total number of samples."""
        if self.access_policy == "real_only":
            return self.num_real
        elif self.access_policy == "generated_only":
            return self.num_augmented
        else:  # "both"
            return self.num_real + self.num_augmented

    def iterate_batches(self, batch_size: int, shuffle: bool = True):
        """
        Iterate through dataset yielding PhiML tensor batches.

        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle samples

        Yields:
            Dict with 'initial_state' and 'targets' as PhiML tensors:
            {
                'initial_state': Tensor(batch=B, x=H, y=W, vector=V),
                'targets': Tensor(batch=B, time=T, x=H, y=W, vector=V)
            }
        """
        # Get sample indices based on access policy
        indices = list(range(len(self)))

        if shuffle:
            random.shuffle(indices)

        # Yield batches
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            yield self._load_batch(batch_indices)

    def _load_batch(self, indices: List[int]) -> Dict[str, Any]:
        """
        Load multiple samples and stack into batch dimension.

        Args:
            indices: List of sample indices

        Returns:
            Dict with batched PhiML tensors
        """
        samples = [self._load_sample(idx) for idx in indices]

        # Stack using PhiML's batch dimension
        initial_states = stack(
            [s['initial'] for s in samples],
            batch_dim('batch')
        )
        targets = stack(
            [s['target'] for s in samples],
            batch_dim('batch')
        )

        return {
            'initial_state': initial_states,
            'targets': targets
        }

    def _load_sample(self, idx: int) -> Dict[str, Any]:
        """
        Load a single sample (initial state + targets).

        Args:
            idx: Global sample index

        Returns:
            Dict with 'initial' and 'target' PhiML tensors
        """
        # Check if augmented or real
        if self._is_augmented_sample(idx):
            return self._get_augmented_sample(idx - self.num_real)
        else:
            return self._extract_real_sample(idx)

    def _extract_real_sample(self, idx: int) -> Dict[str, Any]:
        """
        Extract a windowed sample from real simulations.

        Args:
            idx: Index within real samples

        Returns:
            Dict with 'initial' and 'target' PhiML tensors
        """
        # Apply filtering if enabled
        if self.filtered_indices is not None:
            idx = self.filtered_indices[idx]

        # Compute which simulation and which frame
        sim_idx, start_frame = self._compute_sim_and_frame(idx)

        # Load simulation data (from cache)
        sim_data = self.data_manager.load_simulation(
            sim_idx,
            field_names=self.field_names,
            num_frames=self.num_frames
        )

        # Concatenate all fields along vector dimension
        field_tensors = [sim_data[name] for name in self.field_names]

        # All fields have shape: (time, x, y, vector) or (time, x, y)
        # We need to concatenate along vector dimension
        all_data = phimath.concat(field_tensors, 'vector')

        # Extract window
        initial = all_data.time[start_frame]  # (x, y, vector)
        target_start = start_frame + 1
        target_end = start_frame + 1 + self.rollout_steps
        targets = all_data.time[target_start:target_end]  # (time, x, y, vector)

        return {'initial': initial, 'target': targets}

    def _compute_sim_and_frame(self, idx: int) -> tuple:
        """
        Compute simulation index and start frame from global index.

        Args:
            idx: Global sample index

        Returns:
            Tuple of (sim_index, start_frame)
        """
        sim_offset = idx // self.samples_per_sim
        start_frame = idx % self.samples_per_sim
        sim_idx = self.sim_indices[sim_offset]

        return sim_idx, start_frame

    def _is_augmented_sample(self, idx: int) -> bool:
        """
        Check if sample index corresponds to augmented data.

        Args:
            idx: Global sample index

        Returns:
            True if from augmented data, False if real
        """
        if self.access_policy == "generated_only":
            return True
        elif self.access_policy == "real_only":
            return False
        else:  # 'both'
            return idx >= self.num_real

    def _get_augmented_sample(self, idx: int) -> Dict[str, Any]:
        """
        Get sample from augmented (physically-generated) trajectories.

        Args:
            idx: Index within augmented samples

        Returns:
            Dict with 'initial' and 'target' PhiML tensors
        """
        # Compute which trajectory and which window
        traj_idx = idx // self._samples_per_trajectory
        window_start = idx % self._samples_per_trajectory

        if traj_idx >= self._num_augmented_trajectories:
            raise IndexError(
                f"Augmented index {idx} out of range "
                f"(trajectory {traj_idx} >= {self._num_augmented_trajectories})"
            )

        # Get trajectory data (already PhiML tensors!)
        trajectory_data = self.augmented_samples[traj_idx]

        # Concatenate all fields
        field_tensors = [trajectory_data[name] for name in self.field_names]
        all_data = phimath.concat(field_tensors, 'vector')

        # Extract window
        initial = all_data.time[window_start]
        target_start = window_start + 1
        target_end = window_start + 1 + self.rollout_steps
        targets = all_data.time[target_start:target_end]

        return {'initial': initial, 'target': targets}

    # ==================== Augmentation Management ====================

    def set_augmented_trajectories(self, trajectory_rollouts: List[Dict[str, Field]]):
        """
        Set augmented data from physically-generated Field trajectories.

        Args:
            trajectory_rollouts: List of rollout dicts where each is
                                {'field_name': Field[time, x, y]}
        """
        if not trajectory_rollouts:
            self.augmented_samples = []
            self.num_augmented = 0
            self._num_augmented_trajectories = 0
            return

        logger.debug(f"Converting {len(trajectory_rollouts)} physical trajectories...")

        # Convert each Field trajectory to PhiML tensors
        converted_trajectories = []
        for idx, rollout in enumerate(trajectory_rollouts):
            if idx % 10 == 0 and idx > 0:
                logger.debug(f"  Converted {idx}/{len(trajectory_rollouts)} trajectories...")

            # Extract PhiML tensors from Fields
            tensor_trajectory = {
                field_name: rollout[field_name].values
                for field_name in self.field_names
                if field_name in rollout
            }
            converted_trajectories.append(tensor_trajectory)

        # Store trajectories
        self.augmented_samples = converted_trajectories
        self._num_augmented_trajectories = len(converted_trajectories)

        # Calculate total augmented samples (with windowing)
        self.num_augmented = self._num_augmented_trajectories * self._samples_per_trajectory

        logger.debug(
            f"Set {self._num_augmented_trajectories} augmented trajectories "
            f"({self.num_augmented} windowed samples)"
        )

    def clear_augmented_trajectories(self):
        """Clear all augmented trajectories."""
        self.augmented_samples = []
        self.num_augmented = 0
        self._num_augmented_trajectories = 0
        logger.debug("Cleared augmented trajectories")

    def get_sample_source(self, idx: int) -> str:
        """
        Get the source of a sample.

        Args:
            idx: Sample index

        Returns:
            'real' or 'physical_generated'
        """
        return 'physical_generated' if self._is_augmented_sample(idx) else 'real'

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Dataset(\n"
            f"  simulations={len(self.sim_indices)},\n"
            f"  samples={len(self)} (real={self.num_real}, aug={self.num_augmented}),\n"
            f"  augmented_trajectories={self._num_augmented_trajectories},\n"
            f"  fields={len(self.field_names)},\n"
            f"  frames={self.num_frames},\n"
            f"  predict_steps={self.rollout_steps}\n"
            f")"
        )
