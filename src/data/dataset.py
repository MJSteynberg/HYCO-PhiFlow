from phi.flow import *
from typing import List, Optional
from dataclasses import dataclass
from functools import cached_property
from enum import Enum, auto
import os
import random


class AccessPolicy(Enum):
    """Controls which data samples are accessed during iteration."""
    REAL_ONLY = auto()
    GENERATED_ONLY = auto()
    BOTH = auto()


@dataclass
class Sample:
    """Single training sample - trajectory tensor with time slicing info."""
    trajectory: Tensor  # Tensor(time, x, y?, field)
    time_idx: int
    rollout_steps: int

    @cached_property
    def initial_state(self) -> Tensor:
        """Tensor(x, y?, field) - slice by time."""
        return self.trajectory.time[self.time_idx]

    @cached_property
    def targets(self) -> Tensor:
        """Tensor(time, x, y?, field) - slice time range."""
        start = self.time_idx + 1
        end = start + self.rollout_steps
        return self.trajectory.time[start:end]


@dataclass
class Batch:
    """Batched samples with unified tensors."""
    initial_state: Tensor  # Tensor(batch, x, y?, field)
    targets: Tensor  # Tensor(batch, time, x, y?, field)


class Dataset:
    """Dataset for loading unified tensor simulations."""

    def __init__(self, config: dict, train_sim: List[int], rollout_steps: int):
        self.data_dir = config["data"]["data_dir"]
        self.train_sim = train_sim
        self.rollout_steps = rollout_steps
        self.trajectory_length = config["data"]["trajectory_length"]

        # Load first simulation to infer field info
        sample_data = self._load_simulation(train_sim[0])
        self.num_channels, self.field_names = self._extract_channel_info(sample_data)

        # Load all simulations
        self.simulations = self._load_all_simulations()

        # Sample counts
        self.samples_per_sim = self.trajectory_length - rollout_steps
        self.num_real_samples = len(self.train_sim) * self.samples_per_sim
        self.total_samples = self.num_real_samples

        # Augmentation support
        self.augmented_samples: List[Sample] = []
        self.access_policy = AccessPolicy.BOTH
        self.alpha = 1.0  # Proportion of real data to use (1.0 = all)

    def __len__(self) -> int:
        return self.total_samples

    def _load_simulation(self, sim_idx: int) -> Tensor:
        """Load a single simulation from disk."""
        sim_path = os.path.join(self.data_dir, f"sim_{sim_idx:04d}.npz")
        return math.load(sim_path)

    def _load_all_simulations(self) -> List[Tensor]:
        """Load all training simulations into memory."""
        return [self._load_simulation(sim_idx) for sim_idx in self.train_sim]

    def _extract_channel_info(self, data: Tensor) -> tuple:
        """Extract num_channels and field_names from tensor."""
        if 'field' not in data.shape.names:
            raise ValueError(f"Data has no 'field' channel dimension: {data.shape.names}")

        num_channels = data.shape['field'].size
        raw_names = data.shape['field'].item_names
        field_names = raw_names[0] if raw_names and isinstance(raw_names[0], tuple) else raw_names
        return num_channels, field_names

    def _get_sample(self, sample_idx: int) -> Sample:
        """Get a single sample by index."""
        if sample_idx < self.num_real_samples:
            sim_idx = sample_idx // self.samples_per_sim
            time_idx = sample_idx % self.samples_per_sim
            return Sample(
                trajectory=self.simulations[sim_idx],
                time_idx=time_idx,
                rollout_steps=self.rollout_steps
            )
        else:
            augmented_idx = sample_idx - self.num_real_samples
            return self.augmented_samples[augmented_idx]

    def set_augmented_trajectories(self, trajectories: List[Tensor]):
        """Add augmented trajectories from physical or synthetic models."""
        self.augmented_samples = []
        for trajectory in trajectories:
            traj_length = trajectory.shape['time'].size
            num_samples = traj_length - self.rollout_steps
            for time_idx in range(num_samples):
                self.augmented_samples.append(Sample(
                    trajectory=trajectory,
                    time_idx=time_idx,
                    rollout_steps=self.rollout_steps
                ))
        self.total_samples = self.num_real_samples + len(self.augmented_samples)

    def iterate_batches(self, batch_size: int, shuffle: bool = True):
        """
        Iterate through dataset yielding Batch dataclasses.

        Access policy controls which samples are included.
        Alpha controls the proportion of real data used (randomly sampled).
        """
        # Build indices based on access policy
        if self.access_policy == AccessPolicy.REAL_ONLY:
            real_indices = list(range(self.num_real_samples))
            sample_indices = self._apply_alpha(real_indices)
        elif self.access_policy == AccessPolicy.GENERATED_ONLY:
            sample_indices = list(range(self.num_real_samples, self.total_samples))
        else:  # BOTH
            real_indices = list(range(self.num_real_samples))
            real_indices = self._apply_alpha(real_indices)
            generated_indices = list(range(self.num_real_samples, self.total_samples))
            sample_indices = real_indices + generated_indices

        if shuffle:
            random.shuffle(sample_indices)

        # Yield batches
        for i in range(0, len(sample_indices), batch_size):
            batch_indices = sample_indices[i:i + batch_size]
            samples = [self._get_sample(idx) for idx in batch_indices]

            initial_states = math.stack([s.initial_state for s in samples], math.batch('batch'))
            targets = math.stack([s.targets for s in samples], math.batch('batch'))

            yield Batch(initial_state=initial_states, targets=targets)

    def _apply_alpha(self, indices: List[int]) -> List[int]:
        """Apply alpha sampling to select a proportion of indices."""
        if self.alpha >= 1.0:
            return indices
        num_to_select = max(1, int(len(indices) * self.alpha))
        return random.sample(indices, num_to_select)
