from phi.flow import *
from typing import Dict, List, Any
from dataclasses import dataclass
from functools import cached_property
import os
import random


@dataclass
class Sample:
    """
    Dataclass representing a single training sample.

    Keeps fields separate (no concatenation) for direct use by trainer.
    Uses cached_property for lazy extraction of timesteps.
    """
    sim_data: Dict[str, Any]
    time_idx: int
    rollout_steps: int
    fields: List[str]

    @cached_property
    def initial_state(self) -> Dict[str, Any]:
        """
        Extract initial state at time_idx for each field.

        Returns:
            Dict mapping field_name -> Tensor(x, y, vector)
        """
        return {
            field_name: self.sim_data[field_name].time[self.time_idx]
            for field_name in self.fields
        }

    @cached_property
    def targets(self) -> Dict[str, Any]:
        """
        Extract rollout targets for each field.

        Returns:
            Dict mapping field_name -> Tensor(time, x, y, vector)
        """
        start = self.time_idx + 1
        end = start + self.rollout_steps
        return {
            field_name: self.sim_data[field_name].time[start:end]
            for field_name in self.fields
        }


@dataclass
class Batch:
    """
    Dataclass representing a batch of samples.

    Stores fields separately (no concatenation) for direct trainer use.
    Each field is a batched tensor.
    """
    initial_state: Dict[str, Any]  # {field_name: Tensor(batch=B, x=H, y=W, vector=V)}
    targets: Dict[str, Any]  # {field_name: Tensor(batch=B, time=T, x=H, y=W, vector=V)}

    def __getitem__(self, key):
        """Support dict-like access for backward compatibility."""
        if key == 'initial_state':
            return self.initial_state
        elif key == 'targets':
            return self.targets
        else:
            raise KeyError(f"Unknown key: {key}")


class Dataset:
    """
    Simplified Dataset class for PhiML data handling.

    Loads simulations into memory and yields batched rollout pairs.
    No augmentation, no complex filtering - just simple data loading.
    """

    def __init__(self, config: Dict, train_sim: List[int], rollout_steps: int):
        """
        Initialize dataset.

        Args:
            config: Configuration dictionary
            train_sim: List of simulation indices to load
            rollout_steps: Number of prediction timesteps
        """
        self.data_dir = config["data"]["data_dir"]
        self.train_sim = train_sim
        self.rollout_steps = rollout_steps
        self.trajectory_length = config["data"]["trajectory_length"]
        self.fields = config["data"]["fields"]
        self.fields_scheme = config["data"]["fields_scheme"]

        # Load all simulations into memory
        self.simulations = self._load_all_simulations()

        # Calculate total number of samples (sliding windows)
        # Each simulation of length T gives (T - rollout_steps) samples
        self.samples_per_sim = self.trajectory_length - rollout_steps
        self.num_real_samples = len(self.train_sim) * self.samples_per_sim
        self.total_samples = self.num_real_samples

        # Augmentation support for hybrid training
        self.augmented_samples: List[Sample] = []
        self.access_policy: str = "both"  # "real_only", "generated_only", "both"

    def __len__(self) -> int:
        """Return total number of samples."""
        return self.total_samples

    def _load_simulation(self, sim_idx: int) -> Dict[str, Any]:
        """Load a single simulation from disk."""
        sim_path = os.path.join(self.data_dir, f"sim_{sim_idx:04d}.npz")
        sim_data = math.load(sim_path)
        return sim_data

    def _load_all_simulations(self) -> List[Dict[str, Any]]:
        """Load all training simulations into memory."""
        simulations = []
        for sim_idx in self.train_sim:
            sim_data = self._load_simulation(sim_idx)
            simulations.append(sim_data)
        return simulations

    def _get_sample(self, sample_idx: int) -> Sample:
        """
        Get a single sample (initial state + rollout targets).

        Args:
            sample_idx: Global sample index

        Returns:
            Sample dataclass instance with cached initial_state and targets
        """
        # Check if this is a real or augmented sample
        if sample_idx < self.num_real_samples:
            # Real sample from loaded simulations
            sim_idx = sample_idx // self.samples_per_sim
            time_idx = sample_idx % self.samples_per_sim
            sim_data = self.simulations[sim_idx]

            return Sample(
                sim_data=sim_data,
                time_idx=time_idx,
                rollout_steps=self.rollout_steps,
                fields=self.fields
            )
        else:
            # Augmented sample
            augmented_idx = sample_idx - self.num_real_samples
            return self.augmented_samples[augmented_idx]

    def set_augmented_trajectories(self, trajectories: List[Dict[str, Any]]):
        """
        Add augmented trajectories (from physical or synthetic models) to dataset.

        Converts Fields to PhiML tensors using .values property and windows them
        into training samples.

        Args:
            trajectories: List of trajectory dicts. Each dict contains field_name -> Field or Tensor
                         where Field/Tensor has time dimension: [time, x, y, vector] or [time, x, y]
                         Example: [{'velocity': Field[time=100, x=64, y=64, vector=2]}, ...]
        """
        self.augmented_samples = []
        for trajectory in trajectories:
            # Convert Fields to tensors if needed
            sim_data = {}
            for name, val in trajectory.items():
                # Check if it's a Field (has .values attribute) or already a Tensor
                if isinstance(val, Field):
                    # It's a Field - extract tensor using .values
                    sim_data[name] = val.values
                else:
                    # Already a tensor
                    sim_data[name] = val

            # Get trajectory length from shape
            num_samples = self.trajectory_length - self.rollout_steps

            for time_idx in range(num_samples):
                sample = Sample(
                    sim_data=sim_data,
                    time_idx=time_idx,
                    rollout_steps=self.rollout_steps,
                    fields=self.fields
                )
                
                self.augmented_samples.append(sample)

        # Update total sample count
        self.total_samples = self.num_real_samples + len(self.augmented_samples)

    def set_augmented_predictions(self, predictions: List[Dict[str, Any]]):
        """
        Add augmented predictions (from synthetic model) to dataset.

        This is an alias for set_augmented_trajectories() since both handle
        trajectories in the same way. Predictions are already tensors.

        Args:
            predictions: List of prediction dicts with tensors
                        Example: [{'velocity': Tensor[time=10, x=64, y=64, vector=2]}, ...]
        """
        self.set_augmented_trajectories(predictions)

    def iterate_batches(self, batch_size: int, shuffle: bool = True):
        """
        Iterate through dataset yielding PhiML Batch dataclasses.

        Respects access_policy to control which samples are yielded.

        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle samples

        Yields:
            Batch dataclass with:
                - initial_state: Dict[field_name, Tensor(batch=B, x=H, y=W, vector=V)]
                - targets: Dict[field_name, Tensor(batch=B, time=T, x=H, y=W, vector=V)]
        """
        # Create list of sample indices based on access_policy
        if self.access_policy == "real_only":
            sample_indices = list(range(self.num_real_samples))
        elif self.access_policy == "generated_only":
            sample_indices = list(range(self.num_real_samples, self.total_samples))
        else:  # "both"
            sample_indices = list(range(self.total_samples))

        if shuffle:
            random.shuffle(sample_indices)

        # Yield batches
        for i in range(0, len(sample_indices), batch_size):
            batch_indices = sample_indices[i:i + batch_size]

            # Load samples (Sample dataclasses with cached_property)
            samples = [self._get_sample(idx) for idx in batch_indices]

            # Stack each field separately along batch dimension
            initial_state_batch = {}
            targets_batch = {}

            for field_name in self.fields:
                # Stack initial states for this field
                initial_state_batch[field_name] = math.stack(
                    [s.initial_state[field_name] for s in samples],
                    math.batch('batch')
                )
                # Stack targets for this field
                targets_batch[field_name] = math.stack(
                    [s.targets[field_name] for s in samples],
                    math.batch('batch')
                )

            # Return Batch dataclass
            yield Batch(
                initial_state=initial_state_batch,
                targets=targets_batch
            )