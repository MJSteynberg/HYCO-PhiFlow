"""Dataset module for loading and iterating over simulation data."""

from phi.flow import *
from typing import List, Optional, Tuple
from dataclasses import dataclass
from functools import cached_property
from enum import Enum, auto
import os
import random
from collections import OrderedDict


class AccessPolicy(Enum):
    """Controls which data samples are accessed during iteration."""
    REAL_ONLY = auto()
    GENERATED_ONLY = auto()
    BOTH = auto()


@dataclass
class SeparatedBatch:
    """Batch with real and generated data separated for weighted loss computation."""
    real_initial_state: Optional[Tensor]      # Tensor(batch, x, y?, field) or None
    real_targets: Optional[Tensor]            # Tensor(batch, time, x, y?, field) or None
    generated_initial_state: Optional[Tensor]
    generated_targets: Optional[Tensor]
    
    @property
    def has_real(self) -> bool:
        return self.real_initial_state is not None
    
    @property
    def has_generated(self) -> bool:
        return self.generated_initial_state is not None


class Dataset:
    """
    Dataset for loading unified tensor simulations with lazy loading.
    
    Optimized for hybrid training with separated real/generated batches.
    """

    def __init__(
        self, 
        config: dict, 
        train_sim: List[int], 
        rollout_steps: int, 
        max_cached_sims: int = None
    ):
        self.data_dir = config["data"]["data_dir"]
        self.train_sim = train_sim
        self.rollout_steps = rollout_steps
        self.trajectory_length = config["data"]["trajectory_length"]
        
        # Auto-size cache to hold all training sims (efficiency improvement)
        if max_cached_sims is None:
            max_cached_sims = max(len(train_sim), 4)
        self.max_cached_sims = max_cached_sims
        
        # Manual cache: sim_idx -> Tensor
        self._cache: OrderedDict[int, Tensor] = OrderedDict()

        # Load first simulation to infer field info
        sample_data = self._load_simulation(train_sim[0])
        self.num_channels, self.field_names = self._extract_channel_info(sample_data)

        # Sample counts
        self.samples_per_sim = self.trajectory_length - rollout_steps
        self.num_real_samples = len(self.train_sim) * self.samples_per_sim
        self.total_samples = self.num_real_samples

        # Augmentation support - store trajectories directly (not Sample objects)
        self._augmented_trajectories: List[Tensor] = []
        self._augmented_sample_count = 0
        self.access_policy = AccessPolicy.BOTH
        self.alpha = 1.0  # Proportion of real data to use (1.0 = all)

    def __len__(self) -> int:
        return self.total_samples

    def _load_simulation(self, sim_idx: int) -> Tensor:
        """Load a single simulation from disk with manual LRU caching."""
        # Check cache
        if sim_idx in self._cache:
            self._cache.move_to_end(sim_idx)
            return self._cache[sim_idx]
            
        # Load from disk
        sim_path = os.path.join(self.data_dir, f"sim_{sim_idx:04d}.npz")
        data = math.load(sim_path)
        
        # Update cache
        self._cache[sim_idx] = data
        if len(self._cache) > self.max_cached_sims:
            self._cache.popitem(last=False)  # Remove oldest
            
        return data

    def _extract_channel_info(self, data: Tensor) -> tuple:
        """Extract num_channels and field_names from tensor."""
        if 'field' not in data.shape.names:
            raise ValueError(f"Data has no 'field' channel dimension: {data.shape.names}")

        num_channels = data.shape['field'].size
        raw_names = data.shape['field'].item_names
        field_names = raw_names[0] if raw_names and isinstance(raw_names[0], tuple) else raw_names
        return num_channels, field_names

    def _get_real_sample(self, sample_idx: int) -> Tuple[Tensor, Tensor]:
        """Get real sample initial state and targets by index."""
        sim_list_idx = sample_idx // self.samples_per_sim
        sim_idx = self.train_sim[sim_list_idx]
        time_idx = sample_idx % self.samples_per_sim
        
        trajectory = self._load_simulation(sim_idx)
        initial_state = trajectory.time[time_idx]
        targets = trajectory.time[time_idx + 1 : time_idx + 1 + self.rollout_steps]
        return initial_state, targets

    def _get_augmented_sample(self, aug_idx: int) -> Tuple[Tensor, Tensor]:
        """Get augmented sample by computing trajectory/time index on-the-fly."""
        # Find which trajectory and time index
        cumulative = 0
        for traj in self._augmented_trajectories:
            traj_samples = traj.shape['time'].size - self.rollout_steps
            if aug_idx < cumulative + traj_samples:
                time_idx = aug_idx - cumulative
                initial_state = traj.time[time_idx]
                targets = traj.time[time_idx + 1 : time_idx + 1 + self.rollout_steps]
                return initial_state, targets
            cumulative += traj_samples
        
        raise IndexError(f"Augmented index {aug_idx} out of range")

    def set_augmented_trajectories(self, trajectories: List[Tensor]):
        """
        Store augmented trajectories directly (optimized - no Sample wrapper objects).
        
        Args:
            trajectories: List of trajectory tensors from physical or synthetic models
        """
        self._augmented_trajectories = trajectories
        self._augmented_sample_count = sum(
            traj.shape['time'].size - self.rollout_steps 
            for traj in trajectories
        )
        self.total_samples = self.num_real_samples + self._augmented_sample_count

    def _apply_alpha(self, indices: List[int]) -> List[int]:
        """Apply alpha sampling to select a proportion of indices."""
        if self.alpha >= 1.0:
            return indices
        num_to_select = max(1, int(len(indices) * self.alpha))
        return random.sample(indices, num_to_select)

    def iterate_batches(self, batch_size: int, shuffle: bool = True):
        """
        Iterate through dataset yielding SeparatedBatch with real and generated data separate.
        
        This is the primary iteration method - always yields separated batches for
        weighted loss computation (L for real, I for generated).
        
        Args:
            batch_size: Number of samples per batch (applies to each of real/generated)
            shuffle: Whether to shuffle indices within each group
            
        Yields:
            SeparatedBatch with separate real and generated tensors
        """
        # Build real and generated indices based on access policy
        if self.access_policy == AccessPolicy.REAL_ONLY:
            real_indices = list(range(self.num_real_samples))
            real_indices = self._apply_alpha(real_indices)
            generated_indices = []
        elif self.access_policy == AccessPolicy.GENERATED_ONLY:
            real_indices = []
            generated_indices = list(range(self._augmented_sample_count))
        else:  # BOTH
            real_indices = list(range(self.num_real_samples))
            real_indices = self._apply_alpha(real_indices)
            generated_indices = list(range(self._augmented_sample_count))
        
        # Shuffle within each group
        if shuffle:
            random.shuffle(real_indices)
            random.shuffle(generated_indices)
        
        # Calculate number of batches needed
        num_real_batches = (len(real_indices) + batch_size - 1) // batch_size if real_indices else 0
        num_gen_batches = (len(generated_indices) + batch_size - 1) // batch_size if generated_indices else 0
        max_batches = max(num_real_batches, num_gen_batches, 1)
        
        for i in range(max_batches):
            # Get real batch indices
            real_start = i * batch_size
            real_end = min(real_start + batch_size, len(real_indices))
            real_batch_indices = real_indices[real_start:real_end] if real_start < len(real_indices) else []
            
            # Get generated batch indices  
            gen_start = i * batch_size
            gen_end = min(gen_start + batch_size, len(generated_indices))
            gen_batch_indices = generated_indices[gen_start:gen_end] if gen_start < len(generated_indices) else []
            
            real_init, real_tgt = None, None
            gen_init, gen_tgt = None, None
            
            # Build real batch
            if real_batch_indices:
                real_initial_states = []
                real_targets_list = []
                for idx in real_batch_indices:
                    init, tgt = self._get_real_sample(idx)
                    real_initial_states.append(init)
                    real_targets_list.append(tgt)
                real_init = math.stack(real_initial_states, batch('batch'))
                real_tgt = math.stack(real_targets_list, batch('batch'))
            
            # Build generated batch
            if gen_batch_indices:
                gen_initial_states = []
                gen_targets_list = []
                for idx in gen_batch_indices:
                    init, tgt = self._get_augmented_sample(idx)
                    gen_initial_states.append(init)
                    gen_targets_list.append(tgt)
                gen_init = math.stack(gen_initial_states, batch('batch'))
                gen_tgt = math.stack(gen_targets_list, batch('batch'))
            
            yield SeparatedBatch(real_init, real_tgt, gen_init, gen_tgt)