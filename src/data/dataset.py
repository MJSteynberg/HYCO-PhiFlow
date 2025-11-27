from phi.flow import *
from typing import List, Optional, Dict
from dataclasses import dataclass
from functools import cached_property
from enum import Enum, auto
import os
import random
from collections import OrderedDict

from enum import Enum, auto

class SampleOrigin(Enum):
    """Identifies where a sample came from."""
    REAL = auto()
    GENERATED = auto()

@dataclass
class Sample:
    """Single training sample with origin tracking."""
    trajectory: Tensor
    time_idx: int
    rollout_steps: int
    origin: SampleOrigin = SampleOrigin.REAL  # NEW

@dataclass
class SeparatedBatch:
    """Batch with real and generated data separated."""
    real_initial_state: Optional[Tensor]     # Tensor(batch, x, y?, field) or None
    real_targets: Optional[Tensor]           # Tensor(batch, time, x, y?, field) or None
    generated_initial_state: Optional[Tensor]
    generated_targets: Optional[Tensor]
    
    @property
    def has_real(self) -> bool:
        return self.real_initial_state is not None
    
    @property
    def has_generated(self) -> bool:
        return self.generated_initial_state is not None

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




class Dataset:
    """Dataset for loading unified tensor simulations with lazy loading."""

    def __init__(self, config: dict, train_sim: List[int], rollout_steps: int, max_cached_sims: int = 2):
        self.data_dir = config["data"]["data_dir"]
        self.train_sim = train_sim
        self.rollout_steps = rollout_steps
        self.trajectory_length = config["data"]["trajectory_length"]
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

        # Augmentation support
        self.augmented_samples: List[Sample] = []
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

    def _get_sample(self, sample_idx: int) -> Sample:
        """Get a single sample by index."""
        if sample_idx < self.num_real_samples:
            sim_list_idx = sample_idx // self.samples_per_sim
            sim_idx = self.train_sim[sim_list_idx]
            time_idx = sample_idx % self.samples_per_sim
            return Sample(
                trajectory=self._load_simulation(sim_idx),
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
        Iterate yielding SeparatedBatch with real and generated data separate.
        
        This allows trainers to compute L (real loss) and I (interaction loss) 
        independently and apply different weights.
        """
        # Build real and generated indices
        real_indices = list(range(self.num_real_samples))
        real_indices = self._apply_alpha(real_indices)
        generated_indices = list(range(self.num_real_samples, self.total_samples))
        
        # Shuffle within each group
        if shuffle:
            random.shuffle(real_indices)
            random.shuffle(generated_indices)
        
        # Yield balanced batches containing both real and generated
        max_batches = max(
            len(real_indices) // batch_size + (1 if len(real_indices) % batch_size else 0),
            len(generated_indices) // batch_size + (1 if len(generated_indices) % batch_size else 0)
        )
        
        for i in range(max_batches):
            # Get real batch (with wraparound if needed)
            real_batch_indices = real_indices[i*batch_size:(i+1)*batch_size] if real_indices else []
            gen_batch_indices = generated_indices[i*batch_size:(i+1)*batch_size] if generated_indices else []
            
            real_init, real_tgt = None, None
            gen_init, gen_tgt = None, None
            
            if real_batch_indices:
                real_samples = [self._get_sample(idx) for idx in real_batch_indices]
                real_init = math.stack([s.initial_state for s in real_samples], batch('batch'))
                real_tgt = math.stack([s.targets for s in real_samples], batch('batch'))
            
            if gen_batch_indices:
                gen_samples = [self._get_sample(idx) for idx in gen_batch_indices]
                gen_init = math.stack([s.initial_state for s in gen_samples], batch('batch'))
                gen_tgt = math.stack([s.targets for s in gen_samples], batch('batch'))
            
            yield SeparatedBatch(real_init, real_tgt, gen_init, gen_tgt)


    def _apply_alpha(self, indices: List[int]) -> List[int]:
        """Apply alpha sampling to select a proportion of indices."""
        if self.alpha >= 1.0:
            return indices
        num_to_select = max(1, int(len(indices) * self.alpha))
        return random.sample(indices, num_to_select)
