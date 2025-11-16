"""
TensorDataset - Refactored for Physical Trajectory Augmentation

Key Changes:
- Augmented data = physically-generated trajectories (stored as cache-format)
- These trajectories are windowed exactly like real data
- _get_augmented_sample() applies sliding window to trajectory data
- Clear distinction maintained between real and physically-generated data
"""

from typing import List, Optional, Dict, Any, Tuple
import torch
from phi.field import Field

from .abstract_dataset import AbstractDataset
from .data_manager import DataManager
from .abstract_dataset import FilteringManager
from .augmentation_manager import AugmentationHandler
from src.utils.logger import get_logger, logging

logger = get_logger(__name__)


class TensorDataset(AbstractDataset):
    """
    PyTorch Dataset that returns tensors for synthetic training.
    
    Augmentation source: Physically-generated trajectories
    - Stored in cache-compatible format: {'tensor_data': {field: tensor[T,C,H,W]}}
    - Windowed exactly like real data
    - Remain distinguishable via _is_augmented_trajectory()
    """
    
    def __init__(
        self,
        data_manager: DataManager,
        sim_indices: List[int],
        field_names: List[str],
        num_frames: Optional[int],
        num_predict_steps: int,
        augmentation_config: Optional[Dict[str, Any]] = None,
        access_policy: str = "both",
        max_cached_sims: int = 5,
        pin_memory: bool = True,
        percentage_real_data: float = 1.0,
    ):
        """Initialize TensorDataset."""
        self.pin_memory = pin_memory and torch.cuda.is_available()
        
        # Setup dataset using builder
        logger.debug("Setting up TensorDataset...")
        num_frames, num_real, augmented_samples, index_mapper = self._setup_dataset(
            data_manager, sim_indices, field_names, num_frames, num_predict_steps,
            augmentation_config, percentage_real_data
        )
        
        # Call parent constructor
        super().__init__(
            data_manager=data_manager,
            sim_indices=sim_indices,
            field_names=field_names,
            num_frames=num_frames,
            num_predict_steps=num_predict_steps,
            access_policy=access_policy,
            num_real=num_real,
            augmented_samples=augmented_samples,
            index_mapper=index_mapper,
            max_cached_sims=max_cached_sims,
        )
        
        # Track number of augmented trajectories (for indexing)
        self._num_augmented_trajectories = len(augmented_samples)
        self._samples_per_trajectory = num_frames - num_predict_steps


        self._log_dataset_info()


    # ==================== Setup ====================
    
    def _setup_dataset(
        self,
        data_manager: DataManager,
        sim_indices: List[int],
        field_names: List[str],
        num_frames: Optional[int],
        num_predict_steps: int,
        augmentation_config: Optional[Dict[str, Any]],
        percentage_real_data: float,
    ) -> Tuple[int, int, List[Any], Optional[FilteringManager]]:
        """Setup dataset components."""
        # Use helper functions from AbstractDataset instead of DatasetBuilder
        num_frames = AbstractDataset.setup_cache(
            data_manager, sim_indices, field_names, num_frames, num_predict_steps
        )
        samples_per_sim = AbstractDataset.compute_sliding_window(
            num_frames, num_predict_steps
        )
        total_real_samples = len(sim_indices) * samples_per_sim
        
        # Setup filtering
        index_mapper = None
        if percentage_real_data < 1.0:
            index_mapper = FilteringManager(total_real_samples, percentage_real_data)
            num_real = index_mapper.num_samples
            logger.debug(
                f"  Filtering enabled: using {num_real}/{total_real_samples} "
                f"samples ({percentage_real_data*100:.1f}%)"
            )
        else:
            num_real = total_real_samples
        
        # Setup augmentation (if provided)
        augmented_samples = []
        if augmentation_config:
            augmented_samples = AugmentationHandler.load_augmentation(
                augmentation_config, num_real, num_predict_steps, field_names
            )
        
        return num_frames, num_real, augmented_samples, index_mapper
    
    def _log_dataset_info(self):
        """Log dataset information."""
        if self.access_policy == "both":
            logger.debug(
                f"  Dataset: {self.num_real} real + {self.num_augmented} "
                f"augmented = {len(self)} samples"
            )
        elif self.access_policy == "real_only":
            logger.debug(f"  Dataset: {self.num_real} real samples")
        elif self.access_policy == "generated_only":
            if self.num_augmented == 0:
                logger.warning(
                    "  Dataset: access_policy=generated_only but no augmented samples!"
                )
            logger.debug(f"  Dataset: {self.num_augmented} generated samples")

    # ==================== Implementation of Abstract Methods ====================
    
    def _load_simulation(self, sim_idx: int) -> Dict[str, torch.Tensor]:
        """
        Load simulation data from cache.
        
        Returns tensor_data dict directly.
        """
        full_data = self.data_manager.load_simulation(
            sim_idx, field_names=self.field_names, num_frames=self.num_frames
        )
        
        sim_data = full_data["tensor_data"]


        if self.pin_memory:
            sim_data = {field: tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor for field, tensor in sim_data.items()}

        return sim_data
    
    def _extract_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract a windowed sample from real simulations.
        
        Returns:
            Tuple of (initial_state, rollout_targets)
        """
        sim_idx, start_frame = self._compute_sim_and_frame(idx)
        sim_data = self._cached_load_simulation(sim_idx)

        
        # Concatenate all fields
        all_field_tensors = [sim_data[name] for name in self.field_names]

        all_data = torch.cat(all_field_tensors, dim=0)  # [C_all, T, H, W]

        # Extract window
        initial_state = all_data[:, start_frame:start_frame+1, :, :]  # [V, 1, H, W]
        target_start = start_frame + 1
        target_end = start_frame + 1 + self.num_predict_steps
        rollout_targets = all_data[:, target_start:target_end, :, :]  # [V, T_pred, H, W]
        
        return initial_state, rollout_targets
    
    def _get_augmented_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get sample from augmented (physically-generated) trajectories.
        
        These are stored as full trajectories and windowed on-the-fly,
        exactly like real data.
        
        Args:
            idx: Index within augmented samples
        
        Returns:
            Tuple of (initial_state, rollout_targets)
        """
        # Compute which trajectory and which window within that trajectory
        traj_idx = idx // self._samples_per_trajectory
        window_start = idx % self._samples_per_trajectory
        
        if traj_idx >= self._num_augmented_trajectories:
            raise IndexError(
                f"Augmented index {idx} out of range "
                f"(trajectory {traj_idx} >= {self._num_augmented_trajectories})"
            )
        
        # Get the trajectory data (in cache format)
        trajectory_data = self.augmented_samples[traj_idx]
        
        # Extract tensor_data
        if isinstance(trajectory_data, dict) and 'tensor_data' in trajectory_data:
            tensor_data = trajectory_data['tensor_data']
        else:
            # Assume it's already tensor_data
            tensor_data = trajectory_data
        
        # Concatenate all fields
        all_field_tensors = [tensor_data[name] for name in self.field_names]

        # Augmented trajectories are stored in BVTS cache-format and have been
        # normalized by _normalize_sim_tensor_data earlier. Expect [C_all, T, H, W]
        all_data = torch.cat(all_field_tensors, dim=0).cpu()  # [C_all, T, H, W]


        initial_state = all_data[:, window_start : window_start + 1]  # [C_all, 1, H, W]
        target_start = window_start + 1
        target_end = window_start + 1 + self.num_predict_steps
        rollout_targets = all_data[:, target_start:target_end]  # [C_all, T_pred, H, W]

        return initial_state, rollout_targets

    # ==================== Augmentation Management ====================

    def set_augmented_trajectories(self, trajectory_rollouts: List[Dict[str, Field]]):
        """
        Set augmented data from physically-generated Field trajectories.
        
        Converts Field rollouts to cache-compatible tensor format.
        
        Args:
            trajectory_rollouts: List of rollout dicts where each is
                                {'field_name': Field[time, x, y]}
        """
        if not trajectory_rollouts:
            self.augmented_samples = []
            self.num_augmented = 0
            self._num_augmented_trajectories = 0
            self._total_length = self._compute_length()
            return
        
        logger.debug(f"Converting {len(trajectory_rollouts)} physical trajectories...")
        
        # Convert each Field trajectory to cache format
        converted_trajectories = []
        for idx, rollout in enumerate(trajectory_rollouts):
            if idx % 10 == 0 and idx > 0:
                logger.debug(f"  Converted {idx}/{len(trajectory_rollouts)} trajectories...")
            
            tensor_trajectory =  {field_name: rollout[field_name].values.native('vector,time,x,y') for field_name in self.field_names if field_name in rollout}
            converted_trajectories.append(tensor_trajectory)
        
        # Store trajectories
        self.augmented_samples = converted_trajectories
        self._num_augmented_trajectories = len(converted_trajectories)
        
        # Calculate total augmented samples (with windowing)
        self.num_augmented = self._num_augmented_trajectories * self._samples_per_trajectory
        logger.debug(
            f"Set {self._num_augmented_trajectories} augmented trajectories "
            f"{self._samples_per_trajectory} samples each = "
            f"({self.num_augmented} windowed samples)"
        )
        # Recompute total length
        self._total_length = self._compute_length()
        
        logger.debug(
            f"Set {self._num_augmented_trajectories} augmented trajectories "
            f"({self.num_augmented} windowed samples)"
            f"; total dataset length is now {self._total_length} samples"
            f", where it should be {self.num_augmented + self.num_real} samples."
        )

    def clear_augmented_trajectories(self):
        """Clear all augmented trajectories."""
        self.augmented_samples = []
        self.num_augmented = 0
        self._num_augmented_trajectories = 0
        self._total_length = self._compute_length()
        logger.debug("Cleared augmented trajectories")

    # ==================== Utility Methods ====================

    def _is_augmented_trajectory(self, idx: int) -> bool:
        """
        Check if a sample index corresponds to an augmented trajectory.
        
        Args:
            idx: Global sample index
        
        Returns:
            True if from augmented trajectory, False if real
        """
        if self.access_policy == "generated_only":
            return True
        elif self.access_policy == "real_only":
            return False
        else:  # 'both'
            return idx >= self.num_real

    def get_sample_source(self, idx: int) -> str:
        """
        Get the source of a sample.
        
        Args:
            idx: Sample index
        
        Returns:
            'real' or 'physical_generated'
        """
        return 'physical_generated' if self._is_augmented_trajectory(idx) else 'real'

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TensorDataset(\n"
            f"  simulations={len(self.sim_indices)},\n"
            f"  samples={len(self)} (real={self.num_real}, aug={self.num_augmented}),\n"
            f"  augmented_trajectories={self._num_augmented_trajectories},\n"
            f"  fields={len(self.field_names)},\n"
            f"  frames={self.num_frames},\n"
            f"  predict_steps={self.num_predict_steps},\n"
            f"  pin_memory={self.pin_memory}\n"
            f")"
        )