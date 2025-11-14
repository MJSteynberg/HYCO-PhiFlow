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
from .dataset_utilities import DatasetBuilder, AugmentationHandler, FilteringManager
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
    # Dataset now assumes BVTS as the canonical in-memory layout.
    # All simulation tensors returned by DataManager / builder are normalized
    # to BVTS [B, V, T, *spatial].

        self._log_dataset_info()

        # Normalize any augmented samples we received from the builder so that
        # they follow the same tensor conventions as _load_simulation() returns.
        # This prevents mixed-format augmented samples which can cause collate
        # failures when batches mix real and generated data.
        try:
            self._normalize_all_augmented_samples()
        except Exception:
            # Fail-safe: don't raise during dataset construction; log instead
            logger.debug("Failed to normalize augmented samples at init; continuing")


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
        builder = DatasetBuilder(data_manager)

        num_frames = builder.setup_cache(sim_indices, field_names, num_frames, num_predict_steps)
        samples_per_sim = builder.compute_sliding_window(num_frames, num_predict_steps)
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
        full_data = self.data_manager.get_or_load_simulation(
            sim_idx, field_names=self.field_names, num_frames=self.num_frames
        )
        
        sim_data = full_data["tensor_data"]

        # Always return BVTS in-memory. Convert cached tensors to BVTS using the
        # adapter/helper.
        try:
            from src.data.adapters.bvts_adapter import sim_dict_to_bvts
        except Exception:
            sim_dict_to_bvts = None

        if sim_dict_to_bvts is not None:
            sim_bvts = sim_dict_to_bvts(sim_data)
        else:
            try:
                from src.utils.field_conversion.bvts import to_bvts
            except Exception:
                to_bvts = None

            sim_bvts = {
                k: (to_bvts(v) if to_bvts is not None and isinstance(v, torch.Tensor) else v)
                for k, v in sim_data.items()
            }

        if self.pin_memory:
            sim_bvts = {field: tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor for field, tensor in sim_bvts.items()}

        return sim_bvts
    
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

        # Expect BVTS tensors: [B, C_all, T, H, W]
        all_data = torch.cat(all_field_tensors, dim=1)  # [B, C_all, T, H, W]

        # initial_state: keep time as length-1 dimension -> [B, C_all, 1, H, W]
        initial_state = all_data[:, :, start_frame : start_frame + 1]

        target_start = start_frame + 1
        target_end = start_frame + 1 + self.num_predict_steps
        rollout_targets = all_data[:, :, target_start:target_end]  # [B, C_all, T_pred, H, W]

        # Normalize per-sample shapes: remove the simulation-batch dim when it is singleton
        if initial_state.dim() == 5 and initial_state.size(0) == 1:
            initial_state = initial_state.squeeze(0)
            rollout_targets = rollout_targets.squeeze(0)

        # Ensure initial has a time dim
        if initial_state.dim() == 3:
            initial_state = initial_state.unsqueeze(1)

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

    # Defensive normalization: ensure the augmented trajectory's tensor_data
    # matches the dataset mode (BVTS). This handles the case where
    # augmented_samples were loaded from external sources with differing conventions.
        tensor_data = self._normalize_sim_tensor_data(tensor_data)
        
        # Concatenate all fields
        all_field_tensors = [tensor_data[name] for name in self.field_names]

        # Augmented trajectories are stored in BVTS cache-format and have been
        # normalized by _normalize_sim_tensor_data earlier. Expect [B, C_all, T, H, W]
        all_data = torch.cat(all_field_tensors, dim=1)  # [B, C_all, T, H, W]

        initial_state = all_data[:, :, window_start : window_start + 1]  # [B, C_all, 1, H, W]
        target_start = window_start + 1
        target_end = window_start + 1 + self.num_predict_steps
        rollout_targets = all_data[:, :, target_start:target_end]  # [B, C_all, T_pred, H, W]

        if initial_state.dim() == 5 and initial_state.size(0) == 1:
            initial_state = initial_state.squeeze(0)
            rollout_targets = rollout_targets.squeeze(0)

        if initial_state.dim() == 3:
            initial_state = initial_state.unsqueeze(1)

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
            
            tensor_trajectory = self._convert_field_rollout_to_cache_format(rollout)
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

    def _convert_field_rollout_to_cache_format(
        self,
        rollout: Dict[str, Field]
    ) -> Dict[str, torch.Tensor]:
        """
        Convert a Field rollout to cache-compatible tensor format.
        
        Args:
            rollout: Dict mapping field_name to Field[time, x, y]
        
        Returns:
            Dict with structure: {'tensor_data': {field_name: tensor[T, C, H, W]}}
        """
        from src.utils.field_conversion import make_converter
        
        tensor_data = {}
        
        for field_name in self.field_names:
            if field_name not in rollout:
                continue
            
            field_traj = rollout[field_name]
            num_timesteps = field_traj.shape.get_size('time')
            
            # Try batch conversion first (optimization)
            trajectory_tensor = self._try_batch_convert_trajectory(
                field_traj, num_timesteps
            )
            
            if trajectory_tensor is None:
                # Fallback to sequential conversion
                trajectory_tensor = self._convert_trajectory_sequential(
                    field_traj, num_timesteps
                )
            
            tensor_data[field_name] = trajectory_tensor.cpu()
        
        return {'tensor_data': tensor_data}

    def _try_batch_convert_trajectory(
        self,
        field_traj: Field,
        num_timesteps: int
    ) -> Optional[torch.Tensor]:
        """
        Try to convert entire trajectory at once (fastest path).
        
        Returns:
            Tensor [T, C, H, W] or None if batch conversion not possible
        """
        try:
            if hasattr(field_traj, 'values') and hasattr(field_traj.values, '_native'):
                native_tensor = field_traj.values._native
                
                if native_tensor.dim() == 4:  # Vector field [x, y, vector, time]
                    tensor = native_tensor.permute(3, 2, 0, 1)  # -> [time, vector, x, y]
                elif native_tensor.dim() == 3:  # Scalar field [x, y, time]
                    tensor = native_tensor.permute(2, 0, 1).unsqueeze(1)  # -> [time, 1, x, y]
                else:
                    return None
                
                if tensor.shape[0] == num_timesteps:
                    return tensor
            
            return None
        except Exception:
            return None

    def _convert_trajectory_sequential(
        self,
        field_traj: Field,
        num_timesteps: int
    ) -> torch.Tensor:
        """
        Convert trajectory one timestep at a time (fallback).
        
        Returns:
            Tensor [T, C, H, W]
        """
        from src.utils.field_conversion import make_converter
        
        first_field = field_traj.time[0]
        converter = make_converter(first_field)
        
        tensors = []
        for t in range(num_timesteps):
            field_t = field_traj.time[t]
            tensor_t = converter.field_to_tensor(field_t, ensure_cpu=True)
            
            if tensor_t.dim() == 4:
                tensor_t = tensor_t.squeeze(0)
            
            tensors.append(tensor_t)
        
        return torch.stack(tensors, dim=0)

    # ==================== Normalization Utilities ====================

    def _normalize_sim_tensor_data(self, sim_tensor_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Normalize a simulation's `tensor_data` dict to the canonical BVTS format
        used throughout the codebase: [B, C, T, *spatial].

    This converts tensors to the canonical in-memory BVTS layout
    using the converter helpers. If the BVTS helper is not available
    a RuntimeError is raised to avoid silent misinterpretation.
        """
        try:
            from src.utils.field_conversion.bvts import to_bvts, from_bvts
        except Exception:
            to_bvts = None
            from_bvts = None

        normalized = {}
        for field, tensor in sim_tensor_data.items():
            if not isinstance(tensor, torch.Tensor):
                normalized[field] = tensor
                continue

            # Convert to BVTS canonical in-memory format [B, V, T, *spatial]
            # We require the BVTS helper to be present; treat missing helper as
            # a hard error to avoid silent misinterpretation.
            if to_bvts is None:
                raise RuntimeError(
                    "BVTS conversion helper not available. Ensure 'src.utils.field_conversion.bvts.to_bvts' is importable."
                )

            normalized[field] = to_bvts(tensor)

        return normalized

    def _normalize_all_augmented_samples(self):
        """Normalize all entries in self.augmented_samples in-place."""
        if not hasattr(self, 'augmented_samples') or not self.augmented_samples:
            return

        normalized_list = []
        for entry in self.augmented_samples:
            if isinstance(entry, dict) and 'tensor_data' in entry:
                normalized_list.append({'tensor_data': self._normalize_sim_tensor_data(entry['tensor_data'])})
            elif isinstance(entry, dict):
                # Possibly raw tensor_data
                normalized_list.append({'tensor_data': self._normalize_sim_tensor_data(entry)})
            else:
                # Unknown format: keep as-is
                normalized_list.append(entry)

        self.augmented_samples = normalized_list

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