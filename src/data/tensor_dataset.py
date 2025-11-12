"""
Tensor Dataset for Synthetic Model Training

Returns PyTorch tensors suitable for neural network training.
Parallel structure to FieldDataset with tensor-specific implementations.
"""

from typing import List, Optional, Dict, Any, Tuple
import torch
from phi.field import Field
from src.utils.field_conversion import make_converter
from .abstract_dataset import AbstractDataset
from .data_manager import DataManager
from .dataset_utilities import DatasetBuilder, AugmentationHandler, FilteringManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TensorDataset(AbstractDataset):
    """
    PyTorch Dataset that returns tensors for synthetic training.
    
    Returns samples in format:
    - initial_state: [C_all, H, W] - concatenated tensor of ALL fields
    - rollout_targets: [T, C_all, H, W] - concatenated tensor for T steps
    
    Args:
        data_manager: DataManager for loading cached data
        sim_indices: List of simulation indices
        field_names: List of all field names to load
        num_frames: Number of frames per simulation (None = auto-detect)
        num_predict_steps: Number of prediction steps
        dynamic_fields: List of fields that are predicted
        static_fields: List of fields that are input-only
        augmentation_config: Optional augmentation configuration
        access_policy: 'both', 'real_only', or 'generated_only'
        max_cached_sims: LRU cache size
        pin_memory: If True, pin tensors for faster GPU transfer
        percentage_real_data: Percentage of real data to use (0.0 < p <= 1.0)
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
        """Initialize TensorDataset with validation and setup."""

        self.pin_memory = pin_memory and torch.cuda.is_available()
        
        # Setup dataset using builder
        logger.debug("Setting up TensorDataset...")
        num_frames, num_real, augmented_samples, index_mapper = self._setup_dataset(
            data_manager, sim_indices, field_names, num_frames, num_predict_steps,
            augmentation_config, percentage_real_data
        )
        
        # Call parent constructor with pre-computed values
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
        
        # Log dataset info
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
        """
        Setup dataset components.
        
        Returns:
            Tuple of (num_frames, num_real, augmented_samples, index_mapper)
        """
        builder = DatasetBuilder(data_manager)
        
        # Setup cache and determine num_frames
        num_frames = builder.setup_cache(sim_indices, field_names, num_frames)
        
        # Compute sliding window
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
        
        # Setup augmentation
        augmented_samples = []
        if augmentation_config:
            raw_samples = AugmentationHandler.load_augmentation(
                augmentation_config, num_real, num_predict_steps, field_names
            )
            # Process each augmented sample
            augmented_samples = [
                self._process_augmented_sample(sample) for sample in raw_samples
            ]
        
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
    
    # In tensor_dataset.py, replace _load_simulation (around line 165):

    def _load_simulation(self, sim_idx: int) -> Dict[str, torch.Tensor]:
        """
        Load simulation data (handles both real and synthetic).
        """
        # Check if this is a synthetic simulation
        if hasattr(self, '_synthetic_sims') and sim_idx >= len(self.sim_indices):
            synthetic_idx = sim_idx - len(self.sim_indices)
            if synthetic_idx < len(self._synthetic_sims):
                # Return tensor_data directly from synthetic sim
                sim_data = self._synthetic_sims[synthetic_idx]['tensor_data']
                
                # Pin memory if configured
                if self.pin_memory:
                    sim_data = {
                        field: tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor
                        for field, tensor in sim_data.items()
                    }
                return sim_data
        
        # Otherwise load real simulation from cache
        full_data = self.data_manager.get_or_load_simulation(
            sim_idx, field_names=self.field_names, num_frames=self.num_frames
        )
        
        # Extract tensor_data from the full cache structure
        sim_data = full_data["tensor_data"]
        
        # Pin memory if configured
        if self.pin_memory:
            sim_data = {
                field: tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor
                for field, tensor in sim_data.items()
            }
        
        return sim_data
    
    def _extract_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract a single sample as tensors.
        
        Args:
            idx: Actual (unfiltered) sample index
        
        Returns:
            Tuple of (initial_state, rollout_targets)
            - initial_state: [C_all, H, W]
            - rollout_targets: [T, C_all, H, W]
        """
        # Compute simulation and frame
        sim_idx, start_frame = self._compute_sim_and_frame(idx)
        
        # Load simulation
        sim_data = self._cached_load_simulation(sim_idx)
        
        # Concatenate all fields
        all_field_tensors = [sim_data[name] for name in self.field_names]
        all_data = torch.cat(all_field_tensors, dim=1)  # [T, C_all, H, W]
        
        # Extract initial state
        initial_state = all_data[start_frame]  # [C_all, H, W]
        
        # Extract target rollout
        target_start = start_frame + 1
        target_end = start_frame + 1 + self.num_predict_steps
        rollout_targets = all_data[target_start:target_end]  # [T, C_all, H, W]
        
        return initial_state, rollout_targets
    
    def _process_augmented_sample(self, data: Any) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert a collection of augmented data into a list of tensor samples.
        
        This method acts as a router, handling different input formats:
        - A single batched rollout tensor [B, T, C, H, W].
        - A list of raw trajectory windows (List[List[Dict[str, Field]]]).
        - A list of already-processed (input, target) tensor tuples.
        
        Args:
            data: The raw augmented data to process.
        
        Returns:
            A list of (initial_state, rollout_targets) tensor tuples.
        """
        # Case 1: Input is a single batched rollout tensor
        if isinstance(data, torch.Tensor) and data.dim() == 5:
            return self._convert_trajectory_rollout(data)
        
        # Case 2: Input is a list
        if isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            # Case 2a: List of raw trajectory windows
            if isinstance(first_item, list) and isinstance(first_item[0], dict):
                # Process each window in the list
                return [self._convert_trajectory_window(window) for window in data]
            # Case 2b: Assume it's a list of already-processed tensor tuples
            elif isinstance(first_item, tuple) and isinstance(first_item[0], torch.Tensor):
                return data
        
        # Fallback for unknown or empty data formats
        logger.warning(f"Unsupported augmentation data type: {type(data)}. Returning empty list.")
        return []
    
    def _convert_trajectory_window(
        self,
        window_states: List[Dict[str, Field]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert trajectory window to tensor format.
        
        Args:
            window_states: List of states [Dict[str, Field]]
                         Length = num_predict_steps + 1
        
        Returns:
            Tuple of (initial_state, rollout_targets)
        """
        
        # First state is initial condition
        initial_fields = window_states[0]
        target_states = window_states[1:]
        
        # Convert initial state to tensors and concatenate
        initial_tensors = []
        for field_name in self.field_names:
            field = initial_fields[field_name]
            converter = make_converter(field)
            tensor = converter.field_to_tensor(field, ensure_cpu=True)
            # Remove batch dimension if present
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            initial_tensors.append(tensor)
        
        initial_state = torch.cat(initial_tensors, dim=0)  # [C_all, H, W]
        
        # Convert target states
        target_tensors_list = []
        for state in target_states:
            state_tensors = []
            for field_name in self.field_names:
                field = state[field_name]
                converter = make_converter(field)
                tensor = converter.field_to_tensor(field, ensure_cpu=True)
                if tensor.dim() == 4:
                    tensor = tensor.squeeze(0)
                state_tensors.append(tensor)
            
            state_tensor = torch.cat(state_tensors, dim=0)
            target_tensors_list.append(state_tensor)
        
        rollout_targets = torch.stack(target_tensors_list, dim=0)  # [T, C_all, H, W]
        
        return initial_state, rollout_targets
    
    
    def _convert_trajectory_rollout(
        self,
        rollout: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Efficiently convert a batched rollout tensor into windowed samples.

        Args:
            rollout: A single tensor of shape [B, T, C, H, W] representing
                     B trajectories of length T.

        Returns:
            A list of (initial_state, rollout_targets) tuples.
        """
        num_trajectories, traj_length, _, _, _ = rollout.shape
        num_windows = traj_length - self.num_predict_steps

        if num_windows <= 0:
            return []

        samples = []
        for i in range(num_trajectories):
            trajectory = rollout[i]  # [T, C, H, W]
            for start_frame in range(num_windows):
                # Initial state is a single frame
                initial_state = trajectory[start_frame]  # [C, H, W]

                # Target is a sequence of subsequent frames
                target_start = start_frame + 1
                target_end = start_frame + 1 + self.num_predict_steps
                rollout_targets = trajectory[target_start:target_end]  # [pred_steps, C, H, W]
                samples.append((initial_state, rollout_targets))
        
        return samples


    
    # ==================== Utility Methods ====================
    
    def _compute_sim_and_frame(self, idx: int) -> Tuple[int, int]:
        """
        Compute simulation index and starting frame from sample index.
        Handles both real and synthetic simulations.
        """
        samples_per_sim = self.num_frames - self.num_predict_steps
        
        # Check if this is a synthetic simulation
        if hasattr(self, '_num_real_only'):
            real_samples = self._num_real_only
            if idx >= real_samples:
                # This is a synthetic simulation - compute offset within synthetic data
                synthetic_sample_idx = idx - real_samples
                sim_offset = synthetic_sample_idx // samples_per_sim
                start_frame = synthetic_sample_idx % samples_per_sim
                sim_idx = len(self.sim_indices) + sim_offset
                return sim_idx, start_frame
        
        # Real simulation
        sim_offset = idx // samples_per_sim
        start_frame = idx % samples_per_sim
        sim_idx = self.sim_indices[sim_offset]
        return sim_idx, start_frame
    
    def resample_real_data(self, seed: Optional[int] = None):
        """
        Resample the subset of real data (if filtering is active).
        
        Args:
            seed: Optional random seed for reproducibility
        """
        if self._index_mapper is None:
            logger.debug("No resampling needed (percentage_real_data=1.0)")
            return
        
        self._index_mapper.resample(seed)
    
    def get_field_info(self) -> Dict[str, Any]:
        """Get field configuration information."""
        return {
            "all_fields": self.field_names,
            "num_all_fields": len(self.field_names),
        }
    
    def get_tensor_shapes(self, idx: int = 0) -> Dict[str, tuple]:
        """
        Get tensor shapes for debugging.
        
        Args:
            idx: Sample index to check
        
        Returns:
            Dictionary with shape information
        """
        if idx >= self.num_real:
            raise ValueError(
                f"Index {idx} is augmented, cannot determine shapes from real data"
            )
        
        initial, targets = self._extract_sample(idx)
        
        return {
            "initial_state": tuple(initial.shape),
            "rollout_targets": tuple(targets.shape),
            "num_all_channels": initial.shape[0],
            "spatial_dims": tuple(initial.shape[1:]),
            "num_predict_steps": targets.shape[0],
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TensorDataset(\n"
            f"  simulations={len(self.sim_indices)},\n"
            f"  samples={len(self)} (real={self.num_real}, aug={self.num_augmented}),\n"
            f"  fields={len(self.field_names)} "
            f"  frames={self.num_frames},\n"
            f"  predict_steps={self.num_predict_steps},\n"
            f"  pin_memory={self.pin_memory}\n"
            f")"
        )