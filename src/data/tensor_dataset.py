"""
Tensor Dataset for Synthetic Model Training

Returns PyTorch tensors in format suitable for neural network training.
Inherits common functionality from AbstractDataset.

This dataset is specifically designed for synthetic (neural network) models that
operate on tensor data. It handles:
- Loading tensors from DataManager cache
- Concatenating multiple fields into single tensors
- Separating dynamic (predicted) and static (input-only) fields
- Pin memory for faster GPU transfer
"""

from typing import List, Optional, Dict, Any, Tuple
import torch

from .abstract_dataset import AbstractDataset
from .data_manager import DataManager


class TensorDataset(AbstractDataset):
    """
    PyTorch Dataset that returns tensors for synthetic training.

    Inherits from AbstractDataset:
    - Lazy loading with LRU cache
    - Sliding window support
    - Optional augmentation (built-in)
    - Cache validation and management

    Additional features specific to tensor mode:
    - Static/dynamic field separation
    - Pin memory for GPU transfer
    - Efficient tensor concatenation

    The dataset returns samples in the format expected by synthetic models:
    - initial_state: [C_all, H, W] - concatenated tensor of ALL fields
    - rollout_targets: [T, C_dynamic, H, W] - concatenated tensor of DYNAMIC fields only

    Args:
        data_manager: DataManager instance for loading cached data
        sim_indices: List of simulation indices to include
        field_names: List of all field names to load (e.g., ['velocity', 'density', 'inflow'])
        num_frames: Number of frames per simulation (None = load all)
        num_predict_steps: Number of autoregressive prediction steps
        dynamic_fields: List of fields that are predicted by the model
        static_fields: List of fields that are input-only (not predicted)
        use_sliding_window: If True, create multiple samples per simulation
        augmentation_config: Optional augmentation configuration dict
        max_cached_sims: LRU cache size (number of simulations in memory)
        pin_memory: If True, pin tensors for faster GPU transfer

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - initial_state: [C_all, H, W] - all fields at starting timestep
            - rollout_targets: [T, C_all, H, W] - all fields for next T steps

    Example:
        >>> dataset = TensorDataset(
        ...     data_manager=data_manager,
        ...     sim_indices=[0, 1, 2],
        ...     field_names=['velocity', 'density', 'inflow'],
        ...     num_frames=None,  # Load all frames
        ...     num_predict_steps=10,
        ...     dynamic_fields=['velocity', 'density'],
        ...     static_fields=['inflow'],
        ...     use_sliding_window=True,
        ... )
        >>> initial, targets = dataset[0]
        >>> print(initial.shape)  # [C_all, 64, 64] where C_all = sum of all channels
        >>> print(targets.shape)  # [10, C_all, 64, 64] - all fields for consistency
    """

    def __init__(
        self,
        data_manager: DataManager,
        sim_indices: List[int],
        field_names: List[str],
        num_frames: Optional[int],
        num_predict_steps: int,
        dynamic_fields: List[str],
        static_fields: List[str] = None,
        augmentation_config: Optional[Dict[str, Any]] = None,
        access_policy: str = "both",
        max_cached_sims: int = 5,
        pin_memory: bool = True,
    ):
        """
        Initialize the TensorDataset.

        Validates field specifications and calls parent constructor to handle
        common initialization.
        """
        # Validate field specifications
        if not dynamic_fields:
            raise ValueError("dynamic_fields cannot be empty for TensorDataset")

        # Store tensor-specific attributes
        self.dynamic_fields = dynamic_fields
        self.static_fields = static_fields if static_fields is not None else []
        self.pin_memory = pin_memory and torch.cuda.is_available()

        # Validate that all fields are accounted for
        all_specified = set(self.dynamic_fields + self.static_fields)
        all_fields = set(field_names)
        if all_specified != all_fields:
            missing = all_fields - all_specified
            extra = all_specified - all_fields
            msg = []
            if missing:
                msg.append(f"Missing in dynamic/static: {missing}")
            if extra:
                msg.append(f"Extra in dynamic/static: {extra}")
            raise ValueError(
                f"Field mismatch. {' '.join(msg)}. "
                f"All fields must be classified as dynamic or static."
            )

        # Call parent constructor (handles common initialization)
        super().__init__(
            data_manager=data_manager,
            sim_indices=sim_indices,
            field_names=field_names,
            num_frames=num_frames,
            num_predict_steps=num_predict_steps,
            augmentation_config=augmentation_config,
            access_policy=access_policy,
            max_cached_sims=max_cached_sims,
        )

    # ==================== Implementation of Abstract Methods ====================

    def _load_simulation_uncached(self, sim_idx: int) -> Dict[str, torch.Tensor]:
        """
        Load simulation tensors from cache.

        This method is automatically wrapped with LRU caching by the parent class,
        so it only runs when the simulation is not in the cache.

        Args:
            sim_idx: Simulation index to load

        Returns:
            Dictionary mapping field names to tensors with shape [T, C, H, W]
        """
        # Load full data structure from DataManager
        full_data = self.data_manager.get_or_load_simulation(
            sim_idx, field_names=self.field_names, num_frames=self.num_frames
        )

        # Extract just the tensor data (we don't need metadata for tensor mode)
        sim_data = full_data["tensor_data"]

        # Optionally pin memory for faster GPU transfer
        if self.pin_memory:
            sim_data = {
                field: (
                    tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor
                )
                for field, tensor in sim_data.items()
            }

        return sim_data

    def _get_real_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a real (non-augmented) sample as tensors.

        This method:
        1. Determines simulation and starting frame from index
        2. Loads simulation data using LRU-cached loader
        3. Concatenates fields into tensors
        4. Extracts initial state and rollout targets

        Args:
            idx: Real sample index (0 to num_real-1)

        Returns:
            Tuple of (initial_state, rollout_targets) where:
            - initial_state: [C_all, H, W] tensor with all fields concatenated
            - rollout_targets: [T, C_dynamic, H, W] tensor with dynamic fields concatenated
        """
        # Get simulation and starting frame (inherited utility method)
        sim_idx, start_frame = self.get_simulation_and_frame(idx)

        # Load simulation data (uses LRU cache from parent class)
        sim_data = self._cached_load_simulation(sim_idx)

        # === Concatenate ALL fields for initial state ===
        # Initial state needs all information (dynamic + static)
        all_field_tensors = [sim_data[name] for name in self.field_names]
        all_data = torch.cat(all_field_tensors, dim=1)  # [T, C_all, H, W]

        # Extract initial state at starting frame
        initial_state = all_data[start_frame]  # [C_all, H, W]

        # === Concatenate ALL fields for targets ===
        # For consistency with UNet output structure, targets include all fields
        # Extract target rollout (next num_predict_steps frames)
        target_start = start_frame + 1
        target_end = start_frame + 1 + self.num_predict_steps
        rollout_targets = all_data[target_start:target_end]  # [T, C_all, H, W]

        return initial_state, rollout_targets

    # ==================== Additional Utility Methods ====================

    def get_field_info(self) -> Dict[str, Any]:
        """
        Get information about field configuration.

        Returns:
            Dictionary with field counts and names
        """
        return {
            "all_fields": self.field_names,
            "dynamic_fields": self.dynamic_fields,
            "static_fields": self.static_fields,
            "num_all_fields": len(self.field_names),
            "num_dynamic_fields": len(self.dynamic_fields),
            "num_static_fields": len(self.static_fields),
        }

    def get_tensor_shapes(self, idx: int = 0) -> Dict[str, tuple]:
        """
        Get the shapes of tensors returned by the dataset.

        Useful for debugging and model initialization.

        Args:
            idx: Sample index to check (default: 0)

        Returns:
            Dictionary with shape information
        """
        # Get a sample (only if we have real samples)
        if idx >= self.num_real:
            raise ValueError(
                f"Index {idx} is augmented, cannot determine shapes from real data"
            )

        initial, targets = self._get_real_sample(idx)

        return {
            "initial_state": tuple(initial.shape),
            "rollout_targets": tuple(targets.shape),
            "num_all_channels": initial.shape[0],
            "num_dynamic_channels": targets.shape[1],
            "spatial_dims": tuple(initial.shape[1:]),
            "num_predict_steps": targets.shape[0],
        }

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return (
            f"TensorDataset(\n"
            f"  simulations={len(self.sim_indices)},\n"
            f"  samples={len(self)} (real={self.num_real}, aug={self.num_augmented}),\n"
            f"  fields={len(self.field_names)} (dynamic={len(self.dynamic_fields)}, static={len(self.static_fields)}),\n"
            f"  frames={self.num_frames},\n"
            f"  predict_steps={self.num_predict_steps},\n"
            f"  sliding_window={self.use_sliding_window},\n"
            f"  pin_memory={self.pin_memory}\n"
            f")"
        )
