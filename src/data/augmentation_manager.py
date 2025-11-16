"""
Augmentation Manager - Extracted AugmentationHandler

This module centralizes augmentation loading and processing logic so it can
be reused by both FieldDataset and TensorDataset.
"""
from typing import List, Optional, Dict, Any
from pathlib import Path
import torch
from phi.field import Field
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AugmentationHandler:
    """
    Handles loading and processing of augmented samples.
    """

    @staticmethod
    def load_augmentation(
        cache_dir: str,
        num_real: int,
        rollout_steps: int,
        alpha: float,
        field_names: List[str],
        mode: str = "cache",
        data: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:

        if mode == "memory":
            return AugmentationHandler._load_from_memory(
                cache_dir, rollout_steps, field_names, data
            )
        elif mode == "cache":
            return AugmentationHandler._load_from_cache(
                cache_dir, num_real, alpha
            )
        else:
            raise ValueError(f"Unknown augmentation mode: {mode}")

    @staticmethod
    def _load_from_memory(
        cache_dir: str,
        rollout_steps: int,
        field_names: List[str],
        data: Optional[Dict[str, Any]],
    ) -> List[Any]:

        if AugmentationHandler._is_trajectory_data(data):
            logger.debug("  Processing raw trajectory data...")
            samples = AugmentationHandler._process_trajectory_data(
                data, rollout_steps, field_names
            )
        else:
            samples = data

        logger.debug(f"  Loaded {len(samples)} augmented samples from memory")
        return samples

    @staticmethod
    def _load_from_cache(
        cache_dir: str, num_real: int, alpha: float
    ) -> List[Any]:
        
        expected_count = int(num_real * alpha) if alpha > 0 else None

        cache_path = Path(cache_dir)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Augmentation cache not found: {cache_dir}\n"
                f"Please run cache generation first."
            )

        cache_files = sorted(cache_path.glob("sample_*.pt"))

        if len(cache_files) == 0:
            raise ValueError(
                f"No cached samples found in {cache_dir}\n"
                f"Expected to find sample_*.pt files."
            )

        if expected_count is not None and len(cache_files) != expected_count:
            logger.warning(
                f"Expected {expected_count} cached samples but found {len(cache_files)}"
            )

        samples = []
        for file_path in cache_files:
            try:
                data = torch.load(file_path, map_location="cpu")
                if isinstance(data, dict):
                    samples.append((data["input"], data["target"]))
                else:
                    samples.append(data)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                raise

        logger.debug(f"  Loaded {len(samples)} cached augmented samples")
        return samples

        
    @staticmethod
    def _is_trajectory_data(data: Any) -> bool:
        if not isinstance(data, list) or len(data) == 0:
            return False

        first_item = data[0]

        if isinstance(first_item, list):
            if len(first_item) > 0 and isinstance(first_item[0], dict):
                return True

        return False

    @staticmethod
    def _process_trajectory_data(
        trajectories: List[List[Dict[str, Field]]],
        num_predict_steps: int,
        field_names: List[str],
    ) -> List[List[Dict[str, Field]]]:
        logger.debug(
            f"  Windowing {len(trajectories)} trajectories with "
            f"num_predict_steps={num_predict_steps}"
        )

        all_windows: List[List[Dict[str, Field]]] = []
        for traj_idx, trajectory in enumerate(trajectories):
            traj_length = len(trajectory)
            if traj_length < num_predict_steps + 1:
                logger.warning(
                    f"  Trajectory {traj_idx} too short ({traj_length} steps), "
                    f"need at least {num_predict_steps + 1}. Skipping."
                )
                continue

            num_windows = traj_length - num_predict_steps
            for window_start in range(num_windows):
                window_states = trajectory[
                    window_start : window_start + num_predict_steps + 1
                ]
                all_windows.append(window_states)

        logger.debug(
            f"  Created {len(all_windows)} windowed samples from "
            f"{len(trajectories)} trajectories"
        )
        return all_windows
