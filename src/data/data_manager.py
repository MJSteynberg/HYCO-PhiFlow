"""
Data Manager for Pure PhiML Data Pipeline

This module manages PhiML tensor caching with no PyTorch dependency.
Uses phiml.math.save() and phiml.math.load() for native PhiML tensor storage.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from phiml import math as phimath
from phiml.math import batch, spatial, channel
from phi.field import Field
from phi.torch.flow import Scene, stack

# Skip validation for now - focus on core functionality
# from .validation import CacheValidator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataManager:
    """
    Manages PhiML tensor caching for training data.

    Stores data as PhiML tensors with named dimensions using .npz format.
    No PyTorch dependency - pure PhiML throughout.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PhiML DataManager.

        Args:
            config: Configuration dictionary containing dataset parameters
        """
        self._parse_config(config)
        # Skip validation for now - focus on basic caching
        # self.validator = CacheValidator(config)

        logger.info(f"DataManager initialized: {self.cache_dir}")

    def _parse_config(self, config: Dict[str, Any]):
        """Parse configuration dictionary."""
        self.raw_data_dir = Path(config["data"]["data_dir"])
        self.cache_dir = Path(config["trainer"].get('hybrid', {}).get('augmentation', {}).get('cache_dir', 'data/cache_phiml'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.dset_name = config["data"]['dset_name']
        self.pde_name = config["model"]["physical"]["name"]
        self.pde_params = config["model"]["physical"]["pde_params"]
        self.fields_scheme = config["data"]["fields_scheme"]
        self.resolution = config["model"]["physical"]["resolution"]
        self.dt = config["model"]["physical"]["dt"]
        self.domain = config["model"]["physical"]["domain"]

    def get_cached_path(self, sim_index: int) -> Path:
        """
        Get path for cached PhiML data file.

        Args:
            sim_index: Simulation index

        Returns:
            Path to .npz cache file
        """
        cache_subdir = self.cache_dir / self.dset_name
        cache_subdir.mkdir(parents=True, exist_ok=True)
        return cache_subdir / f"sim_{sim_index:06d}.npz"

    def is_cached(
        self,
        sim_index: int,
        field_names: Optional[List[str]] = None,
        num_frames: Optional[int] = None,
    ) -> bool:
        """
        Check if simulation is cached with matching parameters.

        Args:
            sim_index: Simulation index
            field_names: Optional field names to verify
            num_frames: Optional number of frames to verify

        Returns:
            True if cached and valid, False otherwise
        """
        cache_path = self.get_cached_path(sim_index)

        if not cache_path.exists():
            return False

        if field_names is None and num_frames is None:
            return True

        # Load and validate
        try:
            cached_data = phimath.load(str(cache_path))
            metadata = cached_data.get("metadata", {})

            # Check fields
            if field_names is not None:
                cached_fields = set(cached_data.keys()) - {'metadata'}
                requested_fields = set(field_names)
                if cached_fields != requested_fields:
                    logger.warning(
                        f"Cache invalid for sim_{sim_index:06d}: field mismatch. Removing..."
                    )
                    cache_path.unlink()
                    return False

            # Check num frames
            if num_frames is not None:
                cached_num_frames = metadata.get("num_frames", 0)
                # Handle metadata that may be wrapped in arrays
                if hasattr(cached_num_frames, 'item'):
                    cached_num_frames = cached_num_frames.item()
                if cached_num_frames < num_frames:
                    logger.warning(
                        f"Cache invalid for sim_{sim_index:06d}: insufficient frames. Removing..."
                    )
                    cache_path.unlink()
                    return False

            # Skip enhanced validation for now - just basic checks
            # if field_names is not None:
            #     is_valid, reasons = self.validator.validate_cache(
            #         metadata, field_names, num_frames
            #     )
            #     if not is_valid:
            #         logger.warning(
            #             f"Cache invalid for sim_{sim_index:06d}: {', '.join(reasons)}. Removing..."
            #         )
            #         cache_path.unlink()
            #         return False

            return True

        except Exception as e:
            logger.error(f"Error validating cache for sim_{sim_index:06d}: {e}")
            logger.info(f"Removing corrupted cache...")
            try:
                cache_path.unlink()
            except:
                pass
            return False

    def load_and_cache_simulation(
        self, sim_index: int, field_names: List[str], num_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load simulation from Scene files and cache as PhiML tensors.

        Args:
            sim_index: Simulation index
            field_names: List of field names to load
            num_frames: Optional limit on frames

        Returns:
            Dict with PhiML tensors for each field + metadata
        """
        scene_dir = self.raw_data_dir / f"sim_{sim_index:06d}"
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

        logger.debug(f"Loading simulation {sim_index} from {scene_dir}")

        # Load Scene
        scene = Scene.at(scene_dir)

        # Get all frames
        available_frames = scene.frames
        if num_frames is not None:
            available_frames = available_frames[:num_frames]

        logger.debug(f"  Loading {len(available_frames)} frames for fields {field_names}")

        # Load fields across all frames
        field_data = {}
        for field_name in field_names:
            logger.debug(f"  Loading field '{field_name}'...")

            # Load all frames for this field
            field_frames = []
            for frame in available_frames:
                field = scene.read(field_name, frame=frame)
                field_frames.append(field)

            # Stack into single Field with time dimension
            stacked_field = stack(field_frames, batch('time'))

            # Convert to PhiML tensor
            phiml_tensor = self._field_to_phiml_tensor(stacked_field, field_name)
            field_data[field_name] = phiml_tensor

        # Create metadata
        metadata = self._create_metadata(field_data, field_names, len(available_frames))

        # Save cache
        cache_data = {**field_data, 'metadata': metadata}
        self._save_cache(sim_index, cache_data)

        logger.debug(f"  Cached simulation {sim_index} ({len(field_names)} fields, {len(available_frames)} frames)")

        return cache_data

    def load_simulation(
        self, sim_index: int, field_names: List[str], num_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load simulation data (from cache if available).

        Args:
            sim_index: Simulation index
            field_names: Field names to load
            num_frames: Optional frame limit

        Returns:
            Dict with PhiML tensors + metadata
        """
        if self.is_cached(sim_index, field_names, num_frames):
            return self._load_cache(sim_index)
        else:
            return self.load_and_cache_simulation(sim_index, field_names, num_frames)

    def _field_to_phiml_tensor(self, field: Field, field_name: str):
        """
        Convert PhiFlow Field to PhiML tensor with proper dimension naming.

        PhiFlow Fields already use PhiML tensors internally.
        We just need to ensure proper dimension naming.

        Args:
            field: PhiFlow Field
            field_name: Name of the field

        Returns:
            PhiML Tensor with named dimensions
        """
        # Get values (already a PhiML tensor)
        values = field.values

        # Ensure proper dimension naming
        # Expected: (time, x, y, vector) for vector fields
        #           (time, x, y) for scalar fields

        return values

    def _create_metadata(
        self, field_data: Dict[str, Any], field_names: List[str], num_frames: int
    ) -> Dict[str, Any]:
        """
        Create metadata dictionary for cached data.

        Args:
            field_data: Dictionary of PhiML tensors
            field_names: List of field names
            num_frames: Number of frames

        Returns:
            Metadata dictionary (simplified - no validation for now)
        """
        metadata = {
            "num_frames": num_frames,
            "field_names": field_names,
            "resolution": self.resolution,
            "dt": self.dt,
            "domain": self.domain,
            "pde_name": self.pde_name,
            "pde_params": self.pde_params,
            "fields_scheme": self.fields_scheme,
            "shape_info": {
                name: {
                    "shape": str(field_data[name].shape),
                    "dimension_names": list(field_data[name].shape.names)
                }
                for name in field_names
            }
        }

        return metadata

    def _save_cache(self, sim_index: int, data: Dict[str, Any]):
        """
        Save PhiML tensors to cache using phiml.math.save.

        Args:
            sim_index: Simulation index
            data: Dictionary with PhiML tensors + metadata
        """
        cache_path = self.get_cached_path(sim_index)
        phimath.save(str(cache_path), data)

    def _load_cache(self, sim_index: int) -> Dict[str, Any]:
        """
        Load PhiML tensors from cache using phiml.math.load.

        Args:
            sim_index: Simulation index

        Returns:
            Dictionary with PhiML tensors + metadata
        """
        cache_path = self.get_cached_path(sim_index)
        return phimath.load(str(cache_path))

    def clear_cache(self, sim_index: int):
        """
        Clear cache for a specific simulation.

        Args:
            sim_index: Simulation index
        """
        cache_path = self.get_cached_path(sim_index)
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Cleared cache for sim_{sim_index:06d}")

    def clear_all_caches(self):
        """Clear all cached simulations."""
        cache_subdir = self.cache_dir / self.dset_name
        if cache_subdir.exists():
            for cache_file in cache_subdir.glob("sim_*.npz"):
                cache_file.unlink()
            logger.info(f"Cleared all caches in {cache_subdir}")
