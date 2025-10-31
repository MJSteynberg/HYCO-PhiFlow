"""
Data Manager for Hybrid PDE Modeling

This module provides a centralized data management system that:
1. Loads data from PhiFlow Scene directories
2. Converts Field objects to tensors once and caches them
3. Stores metadata needed to reconstruct Field objects
4. Provides a clean interface for trainers to access data

The goal is to eliminate redundant conversions and provide a unified
data source for both physical and synthetic models.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import torch
from phi.torch.flow import Scene, stack, batch

from .validation import CacheValidator, compute_hash, get_cache_version, get_phiflow_version


class DataManager:
    """
    Manages the conversion and caching of PhiFlow Scene data to tensors.
    
    This class handles the expensive one-time conversion of Field objects
    to tensors and saves them with metadata that allows reconstruction.
    
    Attributes:
        raw_data_dir: Path to directory containing Scene subdirectories
        cache_dir: Path to directory where processed data will be cached
        config: Configuration dictionary for the dataset
    """
    
    def __init__(
        self, 
        raw_data_dir: str, 
        cache_dir: str, 
        config: Dict[str, Any],
        validate_cache: bool = True,
        auto_clear_invalid: bool = False
    ):
        """
        Initialize the DataManager.
        
        Args:
            raw_data_dir: Absolute path to the directory containing Scene data
                         (e.g., "data/burgers_128")
            cache_dir: Absolute path where cached tensors will be stored
                      (e.g., "data/cache")
            config: Configuration dictionary containing dataset parameters
            validate_cache: Whether to validate cached data against current config
            auto_clear_invalid: Whether to automatically remove invalid cache files
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.validate_cache = validate_cache
        self.auto_clear_invalid = auto_clear_invalid
        
        # Create cache validator
        self.validator = CacheValidator(config, strict=False)
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cached_path(self, sim_index: int) -> Path:
        """
        Get the path where cached data for a simulation should be stored.
        
        Args:
            sim_index: Index of the simulation
            
        Returns:
            Path object for the cached file
        """
        # Handle both flat and nested config structures
        if 'dset_name' in self.config:
            dataset_name = self.config['dset_name']
        else:
            dataset_name = self.config.get('data', {}).get('dset_name', 'default')
        
        cache_subdir = self.cache_dir / dataset_name
        cache_subdir.mkdir(parents=True, exist_ok=True)
        return cache_subdir / f"sim_{sim_index:06d}.pt"
    
    def is_cached(
        self, 
        sim_index: int,
        field_names: Optional[List[str]] = None,
        num_frames: Optional[int] = None
    ) -> bool:
        """
        Check if a simulation has already been cached with matching parameters.
        
        This method performs comprehensive validation if validate_cache is True,
        checking PDE parameters, resolution, domain, and more. If validation
        fails and auto_clear_invalid is True, the invalid cache will be removed.
        
        Args:
            sim_index: Index of the simulation
            field_names: Optional list of field names to verify
            num_frames: Optional number of frames to verify
            
        Returns:
            True if cached data exists AND matches parameters, False otherwise
        """
        cache_path = self.get_cached_path(sim_index)
        
        if not cache_path.exists():
            return False
        
        # If no validation parameters provided, just check existence
        if field_names is None and num_frames is None and not self.validate_cache:
            return True
        
        # Load and validate metadata
        try:
            cached_data = torch.load(cache_path, weights_only=False)
            metadata = cached_data.get('metadata', {})
            
            # Basic validation (always performed)
            if field_names is not None:
                cached_fields = set(cached_data['tensor_data'].keys())
                requested_fields = set(field_names)
                if cached_fields != requested_fields:
                    if self.auto_clear_invalid:
                        print(f"Cache invalid for sim_{sim_index:06d}: field mismatch. Removing...")
                        cache_path.unlink()
                    return False
            
            if num_frames is not None:
                cached_num_frames = metadata.get('num_frames', 0)
                if cached_num_frames < num_frames:
                    if self.auto_clear_invalid:
                        print(f"Cache invalid for sim_{sim_index:06d}: insufficient frames. Removing...")
                        cache_path.unlink()
                    return False
            
            # Enhanced validation (if enabled)
            if self.validate_cache and field_names is not None:
                is_valid, reasons = self.validator.validate_cache(
                    metadata, field_names, num_frames
                )
                
                if not is_valid:
                    if self.auto_clear_invalid:
                        print(f"Cache invalid for sim_{sim_index:06d}: {', '.join(reasons)}. Removing...")
                        cache_path.unlink()
                    else:
                        print(f"Cache invalid for sim_{sim_index:06d}: {', '.join(reasons)}")
                    return False
            
            return True
            
        except Exception as e:
            # If we can't load/validate, treat as not cached
            print(f"Error validating cache for sim_{sim_index:06d}: {e}")
            if self.auto_clear_invalid:
                print(f"Removing corrupted cache...")
                try:
                    cache_path.unlink()
                except:
                    pass
            return False
    
    def load_and_cache_simulation(
        self, 
        sim_index: int, 
        field_names: List[str],
        num_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load a simulation from Scene files, convert to tensors, and cache.
        
        This is the core method that performs the expensive conversion once.
        It loads all specified fields from a Scene, converts them to tensors,
        extracts metadata, and saves everything for future fast loading.
        
        Args:
            sim_index: Index of the simulation to load
            field_names: List of field names to load (e.g., ['velocity'])
            num_frames: Optional limit on number of frames to load.
                       If None, loads all available frames.
        
        Returns:
            Dictionary with 'tensor_data' and 'metadata' keys
            
        Raises:
            FileNotFoundError: If the Scene directory doesn't exist
        """
        # Construct path to Scene directory
        scene_name = f"sim_{sim_index:06d}"
        scene_path = self.raw_data_dir / scene_name
        
        # Handle non-zero-padded names as fallback
        if not scene_path.exists():
            scene_path_alt = self.raw_data_dir / f"sim_{sim_index}"
            if scene_path_alt.exists():
                scene_path = scene_path_alt
            else:
                raise FileNotFoundError(
                    f"Scene not found at {scene_path} or {scene_path_alt}"
                )
        
        scene = Scene.at(str(scene_path))
        
        # Load scene metadata from description.json
        description_path = scene_path / "description.json"
        with open(description_path, 'r') as f:
            scene_metadata = json.load(f)
        
        # Determine frames to load
        available_frames = scene.frames
        if num_frames is not None:
            frames_to_load = available_frames[:num_frames]
        else:
            frames_to_load = available_frames
        
        # Dictionary to store tensor data for all fields
        tensor_data = {}
        field_metadata = {}
        
        # Load each field
        for field_name in field_names:
            if field_name not in scene.fieldnames:
                continue
            
            # Load all frames for this field
            field_frames = []
            original_field_type = None  # Track if original was staggered
            
            for frame_idx in frames_to_load:
                field_obj = scene.read_field(
                    field_name,
                    frame=frame_idx,
                    convert_to_backend=True  # Converts to torch backend
                )
                # Remember if the original field was staggered
                if original_field_type is None:
                    original_field_type = 'staggered' if field_obj.is_staggered else 'centered'
                
                # Convert staggered grids to centered grids (like UNet does)
                if field_obj.is_staggered:
                    field_obj = field_obj.at_centers()
                field_frames.append(field_obj)
            
            # Stack along time dimension
            stacked_field = stack(field_frames, batch('time'))
            
            # Extract the underlying native torch tensor
            # PhiFlow's native tensor layout: [x, y, vector_comps, time] for vectors
            # or [x, y, time] for scalars
            # We want: [time, channels, x, y] where channels = vector components (or 1 for scalars)
            tensor = stacked_field.values._native
            
            # Determine if this is a vector or scalar field from the stacked_field shape
            # This is more reliable than checking native tensor dimensions (which can be squeezed)
            is_vector = stacked_field.shape.channel.rank > 0
            num_time_frames = len(frames_to_load)
            
            # Permute dimensions to get [time, channels, x, y]
            if is_vector:
                # Vector field
                if len(tensor.shape) == 4:  # [x, y, vector, time] - normal case
                    tensor = tensor.permute(3, 2, 0, 1)  # -> [time, vector, x, y]
                elif len(tensor.shape) == 3:  # [x, y, vector] - single frame, time dimension squeezed
                    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # -> [1, vector, x, y]
            else:
                # Scalar field
                if len(tensor.shape) == 3:  # [x, y, time] - normal case
                    tensor = tensor.permute(2, 0, 1).unsqueeze(1)  # -> [time, 1, x, y]
                elif len(tensor.shape) == 2:  # [x, y] - single frame, time dimension squeezed
                    tensor = tensor.unsqueeze(0).unsqueeze(0)  # -> [1, 1, x, y]
            
            # Ensure tensor is on CPU for caching (for DataLoader pin_memory compatibility)
            tensor = tensor.cpu()
            
            tensor_data[field_name] = tensor
            
            # Extract metadata needed to reconstruct the Field
            # We need: shape, domain, resolution, extrapolation, boundary, field_type
            first_field = field_frames[0]
            
            # Extract actual bounds values (not just string representation)
            bounds_lower = tuple([float(first_field.bounds.lower[i]) for i in range(len(first_field.bounds.lower))])
            bounds_upper = tuple([float(first_field.bounds.upper[i]) for i in range(len(first_field.bounds.upper))])
            
            field_metadata[field_name] = {
                'shape': str(first_field.shape),  # Full shape info
                'spatial_dims': list(first_field.shape.spatial.names),
                'channel_dims': list(first_field.shape.channel.names) if first_field.shape.channel else [],
                'extrapolation': str(first_field.extrapolation),
                'bounds': str(first_field.bounds),  # Keep for reference
                'bounds_lower': bounds_lower,  # Actual lower bounds values
                'bounds_upper': bounds_upper,  # Actual upper bounds values
                'field_type': original_field_type  # Store original field type (before conversion to centered)
            }
        
        # Prepare data structure to cache with enhanced metadata
        cache_data = {
            'tensor_data': tensor_data,
            'metadata': {
                # Version and timestamp information
                'version': get_cache_version(),
                'created_at': datetime.now().isoformat(),
                'phiflow_version': get_phiflow_version(),
                
                # Original metadata (preserved for backward compatibility)
                'scene_metadata': scene_metadata,
                'field_metadata': field_metadata,
                'num_frames': len(frames_to_load),
                'frame_indices': frames_to_load,
                
                # NEW: Generation parameters for validation
                'generation_params': {
                    'pde_name': self.config.get('model', {}).get('physical', {}).get('name', 'unknown'),
                    'pde_params': self.config.get('model', {}).get('physical', {}).get('pde_params', {}),
                    'domain': self.config.get('model', {}).get('physical', {}).get('domain', {}),
                    'resolution': self.config.get('model', {}).get('physical', {}).get('resolution', {}),
                    'dt': self.config.get('model', {}).get('physical', {}).get('dt', 0.0),
                },
                
                # NEW: Data configuration
                'data_config': {
                    'fields': field_names,
                    'fields_scheme': self.config.get('data', {}).get('fields_scheme', 'unknown'),
                    'dset_name': self.config.get('data', {}).get('dset_name', 'unknown'),
                },
                
                # NEW: Checksums for fast validation
                'checksums': {
                    'pde_params_hash': compute_hash(
                        self.config.get('model', {}).get('physical', {}).get('pde_params', {})
                    ),
                    'resolution_hash': compute_hash(
                        self.config.get('model', {}).get('physical', {}).get('resolution', {})
                    ),
                    'domain_hash': compute_hash(
                        self.config.get('model', {}).get('physical', {}).get('domain', {})
                    ),
                }
            }
        }
        
        # Save to cache
        cache_path = self.get_cached_path(sim_index)
        torch.save(cache_data, cache_path)
        
        return cache_data
    
    def load_from_cache(self, sim_index: int) -> Dict[str, Any]:
        """
        Load cached tensor data for a simulation.
        
        Args:
            sim_index: Index of the simulation
            
        Returns:
            Dictionary with 'tensor_data' and 'metadata' keys
            
        Raises:
            FileNotFoundError: If cached data doesn't exist
        """
        cache_path = self.get_cached_path(sim_index)
        
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cached data not found at {cache_path}. "
                f"Call load_and_cache_simulation() first."
            )
        
        # Use weights_only=False because we're loading our own trusted data
        # and the metadata dict contains non-tensor types
        return torch.load(cache_path, weights_only=False)
    
    def get_or_load_simulation(
        self,
        sim_index: int,
        field_names: List[str],
        num_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get simulation data, loading from cache if available, otherwise loading and caching.
        
        This method validates that cached data matches the requested parameters
        (field names and num_frames) before using it.
        
        Args:
            sim_index: Index of the simulation
            field_names: List of field names to load
            num_frames: Optional limit on number of frames
            
        Returns:
            Dictionary with 'tensor_data' and 'metadata' keys
        """
        # Check if cache exists AND matches requested parameters
        if self.is_cached(sim_index, field_names, num_frames):
            return self.load_from_cache(sim_index)
        else:
            return self.load_and_cache_simulation(
                sim_index, 
                field_names, 
                num_frames
            )
