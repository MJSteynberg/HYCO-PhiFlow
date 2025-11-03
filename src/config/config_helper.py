"""
Configuration Helper

Centralizes all config extraction logic to reduce coupling between components
and configuration structure. Makes refactoring easier by providing a single
point of access to configuration values.

This helper class knows about the structure of the Hydra configuration and
provides clean, typed access to all configuration values needed by the data
loading system.
"""

from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path


class ConfigHelper:
    """
    Helper class for extracting configuration values.
    
    This reduces coupling between components and config structure,
    making refactoring easier. All config path knowledge is centralized here.
    
    The config structure expected:
    ```yaml
    data:
      dset_name: "dataset_name"
      data_dir: "data"
      fields: ["field1", "field2"]
      validate_cache: true
      auto_clear_invalid: false
    
    model:
      synthetic:
        input_specs:
          field1: {...}
          field2: {...}
        output_specs:
          field1: {...}
      physical:
        ...
    
    trainer_params:
      train_sim: [0, 1, 2, ...]
      batch_size: 16
      num_predict_steps: 10
      augmentation:
        enabled: true
        alpha: 0.1
        strategy: "cached"
        cache:
          experiment_name: "..."
    
    cache:
      root: "data/cache"
    ```
    
    Args:
        config: Full configuration dictionary from Hydra
        
    Example:
        >>> cfg = ConfigHelper(config)
        >>> field_names = cfg.get_field_names()
        >>> dynamic, static = cfg.get_field_types()
        >>> batch_size = cfg.get_batch_size()
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ConfigHelper with full configuration.
        
        Args:
            config: Complete configuration dictionary (typically from Hydra)
        """
        self.config = config
        
        # Handle both nested (data: {fields: ...}) and flat (fields: ...) structures
        # This allows ConfigHelper to work with both test configs and real Hydra configs
        if "data" in config and isinstance(config["data"], dict):
            # Nested structure: {data: {fields: [...], ...}}
            self.data_config = config["data"]
        else:
            # Flat structure (Hydra with defaults): {fields: [...], dset_name: ..., ...}
            # Create a virtual data_config from top-level keys
            self.data_config = {
                'fields': config.get('fields', []),
                'dset_name': config.get('dset_name', 'unknown'),
                'data_dir': config.get('data_dir', 'data/'),
                'cache_dir': config.get('cache_dir', 'data/cache'),
                'validate_cache': config.get('validate_cache', True),
                'auto_clear_invalid': config.get('auto_clear_invalid', False),
            }
        
        self.model_config = config.get("model", {})
        self.run_params = config.get("run_params", {})
        self.trainer_config = config.get("trainer_params", {})
        self.augmentation_config = self.trainer_config.get("augmentation", {})
        self.cache_config = config.get("cache", {})
    
    # ==================== Data Configuration ====================
    
    def get_field_names(self) -> List[str]:
        """
        Get list of field names to load.
        
        Returns:
            List of field names (e.g., ['velocity', 'density', 'inflow'])
        """
        return self.data_config.get("fields", [])
    
    def get_dataset_name(self) -> str:
        """
        Get dataset name.
        
        Returns:
            Dataset name (e.g., 'burgers_128', 'smoke_128')
        """
        return self.data_config.get("dset_name", "unknown")
    
    def get_raw_data_dir(self) -> Path:
        """
        Get raw data directory path.
        
        Returns:
            Path to raw data (e.g., Path('data/burgers_128'))
        """
        data_dir = self.data_config.get("data_dir", "data")
        dset_name = self.get_dataset_name()
        return Path(data_dir) / dset_name
    
    def get_cache_dir(self) -> Path:
        """
        Get cache directory path.
        
        Returns:
            Path to cache directory (e.g., Path('data/cache'))
        """
        data_dir = self.data_config.get("data_dir", "data")
        return Path(data_dir) / "cache"
    
    def should_validate_cache(self) -> bool:
        """
        Check if cache validation is enabled.
        
        Returns:
            True if cache should be validated on load
        """
        return self.data_config.get("validate_cache", True)
    
    def should_auto_clear_invalid(self) -> bool:
        """
        Check if auto-clear invalid cache is enabled.
        
        Returns:
            True if invalid cache should be automatically cleared
        """
        return self.data_config.get("auto_clear_invalid", False)
    
    # ==================== Field Specifications ====================
    
    def get_field_types(self) -> Tuple[List[str], List[str]]:
        """
        Get dynamic and static field names.
        
        Dynamic fields are predicted by the model (outputs).
        Static fields are input-only (not predicted).
        
        For synthetic models:
        - Dynamic: Fields in output_specs
        - Static: Fields in input_specs but not in output_specs
        
        For physical models:
        - Dynamic: All fields (physical models predict all fields)
        - Static: Empty list
        
        Returns:
            Tuple of (dynamic_fields, static_fields)
            
        Example:
            >>> dynamic, static = cfg.get_field_types()
            >>> # For synthetic: dynamic=['velocity', 'density'], static=['inflow']
            >>> # For physical: dynamic=['velocity', 'density'], static=[]
        """
        model_type = self.run_params.get("model_type", "synthetic")
        
        if model_type == "synthetic":
            # Extract from model specs
            synthetic_config = self.model_config.get("synthetic", {})
            input_specs = synthetic_config.get("input_specs", {})
            output_specs = synthetic_config.get("output_specs", {})
            
            # Dynamic = output fields
            dynamic_fields = list(output_specs.keys())
            
            # Static = input-only fields (in input but not in output)
            static_fields = [
                field for field in input_specs.keys() 
                if field not in output_specs
            ]
            
            return dynamic_fields, static_fields
        
        else:
            # Physical model: all fields are dynamic
            field_names = self.get_field_names()
            return field_names, []
    
    def get_input_specs(self) -> Dict[str, Any]:
        """
        Get input field specifications for synthetic model.
        
        Returns:
            Dictionary mapping field names to input specs
        """
        return self.model_config.get("synthetic", {}).get("input_specs", {})
    
    def get_output_specs(self) -> Dict[str, Any]:
        """
        Get output field specifications for synthetic model.
        
        Returns:
            Dictionary mapping field names to output specs
        """
        return self.model_config.get("synthetic", {}).get("output_specs", {})
    
    # ==================== Training Configuration ====================
    
    def get_train_sim_indices(self) -> List[int]:
        """
        Get training simulation indices.
        
        Returns:
            List of simulation indices for training
        """
        return self.trainer_config.get("train_sim", [])
    
    def get_val_sim_indices(self) -> List[int]:
        """
        Get validation simulation indices.
        
        Returns:
            List of simulation indices for validation
        """
        return self.trainer_config.get("val_sim", [])
    
    def get_batch_size(self) -> int:
        """
        Get batch size.
        
        Returns:
            Batch size for training
        """
        return self.trainer_config.get("batch_size", 16)
    
    def get_num_predict_steps(self) -> int:
        """
        Get number of prediction steps.
        
        Returns:
            Number of autoregressive prediction steps
        """
        return self.trainer_config.get("num_predict_steps", 10)
    
    def get_num_frames(self, use_sliding_window: bool) -> Optional[int]:
        """
        Get number of frames to load.
        
        Args:
            use_sliding_window: If True, return None (load all frames)
        
        Returns:
            Number of frames to load, or None to load all available frames
        """
        if use_sliding_window:
            return None  # Load all available frames for sliding window
        else:
            # Load just enough for one rollout
            return self.get_num_predict_steps() + 1
    
    def should_use_sliding_window(self) -> bool:
        """
        Check if sliding window should be used.
        
        Returns:
            True if sliding window is enabled in config
        """
        return self.trainer_config.get("use_sliding_window", True)
    
    # ==================== Augmentation Configuration ====================
    
    def is_augmentation_enabled(self) -> bool:
        """
        Check if augmentation is enabled.
        
        Returns:
            True if augmentation is enabled in config
        """
        return self.augmentation_config.get("enabled", False)
    
    def get_augmentation_alpha(self) -> float:
        """
        Get augmentation alpha parameter.
        
        Alpha determines the proportion of augmented samples:
        - alpha=0.1 means 10% augmented samples
        - num_augmented = int(num_real * alpha)
        
        Returns:
            Augmentation proportion (typically 0.1 to 0.5)
        """
        return self.augmentation_config.get("alpha", 0.1)
    
    def get_augmentation_strategy(self) -> str:
        """
        Get augmentation strategy.
        
        Returns:
            Strategy string: 'cached', 'on_the_fly', or 'memory'
        """
        return self.augmentation_config.get("strategy", "cached")
    
    def get_augmentation_mode(self) -> str:
        """
        Get augmentation mode (normalized version of strategy).
        
        Maps strategy names to standardized mode names:
        - 'cached' → 'cache'
        - 'on_the_fly' → 'on_the_fly'
        - 'memory' → 'memory'
        
        Returns:
            Mode string: 'cache', 'on_the_fly', or 'memory'
        """
        strategy = self.get_augmentation_strategy()
        
        # Normalize strategy names
        if strategy == "cached":
            return "cache"
        elif strategy in ["on_the_fly", "on-the-fly"]:
            return "on_the_fly"
        else:
            return strategy
    
    def get_augmentation_config(self) -> Optional[Dict[str, Any]]:
        """
        Get complete augmentation configuration for dataset.
        
        Returns dictionary with all parameters needed by AbstractDataset:
        - mode: 'memory', 'cache', or 'on_the_fly'
        - alpha: Proportion of augmented samples
        - cache_dir: Path to cache directory (for cache mode)
        - data: Pre-loaded data (for memory mode, if available)
        
        Returns:
            Augmentation config dict, or None if augmentation disabled
            
        Example:
            >>> aug_config = cfg.get_augmentation_config()
            >>> # {'mode': 'cache', 'alpha': 0.1, 'cache_dir': 'data/cache/...'}
        """
        if not self.is_augmentation_enabled():
            return None
        
        mode = self.get_augmentation_mode()
        alpha = self.get_augmentation_alpha()
        
        config = {
            'mode': mode,
            'alpha': alpha,
        }
        
        if mode == 'cache':
            # Build cache directory path
            cache_root = self.cache_config.get("root", "data/cache")
            cache_subconfig = self.augmentation_config.get("cache", {})
            experiment_name = cache_subconfig.get(
                "experiment_name", 
                self.get_dataset_name()
            )
            
            # Full path: <cache_root>/hybrid_generated/<experiment_name>
            cache_dir = Path(cache_root) / "hybrid_generated" / experiment_name
            config['cache_dir'] = str(cache_dir)
        
        return config
    
    def get_augmentation_cache_dir(self) -> Optional[Path]:
        """
        Get augmentation cache directory path.
        
        Returns:
            Path to augmentation cache, or None if not using cache mode
        """
        if not self.is_augmentation_enabled():
            return None
        
        if self.get_augmentation_mode() != 'cache':
            return None
        
        aug_config = self.get_augmentation_config()
        return Path(aug_config['cache_dir']) if aug_config else None
    
    # ==================== Model Configuration ====================
    
    def get_model_type(self) -> str:
        """
        Get model type.
        
        Returns:
            Model type: 'synthetic', 'physical', or 'hybrid'
        """
        return self.run_params.get("model_type", "synthetic")
    
    def is_hybrid_training(self) -> bool:
        """
        Check if using hybrid training mode.
        
        Returns:
            True if model_type is 'hybrid'
        """
        return self.get_model_type() == "hybrid"
    
    # ==================== Project Paths ====================
    
    def get_project_root(self) -> Path:
        """
        Get project root directory.
        
        Returns:
            Path to project root
        """
        return Path(self.config.get("project_root", "."))
    
    def get_absolute_raw_data_dir(self) -> Path:
        """
        Get absolute path to raw data directory.
        
        Returns:
            Absolute path: <project_root>/<data_dir>/<dset_name>
        """
        project_root = self.get_project_root()
        return project_root / self.get_raw_data_dir()
    
    def get_absolute_cache_dir(self) -> Path:
        """
        Get absolute path to cache directory.
        
        Returns:
            Absolute path: <project_root>/<data_dir>/cache
        """
        project_root = self.get_project_root()
        return project_root / self.get_cache_dir()
    
    def get_absolute_augmentation_cache_dir(self) -> Optional[Path]:
        """
        Get absolute path to augmentation cache directory.
        
        Returns:
            Absolute path or None if not using cache mode
        """
        cache_dir = self.get_augmentation_cache_dir()
        if cache_dir is None:
            return None
        
        project_root = self.get_project_root()
        return project_root / cache_dir
    
    # ==================== Utility Methods ====================
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all configuration values.
        
        Useful for logging and debugging.
        
        Returns:
            Dictionary with all extracted config values
        """
        dynamic_fields, static_fields = self.get_field_types()
        
        return {
            # Data
            'dataset_name': self.get_dataset_name(),
            'field_names': self.get_field_names(),
            'dynamic_fields': dynamic_fields,
            'static_fields': static_fields,
            'raw_data_dir': str(self.get_raw_data_dir()),
            'cache_dir': str(self.get_cache_dir()),
            
            # Training
            'model_type': self.get_model_type(),
            'train_sims': len(self.get_train_sim_indices()),
            'batch_size': self.get_batch_size(),
            'num_predict_steps': self.get_num_predict_steps(),
            'use_sliding_window': self.should_use_sliding_window(),
            
            # Augmentation
            'augmentation_enabled': self.is_augmentation_enabled(),
            'augmentation_alpha': self.get_augmentation_alpha() if self.is_augmentation_enabled() else None,
            'augmentation_mode': self.get_augmentation_mode() if self.is_augmentation_enabled() else None,
            'augmentation_cache_dir': str(self.get_augmentation_cache_dir()) if self.get_augmentation_cache_dir() else None,
        }
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        
        # Check required fields
        if not self.get_field_names():
            issues.append("No fields specified in data.fields")
        
        if not self.get_train_sim_indices():
            issues.append("No training simulations specified in trainer_params.train_sim")
        
        if self.get_batch_size() <= 0:
            issues.append(f"Invalid batch_size: {self.get_batch_size()}")
        
        if self.get_num_predict_steps() <= 0:
            issues.append(f"Invalid num_predict_steps: {self.get_num_predict_steps()}")
        
        # Check augmentation config
        if self.is_augmentation_enabled():
            alpha = self.get_augmentation_alpha()
            if alpha < 0 or alpha > 1:
                issues.append(f"Invalid augmentation alpha: {alpha} (must be 0-1)")
            
            mode = self.get_augmentation_mode()
            if mode not in ['cache', 'memory', 'on_the_fly']:
                issues.append(f"Invalid augmentation mode: {mode}")
            
            if mode == 'cache':
                cache_dir = self.get_augmentation_cache_dir()
                if cache_dir is None:
                    issues.append("Augmentation cache mode enabled but cache_dir not configured")
        
        return issues
