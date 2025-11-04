"""
Augmentation Configuration Validation and Utilities

This module provides validation and helper functions for augmentation configuration.
Ensures that augmentation settings are valid and compatible with the training setup.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AugmentationConfig:
    """Validates and manages augmentation configuration."""
    
    # Note: Only 'cached' strategy is supported (on_the_fly removed)
    VALID_FORMATS = ['dict', 'tuple']
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize augmentation configuration.
        
        Args:
            config: Configuration dictionary with augmentation settings
        
        Note: Strategy is hardcoded to 'cached' only.
        """
        self.config = config
        self.enabled = config.get('enabled', False)
        
        if self.enabled:
            self._validate()
    
    def _validate(self):
        """Validate augmentation configuration."""
        # Validate alpha
        alpha = self.config.get('alpha', 0.1)
        if not isinstance(alpha, (int, float)):
            raise ValueError(f"alpha must be numeric, got {type(alpha)}")
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
        
        # Note: Strategy is always 'cached' (hardcoded)
        # Validate cache settings
        self._validate_cache_settings()
        
        # Validate device
        device = self.config.get('device', 'cuda')
        if not isinstance(device, str):
            raise ValueError(f"device must be string, got {type(device)}")
    
    def _validate_cache_settings(self):
        """Validate cache-specific settings."""
        cache_config = self.config.get('cache', {})
        
        # Validate experiment_name
        experiment_name = cache_config.get('experiment_name')
        if not experiment_name:
            raise ValueError("cache.experiment_name must be specified")
        
        # Validate format
        format_type = cache_config.get('format', 'dict')
        if format_type not in self.VALID_FORMATS:
            raise ValueError(
                f"cache.format must be one of {self.VALID_FORMATS}, got '{format_type}'"
            )
        
        # Validate max_memory_samples
        max_samples = cache_config.get('max_memory_samples', 1000)
        if not isinstance(max_samples, int) or max_samples <= 0:
            raise ValueError(
                f"cache.max_memory_samples must be positive integer, got {max_samples}"
            )
        
        # Validate reuse_existing
        reuse = cache_config.get('reuse_existing', True)
        if not isinstance(reuse, bool):
            raise ValueError(
                f"cache.reuse_existing must be boolean, got {type(reuse)}"
            )
    
    
    def get_alpha(self) -> float:
        """Get alpha value."""
        return self.config.get('alpha', 0.1)
    
    def get_strategy(self) -> str:
        """Get augmentation strategy (always 'cached')."""
        return 'cached'  # Hardcoded
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration."""
        return self.config.get('cache', {})
    
    def get_device(self) -> str:
        """Get device for prediction generation."""
        return self.config.get('device', 'cuda')
    
    def should_regenerate(self, epoch: int) -> bool:
        """
        Check if predictions should be regenerated at this epoch.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            False (cached strategy never regenerates - hardcoded)
        """
        return False  # Cached strategy doesn't regenerate (hardcoded)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'enabled': self.enabled,
            'alpha': self.get_alpha(),
            'strategy': 'cached',  # Hardcoded
            'cache': self.get_cache_config(),
            'device': self.get_device(),
        }
    
    def __repr__(self) -> str:
        if not self.enabled:
            return "AugmentationConfig(enabled=False)"
        
        return (
            f"AugmentationConfig("
            f"enabled=True, "
            f"alpha={self.get_alpha()}, "
            f"strategy='cached', "  # Hardcoded
            f"device='{self.get_device()}')"
        )


def validate_cache_config(cache_config: Dict[str, Any]) -> List[str]:
    """
    Validate cache configuration from main config.yaml.
    
    Args:
        cache_config: Cache configuration dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Validate root
    root = cache_config.get('root')
    if not root:
        errors.append("cache.root must be specified")
    elif not isinstance(root, str):
        errors.append(f"cache.root must be string, got {type(root)}")
    
    # Validate auto_create
    auto_create = cache_config.get('auto_create', True)
    if not isinstance(auto_create, bool):
        errors.append(f"cache.auto_create must be boolean, got {type(auto_create)}")
    
    # Validate validation settings
    validation_config = cache_config.get('validation', {})
    if not isinstance(validation_config, dict):
        errors.append(f"cache.validation must be dict, got {type(validation_config)}")
    else:
        check_on_load = validation_config.get('check_on_load', True)
        if not isinstance(check_on_load, bool):
            errors.append(
                f"cache.validation.check_on_load must be boolean, got {type(check_on_load)}"
            )
        
        expected_count = validation_config.get('expected_count')
        if expected_count is not None and not isinstance(expected_count, int):
            errors.append(
                f"cache.validation.expected_count must be int or null, got {type(expected_count)}"
            )
    
    # Validate cleanup settings
    cleanup_config = cache_config.get('cleanup', {})
    if not isinstance(cleanup_config, dict):
        errors.append(f"cache.cleanup must be dict, got {type(cleanup_config)}")
    else:
        for key in ['on_start', 'on_error']:
            value = cleanup_config.get(key, False)
            if not isinstance(value, bool):
                errors.append(
                    f"cache.cleanup.{key} must be boolean, got {type(value)}"
                )
    
    return errors


def create_cache_path(cache_root: str, experiment_name: str) -> Path:
    """
    Create cache path for augmented data.
    
    Args:
        cache_root: Root cache directory
        experiment_name: Experiment/dataset name
        
    Returns:
        Path to cache directory
    """
    cache_path = Path(cache_root) / 'hybrid_generated' / experiment_name
    return cache_path


def get_augmentation_summary(config: AugmentationConfig) -> str:
    """
    Get human-readable summary of augmentation configuration.
    
    Args:
        config: Augmentation configuration
        
    Returns:
        Formatted summary string
    """
    if not config.enabled:
        return "Augmentation: Disabled"
    
    lines = [
        "Augmentation Configuration:",
        f"  Enabled: True",
        f"  Alpha: {config.get_alpha()}",
        f"  Strategy: {config.get_strategy()}",
        f"  Device: {config.get_device()}",
    ]
    
    if config.get_strategy() == 'cached':
        cache_config = config.get_cache_config()
        lines.extend([
            "  Cache Settings:",
            f"    Experiment: {cache_config.get('experiment_name')}",
            f"    Format: {cache_config.get('format')}",
            f"    Max Memory Samples: {cache_config.get('max_memory_samples')}",
            f"    Reuse Existing: {cache_config.get('reuse_existing')}",
        ])
    else:
        otf_config = config.get_on_the_fly_config()
        lines.extend([
            "  On-the-fly Settings:",
            f"    Generate Every: {otf_config.get('generate_every')} epochs",
            f"    Batch Size: {otf_config.get('batch_size')}",
            f"    Rollout Steps: {otf_config.get('rollout_steps')}",
        ])
    
    return "\n".join(lines)
