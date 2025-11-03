"""
Utilities for cache warmup and management during training.

This module provides utilities for:
- Pre-populating cache before training (warmup)
- Epoch-based cache regeneration
- Cache statistics and monitoring
"""

import torch
from typing import Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm

from src.data.augmentation.cache_manager import CacheManager
from src.data.augmentation.generation_utils import generate_and_cache_predictions
from src.factories.model_factory import ModelFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)


def warmup_cache(
    config: Dict[str, Any],
    model_checkpoint: Optional[str] = None,
    force_regenerate: bool = False,
) -> bool:
    """
    Pre-populate cache before training starts.
    
    This function:
    1. Checks if augmentation is enabled and uses cached strategy
    2. Validates existing cache or creates new one
    3. Loads model from checkpoint
    4. Generates predictions and populates cache
    
    Args:
        config: Full configuration dictionary
        model_checkpoint: Path to model checkpoint (if available)
        force_regenerate: Force regeneration even if cache exists
        
    Returns:
        True if warmup successful or not needed, False if failed
        
    Example:
        >>> config = load_config("burgers_experiment")
        >>> success = warmup_cache(
        ...     config=config,
        ...     model_checkpoint="results/models/best_model.pt",
        ... )
        >>> if success:
        ...     print("Cache ready for training!")
    """
    logger.info("="*60)
    logger.info("CACHE WARMUP STARTING")
    logger.info("="*60)
    
    # Check if augmentation enabled
    trainer_config = config.get("trainer_params", {})
    aug_params = trainer_config.get("augmentation", {})
    
    if not aug_params.get("enabled", False):
        logger.info("Augmentation disabled - skipping cache warmup")
        return True
    
    strategy = aug_params.get("strategy", "cached")
    if strategy != "cached":
        logger.info(f"Strategy '{strategy}' doesn't use persistent cache - skipping warmup")
        return True
    
    # Get cache config
    cache_config = aug_params.get("cache", {})
    experiment_name = cache_config.get("experiment_name")
    
    if not experiment_name:
        experiment_name = config.get("run_params", {}).get("experiment_name", "default")
        experiment_name = f"{experiment_name}_augmented"
        logger.info(f"Using default experiment_name: {experiment_name}")
    
    cache_root = config.get("cache", {}).get("root", "data/cache")
    project_root = Path(config.get("project_root", "."))
    cache_root = project_root / cache_root
    
    # Create cache manager
    cache_manager = CacheManager(
        cache_root=str(cache_root),
        experiment_name=experiment_name,
        auto_create=True,
    )
    
    # Check if cache already exists
    if not force_regenerate and cache_manager.exists() and not cache_manager.is_empty():
        cache_count = cache_manager.count_samples()
        logger.info(f"Cache already exists with {cache_count} samples")
        
        # Validate cache
        if cache_manager.validate_cache(verbose=True):
            logger.info("✅ Cache validation passed - using existing cache")
            logger.info("="*60)
            return True
        else:
            logger.warning("⚠️ Cache validation failed - regenerating")
    
    # Need to generate cache
    if model_checkpoint is None:
        logger.warning("No model checkpoint provided - cannot generate predictions")
        logger.warning("Cache warmup skipped. Training will use standard DataLoader.")
        logger.info("="*60)
        return False
    
    logger.info(f"Loading model from checkpoint: {model_checkpoint}")
    
    # Load model
    model_type = config.get("run_params", {}).get("model_type", "synthetic")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        if model_type == "synthetic":
            model = ModelFactory.create_synthetic_model(config)
            checkpoint = torch.load(model_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()
            logger.info(f"Loaded synthetic model to {device}")
        elif model_type == "physical":
            model = ModelFactory.create_physical_model(config)
            logger.info("Created physical model")
            # Physical models typically don't need checkpoints
        else:
            logger.error(f"Unknown model type: {model_type}")
            logger.info("="*60)
            return False
        
        # Generate and cache predictions
        logger.info("Generating predictions for cache...")
        
        success = generate_and_cache_predictions(
            config=config,
            model=model,
            model_type=model_type,
            cache_manager=cache_manager,
            device=device,
        )
        
        if success:
            cache_count = cache_manager.count_samples()
            disk_usage = cache_manager.get_disk_usage() / (1024**2)  # MB
            logger.info("="*60)
            logger.info(f"✅ CACHE WARMUP COMPLETE!")
            logger.info(f"  Generated samples: {cache_count}")
            logger.info(f"  Disk usage: {disk_usage:.2f} MB")
            logger.info(f"  Cache location: {cache_manager.cache_dir}")
            logger.info("="*60)
            return True
        else:
            logger.error("❌ Cache generation failed")
            logger.info("="*60)
            return False
            
    except Exception as e:
        logger.error(f"❌ Cache warmup failed with error: {e}")
        import traceback
        traceback.print_exc()
        logger.info("="*60)
        return False


def should_regenerate_cache(
    current_epoch: int,
    augmentation_config: Dict[str, Any],
) -> bool:
    """
    Determine if cache should be regenerated at current epoch.
    
    Cache regeneration allows the system to use fresh predictions as the
    model improves during training. Regeneration happens at specified
    epoch intervals.
    
    Args:
        current_epoch: Current training epoch (0-indexed)
        augmentation_config: Augmentation configuration dict
        
    Returns:
        True if cache should be regenerated, False otherwise
        
    Example:
        >>> aug_config = {"enabled": True, "strategy": "cached", 
        ...               "cache": {"regenerate_epochs": 10}}
        >>> should_regenerate_cache(0, aug_config)  # False (not at epoch 0)
        False
        >>> should_regenerate_cache(10, aug_config)  # True (at interval)
        True
        >>> should_regenerate_cache(15, aug_config)  # False (not at interval)
        False
    """
    if not augmentation_config.get("enabled", False):
        return False
    
    strategy = augmentation_config.get("strategy", "cached")
    if strategy != "cached":
        return False
    
    cache_config = augmentation_config.get("cache", {})
    regenerate_epochs = cache_config.get("regenerate_epochs", None)
    
    if regenerate_epochs is None or regenerate_epochs <= 0:
        # No periodic regeneration
        return False
    
    # Regenerate at specified intervals (but not at epoch 0)
    if current_epoch > 0 and current_epoch % regenerate_epochs == 0:
        logger.info(f"Epoch {current_epoch}: Triggering cache regeneration (interval: {regenerate_epochs})")
        return True
    
    return False


def get_cache_statistics(cache_manager: CacheManager) -> Dict[str, Any]:
    """
    Get comprehensive cache statistics.
    
    Args:
        cache_manager: CacheManager instance
        
    Returns:
        Dictionary with cache statistics including:
        - exists: Whether cache directory exists
        - sample_count: Number of cached samples
        - disk_usage_gb: Disk space used (GB)
        - disk_usage_mb: Disk space used (MB)
        - status: Cache status (not_created, empty, ready)
        - cache_dir: Path to cache directory
        - metadata: Cache metadata dict
        
    Example:
        >>> manager = CacheManager("data/cache", "experiment")
        >>> stats = get_cache_statistics(manager)
        >>> print(f"Cache has {stats['sample_count']} samples")
        >>> print(f"Using {stats['disk_usage_mb']:.2f} MB")
    """
    if not cache_manager.exists():
        return {
            "exists": False,
            "sample_count": 0,
            "disk_usage_gb": 0.0,
            "disk_usage_mb": 0.0,
            "cache_dir": str(cache_manager.cache_dir),
            "status": "not_created",
            "metadata": {},
        }
    
    sample_count = cache_manager.count_samples()
    disk_usage = cache_manager.get_disk_usage()
    disk_usage_gb = disk_usage / (1024**3)  # Convert to GB
    disk_usage_mb = disk_usage / (1024**2)  # Convert to MB
    
    # Get metadata
    metadata = cache_manager.load_metadata()
    
    stats = {
        "exists": True,
        "sample_count": sample_count,
        "disk_usage_gb": disk_usage_gb,
        "disk_usage_mb": disk_usage_mb,
        "cache_dir": str(cache_manager.cache_dir),
        "status": "ready" if sample_count > 0 else "empty",
        "metadata": metadata,
    }
    
    return stats


def log_cache_statistics(cache_manager: CacheManager):
    """
    Log cache statistics in human-readable format.
    
    Args:
        cache_manager: CacheManager instance
        
    Example:
        >>> manager = CacheManager("data/cache", "experiment")
        >>> log_cache_statistics(manager)
        # Logs formatted statistics to console
    """
    stats = get_cache_statistics(cache_manager)
    
    logger.info("Cache Statistics:")
    logger.info(f"  Status: {stats['status']}")
    logger.info(f"  Sample count: {stats['sample_count']}")
    logger.info(f"  Disk usage: {stats['disk_usage_mb']:.2f} MB ({stats['disk_usage_gb']:.3f} GB)")
    logger.info(f"  Location: {stats['cache_dir']}")
    
    if stats['metadata']:
        logger.info("  Metadata:")
        for key, value in stats['metadata'].items():
            logger.info(f"    {key}: {value}")
