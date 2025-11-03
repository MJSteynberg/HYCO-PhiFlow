"""
Cache Validation Script

Validates augmented prediction cache integrity and provides statistics.

Usage:
    # Validate cache for specific experiment
    python scripts/validate_cache.py --config-name=burgers_experiment
    
    # Validate specific cache directory
    python scripts/validate_cache.py --cache-dir=data/cache/hybrid_generated/burgers_128
"""

import sys
import argparse
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation import CacheManager
from src.config import AugmentationConfig, create_cache_path
from src.utils.logger import setup_logger

logger = setup_logger("cache_validation", level=20)  # INFO level


def validate_cache_directory(cache_dir: Path) -> dict:
    """
    Validate a specific cache directory.
    
    Args:
        cache_dir: Path to cache directory
        
    Returns:
        Dictionary with validation results
    """
    cache_dir = Path(cache_dir)
    
    if not cache_dir.exists():
        return {
            'valid': False,
            'error': 'Directory does not exist',
            'path': str(cache_dir)
        }
    
    # Extract experiment name and cache root
    experiment_name = cache_dir.name
    cache_root = str(cache_dir.parent.parent)
    
    # Create CacheManager
    manager = CacheManager(cache_root, experiment_name, auto_create=False)
    
    # Get info
    info = manager.get_info()
    
    # Validate
    validation = manager.validate_cache()
    
    # Get disk usage
    usage = manager.get_disk_usage()
    
    # Load metadata
    try:
        metadata = manager.load_metadata()
    except Exception:
        metadata = {}
    
    return {
        'valid': validation['valid'],
        'path': str(cache_dir),
        'info': info,
        'validation': validation,
        'usage': usage,
        'metadata': metadata,
    }


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Validate cache from config."""
    
    # Parse config
    config = OmegaConf.to_container(cfg, resolve=True)
    config["project_root"] = str(PROJECT_ROOT)
    
    logger.info("=" * 80)
    logger.info("HYCO-PhiFlow Cache Validation")
    logger.info("=" * 80)
    
    # Check if augmentation is enabled
    trainer_config = config.get("trainer_params", {})
    augmentation_config = trainer_config.get("augmentation", {})
    
    if not augmentation_config.get("enabled", False):
        logger.error("Augmentation is not enabled in configuration!")
        logger.error("Set trainer_params.augmentation.enabled=true")
        sys.exit(1)
    
    # Get cache path from config
    aug_config = AugmentationConfig(augmentation_config)
    project_root = Path(config.get("project_root", "."))
    cache_root = config.get("cache", {}).get("root", "data/cache")
    cache_root = project_root / cache_root
    
    data_config = config["data"]
    experiment_name = aug_config.get_cache_config().get("experiment_name", data_config["dset_name"])
    
    cache_path = create_cache_path(str(cache_root), experiment_name)
    
    logger.info(f"Validating cache: {cache_path}")
    logger.info("")
    
    # Validate
    results = validate_cache_directory(cache_path)
    
    if not results['valid']:
        logger.error(f"❌ Validation FAILED: {results.get('error', 'Unknown error')}")
        
        if 'validation' in results:
            validation = results['validation']
            if not validation.get('count_match', True):
                logger.error(f"Count mismatch in metadata")
            if not validation.get('all_loadable', True):
                logger.error(f"Some samples failed to load")
                for error in validation.get('errors', []):
                    logger.error(f"  - {error}")
        
        sys.exit(1)
    
    # Print results
    logger.info("✅ Cache Validation PASSED")
    logger.info("")
    
    info = results['info']
    logger.info("Cache Information:")
    logger.info(f"  Location: {results['path']}")
    logger.info(f"  Exists: {info['exists']}")
    logger.info(f"  Empty: {info['is_empty']}")
    logger.info(f"  Sample Count: {info['num_samples']}")
    logger.info("")
    
    usage = results['usage']
    logger.info("Disk Usage:")
    logger.info(f"  Size: {usage['size_mb']:.2f} MB")
    logger.info(f"  Files: {usage['num_files']}")
    logger.info("")
    
    metadata = results['metadata']
    if metadata:
        logger.info("Metadata:")
        for key, value in metadata.items():
            logger.info(f"  {key}: {value}")
        logger.info("")
    
    validation = results['validation']
    logger.info("Validation Details:")
    logger.info(f"  Count Match: {'✅' if validation['count_match'] else '❌'}")
    logger.info(f"  All Loadable: {'✅' if validation['all_loadable'] else '❌'}")
    
    if validation.get('errors'):
        logger.info("  Errors:")
        for error in validation['errors']:
            logger.info(f"    - {error}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Cache is ready for training!")
    logger.info("=" * 80)


def standalone_validate(cache_dir: str):
    """Validate cache directory without config (for CLI use)."""
    logger.info("=" * 80)
    logger.info("HYCO-PhiFlow Cache Validation (Standalone)")
    logger.info("=" * 80)
    
    results = validate_cache_directory(Path(cache_dir))
    
    if not results['valid']:
        logger.error(f"❌ Validation FAILED: {results.get('error', 'Unknown error')}")
        sys.exit(1)
    
    logger.info("✅ Cache Validation PASSED")
    logger.info("")
    logger.info(f"Path: {results['path']}")
    logger.info(f"Samples: {results['info']['num_samples']}")
    logger.info(f"Size: {results['usage']['size_mb']:.2f} MB")
    logger.info("")
    logger.info("=" * 80)


if __name__ == "__main__":
    # Check if called with --cache-dir argument
    if len(sys.argv) > 1 and sys.argv[1] == "--cache-dir":
        parser = argparse.ArgumentParser()
        parser.add_argument("--cache-dir", required=True, help="Path to cache directory")
        args = parser.parse_args()
        standalone_validate(args.cache_dir)
    else:
        # Use Hydra config
        main()
