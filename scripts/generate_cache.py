"""
Cache Generation Script for Augmented Training

This script pre-generates augmented predictions and caches them to disk
for use with the cached augmentation strategy during training.

Usage:
    # Generate cache for burgers experiment
    python scripts/generate_cache.py --config-name=burgers_experiment
    
    # Force regenerate cache (overwrite existing)
    python scripts/generate_cache.py --config-name=burgers_experiment --force
    
    # Generate with specific checkpoint
    python scripts/generate_cache.py --config-name=burgers_experiment model.synthetic.checkpoint=/path/to/model.pth
"""

import sys
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from src.factories.trainer_factory import TrainerFactory
from src.factories.model_factory import ModelFactory
from src.utils.logger import setup_logger

logger = setup_logger("cache_generation", level=logging.INFO)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Generate and cache augmented predictions."""
    
    # Parse config
    config = OmegaConf.to_container(cfg, resolve=True)
    config["project_root"] = str(PROJECT_ROOT)
    
    # Get force flag from config override
    force_regenerate = config.get("run_params", {}).get("force_regenerate", False)
    
    logger.info("=" * 80)
    logger.info("HYCO-PhiFlow Cache Generation")
    logger.info("=" * 80)
    
    # Check if augmentation is enabled
    trainer_config = config.get("trainer_params", {})
    augmentation_config = trainer_config.get("augmentation", {})
    
    if not augmentation_config.get("enabled", False):
        logger.error("Augmentation is not enabled in configuration!")
        logger.error("Set trainer_params.augmentation.enabled=true to enable augmentation")
        sys.exit(1)
    
    strategy = augmentation_config.get("strategy", "cached")
    if strategy != "cached":
        logger.warning(f"Current strategy is '{strategy}', but cache generation is for 'cached' strategy")
        logger.warning("Consider changing strategy to 'cached' in config")
    
    alpha = augmentation_config.get("alpha", 0.1)
    logger.info(f"Augmentation alpha: {alpha} ({alpha * 100:.1f}% of real data)")
    
    # Determine model type
    model_type = config["run_params"]["model_type"]
    logger.info(f"Model type: {model_type}")
    
    # Load or create model
    if model_type == "synthetic":
        logger.info("Creating synthetic model...")
        model = ModelFactory.create_synthetic_model(config)
        
        # Try to load checkpoint if specified
        model_config = config["model"]["synthetic"]
        checkpoint_path = model_config.get("checkpoint")
        
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.exists():
                logger.info(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                else:
                    model.load_state_dict(checkpoint)
                
                logger.info("Checkpoint loaded successfully")
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                logger.warning("Using randomly initialized model")
        else:
            logger.warning("No checkpoint specified, using randomly initialized model")
            logger.warning("Predictions will be random - train a model first!")
    
    elif model_type == "physical":
        logger.info("Creating physical model...")
        model = ModelFactory.create_physical_model(config)
        logger.info("Physical model created (uses PDE solver, no checkpoint needed)")
    
    else:
        logger.error(f"Unknown model type: {model_type}")
        sys.exit(1)
    
    # Move model to appropriate device
    device = augmentation_config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
    
    if model_type == "synthetic":
        model = model.to(device)
        logger.info(f"Model moved to {device}")
    
    # Generate cache
    logger.info("")
    logger.info("=" * 80)
    logger.info("Generating Augmented Cache")
    logger.info("=" * 80)
    
    try:
        num_cached = TrainerFactory.generate_augmented_cache(
            config=config,
            model=model,
            model_type=model_type,
            force_regenerate=force_regenerate,
        )
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("Cache Generation Complete!")
        logger.info("=" * 80)
        logger.info(f"Successfully cached {num_cached} predictions")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Verify cache with: python scripts/validate_cache.py")
        logger.info("2. Start training with: python run.py --config-name=your_experiment")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Cache generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
