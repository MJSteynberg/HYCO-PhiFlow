"""Test script for hybrid trainer with advection configuration."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.factories.trainer_factory import TrainerFactory
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger("test_hybrid", level="INFO")

def load_config():
    """Load advection config and modify for quick testing."""
    config_path = PROJECT_ROOT / "conf" / "advection.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Reduce iterations for quick testing
    config['trainer']['synthetic']['epochs'] = 2
    config['trainer']['physical']['epochs'] = 1
    config['trainer']['hybrid']['cycles'] = 2
    config['trainer']['hybrid']['warmup'] = 1
    config['trainer']['batch_size'] = 8
    config['trainer']['rollout_steps'] = 2

    # Use only one simulation for testing
    config['trainer']['train_sim'] = [0]

    # Set mode to hybrid
    config['general']['mode'] = 'hybrid'

    return config

def test_hybrid_trainer():
    """Test hybrid trainer initialization and basic training."""
    logger.info("="*60)
    logger.info("Testing Hybrid Trainer")
    logger.info("="*60)

    # Load config
    logger.info("Loading configuration...")
    config = load_config()

    # Create trainer
    logger.info("Creating hybrid trainer...")
    try:
        trainer = TrainerFactory.create_trainer(config)
        logger.info("✓ Hybrid trainer created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create hybrid trainer: {e}")
        raise

    # Test training
    logger.info("\nStarting training test...")
    try:
        results = trainer.train(verbose=True)
        logger.info("✓ Training completed successfully")

        # Print results
        logger.info("\n" + "="*60)
        logger.info("Training Results")
        logger.info("="*60)
        logger.info(f"Cycles completed: {len(results['cycles'])}")
        logger.info(f"Final synthetic loss: {results['synthetic_losses'][-1]:.6f}")
        logger.info(f"Final physical loss: {results['physical_losses'][-1]:.6f}")
        logger.info(f"Total time: {sum(results['cycle_times']):.2f}s")

    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        raise

    logger.info("\n" + "="*60)
    logger.info("All tests passed!")
    logger.info("="*60)

if __name__ == "__main__":
    test_hybrid_trainer()
