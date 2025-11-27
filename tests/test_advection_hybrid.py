"""Integration test for hybrid trainer with advection config."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from omegaconf import OmegaConf
from src.factories.trainer_factory import TrainerFactory
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger("test_advection_hybrid", level="INFO")

def test_advection_hybrid():
    """Test hybrid trainer with advection config."""
    logger.info("="*60)
    logger.info("Testing Advection Hybrid Training")
    logger.info("="*60)

    # Load advection config
    config_path = PROJECT_ROOT / "conf" / "advection.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Reduce for quick testing
    config['trainer']['synthetic']['epochs'] = 1
    config['trainer']['physical']['epochs'] = 1
    config['trainer']['hybrid']['cycles'] = 1
    config['trainer']['hybrid']['warmup'] = 0
    config['trainer']['batch_size'] = 16
    config['trainer']['train_sim'] = [0]

    logger.info(f"Config: mode={config['general']['mode']}")
    logger.info(f"Synthetic epochs: {config['trainer']['synthetic']['epochs']}")
    logger.info(f"Physical epochs: {config['trainer']['physical']['epochs']}")
    logger.info(f"Hybrid cycles: {config['trainer']['hybrid']['cycles']}")

    # Create trainer
    logger.info("\nCreating hybrid trainer...")
    trainer = TrainerFactory.create_trainer(config)
    logger.info("✓ Hybrid trainer created successfully")

    # Run training
    logger.info("\nRunning training...")
    results = trainer.train(verbose=True)

    logger.info("\n" + "="*60)
    logger.info("Training Results")
    logger.info("="*60)
    logger.info(f"Cycles: {len(results['cycles'])}")
    logger.info(f"Final synthetic loss: {results['synthetic_losses'][-1]:.6f}")
    logger.info(f"Final physical loss: {results['physical_losses'][-1]:.6f}")
    logger.info(f"Total time: {sum(results['cycle_times']):.2f}s")

    logger.info("\n" + "="*60)
    logger.info("✓ Integration test passed!")
    logger.info("="*60)

if __name__ == "__main__":
    test_advection_hybrid()
