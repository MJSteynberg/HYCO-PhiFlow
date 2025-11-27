"""Test script for Burgers 2D - all training modes."""

import sys
from pathlib import Path
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.data.data_generator import DataGenerator
from src.factories.trainer_factory import TrainerFactory
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger("test_burgers_2d", level="INFO")

def load_config(mode='synthetic'):
    """Load burgers_2d config and modify for quick testing."""
    config_path = PROJECT_ROOT / "conf" / "burgers_2d.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Reduce iterations for quick testing
    config['data']['num_simulations'] = 2
    config['data']['trajectory_length'] = 20
    config['trainer']['synthetic']['epochs'] = 2
    config['trainer']['physical']['epochs'] = 1
    config['trainer']['hybrid']['cycles'] = 1
    config['trainer']['hybrid']['warmup'] = 0
    config['trainer']['batch_size'] = 8
    config['trainer']['rollout_steps'] = 2
    config['trainer']['train_sim'] = [0]

    # Set mode
    config['general']['mode'] = mode

    return config

def test_data_generation():
    """Test data generation for Burgers 2D."""
    logger.info("="*60)
    logger.info("Testing Burgers 2D Data Generation")
    logger.info("="*60)

    config = load_config()

    # Remove existing data if present
    data_dir = Path(config['data']['data_dir'])
    if data_dir.exists():
        logger.info(f"Removing existing data directory: {data_dir}")
        shutil.rmtree(data_dir)

    # Generate data
    logger.info("Generating data...")
    data_gen = DataGenerator(config)
    data_gen.generate_data()

    # Check data was created
    sim_files = list(data_dir.glob("sim_*.npz"))
    logger.info(f"✓ Generated {len(sim_files)} simulations")

    assert len(sim_files) == config['data']['num_simulations'], "Wrong number of simulations generated"

    return config

def test_synthetic_training(config):
    """Test synthetic model training."""
    logger.info("\n" + "="*60)
    logger.info("Testing Burgers 2D Synthetic Training")
    logger.info("="*60)

    config['general']['mode'] = 'synthetic'

    # Import dataset factory
    from src.factories.dataloader_factory import DataLoaderFactory

    # Create dataset
    dataset = DataLoaderFactory.create_phiml(
        config,
        sim_indices=config['trainer']['train_sim'],
    )

    # Create trainer
    trainer = TrainerFactory.create_trainer(config, num_channels=dataset.num_channels)

    # Train
    num_epochs = config["trainer"]['synthetic']["epochs"]
    results = trainer.train(dataset=dataset, num_epochs=num_epochs)

    logger.info(f"✓ Synthetic training completed: final_loss={results['final_loss']:.6f}")

def test_physical_training(config):
    """Test physical model training."""
    logger.info("\n" + "="*60)
    logger.info("Testing Burgers 2D Physical Training")
    logger.info("="*60)

    config['general']['mode'] = 'physical'

    # Import dataset factory
    from src.factories.dataloader_factory import DataLoaderFactory

    # Create dataset
    dataset = DataLoaderFactory.create_phiml(
        config,
        sim_indices=config['trainer']['train_sim'],
    )

    # Create trainer
    trainer = TrainerFactory.create_trainer(config)

    # Train
    num_epochs = config["trainer"]['physical']["epochs"]
    results = trainer.train(dataset=dataset, num_epochs=num_epochs)

    logger.info(f"✓ Physical training completed: final_loss={results['final_loss']:.6f}")

    # Check learned parameter
    learned_param = float(trainer.model.params.field['diffusion_coeff'])
    target_param = float(eval(config['model']['physical']['pde_params']['value']))
    logger.info(f"  Learned diffusion: {learned_param:.6f}, Target: {target_param:.6f}")

def test_hybrid_training(config):
    """Test hybrid training."""
    logger.info("\n" + "="*60)
    logger.info("Testing Burgers 2D Hybrid Training")
    logger.info("="*60)

    config['general']['mode'] = 'hybrid'

    # Create trainer
    trainer = TrainerFactory.create_trainer(config)

    # Train
    results = trainer.train(verbose=True)

    logger.info(f"✓ Hybrid training completed:")
    logger.info(f"  Final synthetic loss: {results['synthetic_losses'][-1]:.6f}")
    logger.info(f"  Final physical loss: {results['physical_losses'][-1]:.6f}")

if __name__ == "__main__":
    try:
        # Test data generation
        config = test_data_generation()

        # Test synthetic training
        test_synthetic_training(config)

        # Test physical training
        test_physical_training(config)

        # Test hybrid training
        test_hybrid_training(config)

        logger.info("\n" + "="*60)
        logger.info("✓ All Burgers 2D tests passed!")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
