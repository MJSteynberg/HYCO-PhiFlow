"""Test Burgers model with spatially-varying diffusion field."""

import sys
from pathlib import Path
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.data.data_generator import DataGenerator
from src.factories.trainer_factory import TrainerFactory
from src.evaluation import Evaluator
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger("test_field_diffusion", level="INFO")

def test_burgers_1d_field():
    """Test Burgers 1D with field diffusion."""
    logger.info("="*60)
    logger.info("Testing Burgers 1D with Spatially-Varying Diffusion Field")
    logger.info("="*60)

    # Load config
    config_path = PROJECT_ROOT / "conf" / "burgers_1d.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Quick test settings
    config['data']['num_simulations'] = 2
    config['data']['trajectory_length'] = 20
    config['trainer']['physical']['epochs'] = 2
    config['trainer']['physical']['max_iterations'] = 5
    config['trainer']['train_sim'] = [0]

    # Check the config has field type
    assert config['model']['physical']['pde_params']['type'] == 'field', "Expected field type"
    logger.info(f"Diffusion field expression: {config['model']['physical']['pde_params']['value']}")

    # Generate data
    data_dir = Path(config['data']['data_dir'])
    if data_dir.exists():
        logger.info(f"Removing existing data: {data_dir}")
        shutil.rmtree(data_dir)

    logger.info("Step 1: Generating data with spatially-varying diffusion...")
    data_gen = DataGenerator(config)
    data_gen.generate_data()
    logger.info(f"✓ Generated {len(list(data_dir.glob('sim_*.npz')))} simulations")

    # Test that we can create the physical model
    logger.info("\nStep 2: Testing physical model creation...")
    from src.factories.model_factory import ModelFactory
    physical_model = ModelFactory.create_physical_model(config)

    # Check the parameter type
    assert len(physical_model.scalar_param_names) == 0, "Should have no scalar params"
    assert len(physical_model.field_param_names) == 1, "Should have one field param"
    logger.info(f"✓ Physical model created with field parameter: {physical_model.field_param_names}")

    # Get ground truth parameters
    real_params = physical_model.get_real_params()
    logger.info(f"  Ground truth params shape: {real_params.shape}")

    # Train physical model
    logger.info("\nStep 3: Training physical model to learn diffusion field...")
    from src.factories.dataloader_factory import DataLoaderFactory
    dataset = DataLoaderFactory.create_phiml(config, sim_indices=config['trainer']['train_sim'])

    config['general']['mode'] = 'physical'
    trainer = TrainerFactory.create_trainer(config)
    results = trainer.train(dataset=dataset, num_epochs=config['trainer']['physical']['epochs'])

    logger.info(f"✓ Physical training completed: final_loss={results['final_loss']:.6f}")

    # Check learned parameters
    learned_params = trainer.model.params
    logger.info(f"  Learned params shape: {learned_params.shape}")

    # Evaluate
    logger.info("\nStep 4: Running evaluation to visualize learned field...")
    config['evaluation']['test_sim'] = [0]
    config['evaluation']['synthetic_checkpoint'] = 'results/models/burgers_synthetic_model_1d.pth'
    config['evaluation']['physical_checkpoint'] = 'results/models/burgers_physical_model_1d.npz'

    # Check if synthetic model exists
    if Path(config['evaluation']['synthetic_checkpoint']).exists():
        evaluator = Evaluator(config)
        evaluator.evaluate()

        # Check for field parameter visualization
        output_dir = Path(config['evaluation']['output_dir'])
        field_plot = output_dir / 'param_diffusion_field_comparison.png'

        if field_plot.exists():
            logger.info(f"✓ Field parameter visualization created: {field_plot}")
        else:
            logger.info("Note: Field visualization will be created after running full pipeline")
    else:
        logger.info("Note: Skipping evaluation (no synthetic model trained yet)")

    logger.info("\n" + "="*60)
    logger.info("✓ Burgers 1D field diffusion test passed!")
    logger.info("="*60)

def test_burgers_2d_field():
    """Test Burgers 2D with field diffusion."""
    logger.info("\n" + "="*60)
    logger.info("Testing Burgers 2D with Spatially-Varying Diffusion Field")
    logger.info("="*60)

    # Load config
    config_path = PROJECT_ROOT / "conf" / "burgers_2d.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Quick test settings
    config['data']['num_simulations'] = 2
    config['data']['trajectory_length'] = 20
    config['trainer']['physical']['epochs'] = 1
    config['trainer']['physical']['max_iterations'] = 3
    config['trainer']['train_sim'] = [0]

    # Check the config has field type
    assert config['model']['physical']['pde_params']['type'] == 'field', "Expected field type"
    logger.info(f"Diffusion field expression: {config['model']['physical']['pde_params']['value']}")

    # Generate data
    data_dir = Path(config['data']['data_dir'])
    if data_dir.exists():
        logger.info(f"Removing existing data: {data_dir}")
        shutil.rmtree(data_dir)

    logger.info("Step 1: Generating data with spatially-varying diffusion...")
    data_gen = DataGenerator(config)
    data_gen.generate_data()
    logger.info(f"✓ Generated {len(list(data_dir.glob('sim_*.npz')))} simulations")

    # Test that we can create the physical model
    logger.info("\nStep 2: Testing physical model creation...")
    from src.factories.model_factory import ModelFactory
    physical_model = ModelFactory.create_physical_model(config)

    # Check the parameter type
    assert len(physical_model.scalar_param_names) == 0, "Should have no scalar params"
    assert len(physical_model.field_param_names) == 1, "Should have one field param"
    logger.info(f"✓ Physical model created with field parameter: {physical_model.field_param_names}")

    # Get ground truth parameters
    real_params = physical_model.get_real_params()
    logger.info(f"  Ground truth params shape: {real_params.shape}")

    # Train physical model
    logger.info("\nStep 3: Training physical model to learn diffusion field...")
    from src.factories.dataloader_factory import DataLoaderFactory
    dataset = DataLoaderFactory.create_phiml(config, sim_indices=config['trainer']['train_sim'])

    config['general']['mode'] = 'physical'
    trainer = TrainerFactory.create_trainer(config)
    results = trainer.train(dataset=dataset, num_epochs=config['trainer']['physical']['epochs'])

    logger.info(f"✓ Physical training completed: final_loss={results['final_loss']:.6f}")

    # Check learned parameters
    learned_params = trainer.model.params
    logger.info(f"  Learned params shape: {learned_params.shape}")

    logger.info("\n" + "="*60)
    logger.info("✓ Burgers 2D field diffusion test passed!")
    logger.info("="*60)

if __name__ == "__main__":
    try:
        # Test 1D
        test_burgers_1d_field()

        # Test 2D
        test_burgers_2d_field()

        logger.info("\n" + "="*60)
        logger.info("✓ ALL FIELD DIFFUSION TESTS PASSED!")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
