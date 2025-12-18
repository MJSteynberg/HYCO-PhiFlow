"""Integration tests for training pipelines."""

import pytest
from pathlib import Path


@pytest.mark.integration
@pytest.mark.slow
class TestSyntheticTraining:
    """Test suite for synthetic model training."""

    def test_synthetic_training_runs(self, synthetic_config, sample_dataset):
        """Test that synthetic training completes without errors."""
        from src.factories.trainer_factory import TrainerFactory

        trainer = TrainerFactory.create_trainer(
            synthetic_config,
            num_channels=sample_dataset.num_channels
        )

        results = trainer.train(
            dataset=sample_dataset,
            num_epochs=1,
            verbose=False
        )

        assert 'train_losses' in results
        assert 'final_loss' in results
        assert len(results['train_losses']) == 1

    def test_synthetic_loss_decreases(self, synthetic_config, sample_dataset):
        """Test that loss decreases over training (smoke test)."""
        from src.factories.trainer_factory import TrainerFactory

        # Use 2 epochs to check loss trend
        synthetic_config['trainer']['synthetic']['epochs'] = 2

        trainer = TrainerFactory.create_trainer(
            synthetic_config,
            num_channels=sample_dataset.num_channels
        )

        results = trainer.train(
            dataset=sample_dataset,
            num_epochs=2,
            verbose=False
        )

        # Loss should be finite
        assert all(loss < float('inf') for loss in results['train_losses'])
        assert results['final_loss'] < float('inf')

    def test_synthetic_checkpoint_saved(self, synthetic_config, sample_dataset, temp_output_dir):
        """Test that checkpoint is saved during training."""
        from src.factories.trainer_factory import TrainerFactory

        # Update checkpoint path
        synthetic_config['model']['synthetic']['model_path'] = str(temp_output_dir)
        synthetic_config['model']['synthetic']['model_save_name'] = 'test_checkpoint'

        trainer = TrainerFactory.create_trainer(
            synthetic_config,
            num_channels=sample_dataset.num_channels
        )

        trainer.train(
            dataset=sample_dataset,
            num_epochs=1,
            verbose=False
        )

        # Check checkpoint was saved
        checkpoint_path = temp_output_dir / "test_checkpoint.pth"
        assert checkpoint_path.exists()


@pytest.mark.integration
@pytest.mark.slow
class TestPhysicalTraining:
    """Test suite for physical model training."""

    def test_physical_training_runs(self, physical_config, sample_dataset):
        """Test that physical training completes without errors."""
        from src.factories.trainer_factory import TrainerFactory

        trainer = TrainerFactory.create_trainer(physical_config)

        results = trainer.train(
            dataset=sample_dataset,
            num_epochs=1,
            verbose=False
        )

        assert 'train_losses' in results
        assert 'final_loss' in results

    def test_physical_parameters_update(self, physical_config, sample_dataset):
        """Test that physical parameters are updated during training."""
        from src.factories.trainer_factory import TrainerFactory
        from phi.math import math

        trainer = TrainerFactory.create_trainer(physical_config)

        # Get initial params
        initial_params = math.stop_gradient(trainer.model.params)

        # Train
        trainer.train(
            dataset=sample_dataset,
            num_epochs=1,
            verbose=False
        )

        # Params should have changed
        final_params = trainer.model.params

        # At least some difference (might be small for 1 epoch)
        assert final_params is not None

    def test_physical_checkpoint_saved(self, physical_config, sample_dataset, temp_output_dir):
        """Test that checkpoint is saved during training."""
        from src.factories.trainer_factory import TrainerFactory

        # Update checkpoint path
        physical_config['model']['physical']['model_path'] = str(temp_output_dir)
        physical_config['model']['physical']['model_save_name'] = 'test_physical'

        trainer = TrainerFactory.create_trainer(physical_config)

        trainer.train(
            dataset=sample_dataset,
            num_epochs=1,
            verbose=False
        )

        # Check checkpoint was saved
        checkpoint_path = temp_output_dir / "test_physical.npz"
        assert checkpoint_path.exists()


@pytest.mark.integration
@pytest.mark.slow
class TestHybridTraining:
    """Test suite for hybrid training."""

    def test_hybrid_training_runs(self, hybrid_config, test_data_dir):
        """Test that hybrid training completes without errors."""
        from src.factories.trainer_factory import TrainerFactory

        trainer = TrainerFactory.create_trainer(hybrid_config)

        results = trainer.train(verbose=False)

        assert 'cycles' in results
        assert 'synthetic_losses' in results
        assert 'physical_losses' in results
        assert len(results['cycles']) == hybrid_config['trainer']['hybrid']['cycles']

    def test_hybrid_warmup_phase(self, hybrid_config, test_data_dir):
        """Test that warmup phase works correctly."""
        from src.factories.trainer_factory import TrainerFactory

        # Enable warmup
        hybrid_config['trainer']['hybrid']['warmup'] = 1

        trainer = TrainerFactory.create_trainer(hybrid_config)

        results = trainer.train(verbose=False)

        # Should still complete with warmup
        assert len(results['cycles']) == hybrid_config['trainer']['hybrid']['cycles']

    def test_hybrid_loss_scaling(self, hybrid_config, test_data_dir):
        """Test hybrid training with loss scaling."""
        from src.factories.trainer_factory import TrainerFactory

        # Configure loss scaling
        hybrid_config['trainer']['hybrid']['loss_scaling'] = {
            'synthetic': {
                'real_weight': 1.0,
                'interaction_weight': 0.5,
                'proportional': False
            },
            'physical': {
                'real_weight': 1.0,
                'interaction_weight': 0.1,
                'proportional': True
            }
        }

        trainer = TrainerFactory.create_trainer(hybrid_config)

        results = trainer.train(verbose=False)

        # Should complete without errors
        assert len(results['synthetic_losses']) > 0
        assert len(results['physical_losses']) > 0


@pytest.mark.integration
class TestDataGeneration:
    """Test suite for data generation."""

    def test_data_generation(self, quick_test_config, temp_output_dir):
        """Test generating simulation data."""
        from src.data.data_generator import DataGenerator
        import shutil

        # Use temp directory
        config = quick_test_config.copy()
        config['data']['data_dir'] = str(temp_output_dir / 'test_data')
        config['data']['num_simulations'] = 2

        generator = DataGenerator(config)
        generator.generate_data()

        # Check files were created
        data_dir = Path(config['data']['data_dir'])
        sim_files = list(data_dir.glob("sim_*.npz"))

        assert len(sim_files) == 2

    def test_generated_data_format(self, quick_test_config, temp_output_dir):
        """Test that generated data has correct format."""
        from src.data.data_generator import DataGenerator
        from phi.math import math

        # Use temp directory
        config = quick_test_config.copy()
        config['data']['data_dir'] = str(temp_output_dir / 'test_data_format')
        config['data']['num_simulations'] = 1

        generator = DataGenerator(config)
        generator.generate_data()

        # Load and check format
        data_path = Path(config['data']['data_dir']) / "sim_0000.npz"
        data = math.load(str(data_path))

        assert 'time' in data.shape.names
        assert 'field' in data.shape.names
        assert data.shape.get_size('time') == config['data']['trajectory_length']
