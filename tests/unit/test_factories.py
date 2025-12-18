"""Unit tests for factory classes."""

import pytest
from src.factories.model_factory import ModelFactory
from src.factories.trainer_factory import TrainerFactory


class TestModelFactory:
    """Test suite for ModelFactory."""

    @pytest.mark.integration
    def test_create_physical_model(self, quick_test_config, test_data_dir):
        """Test creating a physical model."""
        model = ModelFactory.create_physical_model(quick_test_config)

        assert model is not None
        assert hasattr(model, 'forward')
        assert hasattr(model, 'rollout')
        assert hasattr(model, 'get_initial_state')

    @pytest.mark.integration
    def test_create_synthetic_model(self, quick_test_config, sample_dataset):
        """Test creating a synthetic model."""
        model = ModelFactory.create_synthetic_model(
            quick_test_config,
            num_channels=sample_dataset.num_channels
        )

        assert model is not None
        assert hasattr(model, 'network')
        assert model.num_channels == sample_dataset.num_channels

    def test_list_available_models(self):
        """Test listing available models."""
        models = ModelFactory.list_available_models()

        assert 'physical' in models
        assert 'synthetic' in models
        assert isinstance(models['physical'], list)
        assert isinstance(models['synthetic'], list)

    @pytest.mark.integration
    def test_create_synthetic_with_static_fields(self, quick_test_config, sample_dataset):
        """Test creating synthetic model with static fields."""
        # Create physical model to get static field names
        physical_model = ModelFactory.create_physical_model(quick_test_config)

        model = ModelFactory.create_synthetic_model(
            quick_test_config,
            num_channels=sample_dataset.num_channels,
            physical_model=physical_model
        )

        assert model is not None


class TestTrainerFactory:
    """Test suite for TrainerFactory."""

    def test_list_available_trainers(self):
        """Test listing available trainers."""
        trainers = TrainerFactory.list_available_trainers()

        assert 'synthetic' in trainers
        assert 'physical' in trainers
        assert 'hybrid' in trainers

    @pytest.mark.integration
    def test_create_synthetic_trainer(self, synthetic_config, sample_dataset):
        """Test creating a synthetic trainer."""
        trainer = TrainerFactory.create_trainer(
            synthetic_config,
            num_channels=sample_dataset.num_channels
        )

        assert trainer is not None
        assert hasattr(trainer, 'train')
        assert hasattr(trainer, 'model')

    @pytest.mark.integration
    def test_create_physical_trainer(self, physical_config, test_data_dir):
        """Test creating a physical trainer."""
        trainer = TrainerFactory.create_trainer(physical_config)

        assert trainer is not None
        assert hasattr(trainer, 'train')
        assert hasattr(trainer, 'model')

    @pytest.mark.integration
    @pytest.mark.slow
    def test_create_hybrid_trainer(self, hybrid_config, test_data_dir):
        """Test creating a hybrid trainer."""
        trainer = TrainerFactory.create_trainer(hybrid_config)

        assert trainer is not None
        assert hasattr(trainer, 'train')
        assert hasattr(trainer, 'synthetic_model')
        assert hasattr(trainer, 'physical_model')

    def test_create_trainer_invalid_mode(self, quick_test_config):
        """Test that invalid mode raises ValueError."""
        config = quick_test_config.copy()
        config['general']['mode'] = 'invalid_mode'

        with pytest.raises(ValueError, match="Unknown"):
            TrainerFactory.create_trainer(config)

    def test_synthetic_requires_num_channels(self, synthetic_config):
        """Test that synthetic trainer requires num_channels."""
        with pytest.raises(ValueError, match="num_channels"):
            TrainerFactory.create_trainer(synthetic_config, num_channels=None)
