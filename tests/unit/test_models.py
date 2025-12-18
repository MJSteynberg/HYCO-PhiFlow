"""Unit tests for physical and synthetic models."""

import pytest
from phi.math import math, spatial, channel, batch as phi_batch


@pytest.mark.integration
class TestPhysicalModel:
    """Test suite for physical models."""

    def test_physical_model_properties(self, physical_model):
        """Test physical model has required properties."""
        assert hasattr(physical_model, 'domain')
        assert hasattr(physical_model, 'resolution')
        assert hasattr(physical_model, 'dt')
        assert hasattr(physical_model, 'field_names')
        assert hasattr(physical_model, 'num_channels')

    def test_physical_model_field_names(self, physical_model):
        """Test field names are properly defined."""
        assert physical_model.field_names is not None
        assert len(physical_model.field_names) > 0
        assert physical_model.num_channels == len(physical_model.field_names)

    def test_get_initial_state(self, physical_model):
        """Test generating initial state."""
        state = physical_model.get_initial_state(batch_size=2)

        assert state is not None
        assert 'batch' in state.shape.names
        assert state.shape.get_size('batch') == 2

    def test_get_real_params(self, physical_model):
        """Test getting ground truth parameters."""
        params = physical_model.get_real_params()

        assert params is not None
        assert 'field' in params.shape.names

    def test_get_initial_params(self, physical_model):
        """Test getting initial learnable parameters."""
        params = physical_model.get_initial_params()

        assert params is not None
        assert 'field' in params.shape.names

    def test_forward_pass(self, physical_model):
        """Test single forward step."""
        initial_state = physical_model.get_initial_state(batch_size=1)
        params = physical_model.get_initial_params()

        # Forward pass
        next_state = physical_model.forward(initial_state, params)

        assert next_state is not None
        assert next_state.shape == initial_state.shape

    def test_rollout(self, physical_model):
        """Test multi-step rollout."""
        initial_state = physical_model.get_initial_state(batch_size=1)
        params = physical_model.get_real_params()
        num_steps = 5

        trajectory = physical_model.rollout(initial_state, params, num_steps)

        assert trajectory is not None
        assert 'time' in trajectory.shape.names
        # Rollout includes initial state, so time dimension is num_steps + 1
        assert trajectory.shape.get_size('time') == num_steps + 1

    def test_params_setter(self, physical_model):
        """Test parameter setter enforces constraints."""
        initial_params = physical_model.get_initial_params()

        # Set new params (should be clipped to non-negative)
        new_params = initial_params * 2.0
        physical_model.params = new_params

        # Verify params were set
        assert physical_model.params is not None


@pytest.mark.integration
class TestSyntheticModel:
    """Test suite for synthetic models."""

    def test_synthetic_model_properties(self, synthetic_model):
        """Test synthetic model has required properties."""
        assert hasattr(synthetic_model, 'network')
        assert hasattr(synthetic_model, 'num_channels')
        assert hasattr(synthetic_model, 'static_fields')
        assert hasattr(synthetic_model, 'num_dynamic')

    def test_synthetic_model_forward(self, synthetic_model, sample_dataset):
        """Test synthetic model forward pass."""
        # Get a sample from dataset
        for batch in sample_dataset.iterate_batches(batch_size=1):
            if batch.has_real:
                initial_state = batch.real_initial_state
                break

        # Forward pass
        next_state = synthetic_model(initial_state)

        assert next_state is not None
        # Output should have same spatial shape
        assert next_state.shape.spatial == initial_state.shape.spatial

    def test_synthetic_model_multiple_steps(self, synthetic_model, sample_dataset):
        """Test synthetic model can do multiple forward steps."""
        # Get a sample from dataset
        for batch in sample_dataset.iterate_batches(batch_size=1):
            if batch.has_real:
                current_state = batch.real_initial_state
                break

        # Multiple forward steps
        num_steps = 3
        states = [current_state]
        for _ in range(num_steps):
            current_state = synthetic_model(current_state)
            states.append(current_state)

        assert len(states) == num_steps + 1

    def test_save_and_load(self, synthetic_model, temp_output_dir):
        """Test saving and loading model checkpoint."""
        save_path = temp_output_dir / "test_model.pth"

        # Save
        synthetic_model.save(str(save_path))
        assert save_path.exists()

        # Load
        synthetic_model.load(str(save_path))


class TestModelDimensions:
    """Test suite for model dimension handling."""

    @pytest.mark.integration
    def test_1d_physical_model(self, base_burgers_1d_config, test_data_dir):
        """Test 1D physical model dimensions."""
        from src.factories.model_factory import ModelFactory

        model = ModelFactory.create_physical_model(base_burgers_1d_config)

        assert model.n_spatial_dims == 1
        assert 'x' in model.spatial_dims

    @pytest.mark.integration
    def test_2d_physical_model(self, base_burgers_2d_config):
        """Test 2D physical model dimensions."""
        from src.factories.model_factory import ModelFactory

        # Modify config to use test data path
        config = base_burgers_2d_config.copy()
        config['data']['data_dir'] = 'data/test_burgers_2d'

        model = ModelFactory.create_physical_model(config)

        assert model.n_spatial_dims == 2
        assert 'x' in model.spatial_dims
        assert 'y' in model.spatial_dims
