"""Unit tests for Dataset class."""

import pytest
from phi.math import math, spatial, channel, batch as phi_batch
from src.data.dataset import Dataset, AccessPolicy, SeparatedBatch


class TestAccessPolicy:
    """Test suite for AccessPolicy enum."""

    def test_access_policy_values(self):
        """Test that AccessPolicy has expected values."""
        assert AccessPolicy.REAL_ONLY is not None
        assert AccessPolicy.GENERATED_ONLY is not None
        assert AccessPolicy.BOTH is not None


class TestSeparatedBatch:
    """Test suite for SeparatedBatch dataclass."""

    def test_has_real_true(self):
        """Test has_real returns True when real data present."""
        batch = SeparatedBatch(
            real_initial_state=math.ones(spatial(x=10)),
            real_targets=math.ones(spatial(x=10)),
            generated_initial_state=None,
            generated_targets=None
        )
        assert batch.has_real is True
        assert batch.has_generated is False

    def test_has_generated_true(self):
        """Test has_generated returns True when generated data present."""
        batch = SeparatedBatch(
            real_initial_state=None,
            real_targets=None,
            generated_initial_state=math.ones(spatial(x=10)),
            generated_targets=math.ones(spatial(x=10))
        )
        assert batch.has_real is False
        assert batch.has_generated is True

    def test_has_both(self):
        """Test both flags True when both data present."""
        batch = SeparatedBatch(
            real_initial_state=math.ones(spatial(x=10)),
            real_targets=math.ones(spatial(x=10)),
            generated_initial_state=math.ones(spatial(x=10)),
            generated_targets=math.ones(spatial(x=10))
        )
        assert batch.has_real is True
        assert batch.has_generated is True


@pytest.mark.integration
class TestDataset:
    """Integration tests for Dataset class."""

    def test_dataset_creation(self, quick_test_config, test_data_dir):
        """Test creating a dataset from config."""
        dataset = Dataset(
            config=quick_test_config,
            train_sim=[0],
            rollout_steps=2
        )

        assert dataset.num_channels > 0
        assert dataset.field_names is not None
        assert len(dataset) > 0

    def test_dataset_num_channels(self, quick_test_config, test_data_dir):
        """Test that num_channels is correctly extracted."""
        dataset = Dataset(
            config=quick_test_config,
            train_sim=[0],
            rollout_steps=2
        )

        # Burgers 1D should have 1 channel (velocity)
        assert dataset.num_channels >= 1

    def test_dataset_iteration(self, quick_test_config, test_data_dir):
        """Test iterating through dataset batches."""
        dataset = Dataset(
            config=quick_test_config,
            train_sim=[0],
            rollout_steps=2
        )

        batch_count = 0
        for batch in dataset.iterate_batches(batch_size=4, shuffle=False):
            assert isinstance(batch, SeparatedBatch)
            if batch.has_real:
                assert batch.real_initial_state is not None
                assert batch.real_targets is not None
            batch_count += 1

        assert batch_count > 0

    def test_access_policy_real_only(self, quick_test_config, test_data_dir):
        """Test REAL_ONLY access policy."""
        dataset = Dataset(
            config=quick_test_config,
            train_sim=[0],
            rollout_steps=2
        )
        dataset.access_policy = AccessPolicy.REAL_ONLY

        for batch in dataset.iterate_batches(batch_size=4):
            assert batch.has_real is True
            assert batch.has_generated is False

    def test_set_augmented_trajectories(self, quick_test_config, test_data_dir):
        """Test setting augmented trajectories."""
        dataset = Dataset(
            config=quick_test_config,
            train_sim=[0],
            rollout_steps=2
        )

        # Create fake trajectories using proper PhiML syntax
        trajectory_length = 10
        fake_traj = math.ones(phi_batch(time=trajectory_length), spatial(x=64))
        fake_traj = math.expand(fake_traj, channel(field='vel'))

        initial_total = dataset.total_samples
        dataset.set_augmented_trajectories([fake_traj])

        # Total samples should increase
        assert dataset.total_samples > initial_total

    def test_alpha_sampling(self, quick_test_config, test_data_dir):
        """Test alpha parameter reduces real data samples."""
        dataset = Dataset(
            config=quick_test_config,
            train_sim=[0],
            rollout_steps=2
        )
        dataset.alpha = 0.5
        dataset.access_policy = AccessPolicy.REAL_ONLY

        # Count samples with alpha
        samples_with_alpha = sum(
            1 for batch in dataset.iterate_batches(batch_size=1)
            if batch.has_real
        )

        # Reset alpha
        dataset.alpha = 1.0
        samples_full = sum(
            1 for batch in dataset.iterate_batches(batch_size=1)
            if batch.has_real
        )

        # With alpha=0.5, should have roughly half the samples
        assert samples_with_alpha < samples_full
