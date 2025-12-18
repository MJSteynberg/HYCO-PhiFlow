"""Unit tests for sparsity configuration and masking."""

import pytest
from phi.math import spatial
from src.data.sparsity import (
    TemporalSparsityConfig,
    SpatialSparsityConfig,
    SparsityConfig,
    TemporalMask,
    SpatialMask,
)


class TestTemporalSparsityConfig:
    """Test suite for TemporalSparsityConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TemporalSparsityConfig()
        assert config.enabled is False
        assert config.mode == 'full'
        assert config.start_fraction == 0.1
        assert config.end_fraction == 0.1
        assert config.uniform_stride == 1
        assert config.custom_fractions == []

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TemporalSparsityConfig(
            enabled=True,
            mode='endpoints',
            start_fraction=0.2,
            end_fraction=0.3
        )
        assert config.enabled is True
        assert config.mode == 'endpoints'
        assert config.start_fraction == 0.2
        assert config.end_fraction == 0.3


class TestSpatialSparsityConfig:
    """Test suite for SpatialSparsityConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SpatialSparsityConfig()
        assert config.enabled is False
        assert config.mode == 'full'
        assert config.center_fraction == 0.5

    def test_center_mode(self):
        """Test center mode configuration."""
        config = SpatialSparsityConfig(
            enabled=True,
            mode='center',
            center_fraction=0.6
        )
        assert config.enabled is True
        assert config.mode == 'center'
        assert config.center_fraction == 0.6


class TestTemporalMask:
    """Test suite for TemporalMask."""

    def test_full_mode_all_visible(self):
        """Test that full mode makes all timesteps visible."""
        config = TemporalSparsityConfig(enabled=False)
        mask = TemporalMask(config, trajectory_length=100)

        assert mask.num_visible == 100
        assert len(mask.visible_indices) == 100
        assert mask.is_visible(0)
        assert mask.is_visible(50)
        assert mask.is_visible(99)

    def test_endpoints_mode(self):
        """Test endpoints mode visibility."""
        config = TemporalSparsityConfig(
            enabled=True,
            mode='endpoints',
            start_fraction=0.1,
            end_fraction=0.1
        )
        mask = TemporalMask(config, trajectory_length=100)

        # First 10 and last 10 should be visible
        assert mask.is_visible(0)
        assert mask.is_visible(9)
        assert mask.is_visible(90)
        assert mask.is_visible(99)

        # Middle should not be visible
        assert not mask.is_visible(50)

    def test_uniform_mode(self):
        """Test uniform stride mode."""
        config = TemporalSparsityConfig(
            enabled=True,
            mode='uniform',
            uniform_stride=10
        )
        mask = TemporalMask(config, trajectory_length=100)

        # Every 10th timestep should be visible
        assert mask.is_visible(0)
        assert mask.is_visible(10)
        assert mask.is_visible(20)
        assert not mask.is_visible(5)
        assert not mask.is_visible(15)

    def test_custom_mode(self):
        """Test custom fractions mode."""
        config = TemporalSparsityConfig(
            enabled=True,
            mode='custom',
            custom_fractions=[0.0, 0.5, 1.0]
        )
        mask = TemporalMask(config, trajectory_length=100)

        # Should have 3 visible timesteps at 0%, 50%, 100%
        assert 0 in mask.visible_indices
        assert 49 in mask.visible_indices or 50 in mask.visible_indices
        assert 99 in mask.visible_indices

    def test_visible_mask_tensor(self):
        """Test getting mask as tensor."""
        config = TemporalSparsityConfig(
            enabled=True,
            mode='uniform',
            uniform_stride=5
        )
        mask = TemporalMask(config, trajectory_length=20)
        tensor_mask = mask.get_visible_mask_tensor()

        assert tensor_mask.shape.get_size('time') == 20


class TestSpatialMask:
    """Test suite for SpatialMask."""

    def test_full_mode_all_visible(self):
        """Test that full mode makes entire domain visible."""
        config = SpatialSparsityConfig(enabled=False)
        shape = spatial(x=64, y=64)
        mask = SpatialMask(config, shape)

        assert mask.visible_fraction == 1.0
        assert mask.visible_count == 64 * 64

    def test_center_mode(self):
        """Test center mode creates correct mask."""
        config = SpatialSparsityConfig(
            enabled=True,
            mode='center',
            center_fraction=0.5
        )
        shape = spatial(x=100, y=100)
        mask = SpatialMask(config, shape)

        # Center 50% should be visible (approximately 25% of total area)
        assert 0.2 < mask.visible_fraction < 0.3

    def test_range_mode(self):
        """Test range mode creates correct mask."""
        config = SpatialSparsityConfig(
            enabled=True,
            mode='range',
            x_range=(0.0, 0.5),
            y_range=(0.0, 0.5)
        )
        shape = spatial(x=100, y=100)
        mask = SpatialMask(config, shape)

        # First quarter should be visible
        assert 0.2 < mask.visible_fraction < 0.3

    def test_compute_masked_mse(self):
        """Test MSE computation with mask."""
        from phi.math import zeros, ones

        config = SpatialSparsityConfig(enabled=False)
        shape = spatial(x=10, y=10)
        mask = SpatialMask(config, shape)

        pred = ones(shape)
        target = zeros(shape)

        mse = mask.compute_masked_mse(pred, target)
        assert float(mse) == 1.0  # All ones minus all zeros squared

    def test_apply_to_difference(self):
        """Test applying mask to difference."""
        from phi.math import zeros, ones

        config = SpatialSparsityConfig(
            enabled=True,
            mode='center',
            center_fraction=0.5
        )
        shape = spatial(x=10, y=10)
        mask = SpatialMask(config, shape)

        pred = ones(shape)
        target = zeros(shape)

        masked_diff = mask.apply_to_difference(pred, target)

        # Masked regions should be zero
        # This is a basic check that mask was applied
        assert masked_diff.shape == shape


class TestSparsityConfig:
    """Test suite for combined SparsityConfig."""

    def test_default_both_disabled(self):
        """Test default config has both sparsities disabled."""
        config = SparsityConfig()
        assert config.temporal.enabled is False
        assert config.spatial.enabled is False

    def test_combined_config(self):
        """Test creating combined config."""
        temporal = TemporalSparsityConfig(enabled=True, mode='endpoints')
        spatial = SpatialSparsityConfig(enabled=True, mode='center')
        config = SparsityConfig(temporal=temporal, spatial=spatial)

        assert config.temporal.enabled is True
        assert config.temporal.mode == 'endpoints'
        assert config.spatial.enabled is True
        assert config.spatial.mode == 'center'
