"""
Tests for HybridDataset
"""

import pytest
import tempfile
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.data import DataManager, HybridDataset


@pytest.fixture
def burgers_config():
    """Configuration for burgers dataset."""
    return {
        'dset_name': 'burgers_128'
    }


@pytest.fixture
def smoke_config():
    """Configuration for smoke dataset."""
    return {
        'dset_name': 'smoke_128'
    }


@pytest.fixture
def burgers_dataset(burgers_config):
    """Create a HybridDataset for Burgers."""
    raw_data_dir = Path(__file__).parent.parent.parent / "data" / "burgers_128"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        data_manager = DataManager(
            raw_data_dir=str(raw_data_dir),
            cache_dir=tmpdir,
            config=burgers_config
        )
        
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=[0],
            field_names=['velocity'],
            num_frames=10,
            num_predict_steps=5
        )
        
        yield dataset


@pytest.fixture
def smoke_dataset(smoke_config):
    """Create a HybridDataset for Smoke."""
    raw_data_dir = Path(__file__).parent.parent.parent / "data" / "smoke_128"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        data_manager = DataManager(
            raw_data_dir=str(raw_data_dir),
            cache_dir=tmpdir,
            config=smoke_config
        )
        
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=[0],
            field_names=['velocity', 'density'],
            num_frames=8,
            num_predict_steps=4
        )
        
        yield dataset


class TestHybridDataset:
    """Test cases for HybridDataset with single field."""
    
    def test_dataset_length(self, burgers_dataset):
        """Test that dataset length matches number of simulations."""
        assert len(burgers_dataset) == 1
    
    def test_getitem_returns_tuple(self, burgers_dataset):
        """Test that __getitem__ returns (initial_state, rollout_targets)."""
        initial_state, rollout_targets = burgers_dataset[0]
        
        assert isinstance(initial_state, torch.Tensor)
        assert isinstance(rollout_targets, torch.Tensor)
    
    def test_initial_state_shape(self, burgers_dataset):
        """Test initial state has correct shape [C, H, W]."""
        initial_state, _ = burgers_dataset[0]
        
        assert len(initial_state.shape) == 3
        assert initial_state.shape[0] == 2  # velocity has 2 components
        assert initial_state.shape[1] == 128  # height
        assert initial_state.shape[2] == 128  # width
    
    def test_rollout_targets_shape(self, burgers_dataset):
        """Test rollout targets have correct shape [T, C, H, W]."""
        _, rollout_targets = burgers_dataset[0]
        
        assert len(rollout_targets.shape) == 4
        assert rollout_targets.shape[0] == 5  # num_predict_steps
        assert rollout_targets.shape[1] == 2  # velocity has 2 components
        assert rollout_targets.shape[2] == 128  # height
        assert rollout_targets.shape[3] == 128  # width
    
    def test_validation_error_insufficient_frames(self, burgers_config):
        """Test that error is raised if num_frames < num_predict_steps + 1."""
        raw_data_dir = Path(__file__).parent.parent.parent / "data" / "burgers_128"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data_manager = DataManager(
                raw_data_dir=str(raw_data_dir),
                cache_dir=tmpdir,
                config=burgers_config
            )
            
            with pytest.raises(ValueError, match="num_frames.*must be"):
                HybridDataset(
                    data_manager=data_manager,
                    sim_indices=[0],
                    field_names=['velocity'],
                    num_frames=5,  # Too few
                    num_predict_steps=10  # Too many
                )
    
    def test_dataloader_compatibility(self, burgers_dataset):
        """Test that HybridDataset works with PyTorch DataLoader."""
        dataloader = DataLoader(
            burgers_dataset,
            batch_size=1,
            shuffle=False
        )
        
        # Get one batch
        initial_batch, rollout_batch = next(iter(dataloader))
        
        # Check batch dimension added
        assert initial_batch.shape[0] == 1  # batch size
        assert rollout_batch.shape[0] == 1  # batch size
        
        # Check other dimensions preserved
        assert initial_batch.shape[1] == 2  # channels
        assert rollout_batch.shape[1] == 5  # time steps


class TestHybridDatasetMultiField:
    """Test cases for HybridDataset with multiple fields."""
    
    def test_multiple_fields_concatenation(self, smoke_dataset):
        """Test that multiple fields are concatenated along channel dimension."""
        initial_state, rollout_targets = smoke_dataset[0]
        
        # velocity (2 channels) + density (1 channel) = 3 channels
        assert initial_state.shape[0] == 3
        assert rollout_targets.shape[1] == 3
    
    def test_multiple_fields_shape_consistency(self, smoke_dataset):
        """Test that shapes are consistent for multi-field data."""
        initial_state, rollout_targets = smoke_dataset[0]
        
        # Initial state: [C, H, W]
        assert len(initial_state.shape) == 3
        assert initial_state.shape[1] == 128
        assert initial_state.shape[2] == 128
        
        # Rollout: [T, C, H, W]
        assert len(rollout_targets.shape) == 4
        assert rollout_targets.shape[0] == 4  # num_predict_steps
        assert rollout_targets.shape[2] == 128
        assert rollout_targets.shape[3] == 128
    
    def test_multi_field_dataloader(self, smoke_dataset):
        """Test multi-field dataset with DataLoader."""
        dataloader = DataLoader(
            smoke_dataset,
            batch_size=1,
            shuffle=False
        )
        
        initial_batch, rollout_batch = next(iter(dataloader))
        
        # Check concatenated channels
        assert initial_batch.shape[1] == 3  # velocity (2) + density (1)
        assert rollout_batch.shape[2] == 3  # same for rollout


class TestHybridDatasetMultipleSims:
    """Test cases for HybridDataset with multiple simulations."""
    
    def test_single_simulation_in_list(self, burgers_config):
        """Test dataset with simulation index in list format."""
        raw_data_dir = Path(__file__).parent.parent.parent / "data" / "burgers_128"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data_manager = DataManager(
                raw_data_dir=str(raw_data_dir),
                cache_dir=tmpdir,
                config=burgers_config
            )
            
            # Test with single sim in list (validates multi-sim logic)
            dataset = HybridDataset(
                data_manager=data_manager,
                sim_indices=[0],
                field_names=['velocity'],
                num_frames=10,
                num_predict_steps=5
            )
            
            assert len(dataset) == 1
    
    def test_simulation_indexing(self, burgers_config):
        """Test that indexing works correctly."""
        raw_data_dir = Path(__file__).parent.parent.parent / "data" / "burgers_128"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data_manager = DataManager(
                raw_data_dir=str(raw_data_dir),
                cache_dir=tmpdir,
                config=burgers_config
            )
            
            dataset = HybridDataset(
                data_manager=data_manager,
                sim_indices=[0],
                field_names=['velocity'],
                num_frames=10,
                num_predict_steps=5
            )
            
            # Access the simulation
            initial_0, rollout_0 = dataset[0]
            
            # Should have correct shapes
            assert initial_0.shape[0] == 2  # velocity channels
            assert rollout_0.shape[0] == 5  # time steps
            assert rollout_0.shape[1] == 2  # velocity channels
