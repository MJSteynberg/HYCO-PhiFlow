"""
Test HybridDataset with return_fields option
"""

import pytest
import torch
from phi.field import Field

from src.data import DataManager, HybridDataset


class TestHybridDatasetFields:
    """Test HybridDataset with return_fields=True."""
    
    @pytest.fixture
    def data_manager(self):
        """Create DataManager for burgers test dataset."""
        return DataManager(
            raw_data_dir='data/burgers_128',
            cache_dir='data/cache',
            config={'dset_name': 'burgers_128'}
        )
    
    def test_return_tensors_default(self, data_manager):
        """Test default behavior returns tensors (backward compatibility)."""
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=[0],
            field_names=['velocity'],
            num_frames=10,
            num_predict_steps=5,
            return_fields=False  # Explicit, but this is default
        )
        
        initial_state, rollout_targets = dataset[0]
        
        # Should return tensors
        assert isinstance(initial_state, torch.Tensor)
        assert isinstance(rollout_targets, torch.Tensor)
        assert initial_state.ndim == 3  # [C, H, W]
        assert rollout_targets.ndim == 4  # [T, C, H, W]
    
    def test_return_fields(self, data_manager):
        """Test return_fields=True returns PhiFlow Fields."""
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=[0],
            field_names=['velocity'],
            num_frames=10,
            num_predict_steps=5,
            return_fields=True
        )
        
        initial_fields, target_fields = dataset[0]
        
        # Should return dictionaries
        assert isinstance(initial_fields, dict)
        assert isinstance(target_fields, dict)
        
        # Initial fields should have all field names
        assert 'velocity' in initial_fields
        assert isinstance(initial_fields['velocity'], Field)
        
        # Target fields should have lists of Fields
        assert 'velocity' in target_fields
        assert isinstance(target_fields['velocity'], list)
        assert len(target_fields['velocity']) == 5  # num_predict_steps
        assert all(isinstance(f, Field) for f in target_fields['velocity'])
    
    def test_field_properties(self, data_manager):
        """Test that returned Fields have correct properties."""
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=[0],
            field_names=['velocity'],
            num_frames=10,
            num_predict_steps=2,
            return_fields=True
        )
        
        initial_fields, target_fields = dataset[0]
        
        # Check field structure
        velocity_initial = initial_fields['velocity']
        assert hasattr(velocity_initial, 'shape')
        assert hasattr(velocity_initial, 'values')
        
        # Check target fields
        velocity_targets = target_fields['velocity']
        for field in velocity_targets:
            assert hasattr(field, 'shape')
            assert hasattr(field, 'values')
            # Should have same spatial shape as initial
            assert field.shape.spatial == velocity_initial.shape.spatial
