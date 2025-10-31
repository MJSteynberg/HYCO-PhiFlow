"""
Tests for DataManager cache validation
"""

import pytest
import tempfile
import torch
from pathlib import Path

from src.data import DataManager


@pytest.fixture
def burgers_data_manager():
    """Create a DataManager for Burgers dataset with complete config."""
    raw_data_dir = Path(__file__).parent.parent.parent / "data" / "burgers_128"
    
    # Complete config with all required parameters for validation
    config = {
        'dset_name': 'burgers_128',
        'fields': ['velocity'],
        'model': {
            'physical': {
                'name': 'BurgersModel',
                'resolution': {'x': 128, 'y': 128},
                'domain': {'size_x': 100, 'size_y': 100},
                'dt': 0.8,
                'pde_params': {'batch_size': 1, 'nu': 0.1}
            }
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        dm = DataManager(
            raw_data_dir=str(raw_data_dir),
            cache_dir=tmpdir,
            config=config
        )
        yield dm


class TestCacheValidation:
    """Test cache validation functionality."""
    
    def test_is_cached_with_no_cache(self, burgers_data_manager):
        """Test is_cached returns False when no cache exists."""
        assert not burgers_data_manager.is_cached(0)
        assert not burgers_data_manager.is_cached(0, ['velocity'], 10)
    
    def test_is_cached_after_caching(self, burgers_data_manager):
        """Test is_cached returns True after caching."""
        # Cache with specific parameters
        burgers_data_manager.load_and_cache_simulation(0, ['velocity'], 10)
        
        # Should be cached
        assert burgers_data_manager.is_cached(0)
        assert burgers_data_manager.is_cached(0, ['velocity'], 10)
    
    def test_is_cached_validates_field_names(self, burgers_data_manager):
        """Test that is_cached validates field names."""
        # Cache with 'velocity'
        burgers_data_manager.load_and_cache_simulation(0, ['velocity'], 10)
        
        # Should match with same field
        assert burgers_data_manager.is_cached(0, ['velocity'], 10)
        
        # Should NOT match with different field
        assert not burgers_data_manager.is_cached(0, ['density'], 10)
    
    def test_is_cached_validates_num_frames(self, burgers_data_manager):
        """Test that is_cached validates number of frames."""
        # Cache with 10 frames
        burgers_data_manager.load_and_cache_simulation(0, ['velocity'], 10)
        
        # Should match with same or fewer frames
        assert burgers_data_manager.is_cached(0, ['velocity'], 10)
        assert burgers_data_manager.is_cached(0, ['velocity'], 5)
        
        # Should NOT match with more frames
        assert not burgers_data_manager.is_cached(0, ['velocity'], 15)
    
    def test_is_cached_handles_corrupt_cache(self, burgers_data_manager):
        """Test that is_cached handles corrupt cache files gracefully."""
        # Create a cache file
        burgers_data_manager.load_and_cache_simulation(0, ['velocity'], 10)
        cache_path = burgers_data_manager.get_cached_path(0)
        
        # Corrupt the cache file
        with open(cache_path, 'w') as f:
            f.write("corrupted data")
        
        # Should return False for corrupt cache
        assert not burgers_data_manager.is_cached(0, ['velocity'], 10)
    
    def test_get_or_load_uses_valid_cache(self, burgers_data_manager):
        """Test that get_or_load_simulation uses cache when valid."""
        # Cache with specific parameters
        data1 = burgers_data_manager.load_and_cache_simulation(0, ['velocity'], 10)
        
        # Load again with same parameters - should use cache
        data2 = burgers_data_manager.get_or_load_simulation(0, ['velocity'], 10)
        
        # Should be the same data
        assert torch.equal(data1['tensor_data']['velocity'], data2['tensor_data']['velocity'])
    
    def test_get_or_load_recreates_invalid_cache(self, burgers_data_manager):
        """Test that get_or_load_simulation recreates cache when invalid."""
        # Cache with 10 frames
        burgers_data_manager.load_and_cache_simulation(0, ['velocity'], 10)
        
        # Request with 15 frames - should recreate cache
        data = burgers_data_manager.get_or_load_simulation(0, ['velocity'], 15)
        
        # Should have 15 frames
        assert data['tensor_data']['velocity'].shape[0] == 15
        assert data['metadata']['num_frames'] == 15
    
    def test_cache_validation_with_subset_frames(self, burgers_data_manager):
        """Test using cached data with fewer frames than cached."""
        # Cache with 10 frames
        burgers_data_manager.load_and_cache_simulation(0, ['velocity'], 10)
        
        # Request only 5 frames - should use existing cache
        data = burgers_data_manager.get_or_load_simulation(0, ['velocity'], 5)
        
        # Should have loaded from cache (has 10 frames)
        assert data['tensor_data']['velocity'].shape[0] == 10
        assert data['metadata']['num_frames'] == 10


class TestCacheValidationMultiField:
    """Test cache validation with multiple fields."""
    
    def test_validates_multiple_fields(self):
        """Test validation with multiple fields."""
        raw_data_dir = Path(__file__).parent.parent.parent / "data" / "smoke_128"
        
        # Complete config for smoke dataset
        config = {
            'dset_name': 'smoke_128',
            'fields': ['velocity', 'density'],
            'model': {
                'physical': {
                    'name': 'SmokeModel',
                    'resolution': {'x': 128, 'y': 128},
                    'domain': {'size_x': 100, 'size_y': 100},
                    'dt': 0.8,
                    'pde_params': {'batch_size': 1, 'buoyancy_factor': 0.1}
                }
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManager(
                raw_data_dir=str(raw_data_dir),
                cache_dir=tmpdir,
                config=config
            )
            
            # Cache with velocity and density
            dm.load_and_cache_simulation(0, ['velocity', 'density'], 8)
            
            # Should match with both fields
            assert dm.is_cached(0, ['velocity', 'density'], 8)
            
            # Should NOT match with only one field
            assert not dm.is_cached(0, ['velocity'], 8)
            assert not dm.is_cached(0, ['density'], 8)
            
            # Should NOT match with additional field
            assert not dm.is_cached(0, ['velocity', 'density', 'pressure'], 8)
