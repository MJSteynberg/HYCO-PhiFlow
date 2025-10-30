"""
Tests for the DataManager class.

These tests verify that the DataManager can:
1. Load data from Scene directories
2. Convert Fields to tensors correctly
3. Cache data for fast reloading
4. Store sufficient metadata for Field reconstruction
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from src.data.data_manager import DataManager


class TestDataManager:
    """Test suite for DataManager functionality."""
    
    @pytest.fixture
    def project_root(self):
        """Get the project root directory."""
        # Assuming tests are run from project root
        return Path.cwd()
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Cleanup after test
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def burgers_config(self):
        """Configuration for burgers dataset."""
        return {
            'dset_name': 'burgers_128',
            'fields': ['velocity']
        }
    
    @pytest.fixture
    def smoke_config(self):
        """Configuration for smoke dataset."""
        return {
            'dset_name': 'smoke_128',
            'fields': ['velocity', 'density']
        }
    
    @pytest.fixture
    def burgers_data_manager(self, project_root, temp_cache_dir, burgers_config):
        """Create a DataManager for burgers dataset."""
        raw_data_dir = project_root / "data" / "burgers_128"
        
        # Skip test if data directory doesn't exist
        if not raw_data_dir.exists():
            pytest.skip(f"Data directory not found: {raw_data_dir}")
        
        return DataManager(
            raw_data_dir=str(raw_data_dir),
            cache_dir=str(temp_cache_dir),
            config=burgers_config
        )
    
    def test_initialization(self, burgers_data_manager, temp_cache_dir):
        """Test that DataManager initializes correctly."""
        assert burgers_data_manager.cache_dir.exists()
        assert burgers_data_manager.config['dset_name'] == 'burgers_128'
    
    def test_cache_path_generation(self, burgers_data_manager):
        """Test that cache paths are generated correctly."""
        cache_path = burgers_data_manager.get_cached_path(0)
        
        assert cache_path.name == "sim_000000.pt"
        assert "burgers_128" in str(cache_path)
    
    def test_is_cached_false_initially(self, burgers_data_manager):
        """Test that data is not cached initially."""
        assert not burgers_data_manager.is_cached(0)
    
    def test_load_and_cache_simulation(self, burgers_data_manager):
        """Test loading a simulation and caching it."""
        sim_index = 0
        field_names = ['velocity']
        num_frames = 10  # Load only 10 frames for faster testing
        
        # Load and cache
        data = burgers_data_manager.load_and_cache_simulation(
            sim_index=sim_index,
            field_names=field_names,
            num_frames=num_frames
        )
        
        # Verify structure
        assert 'tensor_data' in data
        assert 'metadata' in data
        
        # Verify tensor data
        assert 'velocity' in data['tensor_data']
        tensor = data['tensor_data']['velocity']
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == num_frames  # Time dimension
        
        # Verify metadata
        metadata = data['metadata']
        assert 'scene_metadata' in metadata
        assert 'field_metadata' in metadata
        assert 'num_frames' in metadata
        assert metadata['num_frames'] == num_frames
        
        # Verify field metadata
        field_meta = metadata['field_metadata']['velocity']
        assert 'shape' in field_meta
        assert 'spatial_dims' in field_meta
        assert 'extrapolation' in field_meta
        assert 'bounds' in field_meta
        
        # Verify cache was created
        assert burgers_data_manager.is_cached(sim_index)
    
    def test_load_from_cache(self, burgers_data_manager):
        """Test loading cached data."""
        sim_index = 0
        field_names = ['velocity']
        num_frames = 5
        
        # First, cache the data
        original_data = burgers_data_manager.load_and_cache_simulation(
            sim_index=sim_index,
            field_names=field_names,
            num_frames=num_frames
        )
        
        # Now load from cache
        cached_data = burgers_data_manager.load_from_cache(sim_index)
        
        # Verify they're the same
        assert 'tensor_data' in cached_data
        assert 'velocity' in cached_data['tensor_data']
        
        # Compare tensor values
        original_tensor = original_data['tensor_data']['velocity']
        cached_tensor = cached_data['tensor_data']['velocity']
        assert torch.allclose(original_tensor, cached_tensor)
        
        # Compare metadata
        assert cached_data['metadata']['num_frames'] == original_data['metadata']['num_frames']
    
    def test_get_or_load_simulation_without_cache(self, burgers_data_manager):
        """Test get_or_load when data is not cached."""
        sim_index = 0
        field_names = ['velocity']
        num_frames = 5
        
        # Should load and cache
        data = burgers_data_manager.get_or_load_simulation(
            sim_index=sim_index,
            field_names=field_names,
            num_frames=num_frames
        )
        
        assert 'tensor_data' in data
        assert burgers_data_manager.is_cached(sim_index)
    
    def test_get_or_load_simulation_with_cache(self, burgers_data_manager):
        """Test get_or_load when data is already cached."""
        sim_index = 0
        field_names = ['velocity']
        num_frames = 5
        
        # Cache the data first
        burgers_data_manager.load_and_cache_simulation(
            sim_index=sim_index,
            field_names=field_names,
            num_frames=num_frames
        )
        
        # Now use get_or_load (should load from cache)
        data = burgers_data_manager.get_or_load_simulation(
            sim_index=sim_index,
            field_names=field_names,
            num_frames=num_frames
        )
        
        assert 'tensor_data' in data
        assert 'velocity' in data['tensor_data']
    
    def test_scene_metadata_preservation(self, burgers_data_manager):
        """Test that scene metadata is correctly preserved."""
        sim_index = 0
        field_names = ['velocity']
        num_frames = 5
        
        data = burgers_data_manager.load_and_cache_simulation(
            sim_index=sim_index,
            field_names=field_names,
            num_frames=num_frames
        )
        
        scene_meta = data['metadata']['scene_metadata']
        
        # Verify key metadata fields are present
        assert 'PDE' in scene_meta
        assert 'Fields' in scene_meta
        assert 'Dt' in scene_meta
        assert 'Domain' in scene_meta
        assert 'Resolution' in scene_meta
        
        # Verify values make sense
        assert scene_meta['PDE'] == 'BurgersModel'
        assert 'velocity' in scene_meta['Fields']
    
    def test_tensor_shape_correctness(self, burgers_data_manager):
        """Test that tensor shapes are correct."""
        sim_index = 0
        field_names = ['velocity']
        num_frames = 10
        
        data = burgers_data_manager.load_and_cache_simulation(
            sim_index=sim_index,
            field_names=field_names,
            num_frames=num_frames
        )
        
        tensor = data['tensor_data']['velocity']
        
        # Shape should be (time, channels, height, width)
        assert len(tensor.shape) == 4
        assert tensor.shape[0] == num_frames  # Time
        
        # Verify resolution matches config
        scene_meta = data['metadata']['scene_metadata']
        resolution = scene_meta['Resolution']
        assert tensor.shape[2] == resolution['x']
        assert tensor.shape[3] == resolution['y']
    
    def test_nonexistent_scene_raises_error(self, burgers_data_manager):
        """Test that loading a non-existent scene raises an error."""
        with pytest.raises(FileNotFoundError):
            burgers_data_manager.load_and_cache_simulation(
                sim_index=99999,  # Non-existent simulation
                field_names=['velocity'],
                num_frames=5
            )
    
    def test_load_uncached_raises_error(self, burgers_data_manager):
        """Test that loading from cache without caching first raises an error."""
        with pytest.raises(FileNotFoundError):
            burgers_data_manager.load_from_cache(sim_index=0)


class TestDataManagerWithMultipleFields:
    """Test DataManager with datasets containing multiple fields (e.g., smoke)."""
    
    @pytest.fixture
    def project_root(self):
        """Get the project root directory."""
        return Path.cwd()
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def smoke_config(self):
        """Configuration for smoke dataset."""
        return {
            'dset_name': 'smoke_128',
            'fields': ['velocity', 'density']
        }
    
    @pytest.fixture
    def smoke_data_manager(self, project_root, temp_cache_dir, smoke_config):
        """Create a DataManager for smoke dataset."""
        raw_data_dir = project_root / "data" / "smoke_128"
        
        if not raw_data_dir.exists():
            pytest.skip(f"Data directory not found: {raw_data_dir}")
        
        return DataManager(
            raw_data_dir=str(raw_data_dir),
            cache_dir=str(temp_cache_dir),
            config=smoke_config
        )
    
    def test_multiple_fields_loading(self, smoke_data_manager):
        """Test loading multiple fields simultaneously."""
        sim_index = 0
        field_names = ['velocity', 'density']
        num_frames = 5
        
        data = smoke_data_manager.load_and_cache_simulation(
            sim_index=sim_index,
            field_names=field_names,
            num_frames=num_frames
        )
        
        # Verify both fields are present
        assert 'velocity' in data['tensor_data']
        assert 'density' in data['tensor_data']
        
        # Verify both have correct time dimension
        assert data['tensor_data']['velocity'].shape[0] == num_frames
        assert data['tensor_data']['density'].shape[0] == num_frames
        
        # Verify metadata for both fields
        assert 'velocity' in data['metadata']['field_metadata']
        assert 'density' in data['metadata']['field_metadata']


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
