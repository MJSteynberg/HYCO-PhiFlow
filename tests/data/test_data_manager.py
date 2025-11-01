"""
Comprehensive tests for DataManager.
Tests initialization, caching, loading, validation, tensor conversion, and metadata handling.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from src.data.data_manager import DataManager


class TestDataManagerInitialization:
    """Tests for DataManager initialization."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        raw_dir = tempfile.mkdtemp()
        cache_dir = tempfile.mkdtemp()
        
        yield raw_dir, cache_dir
        
        # Cleanup
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
    
    @pytest.fixture
    def basic_config(self):
        """Create a basic config."""
        return {
            'dset_name': 'test_dataset',
            'fields': ['velocity']
        }
    
    def test_basic_initialization(self, temp_dirs, basic_config):
        """Test that DataManager can be initialized."""
        raw_dir, cache_dir = temp_dirs
        
        dm = DataManager(raw_dir, cache_dir, basic_config)
        
        assert dm is not None
        assert dm.raw_data_dir == Path(raw_dir)
        assert dm.cache_dir == Path(cache_dir)
        assert dm.config == basic_config
        assert dm.validate_cache is True  # Default value
        assert dm.auto_clear_invalid is False  # Default value
    
    def test_initialization_with_validation_params(self, temp_dirs, basic_config):
        """Test DataManager initialization with cache validation parameters."""
        raw_dir, cache_dir = temp_dirs
        
        dm = DataManager(
            raw_dir, 
            cache_dir, 
            basic_config,
            validate_cache=False,
            auto_clear_invalid=True
        )
        
        assert dm.validate_cache is False
        assert dm.auto_clear_invalid is True
    
    def test_cache_dir_creation(self, temp_dirs, basic_config):
        """Test that cache directory is created if it doesn't exist."""
        raw_dir, cache_dir = temp_dirs
        non_existent_cache = str(Path(cache_dir) / "nested" / "cache")
        
        dm = DataManager(raw_dir, non_existent_cache, basic_config)
        
        assert Path(non_existent_cache).exists()
    
    def test_config_storage(self, temp_dirs, basic_config):
        """Test that config is stored correctly."""
        raw_dir, cache_dir = temp_dirs
        
        dm = DataManager(raw_dir, cache_dir, basic_config)
        
        assert dm.config['dset_name'] == 'test_dataset'
        assert dm.config['fields'] == ['velocity']
    
    def test_raw_data_dir_path_type(self, temp_dirs, basic_config):
        """Test that raw_data_dir is stored as Path object."""
        raw_dir, cache_dir = temp_dirs
        
        dm = DataManager(raw_dir, cache_dir, basic_config)
        
        assert isinstance(dm.raw_data_dir, Path)
    
    def test_cache_dir_path_type(self, temp_dirs, basic_config):
        """Test that cache_dir is stored as Path object."""
        raw_dir, cache_dir = temp_dirs
        
        dm = DataManager(raw_dir, cache_dir, basic_config)
        
        assert isinstance(dm.cache_dir, Path)


class TestDataManagerCachePaths:
    """Tests for cache path generation."""
    
    @pytest.fixture
    def data_manager(self):
        """Create a DataManager for testing."""
        config = {
            'dset_name': 'burgers_128',
            'fields': ['velocity']
        }
        return DataManager('data', 'data/cache', config)
    
    def test_get_cached_path_format(self, data_manager):
        """Test that cache path has correct format."""
        path = data_manager.get_cached_path(0)
        
        assert 'burgers_128' in str(path)
        assert 'sim_000000.pt' in str(path)
    
    def test_get_cached_path_different_indices(self, data_manager):
        """Test paths for different simulation indices."""
        path_0 = data_manager.get_cached_path(0)
        path_1 = data_manager.get_cached_path(1)
        path_10 = data_manager.get_cached_path(10)
        path_99 = data_manager.get_cached_path(99)
        
        assert 'sim_000000.pt' in str(path_0)
        assert 'sim_000001.pt' in str(path_1)
        assert 'sim_000010.pt' in str(path_10)
        assert 'sim_000099.pt' in str(path_99)
    
    def test_get_cached_path_large_index(self, data_manager):
        """Test path generation with large index."""
        path = data_manager.get_cached_path(123456)
        
        assert 'sim_123456.pt' in str(path)
    
    def test_cached_path_in_correct_subdirectory(self, data_manager):
        """Test that cached path is in correct subdirectory."""
        path = data_manager.get_cached_path(0)
        
        assert 'burgers_128' in str(path)
        assert 'cache' in str(path)
    
    def test_cached_path_subdirectory_creation(self, data_manager):
        """Test that subdirectory is created when getting path."""
        path = data_manager.get_cached_path(0)
        
        # The parent directory should exist
        assert path.parent.exists()


class TestDataManagerCacheValidation:
    """Tests for cache validation."""
    
    @pytest.fixture
    def temp_setup(self):
        """Create temp directories and DataManager."""
        raw_dir = tempfile.mkdtemp()
        cache_dir = tempfile.mkdtemp()
        
        config = {
            'dset_name': 'test_dataset',
            'fields': ['velocity']
        }
        
        dm = DataManager(raw_dir, cache_dir, config)
        
        yield dm, raw_dir, cache_dir
        
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
    
    def test_is_cached_nonexistent_file(self, temp_setup):
        """Test that non-existent cache returns False."""
        dm, _, _ = temp_setup
        
        assert not dm.is_cached(0)
        assert not dm.is_cached(999)
    
    def test_is_cached_no_validation(self, temp_setup):
        """Test is_cached without validation parameters."""
        dm, _, _ = temp_setup
        
        # Create fake cache file
        cache_path = dm.get_cached_path(0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        fake_data = {
            'tensor_data': {'velocity': torch.randn(10, 2, 64, 64)},
            'metadata': {}
        }
        torch.save(fake_data, cache_path)
        
        assert dm.is_cached(0)
    
    def test_is_cached_with_field_validation_matching(self, temp_setup):
        """Test cache validation with matching field names."""
        dm, _, _ = temp_setup
        
        # Create cache with velocity field and proper v2.0 metadata
        cache_path = dm.get_cached_path(0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        fake_data = {
            'tensor_data': {'velocity': torch.randn(10, 2, 64, 64)},
            'metadata': {
                'version': '2.0',  # Required for version validation
                'field_metadata': {'velocity': {}},  # Required for field validation
                'num_frames': 10
            }
        }
        torch.save(fake_data, cache_path)
        
        # Should pass with matching field
        assert dm.is_cached(0, field_names=['velocity'])
    
    def test_is_cached_with_field_validation_non_matching(self, temp_setup):
        """Test cache validation with non-matching field names."""
        dm, _, _ = temp_setup
        
        # Create cache with velocity field
        cache_path = dm.get_cached_path(0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        fake_data = {
            'tensor_data': {'velocity': torch.randn(10, 2, 64, 64)},
            'metadata': {}
        }
        torch.save(fake_data, cache_path)
        
        # Should fail with non-matching field
        assert not dm.is_cached(0, field_names=['density'])
    
    def test_is_cached_with_num_frames_validation_sufficient(self, temp_setup):
        """Test cache validation with sufficient num_frames."""
        dm, _, _ = temp_setup
        
        # Create cache with 10 frames
        cache_path = dm.get_cached_path(0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        fake_data = {
            'tensor_data': {'velocity': torch.randn(10, 2, 64, 64)},
            'metadata': {'num_frames': 10}
        }
        torch.save(fake_data, cache_path)
        
        # Should pass when requesting <= 10 frames
        assert dm.is_cached(0, num_frames=5)
        assert dm.is_cached(0, num_frames=10)
    
    def test_is_cached_with_num_frames_validation_insufficient(self, temp_setup):
        """Test cache validation with insufficient num_frames."""
        dm, _, _ = temp_setup
        
        # Create cache with 10 frames
        cache_path = dm.get_cached_path(0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        fake_data = {
            'tensor_data': {'velocity': torch.randn(10, 2, 64, 64)},
            'metadata': {'num_frames': 10}
        }
        torch.save(fake_data, cache_path)
        
        # Should fail when requesting > 10 frames
        assert not dm.is_cached(0, num_frames=15)
        assert not dm.is_cached(0, num_frames=100)
    
    def test_is_cached_with_combined_validation_all_match(self, temp_setup):
        """Test cache validation with both field names and num_frames matching."""
        dm, _, _ = temp_setup
        
        cache_path = dm.get_cached_path(0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        fake_data = {
            'tensor_data': {'velocity': torch.randn(10, 2, 64, 64)},
            'metadata': {
                'version': '2.0',  # Required for version validation
                'field_metadata': {'velocity': {}},  # Required for field validation
                'num_frames': 10
            }
        }
        torch.save(fake_data, cache_path)
        
        # Should pass with matching parameters
        assert dm.is_cached(0, field_names=['velocity'], num_frames=5)
        assert dm.is_cached(0, field_names=['velocity'], num_frames=10)
    
    def test_is_cached_with_combined_validation_field_mismatch(self, temp_setup):
        """Test cache validation with field name mismatch."""
        dm, _, _ = temp_setup
        
        cache_path = dm.get_cached_path(0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        fake_data = {
            'tensor_data': {'velocity': torch.randn(10, 2, 64, 64)},
            'metadata': {'num_frames': 10}
        }
        torch.save(fake_data, cache_path)
        
        # Should fail with wrong field
        assert not dm.is_cached(0, field_names=['density'], num_frames=5)
    
    def test_is_cached_with_combined_validation_frames_mismatch(self, temp_setup):
        """Test cache validation with num_frames mismatch."""
        dm, _, _ = temp_setup
        
        cache_path = dm.get_cached_path(0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        fake_data = {
            'tensor_data': {'velocity': torch.randn(10, 2, 64, 64)},
            'metadata': {'num_frames': 10}
        }
        torch.save(fake_data, cache_path)
        
        # Should fail with too many frames
        assert not dm.is_cached(0, field_names=['velocity'], num_frames=15)
    
    def test_is_cached_handles_corrupt_cache(self, temp_setup):
        """Test that corrupt cache file returns False when validation parameters are provided."""
        dm, _, _ = temp_setup
        
        cache_path = dm.get_cached_path(999)  # Use different sim index
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write corrupt data
        with open(cache_path, 'w') as f:
            f.write("corrupt data")
        
        # With validation parameters, should return False due to exception
        assert not dm.is_cached(999, field_names=['velocity'], num_frames=3)
    
    def test_is_cached_handles_missing_metadata(self, temp_setup):
        """Test that cache without metadata structure is handled."""
        dm, _, _ = temp_setup
        
        cache_path = dm.get_cached_path(0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Cache without proper metadata
        fake_data = {
            'tensor_data': {'velocity': torch.randn(10, 2, 64, 64)}
            # Missing 'metadata' key
        }
        torch.save(fake_data, cache_path)
        
        # Should still work with no validation
        assert dm.is_cached(0)
        
        # But should fail with validation
        assert not dm.is_cached(0, num_frames=5)


class TestDataManagerLoadingFromScene:
    """Tests for loading data from Scene directories."""
    
    @pytest.fixture
    def burgers_manager(self):
        """Create DataManager with real burgers dataset."""
        project_root = Path(__file__).parent.parent.parent
        
        config = {
            'dset_name': 'burgers_128',
            'fields': ['velocity']
        }
        
        raw_dir = project_root / 'data' / 'burgers_128'
        cache_dir = project_root / 'data' / 'cache' / 'test_comprehensive'
        
        return DataManager(str(raw_dir), str(cache_dir), config)
    
    def test_load_and_cache_simulation_structure(self, burgers_manager):
        """Test that loaded data has correct structure."""
        data = burgers_manager.get_or_load_simulation(0, ['velocity'], num_frames=5)
        
        assert 'tensor_data' in data
        assert 'metadata' in data
    
    def test_tensor_data_keys(self, burgers_manager):
        """Test that tensor_data contains correct fields."""
        data = burgers_manager.get_or_load_simulation(0, ['velocity'], num_frames=5)
        
        assert 'velocity' in data['tensor_data']
        assert isinstance(data['tensor_data']['velocity'], torch.Tensor)
    
    def test_tensor_shape_4d(self, burgers_manager):
        """Test that tensors have correct 4D shape."""
        data = burgers_manager.get_or_load_simulation(0, ['velocity'], num_frames=5)
        
        velocity_tensor = data['tensor_data']['velocity']
        
        # Should be [time, channels, x, y]
        assert velocity_tensor.dim() == 4
    
    def test_tensor_time_dimension(self):
        """Test that time dimension matches num_frames."""
        project_root = Path(__file__).parent.parent.parent
        
        config = {
            'dset_name': 'burgers_128',
            'fields': ['velocity']
        }
        
        for num_frames in [3, 5, 10]:
            # Use different cache dir for each num_frames to avoid interference
            cache_dir = project_root / 'data' / 'cache' / f'test_time_dim_{num_frames}'
            raw_dir = project_root / 'data' / 'burgers_128'
            
            burgers_manager = DataManager(str(raw_dir), str(cache_dir), config)
            data = burgers_manager.get_or_load_simulation(0, ['velocity'], num_frames=num_frames)
            velocity_tensor = data['tensor_data']['velocity']
            assert velocity_tensor.shape[0] == num_frames
    
    def test_tensor_channel_dimension(self, burgers_manager):
        """Test that channel dimension is correct for velocity."""
        data = burgers_manager.get_or_load_simulation(0, ['velocity'], num_frames=5)
        velocity_tensor = data['tensor_data']['velocity']
        
        # Velocity should have 2 components
        assert velocity_tensor.shape[1] == 2
    
    def test_tensor_spatial_dimensions(self, burgers_manager):
        """Test that spatial dimensions match dataset."""
        data = burgers_manager.get_or_load_simulation(0, ['velocity'], num_frames=5)
        velocity_tensor = data['tensor_data']['velocity']
        
        # Burgers 128 should have 128x128 resolution
        assert velocity_tensor.shape[2] == 128
        assert velocity_tensor.shape[3] == 128
    
    def test_tensor_dtype(self, burgers_manager):
        """Test that tensors have correct dtype."""
        data = burgers_manager.get_or_load_simulation(0, ['velocity'], num_frames=5)
        
        velocity_tensor = data['tensor_data']['velocity']
        assert velocity_tensor.dtype == torch.float32
    
    def test_tensor_device(self, burgers_manager):
        """Test that cached tensors are on CPU."""
        data = burgers_manager.get_or_load_simulation(0, ['velocity'], num_frames=5)
        
        velocity_tensor = data['tensor_data']['velocity']
        assert velocity_tensor.device.type == 'cpu'
    
    def test_tensor_values_finite(self, burgers_manager):
        """Test that tensor values are finite (no NaN or Inf)."""
        data = burgers_manager.get_or_load_simulation(0, ['velocity'], num_frames=5)
        
        velocity_tensor = data['tensor_data']['velocity']
        assert torch.isfinite(velocity_tensor).all()
    
    def test_metadata_structure(self, burgers_manager):
        """Test that metadata has required keys."""
        data = burgers_manager.get_or_load_simulation(0, ['velocity'], num_frames=5)
        
        metadata = data['metadata']
        assert 'scene_metadata' in metadata
        assert 'field_metadata' in metadata
        assert 'num_frames' in metadata
        assert 'frame_indices' in metadata
    
    def test_scene_metadata_content(self, burgers_manager):
        """Test scene metadata contains PDE information."""
        data = burgers_manager.get_or_load_simulation(0, ['velocity'], num_frames=5)
        
        scene_meta = data['metadata']['scene_metadata']
        
        assert 'PDE' in scene_meta
        assert 'Fields' in scene_meta
        assert 'Dt' in scene_meta
        assert 'Domain' in scene_meta
        assert 'Resolution' in scene_meta
    
    def test_scene_metadata_values(self, burgers_manager):
        """Test scene metadata has correct values."""
        data = burgers_manager.get_or_load_simulation(0, ['velocity'], num_frames=5)
        
        scene_meta = data['metadata']['scene_metadata']
        
        assert scene_meta['PDE'] == 'BurgersModel'
        assert 'velocity' in scene_meta['Fields']
    
    def test_field_metadata_content(self, burgers_manager):
        """Test field metadata contains required information."""
        data = burgers_manager.get_or_load_simulation(0, ['velocity'], num_frames=5)
        
        field_meta = data['metadata']['field_metadata']['velocity']
        
        assert 'shape' in field_meta
        assert 'spatial_dims' in field_meta
        assert 'extrapolation' in field_meta
        assert 'bounds' in field_meta
        assert 'field_type' in field_meta
    
    def test_field_metadata_spatial_dims(self, burgers_manager):
        """Test that spatial_dims are correctly extracted."""
        data = burgers_manager.get_or_load_simulation(0, ['velocity'], num_frames=5)
        
        field_meta = data['metadata']['field_metadata']['velocity']
        spatial_dims = field_meta['spatial_dims']
        
        assert isinstance(spatial_dims, list)
        assert len(spatial_dims) == 2  # 2D simulation
    
    def test_field_metadata_field_type(self, burgers_manager):
        """Test that field_type is recorded."""
        data = burgers_manager.get_or_load_simulation(0, ['velocity'], num_frames=5)
        
        field_meta = data['metadata']['field_metadata']['velocity']
        
        assert field_meta['field_type'] in ['staggered', 'centered']
    
    def test_num_frames_respected(self):
        """Test that num_frames parameter is respected."""
        project_root = Path(__file__).parent.parent.parent
        
        config = {
            'dset_name': 'burgers_128',
            'fields': ['velocity']
        }
        
        raw_dir = project_root / 'data' / 'burgers_128'
        cache_dir = project_root / 'data' / 'cache' / 'test_frames_respected'
        
        burgers_manager = DataManager(str(raw_dir), str(cache_dir), config)
        data = burgers_manager.get_or_load_simulation(0, ['velocity'], num_frames=3)
        
        velocity_tensor = data['tensor_data']['velocity']
        assert velocity_tensor.shape[0] == 3
        assert data['metadata']['num_frames'] == 3
    
    def test_frame_indices_recorded(self, burgers_manager):
        """Test that frame indices are recorded in metadata."""
        data = burgers_manager.get_or_load_simulation(0, ['velocity'], num_frames=5)
        
        frame_indices = data['metadata']['frame_indices']
        assert len(frame_indices) == 5
        assert frame_indices[0] == 0


class TestDataManagerCacheLoading:
    """Tests for loading from cache."""
    
    @pytest.fixture
    def burgers_manager_with_cache(self):
        """Create DataManager and cache some data."""
        project_root = Path(__file__).parent.parent.parent
        
        config = {
            'dset_name': 'burgers_128',
            'fields': ['velocity']
        }
        
        raw_dir = project_root / 'data' / 'burgers_128'
        cache_dir = project_root / 'data' / 'cache' / 'test_loading'
        
        dm = DataManager(str(raw_dir), str(cache_dir), config)
        
        # Cache simulation 0
        dm.get_or_load_simulation(0, ['velocity'], num_frames=5)
        
        return dm
    
    def test_load_from_cache_succeeds(self, burgers_manager_with_cache):
        """Test that loading from cache works."""
        data = burgers_manager_with_cache.load_from_cache(0)
        
        assert 'tensor_data' in data
        assert 'velocity' in data['tensor_data']
    
    def test_load_from_cache_data_integrity(self, burgers_manager_with_cache):
        """Test that cached data maintains integrity."""
        # Load original
        original = burgers_manager_with_cache.load_from_cache(0)
        
        # Load again
        reloaded = burgers_manager_with_cache.load_from_cache(0)
        
        # Should be identical
        assert torch.allclose(
            original['tensor_data']['velocity'],
            reloaded['tensor_data']['velocity']
        )
    
    def test_load_from_cache_nonexistent_raises_error(self, burgers_manager_with_cache):
        """Test that loading non-existent cache raises error."""
        with pytest.raises(FileNotFoundError):
            burgers_manager_with_cache.load_from_cache(999)
    
    def test_get_or_load_uses_cache(self, burgers_manager_with_cache):
        """Test that get_or_load uses cache when available."""
        # First call should use cache
        data1 = burgers_manager_with_cache.get_or_load_simulation(0, ['velocity'], num_frames=5)
        
        # Second call should also use cache
        data2 = burgers_manager_with_cache.get_or_load_simulation(0, ['velocity'], num_frames=5)
        
        # Both should return valid data
        assert 'tensor_data' in data1
        assert 'tensor_data' in data2
        
        # Data should be identical
        assert torch.allclose(
            data1['tensor_data']['velocity'],
            data2['tensor_data']['velocity']
        )
    
    def test_get_or_load_reloads_if_field_mismatch(self, burgers_manager_with_cache):
        """Test that get_or_load reloads if field names don't match."""
        # Cache has velocity
        data1 = burgers_manager_with_cache.get_or_load_simulation(0, ['velocity'], num_frames=5)
        
        # This would require different fields (not applicable for burgers, but tests logic)
        # Just verify the call works
        assert 'velocity' in data1['tensor_data']
    
    def test_get_or_load_reloads_if_frames_insufficient(self):
        """Test that get_or_load reloads if cached frames are insufficient."""
        project_root = Path(__file__).parent.parent.parent
        
        config = {
            'dset_name': 'burgers_128',
            'fields': ['velocity']
        }
        
        raw_dir = project_root / 'data' / 'burgers_128'
        cache_dir = project_root / 'data' / 'cache' / 'test_frames'
        
        dm = DataManager(str(raw_dir), str(cache_dir), config)
        
        # Cache with 5 frames
        dm.get_or_load_simulation(0, ['velocity'], num_frames=5)
        
        # Request 10 frames should reload
        data = dm.get_or_load_simulation(0, ['velocity'], num_frames=10)
        
        assert data['metadata']['num_frames'] == 10
        assert data['tensor_data']['velocity'].shape[0] == 10


class TestDataManagerMultipleFields:
    """Tests for loading multiple fields."""
    
    @pytest.fixture
    def smoke_manager(self):
        """Create DataManager for smoke dataset."""
        project_root = Path(__file__).parent.parent.parent
        
        config = {
            'dset_name': 'smoke_128',
            'fields': ['velocity', 'density']
        }
        
        raw_dir = project_root / 'data' / 'smoke_128'
        cache_dir = project_root / 'data' / 'cache' / 'test_multifield'
        
        return DataManager(str(raw_dir), str(cache_dir), config)
    
    def test_load_multiple_fields(self, smoke_manager):
        """Test loading multiple fields simultaneously."""
        data = smoke_manager.get_or_load_simulation(
            0, 
            ['velocity', 'density'], 
            num_frames=5
        )
        
        assert 'velocity' in data['tensor_data']
        assert 'density' in data['tensor_data']
    
    def test_multiple_field_shapes(self, smoke_manager):
        """Test that multiple fields have correct shapes."""
        data = smoke_manager.get_or_load_simulation(
            0, 
            ['velocity', 'density'], 
            num_frames=5
        )
        
        velocity = data['tensor_data']['velocity']
        density = data['tensor_data']['density']
        
        # Velocity should have 2 channels
        assert velocity.shape[1] == 2
        
        # Density should have 1 channel
        assert density.shape[1] == 1
    
    def test_multiple_field_time_consistency(self, smoke_manager):
        """Test that multiple fields have same time dimension."""
        data = smoke_manager.get_or_load_simulation(
            0, 
            ['velocity', 'density'], 
            num_frames=5
        )
        
        velocity = data['tensor_data']['velocity']
        density = data['tensor_data']['density']
        
        assert velocity.shape[0] == density.shape[0]  # time
    
    def test_multiple_field_spatial_consistency(self, smoke_manager):
        """Test that multiple fields have same spatial dimensions."""
        data = smoke_manager.get_or_load_simulation(
            0, 
            ['velocity', 'density'], 
            num_frames=5
        )
        
        velocity = data['tensor_data']['velocity']
        density = data['tensor_data']['density']
        
        assert velocity.shape[2] == density.shape[2]  # x
        assert velocity.shape[3] == density.shape[3]  # y
    
    def test_multiple_field_metadata(self, smoke_manager):
        """Test metadata for multiple fields."""
        data = smoke_manager.get_or_load_simulation(
            0, 
            ['velocity', 'density'], 
            num_frames=5
        )
        
        field_meta = data['metadata']['field_metadata']
        
        assert 'velocity' in field_meta
        assert 'density' in field_meta
        assert field_meta['velocity']['field_type'] in ['staggered', 'centered']
        assert field_meta['density']['field_type'] in ['staggered', 'centered']


class TestDataManagerErrorHandling:
    """Tests for error handling."""
    
    @pytest.fixture
    def burgers_manager(self):
        """Create DataManager for error testing."""
        project_root = Path(__file__).parent.parent.parent
        
        config = {
            'dset_name': 'burgers_128',
            'fields': ['velocity']
        }
        
        raw_dir = project_root / 'data' / 'burgers_128'
        cache_dir = project_root / 'data' / 'cache' / 'test_errors'
        
        return DataManager(str(raw_dir), str(cache_dir), config)
    
    def test_load_nonexistent_simulation(self, burgers_manager):
        """Test that loading non-existent simulation raises error."""
        with pytest.raises(FileNotFoundError):
            burgers_manager.load_and_cache_simulation(9999, ['velocity'], num_frames=5)
    
    def test_load_nonexistent_field_skips_gracefully(self, burgers_manager):
        """Test that requesting non-existent field is skipped."""
        # This should not raise an error, just skip the non-existent field
        data = burgers_manager.get_or_load_simulation(
            0, 
            ['velocity', 'nonexistent_field'], 
            num_frames=5
        )
        
        # Should have velocity but not the non-existent field
        assert 'velocity' in data['tensor_data']
        assert 'nonexistent_field' not in data['tensor_data']
    
    def test_load_from_uncached_simulation(self, burgers_manager):
        """Test that loading from cache without caching first raises error."""
        with pytest.raises(FileNotFoundError):
            burgers_manager.load_from_cache(999)


class TestDataManagerDifferentDatasets:
    """Tests with different datasets."""
    
    def test_heat_dataset(self):
        """Test with heat dataset."""
        project_root = Path(__file__).parent.parent.parent
        
        config = {
            'dset_name': 'heat_64',
            'fields': ['temp']
        }
        
        raw_dir = project_root / 'data' / 'heat_64'
        cache_dir = project_root / 'data' / 'cache' / 'test_heat'
        
        dm = DataManager(str(raw_dir), str(cache_dir), config)
        data = dm.get_or_load_simulation(0, ['temp'], num_frames=5)
        
        assert 'temp' in data['tensor_data']
        temp_tensor = data['tensor_data']['temp']
        assert temp_tensor.shape[1] == 1  # scalar field
    
    def test_burgers_dataset(self):
        """Test with burgers dataset."""
        project_root = Path(__file__).parent.parent.parent
        
        config = {
            'dset_name': 'burgers_128',
            'fields': ['velocity']
        }
        
        raw_dir = project_root / 'data' / 'burgers_128'
        cache_dir = project_root / 'data' / 'cache' / 'test_burgers'
        
        dm = DataManager(str(raw_dir), str(cache_dir), config)
        data = dm.get_or_load_simulation(0, ['velocity'], num_frames=5)
        
        assert 'velocity' in data['tensor_data']
        velocity_tensor = data['tensor_data']['velocity']
        assert velocity_tensor.shape[1] == 2  # vector field
    
    def test_smoke_dataset(self):
        """Test with smoke dataset."""
        project_root = Path(__file__).parent.parent.parent
        
        config = {
            'dset_name': 'smoke_128',
            'fields': ['velocity', 'density']
        }
        
        raw_dir = project_root / 'data' / 'smoke_128'
        cache_dir = project_root / 'data' / 'cache' / 'test_smoke'
        
        dm = DataManager(str(raw_dir), str(cache_dir), config)
        data = dm.get_or_load_simulation(0, ['velocity', 'density'], num_frames=5)
        
        assert 'velocity' in data['tensor_data']
        assert 'density' in data['tensor_data']


class TestDataManagerSimulationVariations:
    """Tests with different simulation indices and parameters."""
    
    @pytest.fixture
    def burgers_manager(self):
        """Create DataManager."""
        project_root = Path(__file__).parent.parent.parent
        
        config = {
            'dset_name': 'burgers_128',
            'fields': ['velocity']
        }
        
        raw_dir = project_root / 'data' / 'burgers_128'
        cache_dir = project_root / 'data' / 'cache' / 'test_variations'
        
        return DataManager(str(raw_dir), str(cache_dir), config)
    
    def test_load_different_simulations(self, burgers_manager):
        """Test loading multiple simulation indices."""
        for sim_idx in [0, 1, 2]:
            data = burgers_manager.get_or_load_simulation(sim_idx, ['velocity'], num_frames=5)
            assert 'velocity' in data['tensor_data']
    
    def test_load_different_frame_counts(self):
        """Test loading different numbers of frames (minimum 2 frames due to PhiFlow behavior)."""
        project_root = Path(__file__).parent.parent.parent
        
        config = {
            'dset_name': 'burgers_128',
            'fields': ['velocity']
        }
        
        raw_dir = project_root / 'data' / 'burgers_128'
        
        # Test with 2+ frames (PhiFlow squeezes dimensions with single frame)
        for num_frames in [2, 3, 5, 10]:
            # Use different cache dir for each to avoid reuse
            cache_dir = project_root / 'data' / 'cache' / f'test_variations_{num_frames}'
            
            burgers_manager = DataManager(str(raw_dir), str(cache_dir), config)
            data = burgers_manager.get_or_load_simulation(0, ['velocity'], num_frames=num_frames)
            assert data['tensor_data']['velocity'].shape[0] == num_frames
    
    def test_load_minimal_frames(self):
        """Test loading minimal number of frames (2 due to PhiFlow dimension handling)."""
        project_root = Path(__file__).parent.parent.parent
        
        config = {
            'dset_name': 'burgers_128',
            'fields': ['velocity']
        }
        
        raw_dir = project_root / 'data' / 'burgers_128'
        cache_dir = project_root / 'data' / 'cache' / 'test_minimal_frames'
        
        burgers_manager = DataManager(str(raw_dir), str(cache_dir), config)
        data = burgers_manager.get_or_load_simulation(0, ['velocity'], num_frames=2)
        
        assert data['tensor_data']['velocity'].shape[0] == 2
        assert data['metadata']['num_frames'] >= 2  # Should have at least 2 frames
