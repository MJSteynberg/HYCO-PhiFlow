"""
Comprehensive tests for HybridDataset.
Tests initialization, data loading, field handling, tensor shapes, and DataLoader compatibility.
"""

import pytest
import tempfile
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.data import DataManager, HybridDataset


class TestHybridDatasetInitialization:
    """Tests for HybridDataset initialization."""
    
    @pytest.fixture
    def data_manager(self):
        """Create a DataManager for testing."""
        project_root = Path(__file__).parent.parent.parent
        
        config = {
            'dset_name': 'burgers_128'
        }
        
        raw_dir = project_root / 'data' / 'burgers_128'
        cache_dir = project_root / 'data' / 'cache' / 'test_hybrid_init'
        
        return DataManager(str(raw_dir), str(cache_dir), config)
    
    def test_basic_initialization(self, data_manager):
        """Test that HybridDataset can be initialized."""
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=[0],
            field_names=['velocity'],
            num_frames=10,
            num_predict_steps=5
        )
        
        assert dataset is not None
    
    def test_initialization_stores_parameters(self, data_manager):
        """Test that initialization stores parameters correctly."""
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=[0, 1],
            field_names=['velocity'],
            num_frames=10,
            num_predict_steps=5
        )
        
        assert dataset.data_manager == data_manager
        assert dataset.sim_indices == [0, 1]
        assert dataset.field_names == ['velocity']
        assert dataset.num_frames == 10
        assert dataset.num_predict_steps == 5
    
    def test_initialization_default_dynamic_fields(self, data_manager):
        """Test that all fields are dynamic by default."""
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=[0],
            field_names=['velocity'],
            num_frames=10,
            num_predict_steps=5
        )
        
        assert dataset.dynamic_fields == ['velocity']
        assert dataset.static_fields == []
    
    def test_initialization_explicit_dynamic_fields(self, data_manager):
        """Test initialization with explicit dynamic fields."""
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=[0],
            field_names=['velocity'],
            num_frames=10,
            num_predict_steps=5,
            dynamic_fields=['velocity']
        )
        
        assert dataset.dynamic_fields == ['velocity']
    
    def test_initialization_with_static_fields(self, data_manager):
        """Test initialization with static fields."""
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=[0],
            field_names=['velocity'],
            num_frames=10,
            num_predict_steps=5,
            static_fields=[]
        )
        
        assert dataset.static_fields == []
    
    def test_initialization_validation_error(self, data_manager):
        """Test that error is raised for invalid parameters."""
        with pytest.raises(ValueError, match="num_frames.*must be"):
            HybridDataset(
                data_manager=data_manager,
                sim_indices=[0],
                field_names=['velocity'],
                num_frames=5,  # Too few
                num_predict_steps=10  # Too many
            )
    
    def test_initialization_caches_data(self, data_manager):
        """Test that initialization pre-caches data."""
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=[0],
            field_names=['velocity'],
            num_frames=10,
            num_predict_steps=5
        )
        
        # Data should be cached after initialization
        assert data_manager.is_cached(0)


class TestHybridDatasetLength:
    """Tests for dataset length."""
    
    @pytest.fixture
    def data_manager(self):
        """Create a DataManager for testing."""
        project_root = Path(__file__).parent.parent.parent
        
        config = {
            'dset_name': 'burgers_128'
        }
        
        raw_dir = project_root / 'data' / 'burgers_128'
        cache_dir = project_root / 'data' / 'cache' / 'test_hybrid_length'
        
        return DataManager(str(raw_dir), str(cache_dir), config)
    
    def test_length_single_sim(self, data_manager):
        """Test length with single simulation."""
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=[0],
            field_names=['velocity'],
            num_frames=10,
            num_predict_steps=5
        )
        
        assert len(dataset) == 1
    
    def test_length_multiple_sims(self, data_manager):
        """Test length with multiple simulations."""
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=[0, 1, 2],
            field_names=['velocity'],
            num_frames=10,
            num_predict_steps=5
        )
        
        assert len(dataset) == 3
    
    def test_length_many_sims(self, data_manager):
        """Test length with many simulations."""
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=[0, 1, 2, 3],
            field_names=['velocity'],
            num_frames=10,
            num_predict_steps=5
        )
        
        assert len(dataset) == 4


class TestHybridDatasetGetItem:
    """Tests for __getitem__ method."""
    
    @pytest.fixture
    def dataset(self):
        """Create a dataset for testing."""
        project_root = Path(__file__).parent.parent.parent
        
        config = {
            'dset_name': 'burgers_128'
        }
        
        raw_dir = project_root / 'data' / 'burgers_128'
        cache_dir = project_root / 'data' / 'cache' / 'test_hybrid_getitem'
        
        data_manager = DataManager(str(raw_dir), str(cache_dir), config)
        
        return HybridDataset(
            data_manager=data_manager,
            sim_indices=[0],
            field_names=['velocity'],
            num_frames=10,
            num_predict_steps=5
        )
    
    def test_getitem_returns_tuple(self, dataset):
        """Test that __getitem__ returns a tuple."""
        result = dataset[0]
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_getitem_returns_tensors(self, dataset):
        """Test that __getitem__ returns tensors."""
        initial_state, rollout_targets = dataset[0]
        
        assert isinstance(initial_state, torch.Tensor)
        assert isinstance(rollout_targets, torch.Tensor)
    
    def test_getitem_initial_state_shape(self, dataset):
        """Test initial state has correct shape [C, H, W]."""
        initial_state, _ = dataset[0]
        
        assert initial_state.dim() == 3
        assert initial_state.shape[0] == 2  # velocity has 2 components
        assert initial_state.shape[1] == 128  # height
        assert initial_state.shape[2] == 128  # width
    
    def test_getitem_rollout_targets_shape(self, dataset):
        """Test rollout targets have correct shape [T, C, H, W]."""
        _, rollout_targets = dataset[0]
        
        assert rollout_targets.dim() == 4
        assert rollout_targets.shape[0] == 5  # num_predict_steps
        assert rollout_targets.shape[1] == 2  # velocity has 2 components
        assert rollout_targets.shape[2] == 128  # height
        assert rollout_targets.shape[3] == 128  # width
    
    def test_getitem_tensor_dtypes(self, dataset):
        """Test that tensors have correct dtype."""
        initial_state, rollout_targets = dataset[0]
        
        assert initial_state.dtype == torch.float32
        assert rollout_targets.dtype == torch.float32
    
    def test_getitem_tensor_values_finite(self, dataset):
        """Test that tensor values are finite."""
        initial_state, rollout_targets = dataset[0]
        
        assert torch.isfinite(initial_state).all()
        assert torch.isfinite(rollout_targets).all()
    
    def test_getitem_multiple_calls_consistent(self, dataset):
        """Test that multiple calls return consistent data."""
        initial_1, rollout_1 = dataset[0]
        initial_2, rollout_2 = dataset[0]
        
        assert torch.equal(initial_1, initial_2)
        assert torch.equal(rollout_1, rollout_2)


class TestHybridDatasetMultipleFields:
    """Tests for datasets with multiple fields."""
    
    @pytest.fixture
    def smoke_dataset(self):
        """Create a smoke dataset with multiple fields."""
        project_root = Path(__file__).parent.parent.parent
        
        config = {
            'dset_name': 'smoke_128'
        }
        
        raw_dir = project_root / 'data' / 'smoke_128'
        cache_dir = project_root / 'data' / 'cache' / 'test_hybrid_multifield'
        
        data_manager = DataManager(str(raw_dir), str(cache_dir), config)
        
        return HybridDataset(
            data_manager=data_manager,
            sim_indices=[0],
            field_names=['velocity', 'density'],
            num_frames=10,
            num_predict_steps=5
        )
    
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
        assert initial_state.dim() == 3
        assert initial_state.shape[1] == 128
        assert initial_state.shape[2] == 128
        
        # Rollout: [T, C, H, W]
        assert rollout_targets.dim() == 4
        assert rollout_targets.shape[0] == 5  # num_predict_steps
        assert rollout_targets.shape[2] == 128
        assert rollout_targets.shape[3] == 128
    
    def test_multiple_fields_all_dynamic(self, smoke_dataset):
        """Test that all fields are treated as dynamic."""
        assert 'velocity' in smoke_dataset.dynamic_fields
        assert 'density' in smoke_dataset.dynamic_fields
        assert len(smoke_dataset.static_fields) == 0


class TestHybridDatasetStaticFields:
    """Tests for static vs dynamic fields."""
    
    @pytest.fixture
    def data_manager(self):
        """Create a DataManager."""
        project_root = Path(__file__).parent.parent.parent
        
        config = {
            'dset_name': 'smoke_128'
        }
        
        raw_dir = project_root / 'data' / 'smoke_128'
        cache_dir = project_root / 'data' / 'cache' / 'test_hybrid_static'
        
        return DataManager(str(raw_dir), str(cache_dir), config)
    
    def test_static_fields_in_input_only(self, data_manager):
        """Test that static fields appear in input but not output."""
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=[0],
            field_names=['velocity', 'density'],
            num_frames=10,
            num_predict_steps=5,
            dynamic_fields=['velocity'],
            static_fields=['density']
        )
        
        initial_state, rollout_targets = dataset[0]
        
        # Initial state should have all fields: velocity (2) + density (1) = 3
        assert initial_state.shape[0] == 3
        
        # Rollout targets should have only dynamic fields: velocity (2)
        assert rollout_targets.shape[1] == 2
    
    def test_dynamic_fields_identification(self, data_manager):
        """Test that dynamic fields are correctly identified."""
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=[0],
            field_names=['velocity', 'density'],
            num_frames=10,
            num_predict_steps=5,
            dynamic_fields=['velocity', 'density']
        )
        
        assert set(dataset.dynamic_fields) == {'velocity', 'density'}
    
    def test_static_fields_identification(self, data_manager):
        """Test that static fields are correctly identified."""
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=[0],
            field_names=['velocity', 'density'],
            num_frames=10,
            num_predict_steps=5,
            static_fields=['density']
        )
        
        assert 'density' in dataset.static_fields
        assert 'velocity' in dataset.dynamic_fields


class TestHybridDatasetDataLoader:
    """Tests for DataLoader compatibility."""
    
    @pytest.fixture
    def dataset(self):
        """Create a dataset for testing."""
        project_root = Path(__file__).parent.parent.parent
        
        config = {
            'dset_name': 'burgers_128'
        }
        
        raw_dir = project_root / 'data' / 'burgers_128'
        cache_dir = project_root / 'data' / 'cache' / 'test_hybrid_dataloader'
        
        data_manager = DataManager(str(raw_dir), str(cache_dir), config)
        
        return HybridDataset(
            data_manager=data_manager,
            sim_indices=[0, 1],
            field_names=['velocity'],
            num_frames=10,
            num_predict_steps=5
        )
    
    def test_dataloader_creation(self, dataset):
        """Test that DataLoader can be created."""
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False
        )
        
        assert dataloader is not None
    
    def test_dataloader_iteration(self, dataset):
        """Test that DataLoader can be iterated."""
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False
        )
        
        batch_count = 0
        for initial_batch, rollout_batch in dataloader:
            batch_count += 1
        
        assert batch_count == 2  # 2 simulations
    
    def test_dataloader_batch_dimension(self, dataset):
        """Test that batch dimension is added."""
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False
        )
        
        initial_batch, rollout_batch = next(iter(dataloader))
        
        # Check batch dimension added
        assert initial_batch.shape[0] == 1  # batch size
        assert rollout_batch.shape[0] == 1  # batch size
    
    def test_dataloader_preserves_other_dimensions(self, dataset):
        """Test that other dimensions are preserved."""
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False
        )
        
        initial_batch, rollout_batch = next(iter(dataloader))
        
        # Check other dimensions preserved
        assert initial_batch.shape[1] == 2  # channels
        assert rollout_batch.shape[1] == 5  # time steps
        assert rollout_batch.shape[2] == 2  # channels
    
    def test_dataloader_batch_size_2(self, dataset):
        """Test DataLoader with batch size 2."""
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False
        )
        
        initial_batch, rollout_batch = next(iter(dataloader))
        
        assert initial_batch.shape[0] == 2  # batch size
        assert rollout_batch.shape[0] == 2  # batch size
    
    def test_dataloader_shuffle(self, dataset):
        """Test DataLoader with shuffling."""
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True
        )
        
        # Should still be able to iterate
        batch_count = 0
        for initial_batch, rollout_batch in dataloader:
            batch_count += 1
        
        assert batch_count == 2


class TestHybridDatasetDifferentParameters:
    """Tests with different parameter combinations."""
    
    @pytest.fixture
    def data_manager(self):
        """Create a DataManager."""
        project_root = Path(__file__).parent.parent.parent
        
        config = {
            'dset_name': 'burgers_128'
        }
        
        raw_dir = project_root / 'data' / 'burgers_128'
        cache_dir = project_root / 'data' / 'cache' / 'test_hybrid_params'
        
        return DataManager(str(raw_dir), str(cache_dir), config)
    
    def test_different_num_predict_steps(self, data_manager):
        """Test with different num_predict_steps."""
        for num_steps in [1, 3, 5, 10]:
            dataset = HybridDataset(
                data_manager=data_manager,
                sim_indices=[0],
                field_names=['velocity'],
                num_frames=15,
                num_predict_steps=num_steps
            )
            
            _, rollout_targets = dataset[0]
            assert rollout_targets.shape[0] == num_steps
    
    def test_different_num_frames(self, data_manager):
        """Test with different num_frames."""
        for num_frames in [10, 15, 20]:
            dataset = HybridDataset(
                data_manager=data_manager,
                sim_indices=[0],
                field_names=['velocity'],
                num_frames=num_frames,
                num_predict_steps=5
            )
            
            # Should succeed
            initial_state, rollout_targets = dataset[0]
            assert initial_state is not None
    
    def test_minimal_configuration(self, data_manager):
        """Test with minimal valid configuration."""
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=[0],
            field_names=['velocity'],
            num_frames=2,  # Minimum: num_predict_steps + 1
            num_predict_steps=1
        )
        
        initial_state, rollout_targets = dataset[0]
        
        assert initial_state.shape[0] == 2  # velocity channels
        assert rollout_targets.shape[0] == 1  # 1 predict step


class TestHybridDatasetFieldsMode:
    """Tests for return_fields mode (PhiFlow Fields instead of tensors)."""
    
    @pytest.fixture
    def data_manager(self):
        """Create a DataManager."""
        project_root = Path(__file__).parent.parent.parent
        
        config = {
            'dset_name': 'burgers_128'
        }
        
        raw_dir = project_root / 'data' / 'burgers_128'
        cache_dir = project_root / 'data' / 'cache' / 'test_hybrid_fields'
        
        return DataManager(str(raw_dir), str(cache_dir), config)
    
    def test_return_fields_mode(self, data_manager):
        """Test that return_fields mode returns Fields."""
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
    
    def test_return_fields_has_correct_keys(self, data_manager):
        """Test that Fields mode returns correct keys."""
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=[0],
            field_names=['velocity'],
            num_frames=10,
            num_predict_steps=5,
            return_fields=True
        )
        
        initial_fields, target_fields = dataset[0]
        
        assert 'velocity' in initial_fields
        assert 'velocity' in target_fields
    
    def test_return_fields_target_is_list(self, data_manager):
        """Test that target fields are lists of Fields."""
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=[0],
            field_names=['velocity'],
            num_frames=10,
            num_predict_steps=5,
            return_fields=True
        )
        
        initial_fields, target_fields = dataset[0]
        
        # Target fields should be lists (one Field per timestep)
        assert isinstance(target_fields['velocity'], list)
        assert len(target_fields['velocity']) == 5  # num_predict_steps


class TestHybridDatasetIndexing:
    """Tests for dataset indexing."""
    
    @pytest.fixture
    def dataset(self):
        """Create a dataset with multiple simulations."""
        project_root = Path(__file__).parent.parent.parent
        
        config = {
            'dset_name': 'burgers_128'
        }
        
        raw_dir = project_root / 'data' / 'burgers_128'
        cache_dir = project_root / 'data' / 'cache' / 'test_hybrid_indexing'
        
        data_manager = DataManager(str(raw_dir), str(cache_dir), config)
        
        return HybridDataset(
            data_manager=data_manager,
            sim_indices=[0, 1, 2],
            field_names=['velocity'],
            num_frames=10,
            num_predict_steps=5
        )
    
    def test_index_0(self, dataset):
        """Test accessing index 0."""
        initial_state, rollout_targets = dataset[0]
        
        assert initial_state.shape[0] == 2
        assert rollout_targets.shape[0] == 5
    
    def test_index_1(self, dataset):
        """Test accessing index 1."""
        initial_state, rollout_targets = dataset[1]
        
        assert initial_state.shape[0] == 2
        assert rollout_targets.shape[0] == 5
    
    def test_index_2(self, dataset):
        """Test accessing index 2."""
        initial_state, rollout_targets = dataset[2]
        
        assert initial_state.shape[0] == 2
        assert rollout_targets.shape[0] == 5
    
    def test_all_indices_accessible(self, dataset):
        """Test that all indices can be accessed."""
        for idx in range(len(dataset)):
            initial_state, rollout_targets = dataset[idx]
            assert initial_state is not None
            assert rollout_targets is not None
