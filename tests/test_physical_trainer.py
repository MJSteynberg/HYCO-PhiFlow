"""
Test Physical Trainer with HybridDataset integration
"""

import pytest
from src.training.physical.trainer import PhysicalTrainer


class TestPhysicalTrainer:
    """Test PhysicalTrainer basic functionality."""
    
    @pytest.fixture
    def config(self):
        """Basic configuration for testing."""
        return {
            'project_root': '.',
            'data': {
                'data_dir': 'data',
                'dset_name': 'heat_64',
                'fields': ['temp'],
                'cache_dir': 'data/cache'
            },
            'model': {
                'physical': {
                    'name': 'HeatModel',
                    'domain': {'size_x': 1.0, 'size_y': 1.0},
                    'resolution': {'x': 64, 'y': 64},
                    'dt': 0.01,
                    'pde_params': {}
                }
            },
            'trainer_params': {
                'train_sim': [0],
                'epochs': 2,
                'num_predict_steps': 5,
                'learnable_parameters': [
                    {'name': 'diffusivity', 'initial_guess': 0.1}
                ]
            }
        }
    
    def test_trainer_initialization(self, config):
        """Test that trainer initializes correctly with DataManager."""
        trainer = PhysicalTrainer(config)
        
        # Check trainer attributes
        assert trainer.data_manager is not None
        assert trainer.model is not None
        assert len(trainer.initial_guesses) == 1
        assert trainer.num_predict_steps == 5
    
    def test_data_loading_with_fields(self, config):
        """Test that ground truth data loads as Fields."""
        trainer = PhysicalTrainer(config)
        
        # Load ground truth data
        gt_data = trainer._load_ground_truth_data(0)
        
        # Check data structure
        assert 'temp' in gt_data
        assert hasattr(gt_data['temp'], 'shape')
        assert hasattr(gt_data['temp'], 'values')
        
        # Check dimensions (should have batch and time)
        shape = gt_data['temp'].shape
        assert 'batch' in str(shape)
        assert 'time' in str(shape)
