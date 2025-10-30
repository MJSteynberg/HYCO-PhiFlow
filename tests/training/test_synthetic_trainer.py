"""
Tests for SyntheticTrainer with new DataManager pipeline
"""

import pytest
import tempfile
import torch
from pathlib import Path
import yaml

from src.training.synthetic.trainer import SyntheticTrainer


@pytest.fixture
def burgers_config():
    """Create a minimal config for testing."""
    project_root = Path(__file__).parent.parent.parent
    
    config = {
        'project_root': str(project_root),
        'run_params': {
            'experiment_name': 'test_burgers',
            'mode': ['train'],
            'model_type': 'synthetic'
        },
        'data': {
            'data_dir': 'data/',
            'dset_name': 'burgers_128',
            'fields': ['velocity'],
            'fields_scheme': 'VV'
        },
        'model': {
            'synthetic': {
                'name': 'UNet',
                'model_path': 'results/models',
                'model_save_name': 'test_burgers_unet',
                'input_specs': {
                    'velocity': 2
                },
                'output_specs': {
                    'velocity': 2
                },
                'architecture': {
                    'levels': 2,  # Smaller for testing
                    'filters': 16,  # Smaller for testing
                    'batch_norm': True
                }
            }
        },
        'trainer_params': {
            'learning_rate': 1.0e-4,
            'batch_size': 2,  # Small batch for testing
            'epochs': 2,  # Just 2 epochs for testing
            'num_predict_steps': 3,  # Short rollout for testing
            'train_sim': [0]
        }
    }
    return config


class TestSyntheticTrainer:
    """Test cases for the new tensor-based SyntheticTrainer."""
    
    def test_trainer_initialization(self, burgers_config):
        """Test that trainer can be initialized."""
        trainer = SyntheticTrainer(burgers_config)
        
        assert trainer.device is not None
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.train_loader is not None
    
    def test_model_architecture(self, burgers_config):
        """Test that UNet is created with correct architecture."""
        trainer = SyntheticTrainer(burgers_config)
        
        assert trainer.model.in_channels == 2  # velocity has 2 components
        assert trainer.model.out_channels == 2
    
    def test_channel_map_building(self, burgers_config):
        """Test that channel map is built correctly."""
        trainer = SyntheticTrainer(burgers_config)
        
        assert 'velocity' in trainer.channel_map
        assert trainer.channel_map['velocity'] == (0, 2)
        assert trainer.total_channels == 2
    
    def test_data_loader_created(self, burgers_config):
        """Test that DataLoader is created successfully."""
        trainer = SyntheticTrainer(burgers_config)
        
        assert len(trainer.train_loader) > 0
        
        # Get a batch
        initial_state, rollout_targets = next(iter(trainer.train_loader))
        
        # Check shapes
        assert initial_state.shape[0] <= burgers_config['trainer_params']['batch_size']
        assert initial_state.shape[1] == 2  # velocity channels
        assert rollout_targets.shape[1] == 3  # num_predict_steps
        assert rollout_targets.shape[2] == 2  # velocity channels
    
    def test_unpack_tensor_to_dict(self, burgers_config):
        """Test tensor unpacking into field dictionary."""
        trainer = SyntheticTrainer(burgers_config)
        
        # Create a dummy tensor [B, C, H, W]
        dummy_tensor = torch.randn(2, 2, 128, 128)
        
        field_dict = trainer._unpack_tensor_to_dict(dummy_tensor)
        
        assert 'velocity' in field_dict
        assert field_dict['velocity'].shape == (2, 2, 128, 128)
    
    def test_single_training_step(self, burgers_config):
        """Test that a single training step completes without errors."""
        burgers_config['trainer_params']['epochs'] = 1
        trainer = SyntheticTrainer(burgers_config)
        
        # Run one training step
        initial_loss = trainer._train_epoch()
        
        assert isinstance(initial_loss, float)
        assert initial_loss > 0  # Loss should be positive
    
    def test_model_forward_pass(self, burgers_config):
        """Test that model forward pass works with tensor input."""
        trainer = SyntheticTrainer(burgers_config)
        
        # Create dummy input [B, C, H, W]
        dummy_input = torch.randn(2, 2, 128, 128).to(trainer.device)
        
        # Forward pass
        with torch.no_grad():
            output = trainer.model(dummy_input)
        
        # Check output shape
        assert output.shape == (2, 2, 128, 128)


class TestSyntheticTrainerMultiField:
    """Test trainer with multiple fields (like smoke dataset)."""
    
    def test_multi_field_channel_map(self):
        """Test channel mapping for multiple fields."""
        config = {
            'project_root': '.',
            'data': {
                'data_dir': 'data/',
                'dset_name': 'smoke_128',
                'fields': ['velocity', 'density']
            },
            'model': {
                'synthetic': {
                    'input_specs': {'velocity': 2, 'density': 1},
                    'output_specs': {'velocity': 2, 'density': 1},
                    'architecture': {'levels': 2, 'filters': 16, 'batch_norm': True},
                    'model_path': 'results/models',
                    'model_save_name': 'test_smoke'
                }
            },
            'trainer_params': {
                'learning_rate': 1e-4,
                'batch_size': 2,
                'epochs': 1,
                'num_predict_steps': 2,
                'train_sim': [0]
            }
        }
        
        trainer = SyntheticTrainer(config)
        
        assert trainer.channel_map['velocity'] == (0, 2)
        assert trainer.channel_map['density'] == (2, 3)
        assert trainer.total_channels == 3
        assert trainer.model.in_channels == 3
        assert trainer.model.out_channels == 3
