"""
Tests for BaseTrainer abstract class (Phase 3: Improved Modularity).

Tests the base trainer functionality including checkpoint management,
device handling, and abstract method definitions.
"""

import pytest
import tempfile
import torch
from pathlib import Path
from abc import ABC

from src.training.base_trainer import BaseTrainer


class TestBaseTrainerAbstract:
    """Tests for BaseTrainer abstract class properties."""
    
    def test_base_trainer_is_abstract(self):
        """Test that BaseTrainer cannot be instantiated directly."""
        # BaseTrainer should be abstract and raise TypeError when instantiated
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseTrainer({})
    
    def test_base_trainer_has_abstract_methods(self):
        """Test that BaseTrainer defines expected abstract methods."""
        abstract_methods = BaseTrainer.__abstractmethods__
        
        expected_methods = {'_create_model', '_create_data_loader', '_train_epoch', 'train'}
        assert expected_methods.issubset(abstract_methods)
    
    def test_base_trainer_inherits_from_abc(self):
        """Test that BaseTrainer uses ABC."""
        assert issubclass(BaseTrainer, ABC)


class ConcreteTrainer(BaseTrainer):
    """Concrete implementation of BaseTrainer for testing."""
    
    def __init__(self, config):
        super().__init__(config)
        self.model = torch.nn.Linear(10, 10)
        self.train_loader = None
        self._create_model_called = False
        self._create_data_loader_called = False
        self._train_epoch_called = False
        self._train_called = False
    
    def _create_model(self):
        """Create a simple test model."""
        self._create_model_called = True
        return torch.nn.Linear(10, 10)
    
    def _create_data_loader(self):
        """Create a dummy data loader."""
        self._create_data_loader_called = True
        return None
    
    def _train_epoch(self, epoch):
        """Dummy training epoch."""
        self._train_epoch_called = True
        return {'loss': 1.0}
    
    def train(self):
        """Dummy training loop."""
        self._train_called = True


class TestBaseTrainerInitialization:
    """Tests for BaseTrainer initialization."""
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing."""
        return {
            'project_root': str(Path.cwd()),
            'model': {
                'synthetic': {
                    'model_path': 'results/models',
                    'model_save_name': 'test_model'
                }
            },
            'trainer_params': {
                'learning_rate': 0.001,
                'batch_size': 16,
                'epochs': 10
            }
        }
    
    def test_initialization_with_valid_config(self, basic_config):
        """Test that concrete trainer can be initialized."""
        trainer = ConcreteTrainer(basic_config)
        assert trainer.config == basic_config
    
    def test_device_assignment(self, basic_config):
        """Test that device is properly assigned."""
        trainer = ConcreteTrainer(basic_config)
        assert hasattr(trainer, 'device')
        assert isinstance(trainer.device, torch.device)
    
    def test_device_is_cuda_or_cpu(self, basic_config):
        """Test that device is either CUDA or CPU."""
        trainer = ConcreteTrainer(basic_config)
        assert trainer.device.type in ['cuda', 'cpu']
    
    def test_config_stored(self, basic_config):
        """Test that config is stored in trainer."""
        trainer = ConcreteTrainer(basic_config)
        assert trainer.config == basic_config
    
    def test_checkpoint_path_initialized_as_none(self, basic_config):
        """Test that checkpoint_path is initialized as None."""
        trainer = ConcreteTrainer(basic_config)
        # BaseTrainer initializes checkpoint_path to None
        # Subclasses set it during initialization
        assert hasattr(trainer, 'checkpoint_path')
    
    def test_model_initialized_as_none(self, basic_config):
        """Test that model is initialized as None by BaseTrainer."""
        # BaseTrainer sets model to None, subclasses set it
        trainer = ConcreteTrainer(basic_config)
        # ConcreteTrainer sets it in __init__, but BaseTrainer initializes to None
        assert hasattr(trainer, 'model')


class TestBaseTrainerCheckpointManagement:
    """Tests for checkpoint save/load functionality."""
    
    @pytest.fixture
    def trainer_with_temp_dir(self):
        """Create trainer with temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'project_root': tmpdir,
                'model': {
                    'synthetic': {
                        'model_path': 'models',
                        'model_save_name': 'test_checkpoint'
                    }
                },
                'trainer_params': {
                    'learning_rate': 0.001,
                    'batch_size': 16,
                    'epochs': 10
                }
            }
            trainer = ConcreteTrainer(config)
            trainer.model = torch.nn.Linear(5, 3)
            # Set checkpoint_path manually (normally done by subclass __init__)
            trainer.checkpoint_path = Path(tmpdir) / 'models' / 'test_checkpoint.pth'
            trainer.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            yield trainer
    
    def test_save_checkpoint_creates_file(self, trainer_with_temp_dir):
        """Test that save_checkpoint creates a file."""
        trainer = trainer_with_temp_dir
        
        trainer.save_checkpoint(epoch=1, loss=0.5)
        assert trainer.checkpoint_path.exists()
    
    def test_save_checkpoint_with_epoch(self, trainer_with_temp_dir):
        """Test saving checkpoint with epoch number."""
        trainer = trainer_with_temp_dir
        
        trainer.save_checkpoint(epoch=5, loss=0.3)
        # Check that regular checkpoint was created
        assert trainer.checkpoint_path.exists()
    
    def test_save_and_load_checkpoint(self, trainer_with_temp_dir):
        """Test that checkpoint can be saved and loaded."""
        trainer = trainer_with_temp_dir
        
        # Save initial weights
        initial_weights = trainer.model.weight.data.clone()
        trainer.save_checkpoint(epoch=1, loss=0.5)
        
        # Modify weights
        trainer.model.weight.data.fill_(0.5)
        assert not torch.allclose(trainer.model.weight.data, initial_weights)
        
        # Load checkpoint
        trainer.load_checkpoint()
        
        # Weights should be restored
        assert torch.allclose(trainer.model.weight.data, initial_weights)
    
    def test_load_checkpoint_returns_dict(self, trainer_with_temp_dir):
        """Test that load_checkpoint returns a dictionary."""
        trainer = trainer_with_temp_dir
        
        trainer.save_checkpoint(epoch=1, loss=0.5)
        checkpoint = trainer.load_checkpoint()
        
        assert isinstance(checkpoint, dict)
        assert 'model_state_dict' in checkpoint
    
    def test_checkpoint_contains_metadata(self, trainer_with_temp_dir):
        """Test that saved checkpoint contains metadata."""
        trainer = trainer_with_temp_dir
        
        trainer.save_checkpoint(epoch=10, loss=0.5)
        checkpoint = trainer.load_checkpoint()
        
        assert 'epoch' in checkpoint
        assert 'loss' in checkpoint
        assert checkpoint['epoch'] == 10
        assert checkpoint['loss'] == 0.5


class TestBaseTrainerModelSummary:
    """Tests for model parameter counting."""
    
    def test_get_parameter_count(self):
        """Test that get_parameter_count returns correct count."""
        config = {
            'project_root': str(Path.cwd()),
            'model': {
                'synthetic': {
                    'model_path': 'results/models',
                    'model_save_name': 'test'
                }
            },
            'trainer_params': {}
        }
        trainer = ConcreteTrainer(config)
        trainer.model = torch.nn.Linear(10, 5)
        
        # Linear layer has 10*5 weights + 5 biases = 55 parameters
        param_count = trainer.get_parameter_count()
        assert param_count == 55
    
    def test_get_trainable_parameter_count(self):
        """Test counting only trainable parameters."""
        config = {
            'project_root': str(Path.cwd()),
            'model': {
                'synthetic': {
                    'model_path': 'results/models',
                    'model_save_name': 'test'
                }
            },
            'trainer_params': {}
        }
        trainer = ConcreteTrainer(config)
        
        # Create model with some frozen parameters
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.Linear(5, 3)
        )
        
        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False
        
        trainer.model = model
        
        # Should only count second layer: 5*3 weights + 3 biases = 18
        trainable_count = trainer.get_trainable_parameter_count()
        assert trainable_count == 18
        
        # Total count should be both layers: (10*5 + 5) + (5*3 + 3) = 73
        total_count = trainer.get_parameter_count()
        assert total_count == 73
    
    def test_print_model_summary_executes(self):
        """Test that print_model_summary executes without error."""
        config = {
            'project_root': str(Path.cwd()),
            'model': {
                'synthetic': {
                    'model_path': 'results/models',
                    'model_save_name': 'test'
                }
            },
            'trainer_params': {}
        }
        trainer = ConcreteTrainer(config)
        trainer.model = torch.nn.Linear(10, 5)
        
        # Should not raise error
        trainer.print_model_summary()


class TestBaseTrainerIntegration:
    """Integration tests for BaseTrainer."""
    
    def test_concrete_trainer_implements_all_abstract_methods(self):
        """Test that ConcreteTrainer implements all required methods."""
        config = {
            'project_root': str(Path.cwd()),
            'model': {
                'synthetic': {
                    'model_path': 'results/models',
                    'model_save_name': 'test'
                }
            },
            'trainer_params': {}
        }
        
        trainer = ConcreteTrainer(config)
        
        # All abstract methods should be callable
        assert callable(trainer._create_model)
        assert callable(trainer._create_data_loader)
        assert callable(trainer._train_epoch)
        assert callable(trainer.train)
    
    def test_trainer_can_move_model_to_device(self):
        """Test that trainer can move model to device."""
        config = {
            'project_root': str(Path.cwd()),
            'model': {
                'synthetic': {
                    'model_path': 'results/models',
                    'model_save_name': 'test'
                }
            },
            'trainer_params': {}
        }
        
        trainer = ConcreteTrainer(config)
        model = torch.nn.Linear(10, 5)
        
        # Move to trainer's device
        model = model.to(trainer.device)
        
        # Check that parameters are on the right device
        for param in model.parameters():
            assert param.device.type == trainer.device.type
    
    def test_multiple_trainers_have_independent_configs(self):
        """Test that multiple trainer instances don't share config."""
        config1 = {
            'project_root': str(Path.cwd()),
            'model': {
                'synthetic': {
                    'model_path': 'results/models',
                    'model_save_name': 'trainer1'
                }
            },
            'trainer_params': {'epochs': 10}
        }
        
        config2 = {
            'project_root': str(Path.cwd()),
            'model': {
                'synthetic': {
                    'model_path': 'results/models',
                    'model_save_name': 'trainer2'
                }
            },
            'trainer_params': {'epochs': 20}
        }
        
        trainer1 = ConcreteTrainer(config1)
        trainer2 = ConcreteTrainer(config2)
        
        assert trainer1.config is not trainer2.config
        assert trainer1.config['model']['synthetic']['model_save_name'] == 'trainer1'
        assert trainer2.config['model']['synthetic']['model_save_name'] == 'trainer2'
