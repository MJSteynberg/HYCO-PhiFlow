"""
Tests for TrainerFactory (Phase 3: Improved Modularity).

Tests the factory pattern for creating trainers based on model type,
including synthetic and physical trainers.
"""

import pytest
from pathlib import Path
from omegaconf import DictConfig

from src.factories.trainer_factory import TrainerFactory
from src.training.synthetic.trainer import SyntheticTrainer
from src.training.physical.trainer import PhysicalTrainer
from src.training.base_trainer import BaseTrainer


class TestTrainerFactoryRegistration:
    """Tests for trainer registration."""
    
    def test_synthetic_trainer_is_registered(self):
        """Test that SyntheticTrainer is registered."""
        available_trainers = TrainerFactory.list_available_trainers()
        assert 'synthetic' in available_trainers
    
    def test_physical_trainer_is_registered(self):
        """Test that PhysicalTrainer is registered."""
        available_trainers = TrainerFactory.list_available_trainers()
        assert 'physical' in available_trainers
    
    def test_list_available_trainers_returns_list(self):
        """Test that list_available_trainers returns a list."""
        available_trainers = TrainerFactory.list_available_trainers()
        assert isinstance(available_trainers, list)
    
    def test_register_trainer_manually(self):
        """Test manually registering a trainer."""
        # Create a dummy trainer class
        class CustomTrainer(BaseTrainer):
            def _create_model(self):
                pass
            def _create_data_loader(self):
                pass
            def _train_epoch(self):
                pass
            def train(self):
                pass
        
        # Register it
        TrainerFactory.register_trainer('custom', CustomTrainer)
        
        # Check it's available
        available_trainers = TrainerFactory.list_available_trainers()
        assert 'custom' in available_trainers
        
        # Clean up
        del TrainerFactory._trainers['custom']


class TestTrainerFactoryCreateSynthetic:
    """Tests for creating synthetic trainers."""
    
    @pytest.fixture
    def synthetic_config(self):
        """Configuration for synthetic trainer."""
        return DictConfig({
            'project_root': str(Path.cwd()),
            'data': {
                'problem': 'burgers',
                'resolution': 128,
                'cache_validation': {
                    'enabled': True
                }
            },
            'model': {
                'synthetic': {
                    'model_name': 'unet',
                    'model_path': 'results/models',
                    'model_save_name': 'test_synthetic',
                    'in_channels': 1,
                    'out_channels': 1,
                    'base_channels': 32
                }
            },
            'run_params': {
                'model_type': 'synthetic'
            },
            'trainer_params': {
                'learning_rate': 0.001,
                'batch_size': 16,
                'epochs': 10,
                'val_split': 0.2,
                'num_workers': 0
            }
        })
    
    def test_create_synthetic_trainer(self, synthetic_config):
        """Test creating a synthetic trainer."""
        trainer = TrainerFactory.create_trainer(synthetic_config)
        
        assert isinstance(trainer, SyntheticTrainer)
        assert isinstance(trainer, BaseTrainer)
    
    def test_synthetic_trainer_has_config(self, synthetic_config):
        """Test that created trainer has the config."""
        trainer = TrainerFactory.create_trainer(synthetic_config)
        assert trainer.config == synthetic_config
    
    def test_create_synthetic_with_explicit_type(self, synthetic_config):
        """Test creating synthetic trainer with explicit model_type."""
        synthetic_config['run_params']['model_type'] = 'synthetic'
        trainer = TrainerFactory.create_trainer(synthetic_config)
        
        assert isinstance(trainer, SyntheticTrainer)
    
    def test_synthetic_trainer_device_set(self, synthetic_config):
        """Test that synthetic trainer has device set."""
        trainer = TrainerFactory.create_trainer(synthetic_config)
        assert hasattr(trainer, 'device')
        assert trainer.device.type in ['cuda', 'cpu']


class TestTrainerFactoryCreatePhysical:
    """Tests for creating physical trainers."""
    
    @pytest.fixture
    def physical_config(self):
        """Configuration for physical trainer."""
        return DictConfig({
            'project_root': str(Path.cwd()),
            'data': {
                'problem': 'burgers',
                'resolution': 128,
                'cache_validation': {
                    'enabled': True
                }
            },
            'model': {
                'physical': {
                    'model_name': 'spectral',
                    'model_path': 'results/models',
                    'model_save_name': 'test_physical'
                }
            },
            'run_params': {
                'model_type': 'physical'
            },
            'trainer_params': {
                'learning_rate': 0.001,
                'batch_size': 16,
                'epochs': 10,
                'num_workers': 0
            }
        })
    
    def test_create_physical_trainer(self, physical_config):
        """Test creating a physical trainer."""
        trainer = TrainerFactory.create_trainer(physical_config)
        
        assert isinstance(trainer, PhysicalTrainer)
        assert isinstance(trainer, BaseTrainer)
    
    def test_physical_trainer_has_config(self, physical_config):
        """Test that created trainer has the config."""
        trainer = TrainerFactory.create_trainer(physical_config)
        assert trainer.config == physical_config
    
    def test_create_physical_with_explicit_type(self, physical_config):
        """Test creating physical trainer with explicit model_type."""
        physical_config['run_params']['model_type'] = 'physical'
        trainer = TrainerFactory.create_trainer(physical_config)
        
        assert isinstance(trainer, PhysicalTrainer)
    
    def test_physical_trainer_device_set(self, physical_config):
        """Test that physical trainer has device set."""
        trainer = TrainerFactory.create_trainer(physical_config)
        assert hasattr(trainer, 'device')
        assert trainer.device.type in ['cuda', 'cpu']


class TestTrainerFactoryErrorHandling:
    """Tests for error handling in trainer factory."""
    
    def test_unknown_model_type_raises_error(self):
        """Test that unknown model_type raises ValueError."""
        config = DictConfig({
            'project_root': str(Path.cwd()),
            'model': {
                'synthetic': {
                    'model_path': 'results/models',
                    'model_save_name': 'test'
                }
            },
            'run_params': {
                'model_type': 'unknown_type'
            },
            'trainer_params': {}
        })
        
        with pytest.raises(ValueError, match="Unknown model_type"):
            TrainerFactory.create_trainer(config)
    
    def test_missing_run_params_raises_error(self):
        """Test that missing run_params raises KeyError."""
        config = DictConfig({
            'project_root': str(Path.cwd()),
            'model': {
                'synthetic': {
                    'model_path': 'results/models',
                    'model_save_name': 'test'
                }
            },
            'trainer_params': {}
        })
        
        with pytest.raises(KeyError):
            TrainerFactory.create_trainer(config)


class TestTrainerFactoryIntegration:
    """Integration tests for trainer factory."""
    
    def test_factory_and_direct_instantiation_equivalent_synthetic(self):
        """Test that factory creates equivalent trainer to direct instantiation."""
        config = DictConfig({
            'project_root': str(Path.cwd()),
            'data': {
                'problem': 'burgers',
                'resolution': 128,
                'cache_validation': {
                    'enabled': True
                }
            },
            'model': {
                'synthetic': {
                    'model_name': 'unet',
                    'model_path': 'results/models',
                    'model_save_name': 'test',
                    'in_channels': 1,
                    'out_channels': 1,
                    'base_channels': 32
                }
            },
            'run_params': {
                'model_type': 'synthetic'
            },
            'trainer_params': {
                'learning_rate': 0.001,
                'batch_size': 16,
                'epochs': 10,
                'val_split': 0.2,
                'num_workers': 0
            }
        })
        
        # Create via factory
        factory_trainer = TrainerFactory.create_trainer(config)
        
        # Create directly
        direct_trainer = SyntheticTrainer(config)
        
        # Both should be same type
        assert type(factory_trainer) == type(direct_trainer)
        assert factory_trainer.config == direct_trainer.config
    
    def test_factory_and_direct_instantiation_equivalent_physical(self):
        """Test that factory creates equivalent trainer to direct instantiation."""
        config = DictConfig({
            'project_root': str(Path.cwd()),
            'data': {
                'problem': 'burgers',
                'resolution': 128,
                'cache_validation': {
                    'enabled': True
                }
            },
            'model': {
                'physical': {
                    'model_name': 'spectral',
                    'model_path': 'results/models',
                    'model_save_name': 'test'
                }
            },
            'run_params': {
                'model_type': 'physical'
            },
            'trainer_params': {
                'learning_rate': 0.001,
                'batch_size': 16,
                'epochs': 10,
                'num_workers': 0
            }
        })
        
        # Create via factory
        factory_trainer = TrainerFactory.create_trainer(config)
        
        # Create directly
        direct_trainer = PhysicalTrainer(config)
        
        # Both should be same type
        assert type(factory_trainer) == type(direct_trainer)
        assert factory_trainer.config == direct_trainer.config
    
    def test_multiple_trainers_independent(self):
        """Test that multiple trainers created by factory are independent."""
        config1 = DictConfig({
            'project_root': str(Path.cwd()),
            'data': {
                'problem': 'burgers',
                'resolution': 128,
                'cache_validation': {'enabled': True}
            },
            'model': {
                'synthetic': {
                    'model_name': 'unet',
                    'model_path': 'results/models',
                    'model_save_name': 'trainer1',
                    'in_channels': 1,
                    'out_channels': 1,
                    'base_channels': 32
                }
            },
            'run_params': {
                'model_type': 'synthetic'
            },
            'trainer_params': {
                'learning_rate': 0.001,
                'batch_size': 16,
                'epochs': 10,
                'val_split': 0.2,
                'num_workers': 0
            }
        })
        
        config2 = DictConfig({
            'project_root': str(Path.cwd()),
            'data': {
                'problem': 'smoke',
                'resolution': 64,
                'cache_validation': {'enabled': True}
            },
            'model': {
                'physical': {
                    'model_name': 'spectral',
                    'model_path': 'results/models',
                    'model_save_name': 'trainer2'
                }
            },
            'run_params': {
                'model_type': 'physical'
            },
            'trainer_params': {
                'learning_rate': 0.002,
                'batch_size': 32,
                'epochs': 20,
                'num_workers': 0
            }
        })
        
        trainer1 = TrainerFactory.create_trainer(config1)
        trainer2 = TrainerFactory.create_trainer(config2)
        
        # Trainers should be different types
        assert type(trainer1) != type(trainer2)
        assert isinstance(trainer1, SyntheticTrainer)
        assert isinstance(trainer2, PhysicalTrainer)
        
        # Configs should be independent
        assert trainer1.config is not trainer2.config
        assert trainer1.config['model']['synthetic']['model_save_name'] == 'trainer1'
        assert trainer2.config['model']['physical']['model_save_name'] == 'trainer2'
    
    def test_list_available_trainers_complete(self):
        """Test that all expected trainers are available."""
        available_trainers = TrainerFactory.list_available_trainers()
        
        # Should have at least synthetic and physical
        assert 'synthetic' in available_trainers
        assert 'physical' in available_trainers
