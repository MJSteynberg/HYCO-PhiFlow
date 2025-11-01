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
from src.training.abstract_trainer import AbstractTrainer


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
        class CustomTrainer(AbstractTrainer):
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
                },
                'data_dir': 'data/',
                'dset_name': 'burgers_128',
                'fields': ['velocity']
            },
            'model': {
                'synthetic': {
                    'name': 'UNet',
                    'model_path': 'results/models',
                    'model_save_name': 'test_synthetic',
                    'input_specs': {'velocity': 2},
                    'output_specs': {'velocity': 2},
                    'architecture': {
                        'levels': 2,
                        'filters': 16
                    }
                }
            },
            'run_params': {
                'model_type': 'synthetic'
            },
            'trainer_params': {
                'learning_rate': 0.001,
                'batch_size': 2,
                'epochs': 1,
                'num_predict_steps': 2,
                'train_sim': [0]
            }
        })
    
    def test_create_synthetic_trainer(self, synthetic_config):
        """Test creating a synthetic trainer."""
        trainer = TrainerFactory.create_trainer(synthetic_config)
        
        assert isinstance(trainer, SyntheticTrainer)
        assert isinstance(trainer, AbstractTrainer)
    
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
                },
                'data_dir': 'data',
                'dset_name': 'heat_64',
                'fields': ['temp'],
                'cache_dir': 'data/cache'
            },
            'model': {
                'physical': {
                    'name': 'HeatModel',
                    'model_path': 'results/models',
                    'model_save_name': 'test_physical',
                    'domain': {'size_x': 100.0, 'size_y': 100.0},
                    'resolution': {'x': 64, 'y': 64},
                    'dt': 0.1,
                    'pde_params': {}
                }
            },
            'run_params': {
                'model_type': 'physical'
            },
            'trainer_params': {
                'learning_rate': 0.001,
                'batch_size': 16,
                'epochs': 1,
                'train_sim': [0],
                'num_predict_steps': 5,
                'learnable_parameters': [
                    {'name': 'diffusivity', 'initial_guess': 0.5}
                ]
            }
        })
    
    def test_create_physical_trainer(self, physical_config):
        """Test creating a physical trainer."""
        trainer = TrainerFactory.create_trainer(physical_config)
        
        assert isinstance(trainer, PhysicalTrainer)
        assert isinstance(trainer, AbstractTrainer)
    
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
        """Test that physical trainer is created successfully."""
        trainer = TrainerFactory.create_trainer(physical_config)
        # Physical trainers don't have a device attribute (they use PhiFlow)
        assert hasattr(trainer, 'model')
        assert trainer.model is not None


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
                },
                'data_dir': 'data/',
                'dset_name': 'burgers_128',
                'fields': ['velocity']
            },
            'model': {
                'synthetic': {
                    'name': 'UNet',
                    'model_path': 'results/models',
                    'model_save_name': 'test',
                    'input_specs': {'velocity': 2},
                    'output_specs': {'velocity': 2},
                    'architecture': {
                        'levels': 2,
                        'filters': 16
                    }
                }
            },
            'run_params': {
                'model_type': 'synthetic'
            },
            'trainer_params': {
                'learning_rate': 0.001,
                'batch_size': 2,
                'epochs': 1,
                'num_predict_steps': 2,
                'train_sim': [0]
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
                },
                'data_dir': 'data',
                'dset_name': 'heat_64',
                'fields': ['temp'],
                'cache_dir': 'data/cache'
            },
            'model': {
                'physical': {
                    'name': 'HeatModel',
                    'model_path': 'results/models',
                    'model_save_name': 'test',
                    'domain': {'size_x': 100.0, 'size_y': 100.0},
                    'resolution': {'x': 64, 'y': 64},
                    'dt': 0.1,
                    'pde_params': {}
                }
            },
            'run_params': {
                'model_type': 'physical'
            },
            'trainer_params': {
                'learning_rate': 0.001,
                'batch_size': 16,
                'epochs': 1,
                'train_sim': [0],
                'num_predict_steps': 5,
                'learnable_parameters': [
                    {'name': 'diffusivity', 'initial_guess': 0.5}
                ]
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
                'cache_validation': {'enabled': True},
                'data_dir': 'data/',
                'dset_name': 'burgers_128',
                'fields': ['velocity']
            },
            'model': {
                'synthetic': {
                    'name': 'UNet',
                    'model_path': 'results/models',
                    'model_save_name': 'trainer1',
                    'input_specs': {'velocity': 2},
                    'output_specs': {'velocity': 2},
                    'architecture': {
                        'levels': 2,
                        'filters': 16
                    }
                }
            },
            'run_params': {
                'model_type': 'synthetic'
            },
            'trainer_params': {
                'learning_rate': 0.001,
                'batch_size': 2,
                'epochs': 1,
                'num_predict_steps': 2,
                'train_sim': [0]
            }
        })
        
        config2 = DictConfig({
            'project_root': str(Path.cwd()),
            'data': {
                'problem': 'smoke',
                'resolution': 64,
                'cache_validation': {'enabled': True},
                'data_dir': 'data',
                'dset_name': 'heat_64',
                'fields': ['temp'],
                'cache_dir': 'data/cache'
            },
            'model': {
                'physical': {
                    'name': 'HeatModel',
                    'model_path': 'results/models',
                    'model_save_name': 'trainer2',
                    'domain': {'size_x': 100.0, 'size_y': 100.0},
                    'resolution': {'x': 64, 'y': 64},
                    'dt': 0.1,
                    'pde_params': {}
                }
            },
            'run_params': {
                'model_type': 'physical'
            },
            'trainer_params': {
                'learning_rate': 0.002,
                'batch_size': 32,
                'epochs': 1,
                'train_sim': [0],
                'num_predict_steps': 5,
                'learnable_parameters': [
                    {'name': 'diffusivity', 'initial_guess': 0.5}
                ]
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
