"""
Comprehensive tests for PhysicalTrainer.
Tests inverse problem solving, parameter optimization, and data integration.
"""

import pytest
import os
import torch
from pathlib import Path
from phi.flow import Box, spatial, math
from phi.math import Tensor

from src.training.physical.trainer import PhysicalTrainer
from src.models.physical import BurgersModel, HeatModel, SmokeModel


class TestPhysicalTrainerInitialization:
    """Tests for PhysicalTrainer initialization."""
    
    @pytest.fixture
    def heat_config(self):
        """Basic configuration for HeatModel testing."""
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
                    'domain': {'size_x': 100.0, 'size_y': 100.0},
                    'resolution': {'x': 64, 'y': 64},
                    'dt': 0.1,
                    'pde_params': {}
                }
            },
            'trainer_params': {
                'train_sim': [0],
                'epochs': 2,
                'num_predict_steps': 5,
                'learnable_parameters': [
                    {'name': 'diffusivity', 'initial_guess': 0.5}
                ]
            }
        }
    
    @pytest.fixture
    def burgers_config(self):
        """Configuration for BurgersModel testing."""
        return {
            'project_root': '.',
            'data': {
                'data_dir': 'data',
                'dset_name': 'burgers_128',
                'fields': ['velocity'],
                'cache_dir': 'data/cache'
            },
            'model': {
                'physical': {
                    'name': 'BurgersModel',
                    'domain': {'size_x': 1.0, 'size_y': 1.0},
                    'resolution': {'x': 128, 'y': 128},
                    'dt': 0.01,
                    'pde_params': {
                        'nu': 0.01,
                        'batch_size': 1
                    }
                }
            },
            'trainer_params': {
                'train_sim': [0],
                'epochs': 3,
                'num_predict_steps': 10,
                'learnable_parameters': [
                    {'name': 'nu', 'initial_guess': 0.02}
                ]
            }
        }
    
    def test_basic_initialization(self, heat_config):
        """Test basic trainer initialization."""
        trainer = PhysicalTrainer(heat_config)
        
        assert trainer is not None
        assert trainer.config == heat_config
        assert trainer.project_root == '.'
    
    def test_config_parsing(self, heat_config):
        """Test that config sections are parsed correctly."""
        trainer = PhysicalTrainer(heat_config)
        
        assert trainer.data_config is not None
        assert trainer.model_config is not None
        assert trainer.trainer_config is not None
    
    def test_parameter_extraction(self, heat_config):
        """Test extraction of training parameters."""
        trainer = PhysicalTrainer(heat_config)
        
        assert trainer.train_sims == [0]
        assert trainer.num_epochs == 2
        assert trainer.num_predict_steps == 5
    
    def test_data_manager_creation(self, heat_config):
        """Test that DataManager is created successfully."""
        trainer = PhysicalTrainer(heat_config)
        
        assert trainer.data_manager is not None
        assert hasattr(trainer.data_manager, 'raw_data_dir')
        assert hasattr(trainer.data_manager, 'cache_dir')
    
    def test_initial_guesses_setup(self, heat_config):
        """Test that initial guesses are set up correctly."""
        trainer = PhysicalTrainer(heat_config)
        
        assert len(trainer.initial_guesses) == 1
        assert isinstance(trainer.initial_guesses[0], Tensor)
    
    def test_model_creation(self, heat_config):
        """Test that physical model is created."""
        trainer = PhysicalTrainer(heat_config)
        
        assert trainer.model is not None
        assert hasattr(trainer.model, 'step')
        assert hasattr(trainer.model, 'get_initial_state')
    
    def test_multiple_learnable_parameters(self):
        """Test initialization with multiple learnable parameters."""
        config = {
            'project_root': '.',
            'data': {
                'data_dir': 'data',
                'dset_name': 'smoke_128',
                'fields': ['velocity', 'density'],
                'cache_dir': 'data/cache'
            },
            'model': {
                'physical': {
                    'name': 'SmokeModel',
                    'domain': {'size_x': 80.0, 'size_y': 80.0},
                    'resolution': {'x': 128, 'y': 128},
                    'dt': 1.0,
                    'pde_params': {}
                }
            },
            'trainer_params': {
                'train_sim': [0],
                'epochs': 2,
                'num_predict_steps': 5,
                'learnable_parameters': [
                    {'name': 'nu', 'initial_guess': 0.0},
                    {'name': 'buoyancy', 'initial_guess': 0.5}
                ]
            }
        }
        
        trainer = PhysicalTrainer(config)
        assert len(trainer.initial_guesses) == 2
    
    def test_burgers_model_initialization(self, burgers_config):
        """Test initialization with BurgersModel."""
        trainer = PhysicalTrainer(burgers_config)
        
        assert isinstance(trainer.model, BurgersModel)
        assert trainer.model.nu is not None
    
    def test_gt_fields_parsing(self, heat_config):
        """Test ground truth fields are parsed correctly."""
        trainer = PhysicalTrainer(heat_config)
        
        assert trainer.gt_fields == ['temp']
        assert 'temp' in trainer.gt_fields
    
    def test_learnable_params_config_storage(self, heat_config):
        """Test learnable parameters config is stored."""
        trainer = PhysicalTrainer(heat_config)
        
        assert len(trainer.learnable_params_config) == 1
        assert trainer.learnable_params_config[0]['name'] == 'diffusivity'
        assert trainer.learnable_params_config[0]['initial_guess'] == 0.5


class TestPhysicalTrainerDataLoading:
    """Tests for ground truth data loading."""
    
    @pytest.fixture
    def trainer(self):
        """Create a trainer instance for testing."""
        config = {
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
                    'domain': {'size_x': 100.0, 'size_y': 100.0},
                    'resolution': {'x': 64, 'y': 64},
                    'dt': 0.1,
                    'pde_params': {}
                }
            },
            'trainer_params': {
                'train_sim': [0],
                'epochs': 2,
                'num_predict_steps': 5,
                'learnable_parameters': [
                    {'name': 'diffusivity', 'initial_guess': 0.5}
                ]
            }
        }
        return PhysicalTrainer(config)
    
    def test_load_ground_truth_structure(self, trainer):
        """Test that ground truth data loads with correct structure."""
        gt_data = trainer._load_ground_truth_data(0)
        
        assert isinstance(gt_data, dict)
        assert 'temp' in gt_data
    
    def test_ground_truth_is_field(self, trainer):
        """Test that loaded data are PhiFlow Fields."""
        gt_data = trainer._load_ground_truth_data(0)
        
        for field_name, field_value in gt_data.items():
            assert hasattr(field_value, 'shape')
            assert hasattr(field_value, 'values')
    
    def test_ground_truth_dimensions(self, trainer):
        """Test that ground truth has correct dimensions."""
        gt_data = trainer._load_ground_truth_data(0)
        
        for field_name, field_value in gt_data.items():
            shape = field_value.shape
            # Should have batch and time dimensions
            assert 'batch' in str(shape).lower() or shape.batch.volume > 0
            assert 'time' in str(shape).lower() or hasattr(shape, 'time')
    
    def test_ground_truth_time_length(self, trainer):
        """Test that ground truth has correct number of time steps."""
        gt_data = trainer._load_ground_truth_data(0)
        
        expected_steps = trainer.num_predict_steps + 1  # Initial + rollout
        
        for field_name, field_value in gt_data.items():
            # Access time dimension
            if hasattr(field_value.shape, 'time'):
                assert field_value.shape.time == expected_steps
    
    def test_ground_truth_spatial_resolution(self, trainer):
        """Test that ground truth has correct spatial resolution."""
        gt_data = trainer._load_ground_truth_data(0)
        
        for field_name, field_value in gt_data.items():
            shape = field_value.shape
            assert shape.spatial.volume == trainer.model.resolution.volume
    
    def test_true_pde_params_loading(self, trainer):
        """Test that true PDE parameters are loaded from metadata."""
        gt_data = trainer._load_ground_truth_data(0)
        
        # Check that true_pde_params attribute exists
        assert hasattr(trainer, 'true_pde_params')
        assert isinstance(trainer.true_pde_params, dict)


class TestPhysicalTrainerLossFunction:
    """Tests for loss function and optimization."""
    
    @pytest.fixture
    def trainer(self):
        """Create a simple trainer for loss testing."""
        config = {
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
                    'domain': {'size_x': 100.0, 'size_y': 100.0},
                    'resolution': {'x': 64, 'y': 64},
                    'dt': 0.1,
                    'pde_params': {}
                }
            },
            'trainer_params': {
                'train_sim': [0],
                'epochs': 2,
                'num_predict_steps': 3,  # Short for testing
                'learnable_parameters': [
                    {'name': 'diffusivity', 'initial_guess': 1.0}
                ]
            }
        }
        return PhysicalTrainer(config)
    
    def test_model_has_learnable_parameter(self, trainer):
        """Test that model has the learnable parameter set."""
        assert hasattr(trainer.model, 'diffusivity')
    
    def test_model_step_produces_state(self, trainer):
        """Test that model can perform a step."""
        initial_state = trainer.model.get_initial_state()
        next_state = trainer.model.step(initial_state)
        
        assert isinstance(next_state, dict)
        assert len(next_state) > 0
    
    def test_parameter_update_affects_model(self, trainer):
        """Test that updating parameter affects model."""
        # Get initial parameter value
        initial_param = trainer.model.diffusivity
        
        # Set new value
        new_value = math.tensor(2.0)
        trainer.model.diffusivity = new_value
        
        # Check it changed - math.close returns a boolean tensor
        assert not math.all(math.close(trainer.model.diffusivity, initial_param))


class TestPhysicalTrainerTraining:
    """Tests for training/optimization process."""
    
    @pytest.fixture
    def simple_config(self):
        """Minimal config for quick training test."""
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
                    'domain': {'size_x': 100.0, 'size_y': 100.0},
                    'resolution': {'x': 64, 'y': 64},
                    'dt': 0.1,
                    'pde_params': {}
                }
            },
            'trainer_params': {
                'train_sim': [0],
                'epochs': 1,  # Just 1 iteration
                'num_predict_steps': 2,  # Minimal rollout
                'learnable_parameters': [
                    {'name': 'diffusivity', 'initial_guess': 1.0}
                ]
            }
        }
    
    def test_train_method_exists(self, simple_config):
        """Test that train method exists and is callable."""
        trainer = PhysicalTrainer(simple_config)
        assert hasattr(trainer, 'train')
        assert callable(trainer.train)
    
    def test_training_completes_without_error(self, simple_config):
        """Test that training runs to completion."""
        trainer = PhysicalTrainer(simple_config)
        
        # This should complete without raising exceptions
        try:
            trainer.train()
            training_completed = True
        except Exception as e:
            training_completed = False
            print(f"Training failed with: {e}")
        
        assert training_completed


class TestPhysicalTrainerMultipleSimulations:
    """Tests for handling multiple simulations."""
    
    def test_multiple_sims_in_config(self):
        """Test configuration with multiple training simulations."""
        config = {
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
                    'domain': {'size_x': 100.0, 'size_y': 100.0},
                    'resolution': {'x': 64, 'y': 64},
                    'dt': 0.1,
                    'pde_params': {}
                }
            },
            'trainer_params': {
                'train_sim': [0, 1, 2],  # Multiple sims
                'epochs': 2,
                'num_predict_steps': 5,
                'learnable_parameters': [
                    {'name': 'diffusivity', 'initial_guess': 0.5}
                ]
            }
        }
        
        trainer = PhysicalTrainer(config)
        assert len(trainer.train_sims) == 3
        assert trainer.train_sims == [0, 1, 2]


class TestPhysicalTrainerErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_no_learnable_parameters_raises_error(self):
        """Test that missing learnable parameters raises error."""
        config = {
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
                    'domain': {'size_x': 100.0, 'size_y': 100.0},
                    'resolution': {'x': 64, 'y': 64},
                    'dt': 0.1,
                    'pde_params': {}
                }
            },
            'trainer_params': {
                'train_sim': [0],
                'epochs': 2,
                'num_predict_steps': 5,
                'learnable_parameters': []  # Empty!
            }
        }
        
        with pytest.raises(ValueError, match="No 'learnable_parameters'"):
            trainer = PhysicalTrainer(config)
    
    def test_invalid_model_name_raises_error(self):
        """Test that invalid model name raises error."""
        config = {
            'project_root': '.',
            'data': {
                'data_dir': 'data',
                'dset_name': 'heat_64',
                'fields': ['temp'],
                'cache_dir': 'data/cache'
            },
            'model': {
                'physical': {
                    'name': 'NonExistentModel',  # Invalid!
                    'domain': {'size_x': 100.0, 'size_y': 100.0},
                    'resolution': {'x': 64, 'y': 64},
                    'dt': 0.1,
                    'pde_params': {}
                }
            },
            'trainer_params': {
                'train_sim': [0],
                'epochs': 2,
                'num_predict_steps': 5,
                'learnable_parameters': [
                    {'name': 'diffusivity', 'initial_guess': 0.5}
                ]
            }
        }
        
        with pytest.raises(ImportError, match="Model .* not found"):
            trainer = PhysicalTrainer(config)


class TestPhysicalTrainerModelSpecific:
    """Tests for specific physical models."""
    
    def test_burgers_model_trainer(self):
        """Test trainer with BurgersModel."""
        config = {
            'project_root': '.',
            'data': {
                'data_dir': 'data',
                'dset_name': 'burgers_128',
                'fields': ['velocity'],
                'cache_dir': 'data/cache'
            },
            'model': {
                'physical': {
                    'name': 'BurgersModel',
                    'domain': {'size_x': 1.0, 'size_y': 1.0},
                    'resolution': {'x': 128, 'y': 128},
                    'dt': 0.01,
                    'pde_params': {
                        'batch_size': 1
                    }
                }
            },
            'trainer_params': {
                'train_sim': [0],
                'epochs': 2,
                'num_predict_steps': 5,
                'learnable_parameters': [
                    {'name': 'nu', 'initial_guess': 0.01}
                ]
            }
        }
        
        trainer = PhysicalTrainer(config)
        assert isinstance(trainer.model, BurgersModel)
        assert hasattr(trainer.model, 'nu')
    
    def test_smoke_model_trainer(self):
        """Test trainer with SmokeModel."""
        config = {
            'project_root': '.',
            'data': {
                'data_dir': 'data',
                'dset_name': 'smoke_128',
                'fields': ['velocity', 'density'],
                'cache_dir': 'data/cache'
            },
            'model': {
                'physical': {
                    'name': 'SmokeModel',
                    'domain': {'size_x': 80.0, 'size_y': 80.0},
                    'resolution': {'x': 128, 'y': 128},
                    'dt': 1.0,
                    'pde_params': {
                        'inflow_center': (40.0, 20.0)
                    }
                }
            },
            'trainer_params': {
                'train_sim': [0],
                'epochs': 2,
                'num_predict_steps': 5,
                'learnable_parameters': [
                    {'name': 'buoyancy', 'initial_guess': 1.0}
                ]
            }
        }
        
        trainer = PhysicalTrainer(config)
        assert isinstance(trainer.model, SmokeModel)
        assert hasattr(trainer.model, 'buoyancy')
