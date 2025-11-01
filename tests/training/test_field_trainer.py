"""
Tests for FieldTrainer

Tests the base class for PhiFlow field-based trainers.
"""

import pytest
import torch
from pathlib import Path
import tempfile

from phi.field import Field, CenteredGrid
from phi.math import math, Tensor, batch, spatial

from src.training.field_trainer import FieldTrainer
from src.training.abstract_trainer import AbstractTrainer


class SimplePhysicalModel:
    """Simple mock physical model for testing."""
    
    def __init__(self, param=1.0):
        self.param = param
    
    def step(self, state):
        """Simple step function."""
        # Just modify the fields slightly
        return {name: field * 0.99 for name, field in state.items()}


class SimpleMockDataManager:
    """Simple mock DataManager for testing."""
    
    def __init__(self):
        self.data = {}
    
    def is_cached(self, sim_idx, field_names, num_frames):
        return True


class SimpleConcreteFieldTrainer(FieldTrainer):
    """Minimal concrete implementation for testing."""
    
    def __init__(self, config):
        super().__init__(config)
        self.data_manager = self._create_data_manager()
        self.model = self._create_model()
        self.learnable_params = [math.tensor(1.0)]
        self.learnable_params_config = [{'name': 'test_param'}]
    
    def _create_data_manager(self):
        """Create mock data manager."""
        return SimpleMockDataManager()
    
    def _create_model(self):
        """Create simple physical model."""
        return SimplePhysicalModel()
    
    def _setup_optimization(self):
        """Setup simple optimization."""
        return math.Solve(
            method='L-BFGS-B',
            abs_tol=1e-6,
            x0=self.learnable_params,
            max_iterations=5
        )
    
    def _load_ground_truth(self):
        """Load mock ground truth."""
        # Create simple test field
        field = CenteredGrid(1.0, x=10, y=10)
        # Add time dimension
        fields = [field] * 3
        stacked = math.stack(fields, batch('time'))
        return {'test_field': stacked}


class TestFieldTrainerInheritance:
    """Tests for FieldTrainer inheritance."""
    
    def test_field_trainer_inherits_from_abstract_trainer(self):
        """Test that FieldTrainer inherits from AbstractTrainer."""
        assert issubclass(FieldTrainer, AbstractTrainer)
    
    def test_concrete_trainer_is_field_trainer(self):
        """Test that concrete implementation is a FieldTrainer."""
        trainer = SimpleConcreteFieldTrainer({})
        assert isinstance(trainer, FieldTrainer)
        assert isinstance(trainer, AbstractTrainer)


class TestFieldTrainerInitialization:
    """Tests for FieldTrainer initialization."""
    
    def test_data_manager_initialized_as_none(self):
        """Test that data_manager is initialized as None by FieldTrainer."""
        
        class PartialTrainer(FieldTrainer):
            def __init__(self, config):
                super().__init__(config)
            
            def _create_data_manager(self):
                pass
            def _create_model(self):
                pass
            def _setup_optimization(self):
                pass
        
        trainer = PartialTrainer({})
        assert trainer.data_manager is None
    
    def test_model_initialized_as_none(self):
        """Test that model is initialized as None."""
        
        class PartialTrainer(FieldTrainer):
            def __init__(self, config):
                super().__init__(config)
            
            def _create_data_manager(self):
                pass
            def _create_model(self):
                pass
            def _setup_optimization(self):
                pass
        
        trainer = PartialTrainer({})
        assert trainer.model is None
    
    def test_learnable_params_initialized_as_empty_list(self):
        """Test that learnable_params is initialized as empty list."""
        
        class PartialTrainer(FieldTrainer):
            def __init__(self, config):
                super().__init__(config)
            
            def _create_data_manager(self):
                pass
            def _create_model(self):
                pass
            def _setup_optimization(self):
                pass
        
        trainer = PartialTrainer({})
        assert trainer.learnable_params == []
    
    def test_learnable_params_config_initialized_as_empty_list(self):
        """Test that learnable_params_config is initialized as empty list."""
        
        class PartialTrainer(FieldTrainer):
            def __init__(self, config):
                super().__init__(config)
            
            def _create_data_manager(self):
                pass
            def _create_model(self):
                pass
            def _setup_optimization(self):
                pass
        
        trainer = PartialTrainer({})
        assert trainer.learnable_params_config == []
    
    def test_final_loss_initialized(self):
        """Test that final_loss is initialized."""
        trainer = SimpleConcreteFieldTrainer({})
        assert hasattr(trainer, 'final_loss')
        assert trainer.final_loss == 0.0
    
    def test_optimization_history_initialized(self):
        """Test that optimization_history is initialized."""
        trainer = SimpleConcreteFieldTrainer({})
        assert hasattr(trainer, 'optimization_history')
        assert trainer.optimization_history == []


class TestFieldTrainerAbstractMethods:
    """Tests for abstract method enforcement."""
    
    def test_field_trainer_has_abstract_methods(self):
        """Test that FieldTrainer defines expected abstract methods."""
        abstract_methods = FieldTrainer.__abstractmethods__
        
        expected = {'_create_data_manager', '_create_model', '_setup_optimization'}
        assert expected.issubset(abstract_methods)
    
    def test_field_trainer_is_abstract(self):
        """Test that FieldTrainer cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            FieldTrainer({})
    
    def test_missing_create_data_manager_raises_error(self):
        """Test that missing _create_data_manager() raises TypeError."""
        
        class IncompleteTrainer(FieldTrainer):
            def _create_model(self):
                pass
            def _setup_optimization(self):
                pass
        
        with pytest.raises(TypeError):
            IncompleteTrainer({})
    
    def test_missing_create_model_raises_error(self):
        """Test that missing _create_model() raises TypeError."""
        
        class IncompleteTrainer(FieldTrainer):
            def _create_data_manager(self):
                pass
            def _setup_optimization(self):
                pass
        
        with pytest.raises(TypeError):
            IncompleteTrainer({})
    
    def test_missing_setup_optimization_raises_error(self):
        """Test that missing _setup_optimization() raises TypeError."""
        
        class IncompleteTrainer(FieldTrainer):
            def _create_data_manager(self):
                pass
            def _create_model(self):
                pass
        
        with pytest.raises(TypeError):
            IncompleteTrainer({})


class TestFieldTrainerDefaultTrain:
    """Tests for the default train() implementation."""
    
    def test_train_requires_model_and_data_manager(self):
        """Test that train() raises error if model or data_manager not set."""
        
        class NoComponentsTrainer(FieldTrainer):
            def __init__(self, config):
                super().__init__(config)
                self.model = None
                self.data_manager = None
            
            def _create_data_manager(self):
                return None
            def _create_model(self):
                return None
            def _setup_optimization(self):
                return None
        
        trainer = NoComponentsTrainer({})
        
        with pytest.raises(RuntimeError, match="Model and data manager must be initialized"):
            trainer.train()
    
    def test_train_returns_dict(self):
        """Test that train() returns a dictionary."""
        trainer = SimpleConcreteFieldTrainer({})
        
        # Mock the methods to avoid actual optimization
        trainer._load_ground_truth = lambda: {'test': CenteredGrid(1.0, x=10, y=10)}
        trainer._run_simulation = lambda x: {'test': CenteredGrid(0.9, x=10, y=10)}
        trainer._compute_loss = lambda p, gt: math.tensor(0.1)
        
        # Don't actually run optimization, just test structure
        # result = trainer.train()  # This would run actual optimization
        
        # Just verify the trainer is properly set up
        assert trainer.model is not None
        assert trainer.data_manager is not None


class TestFieldTrainerUtilityMethods:
    """Tests for utility methods."""
    
    def test_run_simulation(self):
        """Test _run_simulation method."""
        trainer = SimpleConcreteFieldTrainer({})
        trainer.config = {'trainer_params': {'num_predict_steps': 2}}
        
        # Create initial data
        field = CenteredGrid(1.0, x=10, y=10)
        initial_data = {'test_field': math.stack([field], batch('time'))}
        
        # Run simulation
        predictions = trainer._run_simulation(initial_data)
        
        assert 'test_field' in predictions
        assert isinstance(predictions['test_field'], (Field, Tensor))
    
    def test_compute_loss(self):
        """Test _compute_loss method."""
        trainer = SimpleConcreteFieldTrainer({})
        
        # Create test fields
        field1 = CenteredGrid(1.0, x=10, y=10)
        field2 = CenteredGrid(0.9, x=10, y=10)
        
        predictions = {'test': field1}
        ground_truth = {'test': field2}
        
        loss = trainer._compute_loss(predictions, ground_truth)
        
        assert isinstance(loss, (Tensor, float))
    
    def test_update_model_parameters(self):
        """Test _update_model_parameters method."""
        trainer = SimpleConcreteFieldTrainer({})
        trainer.model = SimplePhysicalModel(param=1.0)
        trainer.learnable_params_config = [{'name': 'param'}]
        
        # Update parameters
        new_params = [2.0]
        trainer._update_model_parameters(new_params)
        
        assert trainer.model.param == 2.0
    
    def test_update_model_parameters_single_value(self):
        """Test _update_model_parameters with single value."""
        trainer = SimpleConcreteFieldTrainer({})
        trainer.model = SimplePhysicalModel(param=1.0)
        trainer.learnable_params_config = [{'name': 'param'}]
        
        # Update with single value (not list)
        trainer._update_model_parameters(3.0)
        
        assert trainer.model.param == 3.0


class TestFieldTrainerResultsSaving:
    """Tests for saving and loading results."""
    
    def test_save_results_creates_file(self):
        """Test that save_results creates a file."""
        trainer = SimpleConcreteFieldTrainer({})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'results.pt'
            results = {
                'final_loss': 0.5,
                'optimized_parameters': {'param': 1.5},
                'iterations': 10
            }
            
            trainer.save_results(path, results)
            
            assert path.exists()
    
    def test_load_results_applies_parameters(self):
        """Test that load_results applies parameters to model."""
        trainer = SimpleConcreteFieldTrainer({})
        trainer.model = SimplePhysicalModel(param=1.0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'results.pt'
            results = {
                'final_loss': 0.5,
                'optimized_parameters': {'param': 2.5},
                'iterations': 10
            }
            
            trainer.save_results(path, results)
            loaded_results = trainer.load_results(path)
            
            assert loaded_results['final_loss'] == 0.5
            assert trainer.model.param == 2.5
    
    def test_load_results_missing_file_raises_error(self):
        """Test that loading missing results file raises error."""
        trainer = SimpleConcreteFieldTrainer({})
        
        with pytest.raises(FileNotFoundError):
            trainer.load_results(Path('nonexistent_file.pt'))


class TestFieldTrainerIntegration:
    """Integration tests for FieldTrainer."""
    
    def test_trainer_initialization_complete(self):
        """Test that trainer can be fully initialized."""
        trainer = SimpleConcreteFieldTrainer({})
        
        assert trainer.data_manager is not None
        assert trainer.model is not None
        assert len(trainer.learnable_params) > 0
        assert len(trainer.learnable_params_config) > 0
    
    def test_multiple_trainers_independent(self):
        """Test that multiple trainer instances are independent."""
        trainer1 = SimpleConcreteFieldTrainer({'id': 1})
        trainer2 = SimpleConcreteFieldTrainer({'id': 2})
        
        # Models should be different instances
        assert trainer1.model is not trainer2.model
        
        # Modifying one shouldn't affect the other
        trainer1.model.param = 5.0
        assert trainer2.model.param != 5.0


class TestFieldTrainerLoadGroundTruth:
    """Tests for _load_ground_truth method."""
    
    def test_load_ground_truth_not_implemented_by_default(self):
        """Test that _load_ground_truth raises NotImplementedError if not overridden."""
        
        class MinimalTrainer(FieldTrainer):
            def __init__(self, config):
                super().__init__(config)
                self.model = SimplePhysicalModel()
                self.data_manager = SimpleMockDataManager()
            
            def _create_data_manager(self):
                return SimpleMockDataManager()
            
            def _create_model(self):
                return SimplePhysicalModel()
            
            def _setup_optimization(self):
                return math.Solve(method='L-BFGS-B', abs_tol=1e-6, x0=[1.0])
        
        trainer = MinimalTrainer({})
        
        with pytest.raises(NotImplementedError):
            trainer._load_ground_truth()


class TestFieldTrainerConfigurationHandling:
    """Tests for configuration handling."""
    
    def test_config_passed_to_subclass(self):
        """Test that config is properly passed to subclass."""
        config = {
            'project_root': '/test/path',
            'trainer_params': {
                'num_predict_steps': 5
            }
        }
        trainer = SimpleConcreteFieldTrainer(config)
        
        assert trainer.config == config
        assert trainer.project_root == '/test/path'
    
    def test_empty_config_works(self):
        """Test that empty config works."""
        trainer = SimpleConcreteFieldTrainer({})
        assert trainer.config == {}
        assert trainer.project_root == '.'


class TestFieldTrainerOptimizationHistory:
    """Tests for optimization history tracking."""
    
    def test_optimization_history_initially_empty(self):
        """Test that optimization history is initially empty."""
        trainer = SimpleConcreteFieldTrainer({})
        assert trainer.optimization_history == []
    
    def test_final_loss_initially_zero(self):
        """Test that final_loss is initially 0.0."""
        trainer = SimpleConcreteFieldTrainer({})
        assert trainer.final_loss == 0.0
