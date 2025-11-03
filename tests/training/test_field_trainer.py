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
    """Minimal concrete implementation for testing (Phase 1)."""

    def __init__(self, config):
        # Phase 1: Pass model and learnable_params to parent
        model = SimplePhysicalModel()
        learnable_params = [torch.nn.Parameter(torch.tensor(1.0))]
        super().__init__(config, model, learnable_params)
        
        # Store additional test attributes
        self.learnable_params_config = [{"name": "test_param"}]

    def _train_sample(self, initial_fields, target_fields):
        """Train on a single sample (Phase 1 abstract method)."""
        # Simple mock training: just return a loss value
        return 0.5


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
        """Test that data_manager attribute doesn't exist in Phase 1."""
        
        class PartialTrainer(FieldTrainer):
            def __init__(self, config):
                model = SimplePhysicalModel()
                learnable_params = [torch.nn.Parameter(torch.tensor(1.0))]
                super().__init__(config, model, learnable_params)

            def _train_sample(self, initial_fields, target_fields):
                return 0.5

        trainer = PartialTrainer({})
        # Phase 1: No data_manager attribute
        assert not hasattr(trainer, "data_manager")

    def test_model_initialized_as_none(self):
        """Test that model is provided in __init__."""

        class PartialTrainer(FieldTrainer):
            def __init__(self, config):
                model = SimplePhysicalModel()
                learnable_params = [torch.nn.Parameter(torch.tensor(1.0))]
                super().__init__(config, model, learnable_params)

            def _train_sample(self, initial_fields, target_fields):
                return 0.5

        trainer = PartialTrainer({})
        # Phase 1: Model is passed in and stored
        assert trainer.model is not None

    def test_learnable_params_initialized_as_empty_list(self):
        """Test that learnable_params is provided in __init__."""

        class PartialTrainer(FieldTrainer):
            def __init__(self, config):
                model = SimplePhysicalModel()
                learnable_params = []  # Empty list
                super().__init__(config, model, learnable_params)

            def _train_sample(self, initial_fields, target_fields):
                return 0.5

        trainer = PartialTrainer({})
        # Phase 1: Learnable params are passed in
        assert trainer.learnable_params == []

    def test_learnable_params_config_initialized_as_empty_list(self):
        """Test that learnable_params_config doesn't exist by default."""

        class PartialTrainer(FieldTrainer):
            def __init__(self, config):
                model = SimplePhysicalModel()
                learnable_params = []
                super().__init__(config, model, learnable_params)

            def _train_sample(self, initial_fields, target_fields):
                return 0.5

        trainer = PartialTrainer({})
        # Phase 1: learnable_params_config is not set by FieldTrainer
        assert not hasattr(trainer, "learnable_params_config")

    def test_final_loss_initialized(self):
        """Test that final_loss is initialized."""
        trainer = SimpleConcreteFieldTrainer({})
        assert hasattr(trainer, "final_loss")
        assert trainer.final_loss == 0.0

    def test_optimization_history_initialized(self):
        """Test that training_history is initialized (Phase 1 renamed)."""
        trainer = SimpleConcreteFieldTrainer({})
        # Phase 1: renamed from optimization_history to training_history
        assert hasattr(trainer, "training_history")
        assert trainer.training_history == []


class TestFieldTrainerAbstractMethods:
    """Tests for abstract method enforcement."""

    def test_field_trainer_has_abstract_methods(self):
        """Test that FieldTrainer defines expected abstract methods."""
        abstract_methods = FieldTrainer.__abstractmethods__

        # Phase 1: Only _train_sample is abstract
        expected = {"_train_sample"}
        assert expected.issubset(abstract_methods)

    def test_field_trainer_is_abstract(self):
        """Test that FieldTrainer cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            # Phase 1: Need to pass model and params, but will fail anyway due to abstract method
            FieldTrainer({}, None, [])

    def test_missing_train_sample_raises_error(self):
        """Test that missing _train_sample() raises TypeError."""

        class IncompleteTrainer(FieldTrainer):
            def __init__(self, config):
                model = SimplePhysicalModel()
                learnable_params = []
                super().__init__(config, model, learnable_params)

        with pytest.raises(TypeError, match="abstract method"):
            IncompleteTrainer({})


class TestFieldTrainerDefaultTrain:
    """Tests for the default train() implementation."""

    def test_train_requires_model(self):
        """Test that train() raises error if model is None."""

        class NoModelTrainer(FieldTrainer):
            def __init__(self, config):
                # Pass None as model
                super().__init__(config, None, [])

            def _train_sample(self, initial_fields, target_fields):
                return 0.5

        trainer = NoModelTrainer({})

        with pytest.raises(RuntimeError, match="Model must be initialized"):
            trainer.train([], num_epochs=1)

    def test_train_returns_dict(self):
        """Test that train() returns a dictionary."""
        trainer = SimpleConcreteFieldTrainer({})

        # Create mock data source
        mock_data = [({"field1": None}, {"field1": None})]  # Minimal mock data
        
        # Train should return a dict with results
        result = trainer.train(mock_data, num_epochs=1)
        
        assert isinstance(result, dict)
        assert "num_epochs" in result


class TestFieldTrainerUtilityMethods:
    """Tests for utility methods."""

    def test_model_is_stored(self):
        """Test that model is stored in trainer."""
        trainer = SimpleConcreteFieldTrainer({})
        assert trainer.model is not None
        assert isinstance(trainer.model, SimplePhysicalModel)

    def test_learnable_params_are_stored(self):
        """Test that learnable_params are stored."""
        trainer = SimpleConcreteFieldTrainer({})
        assert trainer.learnable_params is not None
        assert len(trainer.learnable_params) > 0


class TestFieldTrainerResultsSaving:
    """Tests for saving and loading results."""

    def test_save_results_creates_file(self):
        """Test that save_results creates a file."""
        trainer = SimpleConcreteFieldTrainer({})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.pt"
            results = {
                "final_loss": 0.5,
                "optimized_parameters": {"param": 1.5},
                "iterations": 10,
            }

            trainer.save_results(path, results)

            assert path.exists()

    def test_load_results_applies_parameters(self):
        """Test that load_results loads parameters (Phase 1: doesn't auto-apply)."""
        trainer = SimpleConcreteFieldTrainer({})
        trainer.model = SimplePhysicalModel(param=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.pt"
            results = {
                "final_loss": 0.5,
                "optimized_parameters": {"param": 2.5},
                "iterations": 10,
            }

            trainer.save_results(path, results)
            loaded_results = trainer.load_results(path)

            # Phase 1: load_results just returns data, doesn't auto-apply
            assert loaded_results["final_loss"] == 0.5
            assert loaded_results["optimized_parameters"]["param"] == 2.5
            # Model parameter unchanged (user must apply manually)
            assert trainer.model.param == 1.0

    def test_load_results_missing_file_raises_error(self):
        """Test that loading missing results file raises error."""
        trainer = SimpleConcreteFieldTrainer({})

        with pytest.raises(FileNotFoundError):
            trainer.load_results(Path("nonexistent_file.pt"))


class TestFieldTrainerIntegration:
    """Integration tests for FieldTrainer."""

    def test_trainer_initialization_complete(self):
        """Test that trainer can be fully initialized."""
        trainer = SimpleConcreteFieldTrainer({})

        # Phase 1: Check for expected attributes
        assert trainer.model is not None
        assert len(trainer.learnable_params) > 0
        assert len(trainer.learnable_params_config) > 0

    def test_multiple_trainers_independent(self):
        """Test that multiple trainer instances are independent."""
        trainer1 = SimpleConcreteFieldTrainer({"id": 1})
        trainer2 = SimpleConcreteFieldTrainer({"id": 2})

        # Models should be different instances
        assert trainer1.model is not trainer2.model

        # Modifying one shouldn't affect the other
        trainer1.model.param = 5.0
        assert trainer2.model.param != 5.0


class TestFieldTrainerLoadGroundTruth:
    """Tests for ground truth loading (Phase 1: done externally)."""

    def test_ground_truth_loading_is_external(self):
        """Test that Phase 1 trainers don't have _load_ground_truth method."""
        
        class MinimalTrainer(FieldTrainer):
            def __init__(self, config):
                model = SimplePhysicalModel()
                learnable_params = [torch.nn.Parameter(torch.tensor(1.0))]
                super().__init__(config, model, learnable_params)

            def _train_sample(self, initial_fields, target_fields):
                return 0.5

        trainer = MinimalTrainer({})

        # Phase 1: No _load_ground_truth method (data is external)
        assert not hasattr(trainer, "_load_ground_truth") or not callable(getattr(trainer, "_load_ground_truth", None))


class TestFieldTrainerConfigurationHandling:
    """Tests for configuration handling."""

    def test_config_passed_to_subclass(self):
        """Test that config is properly passed to subclass."""
        config = {
            "project_root": "/test/path",
            "trainer_params": {"num_predict_steps": 5},
        }
        trainer = SimpleConcreteFieldTrainer(config)

        assert trainer.config == config
        assert trainer.project_root == "/test/path"

    def test_empty_config_works(self):
        """Test that empty config works."""
        trainer = SimpleConcreteFieldTrainer({})
        assert trainer.config == {}
        assert trainer.project_root == "."


class TestFieldTrainerOptimizationHistory:
    """Tests for optimization history tracking."""

    def test_training_history_initially_empty(self):
        """Test that training_history is initially empty (Phase 1 renamed)."""
        trainer = SimpleConcreteFieldTrainer({})
        assert trainer.training_history == []

    def test_final_loss_initially_zero(self):
        """Test that final_loss is initially 0.0."""
        trainer = SimpleConcreteFieldTrainer({})
        assert trainer.final_loss == 0.0
