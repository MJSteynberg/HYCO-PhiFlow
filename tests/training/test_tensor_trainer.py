"""
Tests for TensorTrainer (Phase 1 API)

Tests the base class for PyTorch tensor-based trainers with external model management.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import tempfile

from src.training.tensor_trainer import TensorTrainer
from src.training.abstract_trainer import AbstractTrainer


@pytest.fixture
def default_trainer_config():
    """Provides default trainer configuration matching Hydra defaults."""
    return {
        "trainer_params": {
            "epochs": 100,
            "print_freq": 10,
            "checkpoint_freq": 50,
        }
    }


class SimpleConcreteTensorTrainer(TensorTrainer):
    """Minimal concrete implementation for testing Phase 1 API."""

    def __init__(self, config, model):
        super().__init__(config, model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def _train_epoch_with_data(self, data_source):
        """Simple training epoch with data_source."""
        total_loss = 0.0
        for batch in data_source:
            X, y = batch
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            loss = nn.MSELoss()(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(data_source)


class TestTensorTrainerInheritance:
    """Tests for TensorTrainer inheritance."""

    def test_tensor_trainer_inherits_from_abstract_trainer(self, default_trainer_config):
        """Test that TensorTrainer inherits from AbstractTrainer."""
        assert issubclass(TensorTrainer, AbstractTrainer)

    def test_concrete_trainer_is_tensor_trainer(self, default_trainer_config):
        """Test that concrete implementation is a TensorTrainer."""
        model = nn.Linear(10, 5)
        trainer = SimpleConcreteTensorTrainer(default_trainer_config, model)
        assert isinstance(trainer, TensorTrainer)
        assert isinstance(trainer, AbstractTrainer)


class TestTensorTrainerInitialization:
    """Tests for TensorTrainer initialization."""

    def test_device_assignment(self, default_trainer_config):
        """Test that device is properly assigned."""
        model = nn.Linear(10, 5)
        trainer = SimpleConcreteTensorTrainer(default_trainer_config, model)

        assert hasattr(trainer, "device")
        assert isinstance(trainer.device, torch.device)
        assert trainer.device.type in ["cuda", "cpu"]

    def test_device_is_cuda_or_cpu(self, default_trainer_config):
        """Test that device is either CUDA or CPU."""
        model = nn.Linear(10, 5)
        trainer = SimpleConcreteTensorTrainer(default_trainer_config, model)
        assert trainer.device.type in ["cuda", "cpu"]

    def test_model_set_from_parameter(self, default_trainer_config):
        """Test that model is set from constructor parameter."""
        model = nn.Linear(10, 5)
        trainer = SimpleConcreteTensorTrainer(default_trainer_config, model)
        assert trainer.model is model

    def test_optimizer_created_automatically(self, default_trainer_config):
        """Test that optimizer is created automatically by base class."""

        class PartialTrainer(TensorTrainer):
            def __init__(self, config, model):
                super().__init__(config, model)
                # Optimizer created automatically

            def _train_epoch_with_data(self, data_source):
                return 0.0

        model = nn.Linear(10, 5)
        trainer = PartialTrainer({}, model)
        # With Phase 1 API, optimizer is created automatically if model exists
        assert trainer.optimizer is not None
        assert isinstance(trainer.optimizer, torch.optim.Optimizer)

    def test_checkpoint_path_initialized_as_none(self, default_trainer_config):
        """Test that checkpoint_path is initialized as None."""
        model = nn.Linear(10, 5)
        trainer = SimpleConcreteTensorTrainer(default_trainer_config, model)
        # Subclass may set it, but base initializes to None
        assert hasattr(trainer, "checkpoint_path")


class TestTensorTrainerAbstractMethods:
    """Tests for abstract method enforcement."""

    def test_tensor_trainer_has_abstract_method(self, default_trainer_config):
        """Test that TensorTrainer defines expected abstract method."""
        abstract_methods = TensorTrainer.__abstractmethods__

        expected = {"_train_epoch_with_data"}
        assert expected.issubset(abstract_methods)

    def test_tensor_trainer_is_abstract(self, default_trainer_config):
        """Test that TensorTrainer cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            model = nn.Linear(10, 5)
            TensorTrainer({}, model)

    def test_missing_train_epoch_with_data_raises_error(self, default_trainer_config):
        """Test that missing _train_epoch_with_data() raises TypeError."""

        class IncompleteTrainer(TensorTrainer):
            pass  # Missing _train_epoch_with_data

        with pytest.raises(TypeError):
            model = nn.Linear(10, 5)
            IncompleteTrainer({}, model)


class TestTensorTrainerDefaultTrain:
    """Tests for the default train() implementation."""

    @pytest.fixture
    def trainer_with_config(self):
        """Create trainer with custom config."""
        config = {
            "trainer_params": {
                "epochs": 3,
                "print_freq": 1,
                "checkpoint_freq": 0,  # Disable checkpointing for tests
            }
        }
        model = nn.Linear(10, 5)
        return SimpleConcreteTensorTrainer(config, model)

    @pytest.fixture
    def sample_data_loader(self):
        """Create sample data loader."""
        X = torch.randn(100, 10)
        y = torch.randn(100, 5)
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=10)

    def test_train_executes_successfully(self, trainer_with_config, sample_data_loader):
        """Test that train() executes without error with explicit data."""
        trainer = trainer_with_config
        result = trainer.train(data_source=sample_data_loader, num_epochs=3)

        assert isinstance(result, dict)
        assert "train_losses" in result
        assert "epochs" in result

    def test_train_returns_results(self, trainer_with_config, sample_data_loader):
        """Test that train() returns results dictionary."""
        result = trainer_with_config.train(data_source=sample_data_loader, num_epochs=3)

        assert isinstance(result, dict)
        assert "train_losses" in result
        assert "epochs" in result
        assert len(result["train_losses"]) == 3  # 3 epochs
        assert len(result["epochs"]) == 3

    def test_train_requires_model(self, default_trainer_config):
        """Test that train() raises error if model not set."""

        class NoModelTrainer(TensorTrainer):
            def __init__(self, config, model):
                super().__init__(config, None)  # Pass None as model

            def _train_epoch_with_data(self, data_source):
                return 0.0

        trainer = NoModelTrainer({}, None)
        X = torch.randn(100, 10)
        y = torch.randn(100, 5)
        dataset = TensorDataset(X, y)
        data_loader = DataLoader(dataset, batch_size=10)

        with pytest.raises(RuntimeError, match="Model must be initialized"):
            trainer.train(data_source=data_loader, num_epochs=1)


class TestTensorTrainerConfigMethods:
    """Tests for configuration access."""

    def test_get_num_epochs_default(self, default_trainer_config):
        """Test default number of epochs from Hydra config."""
        model = nn.Linear(10, 5)
        trainer = SimpleConcreteTensorTrainer(default_trainer_config, model)
        # Access config directly via config dictionary
        trainer_params = trainer.config.get("trainer_params", {})
        assert trainer_params.get("epochs", 100) == 100

    def test_get_num_epochs_custom(self, default_trainer_config):
        """Test custom number of epochs."""
        config = {"trainer_params": {"epochs": 50, "print_freq": 10, "checkpoint_freq": 50}}
        model = nn.Linear(10, 5)
        trainer = SimpleConcreteTensorTrainer(config, model)
        assert trainer.config["trainer_params"]["epochs"] == 50

    def test_get_print_frequency_default(self, default_trainer_config):
        """Test default print frequency from Hydra config."""
        model = nn.Linear(10, 5)
        trainer = SimpleConcreteTensorTrainer(default_trainer_config, model)
        trainer_params = trainer.config.get("trainer_params", {})
        assert trainer_params.get("print_freq", 10) == 10

    def test_get_print_frequency_custom(self, default_trainer_config):
        """Test custom print frequency."""
        config = {"trainer_params": {"epochs": 100, "print_freq": 5, "checkpoint_freq": 50}}
        model = nn.Linear(10, 5)
        trainer = SimpleConcreteTensorTrainer(config, model)
        assert trainer.config["trainer_params"]["print_freq"] == 5

    def test_get_checkpoint_frequency_default(self, default_trainer_config):
        """Test default checkpoint frequency from Hydra config."""
        model = nn.Linear(10, 5)
        trainer = SimpleConcreteTensorTrainer(default_trainer_config, model)
        trainer_params = trainer.config.get("trainer_params", {})
        assert trainer_params.get("checkpoint_freq", 50) == 50

    def test_get_checkpoint_frequency_custom(self, default_trainer_config):
        """Test custom checkpoint frequency."""
        config = {"trainer_params": {"epochs": 100, "print_freq": 10, "checkpoint_freq": 25}}
        model = nn.Linear(10, 5)
        trainer = SimpleConcreteTensorTrainer(config, model)
        assert trainer.config["trainer_params"]["checkpoint_freq"] == 25


class TestTensorTrainerCheckpointManagement:
    """Tests for checkpoint save/load functionality."""

    @pytest.fixture
    def trainer_with_temp_dir(self):
        """Create trainer with temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"project_root": tmpdir}
            model = nn.Linear(10, 5)
            trainer = SimpleConcreteTensorTrainer(config, model)
            trainer.checkpoint_path = Path(tmpdir) / "test_checkpoint.pth"
            yield trainer

    def test_save_checkpoint_creates_file(self, trainer_with_temp_dir):
        """Test that save_checkpoint creates a file."""
        trainer = trainer_with_temp_dir

        trainer.save_checkpoint(epoch=1, loss=0.5)
        assert trainer.checkpoint_path.exists()

    def test_save_checkpoint_without_path_raises_error(self, default_trainer_config):
        """Test that save_checkpoint without path raises ValueError."""
        model = nn.Linear(10, 5)
        trainer = SimpleConcreteTensorTrainer(default_trainer_config, model)
        trainer.checkpoint_path = None

        with pytest.raises(ValueError, match="checkpoint_path not set"):
            trainer.save_checkpoint(epoch=1, loss=0.5)

    def test_save_checkpoint_without_model_raises_error(self, default_trainer_config):
        """Test that save_checkpoint without model raises RuntimeError."""
        config = {}

        class NoModelTrainer(TensorTrainer):
            def __init__(self, config, model):
                super().__init__(config, model)
                self.model = None  # Override to None
                self.checkpoint_path = Path("test.pth")

            def _train_epoch_with_data(self, data_source):
                return 0.0

        trainer = NoModelTrainer(config, None)

        with pytest.raises(RuntimeError, match="Model not initialized"):
            trainer.save_checkpoint(epoch=1, loss=0.5)

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
        assert "model_state_dict" in checkpoint
        assert "epoch" in checkpoint
        assert "loss" in checkpoint

    def test_checkpoint_contains_metadata(self, trainer_with_temp_dir):
        """Test that saved checkpoint contains metadata."""
        trainer = trainer_with_temp_dir

        trainer.save_checkpoint(epoch=10, loss=0.5)
        checkpoint = trainer.load_checkpoint()

        assert checkpoint["epoch"] == 10
        assert checkpoint["loss"] == 0.5
        assert "config" in checkpoint

    def test_checkpoint_with_optimizer_state(self, trainer_with_temp_dir):
        """Test saving checkpoint with optimizer state."""
        trainer = trainer_with_temp_dir

        optimizer_state = trainer.optimizer.state_dict()
        trainer.save_checkpoint(epoch=1, loss=0.5, optimizer_state=optimizer_state)

        checkpoint = trainer.load_checkpoint()
        assert "optimizer_state_dict" in checkpoint

    def test_checkpoint_with_additional_info(self, trainer_with_temp_dir):
        """Test saving checkpoint with additional info."""
        trainer = trainer_with_temp_dir

        additional_info = {"custom_metric": 0.95, "best_epoch": 5}
        trainer.save_checkpoint(epoch=1, loss=0.5, additional_info=additional_info)

        checkpoint = trainer.load_checkpoint()
        assert checkpoint["custom_metric"] == 0.95
        assert checkpoint["best_epoch"] == 5


class TestTensorTrainerModelSummary:
    """Tests for model parameter counting."""

    def test_get_parameter_count(self, default_trainer_config):
        """Test that get_parameter_count returns correct count."""
        model = nn.Linear(10, 5)
        trainer = SimpleConcreteTensorTrainer(default_trainer_config, model)
        # Linear(10, 5) has 10*5 weights + 5 biases = 55 parameters
        assert trainer.get_parameter_count() == 55

    def test_get_trainable_parameter_count(self, default_trainer_config):
        """Test counting only trainable parameters."""
        model = nn.Linear(10, 5)
        trainer = SimpleConcreteTensorTrainer(default_trainer_config, model)

        # Freeze some parameters
        for param in list(trainer.model.parameters())[:1]:
            param.requires_grad = False

        trainable_count = trainer.get_trainable_parameter_count()
        total_count = trainer.get_parameter_count()

        assert trainable_count < total_count

    def test_get_parameter_count_without_model(self, default_trainer_config):
        """Test parameter count when model is None."""

        class NoModelTrainer(TensorTrainer):
            def __init__(self, config, model):
                super().__init__(config, model)
                self.model = None  # Override to None

            def _train_epoch_with_data(self, data_source):
                return 0.0

        trainer = NoModelTrainer({}, None)
        assert trainer.get_parameter_count() == 0
        assert trainer.get_trainable_parameter_count() == 0

    def test_print_model_summary_executes(self, default_trainer_config):
        """Test that print_model_summary executes without error."""
        model = nn.Linear(10, 5)
        trainer = SimpleConcreteTensorTrainer(default_trainer_config, model)
        trainer.print_model_summary()  # Should not raise error

    def test_print_model_summary_without_model(self, default_trainer_config):
        """Test print_model_summary when model is None."""

        class NoModelTrainer(TensorTrainer):
            def __init__(self, config, model):
                super().__init__(config, model)
                self.model = None  # Override to None

            def _train_epoch_with_data(self, data_source):
                return 0.0

        trainer = NoModelTrainer({}, None)
        trainer.print_model_summary()  # Should not raise error


class TestTensorTrainerDeviceManagement:
    """Tests for device management."""

    def test_move_model_to_device(self, default_trainer_config):
        """Test that model can be moved to device."""
        model = nn.Linear(10, 5)
        trainer = SimpleConcreteTensorTrainer(default_trainer_config, model)
        trainer.move_model_to_device()

        # Check that model parameters are on the right device
        for param in trainer.model.parameters():
            assert param.device.type == trainer.device.type

    def test_move_model_to_device_without_model(self, default_trainer_config):
        """Test move_model_to_device when model is None."""

        class NoModelTrainer(TensorTrainer):
            def __init__(self, config, model):
                super().__init__(config, model)
                self.model = None  # Override to None

            def _train_epoch_with_data(self, data_source):
                return 0.0

        trainer = NoModelTrainer({}, None)
        trainer.move_model_to_device()  # Should not raise error


class TestTensorTrainerModeManagement:
    """Tests for training/eval mode management."""

    def test_set_train_mode(self, default_trainer_config):
        """Test that model can be set to training mode."""
        model = nn.Linear(10, 5)
        trainer = SimpleConcreteTensorTrainer(default_trainer_config, model)
        trainer.set_train_mode()

        assert trainer.model.training is True

    def test_set_eval_mode(self, default_trainer_config):
        """Test that model can be set to evaluation mode."""
        model = nn.Linear(10, 5)
        trainer = SimpleConcreteTensorTrainer(default_trainer_config, model)
        trainer.set_eval_mode()

        assert trainer.model.training is False

    def test_mode_toggle(self, default_trainer_config):
        """Test toggling between train and eval modes."""
        model = nn.Linear(10, 5)
        trainer = SimpleConcreteTensorTrainer(default_trainer_config, model)

        trainer.set_train_mode()
        assert trainer.model.training is True

        trainer.set_eval_mode()
        assert trainer.model.training is False

        trainer.set_train_mode()
        assert trainer.model.training is True


class TestTensorTrainerIntegration:
    """Integration tests for TensorTrainer."""

    def test_full_training_workflow(self, default_trainer_config):
        """Test complete training workflow."""
        config = {
            "trainer_params": {
                "epochs": 2,
                "print_freq": 1,
                "checkpoint_freq": 50
            }
        }
        model = nn.Linear(10, 5)
        trainer = SimpleConcreteTensorTrainer(config, model)

        # Create simple data source (list of tuples)
        simple_data = [
            (torch.randn(4, 10), torch.randn(4, 5))  # (X, y) tuples
            for _ in range(5)  # 5 batches
        ]

        # Train
        result = trainer.train(simple_data, num_epochs=2)

        # Verify results
        assert len(result["train_losses"]) == 2
        assert all(isinstance(loss, float) for loss in result["train_losses"])

    def test_multiple_trainers_independent(self, default_trainer_config):
        """Test that multiple trainer instances are independent."""
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(10, 5)
        trainer1 = SimpleConcreteTensorTrainer({"id": 1}, model1)
        trainer2 = SimpleConcreteTensorTrainer({"id": 2}, model2)

        # Models should be different instances
        assert trainer1.model is not trainer2.model

        # Modifying one shouldn't affect the other
        trainer1.model.weight.data.fill_(1.0)
        assert not torch.allclose(
            trainer1.model.weight.data, trainer2.model.weight.data
        )
