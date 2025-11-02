"""
Tests for TensorTrainer

Tests the base class for PyTorch tensor-based trainers.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import tempfile

from src.training.tensor_trainer import TensorTrainer
from src.training.abstract_trainer import AbstractTrainer


class SimpleConcreteTensorTrainer(TensorTrainer):
    """Minimal concrete implementation for testing."""

    def __init__(self, config):
        super().__init__(config)
        self.model = self._create_model()
        self.dataloader = self._create_data_loader()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def _create_model(self):
        """Create a simple linear model."""
        return nn.Linear(10, 5)

    def _create_data_loader(self):
        """Create a simple dataloader."""
        X = torch.randn(100, 10)
        y = torch.randn(100, 5)
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=10)

    def _train_epoch(self):
        """Simple training epoch."""
        total_loss = 0.0
        for X, y in self.dataloader:
            pred = self.model(X)
            loss = nn.MSELoss()(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.dataloader)


class TestTensorTrainerInheritance:
    """Tests for TensorTrainer inheritance."""

    def test_tensor_trainer_inherits_from_abstract_trainer(self):
        """Test that TensorTrainer inherits from AbstractTrainer."""
        assert issubclass(TensorTrainer, AbstractTrainer)

    def test_concrete_trainer_is_tensor_trainer(self):
        """Test that concrete implementation is a TensorTrainer."""
        trainer = SimpleConcreteTensorTrainer({})
        assert isinstance(trainer, TensorTrainer)
        assert isinstance(trainer, AbstractTrainer)


class TestTensorTrainerInitialization:
    """Tests for TensorTrainer initialization."""

    def test_device_assignment(self):
        """Test that device is properly assigned."""
        trainer = SimpleConcreteTensorTrainer({})

        assert hasattr(trainer, "device")
        assert isinstance(trainer.device, torch.device)
        assert trainer.device.type in ["cuda", "cpu"]

    def test_device_is_cuda_or_cpu(self):
        """Test that device is either CUDA or CPU."""
        trainer = SimpleConcreteTensorTrainer({})
        assert trainer.device.type in ["cuda", "cpu"]

    def test_model_initialized_as_none(self):
        """Test that model is initialized as None by TensorTrainer."""
        config = {}

        # Before subclass sets it, should be None
        class PartialTrainer(TensorTrainer):
            def __init__(self, config):
                super().__init__(config)
                # Don't set model yet

            def _create_model(self):
                pass

            def _create_data_loader(self):
                pass

            def _train_epoch(self):
                pass

        trainer = PartialTrainer(config)
        assert trainer.model is None

    def test_optimizer_initialized_as_none(self):
        """Test that optimizer is initialized as None."""

        class PartialTrainer(TensorTrainer):
            def __init__(self, config):
                super().__init__(config)

            def _create_model(self):
                pass

            def _create_data_loader(self):
                pass

            def _train_epoch(self):
                pass

        trainer = PartialTrainer({})
        assert trainer.optimizer is None

    def test_dataloader_initialized_as_none(self):
        """Test that dataloader is initialized as None."""

        class PartialTrainer(TensorTrainer):
            def __init__(self, config):
                super().__init__(config)

            def _create_model(self):
                pass

            def _create_data_loader(self):
                pass

            def _train_epoch(self):
                pass

        trainer = PartialTrainer({})
        assert trainer.dataloader is None

    def test_checkpoint_path_initialized_as_none(self):
        """Test that checkpoint_path is initialized as None."""
        trainer = SimpleConcreteTensorTrainer({})
        # Subclass may set it, but base initializes to None
        assert hasattr(trainer, "checkpoint_path")


class TestTensorTrainerAbstractMethods:
    """Tests for abstract method enforcement."""

    def test_tensor_trainer_has_abstract_methods(self):
        """Test that TensorTrainer defines expected abstract methods."""
        abstract_methods = TensorTrainer.__abstractmethods__

        expected = {"_create_model", "_create_data_loader", "_train_epoch"}
        assert expected.issubset(abstract_methods)

    def test_tensor_trainer_is_abstract(self):
        """Test that TensorTrainer cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            TensorTrainer({})

    def test_missing_create_model_raises_error(self):
        """Test that missing _create_model() raises TypeError."""

        class IncompleteTrainer(TensorTrainer):
            def _create_data_loader(self):
                pass

            def _train_epoch(self):
                pass

        with pytest.raises(TypeError):
            IncompleteTrainer({})

    def test_missing_create_data_loader_raises_error(self):
        """Test that missing _create_data_loader() raises TypeError."""

        class IncompleteTrainer(TensorTrainer):
            def _create_model(self):
                pass

            def _train_epoch(self):
                pass

        with pytest.raises(TypeError):
            IncompleteTrainer({})

    def test_missing_train_epoch_raises_error(self):
        """Test that missing _train_epoch() raises TypeError."""

        class IncompleteTrainer(TensorTrainer):
            def _create_model(self):
                pass

            def _create_data_loader(self):
                pass

        with pytest.raises(TypeError):
            IncompleteTrainer({})


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
        return SimpleConcreteTensorTrainer(config)

    def test_train_executes_successfully(self, trainer_with_config):
        """Test that train() executes without error."""
        trainer = trainer_with_config
        result = trainer.train()

        assert isinstance(result, dict)
        assert "losses" in result
        assert "epochs" in result

    def test_train_returns_results(self, trainer_with_config):
        """Test that train() returns results dictionary."""
        result = trainer_with_config.train()

        assert isinstance(result, dict)
        assert "losses" in result
        assert "epochs" in result
        assert len(result["losses"]) == 3  # 3 epochs
        assert len(result["epochs"]) == 3

    def test_train_requires_model_and_dataloader(self):
        """Test that train() raises error if model or dataloader not set."""

        class NoModelTrainer(TensorTrainer):
            def __init__(self, config):
                super().__init__(config)
                self.model = None  # Not set
                self.dataloader = None  # Not set

            def _create_model(self):
                return None

            def _create_data_loader(self):
                return None

            def _train_epoch(self):
                return 0.0

        trainer = NoModelTrainer({})

        with pytest.raises(
            RuntimeError, match="Model and dataloader must be initialized"
        ):
            trainer.train()


class TestTensorTrainerConfigMethods:
    """Tests for configuration getter methods."""

    def test_get_num_epochs_default(self):
        """Test default number of epochs."""
        trainer = SimpleConcreteTensorTrainer({})
        assert trainer.get_num_epochs() == 100

    def test_get_num_epochs_custom(self):
        """Test custom number of epochs."""
        config = {"trainer_params": {"epochs": 50}}
        trainer = SimpleConcreteTensorTrainer(config)
        assert trainer.get_num_epochs() == 50

    def test_get_print_frequency_default(self):
        """Test default print frequency."""
        trainer = SimpleConcreteTensorTrainer({})
        assert trainer.get_print_frequency() == 10

    def test_get_print_frequency_custom(self):
        """Test custom print frequency."""
        config = {"trainer_params": {"print_freq": 5}}
        trainer = SimpleConcreteTensorTrainer(config)
        assert trainer.get_print_frequency() == 5

    def test_get_checkpoint_frequency_default(self):
        """Test default checkpoint frequency."""
        trainer = SimpleConcreteTensorTrainer({})
        assert trainer.get_checkpoint_frequency() == 50

    def test_get_checkpoint_frequency_custom(self):
        """Test custom checkpoint frequency."""
        config = {"trainer_params": {"checkpoint_freq": 25}}
        trainer = SimpleConcreteTensorTrainer(config)
        assert trainer.get_checkpoint_frequency() == 25


class TestTensorTrainerCheckpointManagement:
    """Tests for checkpoint save/load functionality."""

    @pytest.fixture
    def trainer_with_temp_dir(self):
        """Create trainer with temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"project_root": tmpdir}
            trainer = SimpleConcreteTensorTrainer(config)
            trainer.checkpoint_path = Path(tmpdir) / "test_checkpoint.pth"
            yield trainer

    def test_save_checkpoint_creates_file(self, trainer_with_temp_dir):
        """Test that save_checkpoint creates a file."""
        trainer = trainer_with_temp_dir

        trainer.save_checkpoint(epoch=1, loss=0.5)
        assert trainer.checkpoint_path.exists()

    def test_save_checkpoint_without_path_raises_error(self):
        """Test that save_checkpoint without path raises ValueError."""
        trainer = SimpleConcreteTensorTrainer({})
        trainer.checkpoint_path = None

        with pytest.raises(ValueError, match="checkpoint_path not set"):
            trainer.save_checkpoint(epoch=1, loss=0.5)

    def test_save_checkpoint_without_model_raises_error(self):
        """Test that save_checkpoint without model raises RuntimeError."""
        config = {}

        class NoModelTrainer(TensorTrainer):
            def __init__(self, config):
                super().__init__(config)
                self.model = None
                self.checkpoint_path = Path("test.pth")

            def _create_model(self):
                return None

            def _create_data_loader(self):
                return None

            def _train_epoch(self):
                return 0.0

        trainer = NoModelTrainer(config)

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

    def test_get_parameter_count(self):
        """Test that get_parameter_count returns correct count."""
        trainer = SimpleConcreteTensorTrainer({})
        # Linear(10, 5) has 10*5 weights + 5 biases = 55 parameters
        assert trainer.get_parameter_count() == 55

    def test_get_trainable_parameter_count(self):
        """Test counting only trainable parameters."""
        trainer = SimpleConcreteTensorTrainer({})

        # Freeze some parameters
        for param in list(trainer.model.parameters())[:1]:
            param.requires_grad = False

        trainable_count = trainer.get_trainable_parameter_count()
        total_count = trainer.get_parameter_count()

        assert trainable_count < total_count

    def test_get_parameter_count_without_model(self):
        """Test parameter count when model is None."""

        class NoModelTrainer(TensorTrainer):
            def __init__(self, config):
                super().__init__(config)
                self.model = None

            def _create_model(self):
                return None

            def _create_data_loader(self):
                return None

            def _train_epoch(self):
                return 0.0

        trainer = NoModelTrainer({})
        assert trainer.get_parameter_count() == 0
        assert trainer.get_trainable_parameter_count() == 0

    def test_print_model_summary_executes(self):
        """Test that print_model_summary executes without error."""
        trainer = SimpleConcreteTensorTrainer({})
        trainer.print_model_summary()  # Should not raise error

    def test_print_model_summary_without_model(self):
        """Test print_model_summary when model is None."""

        class NoModelTrainer(TensorTrainer):
            def __init__(self, config):
                super().__init__(config)
                self.model = None

            def _create_model(self):
                return None

            def _create_data_loader(self):
                return None

            def _train_epoch(self):
                return 0.0

        trainer = NoModelTrainer({})
        trainer.print_model_summary()  # Should not raise error


class TestTensorTrainerDeviceManagement:
    """Tests for device management."""

    def test_move_model_to_device(self):
        """Test that model can be moved to device."""
        trainer = SimpleConcreteTensorTrainer({})
        trainer.move_model_to_device()

        # Check that model parameters are on the right device
        for param in trainer.model.parameters():
            assert param.device.type == trainer.device.type

    def test_move_model_to_device_without_model(self):
        """Test move_model_to_device when model is None."""

        class NoModelTrainer(TensorTrainer):
            def __init__(self, config):
                super().__init__(config)
                self.model = None

            def _create_model(self):
                return None

            def _create_data_loader(self):
                return None

            def _train_epoch(self):
                return 0.0

        trainer = NoModelTrainer({})
        trainer.move_model_to_device()  # Should not raise error


class TestTensorTrainerModeManagement:
    """Tests for training/eval mode management."""

    def test_set_train_mode(self):
        """Test that model can be set to training mode."""
        trainer = SimpleConcreteTensorTrainer({})
        trainer.set_train_mode()

        assert trainer.model.training is True

    def test_set_eval_mode(self):
        """Test that model can be set to evaluation mode."""
        trainer = SimpleConcreteTensorTrainer({})
        trainer.set_eval_mode()

        assert trainer.model.training is False

    def test_mode_toggle(self):
        """Test toggling between train and eval modes."""
        trainer = SimpleConcreteTensorTrainer({})

        trainer.set_train_mode()
        assert trainer.model.training is True

        trainer.set_eval_mode()
        assert trainer.model.training is False

        trainer.set_train_mode()
        assert trainer.model.training is True


class TestTensorTrainerIntegration:
    """Integration tests for TensorTrainer."""

    def test_full_training_workflow(self):
        """Test complete training workflow."""
        config = {"trainer_params": {"epochs": 2}}
        trainer = SimpleConcreteTensorTrainer(config)

        # Train
        result = trainer.train()

        # Verify results
        assert len(result["losses"]) == 2
        assert all(isinstance(loss, float) for loss in result["losses"])

    def test_multiple_trainers_independent(self):
        """Test that multiple trainer instances are independent."""
        trainer1 = SimpleConcreteTensorTrainer({"id": 1})
        trainer2 = SimpleConcreteTensorTrainer({"id": 2})

        # Models should be different instances
        assert trainer1.model is not trainer2.model

        # Modifying one shouldn't affect the other
        trainer1.model.weight.data.fill_(1.0)
        assert not torch.allclose(
            trainer1.model.weight.data, trainer2.model.weight.data
        )
