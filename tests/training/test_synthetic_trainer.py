"""
Comprehensive tests for SyntheticTrainer with DataManager pipeline.
Tests initialization, data loading, training, and model integration.
"""

import pytest
import tempfile
import torch
import torch.nn as nn
from pathlib import Path

from src.training.synthetic.trainer import SyntheticTrainer
from src.models.synthetic.unet import UNet


# Get the device that models will use
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestSyntheticTrainerInitialization:
    """Tests for SyntheticTrainer initialization."""

    @pytest.fixture
    def burgers_config(self):
        """Create a minimal config for Burgers testing."""
        project_root = Path(__file__).parent.parent.parent

        return {
            "project_root": str(project_root),
            "run_params": {
                "experiment_name": "test_burgers",
                "mode": ["train"],
                "model_type": "synthetic",
            },
            "data": {
                "data_dir": "data/",
                "dset_name": "burgers_128",
                "fields": ["velocity"],
                "fields_scheme": "VV",
            },
            "model": {
                "synthetic": {
                    "name": "UNet",
                    "model_path": "results/models",
                    "model_save_name": "test_burgers_unet",
                    "input_specs": {"velocity": 2},
                    "output_specs": {"velocity": 2},
                    "architecture": {"levels": 2, "filters": 16, "batch_norm": True},
                }
            },
            "trainer_params": {
                "learning_rate": 1.0e-4,
                "batch_size": 2,
                "epochs": 2,
                "num_predict_steps": 3,
                "train_sim": [0],
            },
        }

    @pytest.fixture
    def heat_config(self):
        """Config for heat equation."""
        project_root = Path(__file__).parent.parent.parent

        return {
            "project_root": str(project_root),
            "data": {"data_dir": "data/", "dset_name": "heat_64", "fields": ["temp"]},
            "model": {
                "synthetic": {
                    "name": "UNet",
                    "model_path": "results/models",
                    "model_save_name": "test_heat_unet",
                    "input_specs": {"temp": 1},
                    "output_specs": {"temp": 1},
                    "architecture": {"levels": 2, "filters": 16, "batch_norm": True},
                }
            },
            "trainer_params": {
                "learning_rate": 1e-4,
                "batch_size": 2,
                "epochs": 1,
                "num_predict_steps": 2,
                "train_sim": [0],
            },
        }

    @pytest.fixture
    def smoke_config(self):
        """Config for smoke simulation with multiple fields."""
        project_root = Path(__file__).parent.parent.parent

        return {
            "project_root": str(project_root),
            "data": {
                "data_dir": "data/",
                "dset_name": "smoke_128",
                "fields": ["velocity", "density"],
            },
            "model": {
                "synthetic": {
                    "name": "UNet",
                    "model_path": "results/models",
                    "model_save_name": "test_smoke_unet",
                    "input_specs": {"velocity": 2, "density": 1},
                    "output_specs": {"velocity": 2, "density": 1},
                    "architecture": {"levels": 2, "filters": 16, "batch_norm": True},
                }
            },
            "trainer_params": {
                "learning_rate": 1e-4,
                "batch_size": 2,
                "epochs": 1,
                "num_predict_steps": 2,
                "train_sim": [0],
            },
        }

    def test_basic_initialization(self, burgers_config):
        """Test that trainer can be initialized."""
        trainer = SyntheticTrainer(burgers_config)

        assert trainer is not None
        assert trainer.config == burgers_config

    def test_device_detection(self, burgers_config):
        """Test that device is detected correctly."""
        trainer = SyntheticTrainer(burgers_config)

        assert trainer.device is not None
        assert trainer.device.type in ["cuda", "cpu"]

    def test_config_parsing(self, burgers_config):
        """Test that config sections are parsed."""
        trainer = SyntheticTrainer(burgers_config)

        assert trainer.data_config is not None
        assert trainer.model_config is not None
        assert trainer.trainer_config is not None

    def test_field_specifications(self, burgers_config):
        """Test field specs are extracted correctly."""
        trainer = SyntheticTrainer(burgers_config)

        assert trainer.field_names == ["velocity"]
        assert trainer.input_specs == {"velocity": 2}
        assert trainer.output_specs == {"velocity": 2}

    def test_dynamic_and_static_fields(self, burgers_config):
        """Test identification of dynamic vs static fields."""
        trainer = SyntheticTrainer(burgers_config)

        assert "velocity" in trainer.dynamic_fields
        assert len(trainer.static_fields) == 0

    def test_static_field_identification(self):
        """Test correct identification of static fields."""
        config = {
            "project_root": ".",
            "data": {
                "data_dir": "data/",
                "dset_name": "smoke_128",
                "fields": ["velocity", "density", "inflow"],
            },
            "model": {
                "synthetic": {
                    "name": "UNet",
                    "model_path": "results/models",
                    "model_save_name": "test",
                    "input_specs": {"velocity": 2, "density": 1, "inflow": 1},
                    "output_specs": {"velocity": 2, "density": 1},  # No inflow
                    "architecture": {"levels": 2, "filters": 8},
                }
            },
            "trainer_params": {
                "learning_rate": 1e-4,
                "batch_size": 2,
                "epochs": 1,
                "num_predict_steps": 2,
                "train_sim": [0],
            },
        }

        trainer = SyntheticTrainer(config)
        assert "inflow" in trainer.static_fields
        assert "velocity" in trainer.dynamic_fields
        assert "density" in trainer.dynamic_fields

    def test_training_parameters(self, burgers_config):
        """Test training parameters are set correctly."""
        trainer = SyntheticTrainer(burgers_config)

        assert trainer.learning_rate == 1.0e-4
        assert trainer.epochs == 2
        assert trainer.batch_size == 2
        assert trainer.num_predict_steps == 3

    def test_checkpoint_path_creation(self, burgers_config):
        """Test checkpoint path is created."""
        trainer = SyntheticTrainer(burgers_config)

        assert trainer.checkpoint_path is not None
        assert "test_burgers_unet" in trainer.checkpoint_path
        assert trainer.checkpoint_path.endswith(".pth")

    def test_model_creation(self, burgers_config):
        """Test that UNet model is created."""
        trainer = SyntheticTrainer(burgers_config)

        assert trainer.model is not None
        assert isinstance(trainer.model, UNet)

    def test_optimizer_creation(self, burgers_config):
        """Test that optimizer is created."""
        trainer = SyntheticTrainer(burgers_config)

        assert trainer.optimizer is not None
        assert isinstance(trainer.optimizer, torch.optim.Adam)

    def test_scheduler_creation(self, burgers_config):
        """Test that learning rate scheduler is created."""
        trainer = SyntheticTrainer(burgers_config)

        assert trainer.scheduler is not None

    def test_loss_function_creation(self, burgers_config):
        """Test that loss function is created."""
        trainer = SyntheticTrainer(burgers_config)

        assert trainer.loss_fn is not None
        assert isinstance(trainer.loss_fn, nn.MSELoss)

    def test_heat_config_initialization(self, heat_config):
        """Test initialization with heat equation config."""
        trainer = SyntheticTrainer(heat_config)

        assert trainer.model.in_channels == 1
        assert trainer.model.out_channels == 1

    def test_smoke_config_initialization(self, smoke_config):
        """Test initialization with smoke config."""
        trainer = SyntheticTrainer(smoke_config)

        assert trainer.model.in_channels == 3  # velocity(2) + density(1)
        assert trainer.model.out_channels == 3


class TestSyntheticTrainerChannelMapping:
    """Tests for channel mapping functionality."""

    @pytest.fixture
    def trainer(self):
        """Create trainer with multiple fields."""
        config = {
            "project_root": ".",
            "data": {
                "data_dir": "data/",
                "dset_name": "smoke_128",
                "fields": ["velocity", "density"],
            },
            "model": {
                "synthetic": {
                    "input_specs": {"velocity": 2, "density": 1},
                    "output_specs": {"velocity": 2, "density": 1},
                    "architecture": {"levels": 2, "filters": 16},
                    "model_path": "results/models",
                    "model_save_name": "test",
                }
            },
            "trainer_params": {
                "learning_rate": 1e-4,
                "batch_size": 2,
                "epochs": 1,
                "num_predict_steps": 2,
                "train_sim": [0],
            },
        }
        return SyntheticTrainer(config)

    def test_channel_map_built(self, trainer):
        """Test that channel map is built."""
        assert hasattr(trainer, "channel_map")
        assert isinstance(trainer.channel_map, dict)

    def test_channel_map_keys(self, trainer):
        """Test channel map contains all fields."""
        assert "velocity" in trainer.channel_map
        assert "density" in trainer.channel_map

    def test_channel_map_values(self, trainer):
        """Test channel map values are correct."""
        assert trainer.channel_map["velocity"] == (0, 2)
        assert trainer.channel_map["density"] == (2, 3)

    def test_total_channels(self, trainer):
        """Test total channel count is correct."""
        assert trainer.total_channels == 3

    def test_channel_map_single_field(self):
        """Test channel map with single field."""
        config = {
            "project_root": ".",
            "data": {"data_dir": "data/", "dset_name": "heat_64", "fields": ["temp"]},
            "model": {
                "synthetic": {
                    "input_specs": {"temp": 1},
                    "output_specs": {"temp": 1},
                    "architecture": {"levels": 2, "filters": 16},
                    "model_path": "results/models",
                    "model_save_name": "test",
                }
            },
            "trainer_params": {
                "learning_rate": 1e-4,
                "batch_size": 2,
                "epochs": 1,
                "num_predict_steps": 2,
                "train_sim": [0],
            },
        }

        trainer = SyntheticTrainer(config)
        assert trainer.channel_map["temp"] == (0, 1)
        assert trainer.total_channels == 1

    def test_channel_map_no_overlap(self, trainer):
        """Test that channel ranges don't overlap."""
        ranges = list(trainer.channel_map.values())

        for i, (start1, end1) in enumerate(ranges):
            for j, (start2, end2) in enumerate(ranges):
                if i != j:
                    # Ranges should not overlap
                    assert end1 <= start2 or end2 <= start1


class TestSyntheticTrainerDataLoader:
    """Tests for DataLoader functionality."""

    @pytest.fixture
    def trainer(self):
        """Create trainer for data loader testing."""
        project_root = Path(__file__).parent.parent.parent

        config = {
            "project_root": str(project_root),
            "data": {
                "data_dir": "data/",
                "dset_name": "burgers_128",
                "fields": ["velocity"],
            },
            "model": {
                "synthetic": {
                    "input_specs": {"velocity": 2},
                    "output_specs": {"velocity": 2},
                    "architecture": {"levels": 2, "filters": 16},
                    "model_path": "results/models",
                    "model_save_name": "test",
                }
            },
            "trainer_params": {
                "learning_rate": 1e-4,
                "batch_size": 2,
                "epochs": 1,
                "num_predict_steps": 3,
                "train_sim": [0],
            },
        }
        return SyntheticTrainer(config)

    def test_data_loader_created(self, trainer):
        """Test that DataLoader is created."""
        assert trainer.train_loader is not None
        assert hasattr(trainer.train_loader, "__iter__")

    def test_data_loader_length(self, trainer):
        """Test DataLoader has samples."""
        assert len(trainer.train_loader) > 0

    def test_data_loader_batch_structure(self, trainer):
        """Test that batches have correct structure."""
        initial_state, rollout_targets = next(iter(trainer.train_loader))

        # Check we got two tensors
        assert initial_state is not None
        assert rollout_targets is not None

    def test_initial_state_shape(self, trainer):
        """Test initial state tensor shape."""
        initial_state, _ = next(iter(trainer.train_loader))

        # Should be [B, C, H, W]
        assert initial_state.dim() == 4
        assert initial_state.shape[1] == 2  # velocity has 2 channels

    def test_rollout_targets_shape(self, trainer):
        """Test rollout targets tensor shape."""
        _, rollout_targets = next(iter(trainer.train_loader))

        # Should be [B, T, C, H, W]
        assert rollout_targets.dim() == 5
        assert rollout_targets.shape[1] == 3  # num_predict_steps
        assert rollout_targets.shape[2] == 2  # velocity channels

    def test_batch_size_respected(self, trainer):
        """Test that batch size is respected."""
        initial_state, _ = next(iter(trainer.train_loader))

        assert initial_state.shape[0] <= trainer.batch_size

    def test_multiple_batches(self, trainer):
        """Test iterating through multiple batches."""
        batch_count = 0
        for initial_state, rollout_targets in trainer.train_loader:
            batch_count += 1
            assert initial_state.shape[1] == 2
            assert rollout_targets.shape[1] == 3

        assert batch_count > 0


class TestSyntheticTrainerTensorUnpacking:
    """Tests for tensor unpacking functionality."""

    @pytest.fixture
    def trainer(self):
        """Create trainer with multiple fields."""
        config = {
            "project_root": ".",
            "data": {
                "data_dir": "data/",
                "dset_name": "smoke_128",
                "fields": ["velocity", "density"],
            },
            "model": {
                "synthetic": {
                    "input_specs": {"velocity": 2, "density": 1},
                    "output_specs": {"velocity": 2, "density": 1},
                    "architecture": {"levels": 2, "filters": 16},
                    "model_path": "results/models",
                    "model_save_name": "test",
                }
            },
            "trainer_params": {
                "learning_rate": 1e-4,
                "batch_size": 2,
                "epochs": 1,
                "num_predict_steps": 2,
                "train_sim": [0],
            },
        }
        return SyntheticTrainer(config)

    def test_unpack_4d_tensor(self, trainer):
        """Test unpacking 4D tensor [B, C, H, W]."""
        dummy_tensor = torch.randn(2, 3, 128, 128)

        field_dict = trainer._unpack_tensor_to_dict(dummy_tensor)

        assert isinstance(field_dict, dict)
        assert "velocity" in field_dict
        assert "density" in field_dict

    def test_unpack_field_shapes(self, trainer):
        """Test unpacked field shapes are correct."""
        dummy_tensor = torch.randn(2, 3, 128, 128)

        field_dict = trainer._unpack_tensor_to_dict(dummy_tensor)

        assert field_dict["velocity"].shape == (2, 2, 128, 128)
        assert field_dict["density"].shape == (2, 1, 128, 128)

    def test_unpack_5d_tensor(self, trainer):
        """Test unpacking 5D tensor [B, T, C, H, W]."""
        dummy_tensor = torch.randn(2, 3, 3, 128, 128)  # B=2, T=3, C=3

        field_dict = trainer._unpack_tensor_to_dict(dummy_tensor)

        assert field_dict["velocity"].shape == (2, 3, 2, 128, 128)
        assert field_dict["density"].shape == (2, 3, 1, 128, 128)

    def test_unpack_preserves_device(self, trainer):
        """Test that unpacking preserves tensor device."""
        dummy_tensor = torch.randn(2, 3, 128, 128).to(DEVICE)

        field_dict = trainer._unpack_tensor_to_dict(dummy_tensor)

        for field_name, field_tensor in field_dict.items():
            assert field_tensor.device == dummy_tensor.device


class TestSyntheticTrainerModelIntegration:
    """Tests for model integration."""

    @pytest.fixture
    def trainer(self):
        """Create trainer for model testing."""
        project_root = Path(__file__).parent.parent.parent

        config = {
            "project_root": str(project_root),
            "data": {
                "data_dir": "data/",
                "dset_name": "burgers_128",
                "fields": ["velocity"],
            },
            "model": {
                "synthetic": {
                    "input_specs": {"velocity": 2},
                    "output_specs": {"velocity": 2},
                    "architecture": {"levels": 2, "filters": 16, "batch_norm": True},
                    "model_path": "results/models",
                    "model_save_name": "test",
                }
            },
            "trainer_params": {
                "learning_rate": 1e-4,
                "batch_size": 2,
                "epochs": 1,
                "num_predict_steps": 2,
                "train_sim": [0],
            },
        }
        return SyntheticTrainer(config)

    def test_model_architecture(self, trainer):
        """Test model architecture matches config."""
        assert trainer.model.in_channels == 2
        assert trainer.model.out_channels == 2

    def test_model_on_correct_device(self, trainer):
        """Test model is on the correct device."""
        for param in trainer.model.parameters():
            assert param.device.type == trainer.device.type

    def test_model_forward_pass(self, trainer):
        """Test model forward pass works."""
        dummy_input = torch.randn(2, 2, 128, 128).to(trainer.device)

        with torch.no_grad():
            output = trainer.model(dummy_input)

        assert output.shape == (2, 2, 128, 128)

    def test_model_trainable(self, trainer):
        """Test model parameters are trainable."""
        trainable_params = sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        )
        assert trainable_params > 0


class TestSyntheticTrainerTraining:
    """Tests for training functionality."""

    @pytest.fixture
    def trainer(self):
        """Create trainer for training tests."""
        project_root = Path(__file__).parent.parent.parent

        config = {
            "project_root": str(project_root),
            "data": {
                "data_dir": "data/",
                "dset_name": "burgers_128",
                "fields": ["velocity"],
            },
            "model": {
                "synthetic": {
                    "input_specs": {"velocity": 2},
                    "output_specs": {"velocity": 2},
                    "architecture": {"levels": 2, "filters": 16},
                    "model_path": "results/models",
                    "model_save_name": "test_training",
                }
            },
            "trainer_params": {
                "learning_rate": 1e-4,
                "batch_size": 2,
                "epochs": 1,
                "num_predict_steps": 2,
                "train_sim": [0],
            },
        }
        return SyntheticTrainer(config)

    def test_train_epoch_method_exists(self, trainer):
        """Test _train_epoch method exists."""
        assert hasattr(trainer, "_train_epoch")
        assert callable(trainer._train_epoch)

    def test_train_epoch_returns_loss(self, trainer):
        """Test that _train_epoch returns a loss value."""
        loss = trainer._train_epoch()

        assert isinstance(loss, float)
        assert loss >= 0

    def test_train_method_exists(self, trainer):
        """Test train method exists."""
        assert hasattr(trainer, "train")
        assert callable(trainer.train)

    def test_training_completes(self, trainer):
        """Test that full training loop completes."""
        try:
            trainer.train()
            training_completed = True
        except Exception as e:
            training_completed = False
            print(f"Training failed with: {e}")

        assert training_completed

    def test_gradient_computation(self, trainer):
        """Test that gradients are computed during training."""
        # Run one training step
        trainer._train_epoch()

        # Check that model parameters have gradients
        has_gradients = False
        for param in trainer.model.parameters():
            if param.grad is not None:
                has_gradients = True
                break

        assert has_gradients

    def test_optimizer_step_updates_parameters(self, trainer):
        """Test that optimizer actually updates parameters."""
        # Get initial parameters
        initial_params = [p.clone() for p in trainer.model.parameters()]

        # Run training epoch
        trainer._train_epoch()

        # Check that at least some parameters changed
        params_changed = False
        for initial, current in zip(initial_params, trainer.model.parameters()):
            if not torch.equal(initial, current):
                params_changed = True
                break

        assert params_changed


class TestSyntheticTrainerMultiFieldScenarios:
    """Tests for multi-field scenarios."""

    def test_smoke_simulation(self):
        """Test with smoke simulation (velocity + density)."""
        project_root = Path(__file__).parent.parent.parent

        config = {
            "project_root": str(project_root),
            "data": {
                "data_dir": "data/",
                "dset_name": "smoke_128",
                "fields": ["velocity", "density"],
            },
            "model": {
                "synthetic": {
                    "input_specs": {"velocity": 2, "density": 1},
                    "output_specs": {"velocity": 2, "density": 1},
                    "architecture": {"levels": 2, "filters": 16},
                    "model_path": "results/models",
                    "model_save_name": "test_smoke",
                }
            },
            "trainer_params": {
                "learning_rate": 1e-4,
                "batch_size": 2,
                "epochs": 1,
                "num_predict_steps": 2,
                "train_sim": [0],
            },
        }

        trainer = SyntheticTrainer(config)

        assert trainer.model.in_channels == 3
        assert trainer.model.out_channels == 3
        assert len(trainer.dynamic_fields) == 2

    def test_with_static_fields(self):
        """Test configuration with static fields."""
        project_root = Path(__file__).parent.parent.parent

        config = {
            "project_root": str(project_root),
            "data": {
                "data_dir": "data/",
                "dset_name": "smoke_128",
                "fields": ["velocity", "density", "inflow"],
            },
            "model": {
                "synthetic": {
                    "input_specs": {"velocity": 2, "density": 1, "inflow": 1},
                    "output_specs": {"velocity": 2, "density": 1},  # No inflow output
                    "architecture": {"levels": 2, "filters": 16},
                    "model_path": "results/models",
                    "model_save_name": "test_static",
                }
            },
            "trainer_params": {
                "learning_rate": 1e-4,
                "batch_size": 2,
                "epochs": 1,
                "num_predict_steps": 2,
                "train_sim": [0],
            },
        }

        trainer = SyntheticTrainer(config)

        assert "inflow" in trainer.static_fields
        assert "inflow" not in trainer.dynamic_fields
        assert trainer.model.in_channels == 4  # All fields
        assert trainer.model.out_channels == 3  # Only dynamic


class TestSyntheticTrainerErrorHandling:
    """Tests for error handling."""

    def test_missing_field_in_specs_raises_error(self):
        """Test that missing field in specs raises error."""
        config = {
            "project_root": ".",
            "data": {
                "data_dir": "data/",
                "dset_name": "test",
                "fields": ["velocity", "density"],
            },
            "model": {
                "synthetic": {
                    "input_specs": {"velocity": 2},  # Missing density!
                    "output_specs": {"velocity": 2},
                    "architecture": {"levels": 2, "filters": 16},
                    "model_path": "results/models",
                    "model_save_name": "test",
                }
            },
            "trainer_params": {
                "learning_rate": 1e-4,
                "batch_size": 2,
                "epochs": 1,
                "num_predict_steps": 2,
                "train_sim": [0],
            },
        }

        with pytest.raises(ValueError, match="not found in specs"):
            trainer = SyntheticTrainer(config)


class TestSyntheticTrainerDifferentConfigurations:
    """Tests with different configurations."""

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        project_root = Path(__file__).parent.parent.parent

        for batch_size in [1, 2, 4]:
            config = {
                "project_root": str(project_root),
                "data": {
                    "data_dir": "data/",
                    "dset_name": "burgers_128",
                    "fields": ["velocity"],
                },
                "model": {
                    "synthetic": {
                        "input_specs": {"velocity": 2},
                        "output_specs": {"velocity": 2},
                        "architecture": {"levels": 2, "filters": 8},
                        "model_path": "results/models",
                        "model_save_name": f"test_bs{batch_size}",
                    }
                },
                "trainer_params": {
                    "learning_rate": 1e-4,
                    "batch_size": batch_size,
                    "epochs": 1,
                    "num_predict_steps": 2,
                    "train_sim": [0],
                },
            }

            trainer = SyntheticTrainer(config)
            assert trainer.batch_size == batch_size

    def test_different_learning_rates(self):
        """Test with different learning rates."""
        project_root = Path(__file__).parent.parent.parent

        for lr in [1e-3, 1e-4, 1e-5]:
            config = {
                "project_root": str(project_root),
                "data": {
                    "data_dir": "data/",
                    "dset_name": "burgers_128",
                    "fields": ["velocity"],
                },
                "model": {
                    "synthetic": {
                        "input_specs": {"velocity": 2},
                        "output_specs": {"velocity": 2},
                        "architecture": {"levels": 2, "filters": 8},
                        "model_path": "results/models",
                        "model_save_name": f"test_lr{lr}",
                    }
                },
                "trainer_params": {
                    "learning_rate": lr,
                    "batch_size": 2,
                    "epochs": 1,
                    "num_predict_steps": 2,
                    "train_sim": [0],
                },
            }

            trainer = SyntheticTrainer(config)
            assert trainer.learning_rate == lr

    def test_different_rollout_lengths(self):
        """Test with different rollout lengths."""
        project_root = Path(__file__).parent.parent.parent

        for steps in [1, 3, 5]:
            config = {
                "project_root": str(project_root),
                "data": {
                    "data_dir": "data/",
                    "dset_name": "burgers_128",
                    "fields": ["velocity"],
                },
                "model": {
                    "synthetic": {
                        "input_specs": {"velocity": 2},
                        "output_specs": {"velocity": 2},
                        "architecture": {"levels": 2, "filters": 8},
                        "model_path": "results/models",
                        "model_save_name": f"test_steps{steps}",
                    }
                },
                "trainer_params": {
                    "learning_rate": 1e-4,
                    "batch_size": 2,
                    "epochs": 1,
                    "num_predict_steps": steps,
                    "train_sim": [0],
                },
            }

            trainer = SyntheticTrainer(config)
            assert trainer.num_predict_steps == steps
            assert trainer.num_frames == steps + 1
