"""
Tests for Model Factory Pattern (Phase 3: Improved Modularity).

Tests the factory pattern for creating models from configuration,
providing a higher-level interface than the registry.
"""

import pytest
import torch.nn as nn
from src.factories.model_factory import ModelFactory


class TestModelFactoryPhysical:
    """Tests for physical model factory."""

    def test_create_burgers_model(self):
        """Test creating BurgersModel through factory."""
        config = {
            "model": {
                "physical": {
                    "name": "BurgersModel",
                    "domain": {"size_x": 100, "size_y": 100},
                    "resolution": {"x": 64, "y": 64},
                    "dt": 0.01,
                    "pde_params": {"batch_size": 1, "nu": 0.1},
                }
            }
        }

        model = ModelFactory.create_physical_model(config)
        assert model is not None
        assert hasattr(model, "step")
        assert hasattr(model, "get_initial_state")

    def test_create_smoke_model(self):
        """Test creating SmokeModel through factory."""
        config = {
            "model": {
                "physical": {
                    "name": "SmokeModel",
                    "domain": {"size_x": 100, "size_y": 100},
                    "resolution": {"x": 64, "y": 64},
                    "dt": 0.01,
                    "pde_params": {"batch_size": 1, "buoyancy_factor": 0.1},
                }
            }
        }

        model = ModelFactory.create_physical_model(config)
        assert model is not None

    def test_create_heat_model(self):
        """Test creating HeatModel through factory."""
        config = {
            "model": {
                "physical": {
                    "name": "HeatModel",
                    "domain": {"size_x": 100, "size_y": 100},
                    "resolution": {"x": 64, "y": 64},
                    "dt": 0.01,
                    "pde_params": {"batch_size": 1, "diffusivity": 0.1},
                }
            }
        }

        model = ModelFactory.create_physical_model(config)
        assert model is not None

    def test_factory_extracts_name_correctly(self):
        """Test that factory correctly extracts model name from config."""
        config = {
            "model": {
                "physical": {
                    "name": "BurgersModel",
                    "domain": {"size_x": 100, "size_y": 100},
                    "resolution": {"x": 64, "y": 64},
                    "dt": 0.01,
                    "pde_params": {"batch_size": 1, "nu": 0.1},
                }
            }
        }

        # Should not raise error
        model = ModelFactory.create_physical_model(config)
        assert model is not None

    def test_factory_passes_config_correctly(self):
        """Test that factory passes configuration to model correctly."""
        nu_value = 0.123
        config = {
            "model": {
                "physical": {
                    "name": "BurgersModel",
                    "domain": {"size_x": 100, "size_y": 100},
                    "resolution": {"x": 64, "y": 64},
                    "dt": 0.01,
                    "pde_params": {"batch_size": 1, "nu": nu_value},
                }
            }
        }

        model = ModelFactory.create_physical_model(config)
        assert model.nu == nu_value


class TestModelFactorySynthetic:
    """Tests for synthetic model factory."""

    def test_create_unet_model(self):
        """Test creating UNet through factory."""
        config = {
            "model": {
                "synthetic": {
                    "name": "UNet",
                    "input_specs": {"velocity": 2},
                    "output_specs": {"velocity": 2},
                    "architecture": {"levels": 4, "filters": 64, "batch_norm": True},
                }
            }
        }

        model = ModelFactory.create_synthetic_model(config)
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_factory_creates_torch_module(self):
        """Test that factory creates PyTorch module."""
        config = {
            "model": {
                "synthetic": {
                    "name": "UNet",
                    "input_specs": {"velocity": 2},
                    "output_specs": {"velocity": 2},
                    "architecture": {"levels": 4, "filters": 64, "batch_norm": True},
                }
            }
        }

        model = ModelFactory.create_synthetic_model(config)
        assert hasattr(model, "forward")
        assert callable(model.forward)

    def test_factory_passes_architecture_config(self):
        """Test that factory passes architecture config correctly."""
        config = {
            "model": {
                "synthetic": {
                    "name": "UNet",
                    "input_specs": {"velocity": 2},
                    "output_specs": {"velocity": 2},
                    "architecture": {"levels": 3, "filters": 32, "batch_norm": False},
                }
            }
        }

        model = ModelFactory.create_synthetic_model(config)
        # Model should be created successfully with custom architecture
        assert model is not None


class TestModelFactoryListModels:
    """Tests for listing available models."""

    def test_list_available_models_structure(self):
        """Test that list_available_models returns correct structure."""
        models = ModelFactory.list_available_models()

        assert isinstance(models, dict)
        assert "physical" in models
        assert "synthetic" in models
        assert isinstance(models["physical"], list)
        assert isinstance(models["synthetic"], list)

    def test_list_available_models_has_physical(self):
        """Test that physical models are listed."""
        models = ModelFactory.list_available_models()

        assert "BurgersModel" in models["physical"]
        assert "SmokeModel" in models["physical"]
        assert "HeatModel" in models["physical"]

    def test_list_available_models_has_synthetic(self):
        """Test that synthetic models are listed."""
        models = ModelFactory.list_available_models()

        assert "UNet" in models["synthetic"]

    def test_list_includes_all_registered_models(self):
        """Test that list includes all registered models."""
        from src.models.registry import ModelRegistry

        factory_models = ModelFactory.list_available_models()
        registry_physical = ModelRegistry.list_physical_models()
        registry_synthetic = ModelRegistry.list_synthetic_models()

        # Factory should list same models as registry
        assert set(factory_models["physical"]) == set(registry_physical)
        assert set(factory_models["synthetic"]) == set(registry_synthetic)


class TestModelFactoryIntegration:
    """Integration tests for model factory."""

    def test_factory_and_registry_consistency(self):
        """Test that factory and registry produce same models."""
        from src.models.registry import ModelRegistry

        config_dict = {
            "model": {
                "physical": {
                    "name": "BurgersModel",
                    "domain": {"size_x": 100, "size_y": 100},
                    "resolution": {"x": 64, "y": 64},
                    "dt": 0.01,
                    "pde_params": {"batch_size": 1, "nu": 0.1},
                }
            }
        }

        # Create through factory
        factory_model = ModelFactory.create_physical_model(config_dict)

        # Create through registry
        registry_model = ModelRegistry.get_physical_model(
            "BurgersModel", config_dict["model"]["physical"]
        )

        # Both should have same properties
        assert type(factory_model).__name__ == type(registry_model).__name__
        assert factory_model.nu == registry_model.nu
        assert factory_model.dt == registry_model.dt

    def test_factory_with_hydra_config_structure(self):
        """Test factory with Hydra-style nested config."""
        config = {
            "model": {
                "physical": {
                    "name": "BurgersModel",
                    "domain": {"size_x": 100, "size_y": 100},
                    "resolution": {"x": 128, "y": 128},
                    "dt": 0.5,
                    "pde_params": {"batch_size": 1, "nu": 0.1},
                },
                "synthetic": {
                    "name": "UNet",
                    "input_specs": {"velocity": 2},
                    "output_specs": {"velocity": 2},
                    "architecture": {"levels": 4, "filters": 64, "batch_norm": True},
                },
            },
            "data": {"dset_name": "burgers_128"},
            "trainer_params": {"epochs": 100},
        }

        # Factory should work with full Hydra config
        physical_model = ModelFactory.create_physical_model(config)
        synthetic_model = ModelFactory.create_synthetic_model(config)

        assert physical_model is not None
        assert synthetic_model is not None

    def test_multiple_models_created_independently(self):
        """Test that multiple models created through factory are independent."""
        config = {
            "model": {
                "physical": {
                    "name": "BurgersModel",
                    "domain": {"size_x": 100, "size_y": 100},
                    "resolution": {"x": 64, "y": 64},
                    "dt": 0.01,
                    "pde_params": {"batch_size": 1, "nu": 0.1},
                }
            }
        }

        model1 = ModelFactory.create_physical_model(config)
        model2 = ModelFactory.create_physical_model(config)

        # Should be different instances
        assert model1 is not model2

        # Modifying one shouldn't affect the other
        model1.nu = 0.5
        assert model2.nu == 0.1

    def test_factory_with_different_configs(self):
        """Test factory with various model configurations."""
        configs = [
            {
                "model": {
                    "physical": {
                        "name": "BurgersModel",
                        "domain": {"size_x": 50, "size_y": 50},
                        "resolution": {"x": 32, "y": 32},
                        "dt": 0.05,
                        "pde_params": {"batch_size": 1, "nu": 0.05},
                    }
                }
            },
            {
                "model": {
                    "physical": {
                        "name": "SmokeModel",
                        "domain": {"size_x": 100, "size_y": 100},
                        "resolution": {"x": 128, "y": 128},
                        "dt": 0.1,
                        "pde_params": {"batch_size": 2, "buoyancy_factor": 0.2},
                    }
                }
            },
            {
                "model": {
                    "physical": {
                        "name": "HeatModel",
                        "domain": {"size_x": 200, "size_y": 200},
                        "resolution": {"x": 64, "y": 64},
                        "dt": 0.01,
                        "pde_params": {"batch_size": 1, "diffusivity": 0.15},
                    }
                }
            },
        ]

        for config in configs:
            model = ModelFactory.create_physical_model(config)
            assert model is not None
            assert hasattr(model, "step")
