"""
Tests for Model Registry Pattern (Phase 3: Improved Modularity).

Tests the model registration and discovery system that allows
dynamic model loading without hard-coded imports.
"""

import pytest
from src.models.registry import ModelRegistry


class TestModelRegistryPhysical:
    """Tests for physical model registration."""

    def test_burgers_model_registered(self):
        """Test that BurgersModel is registered."""
        assert "BurgersModel" in ModelRegistry.list_physical_models()

    def test_smoke_model_registered(self):
        """Test that SmokeModel is registered."""
        assert "SmokeModel" in ModelRegistry.list_physical_models()

    def test_heat_model_registered(self):
        """Test that HeatModel is registered."""
        assert "HeatModel" in ModelRegistry.list_physical_models()

    def test_get_burgers_model(self):
        """Test getting BurgersModel from registry."""
        config = {
            "domain": {"size_x": 100, "size_y": 100},
            "resolution": {"x": 64, "y": 64},
            "dt": 0.01,
            "pde_params": {"batch_size": 1, "nu": 0.1},
        }

        model = ModelRegistry.get_physical_model("BurgersModel", config)
        assert model is not None
        assert hasattr(model, "step")
        assert hasattr(model, "get_initial_state")

    def test_get_smoke_model(self):
        """Test getting SmokeModel from registry."""
        config = {
            "domain": {"size_x": 100, "size_y": 100},
            "resolution": {"x": 64, "y": 64},
            "dt": 0.01,
            "pde_params": {"batch_size": 1, "buoyancy_factor": 0.1},
        }

        model = ModelRegistry.get_physical_model("SmokeModel", config)
        assert model is not None
        assert hasattr(model, "step")
        assert hasattr(model, "get_initial_state")

    def test_get_heat_model(self):
        """Test getting HeatModel from registry."""
        config = {
            "domain": {"size_x": 100, "size_y": 100},
            "resolution": {"x": 64, "y": 64},
            "dt": 0.01,
            "pde_params": {"batch_size": 1, "diffusivity": 0.1},
        }

        model = ModelRegistry.get_physical_model("HeatModel", config)
        assert model is not None
        assert hasattr(model, "step")
        assert hasattr(model, "get_initial_state")

    def test_get_unknown_physical_model(self):
        """Test that getting unknown model raises error."""
        with pytest.raises(ValueError, match="Physical model.*not found"):
            ModelRegistry.get_physical_model("UnknownModel", {})

    def test_list_physical_models_returns_list(self):
        """Test that list_physical_models returns a list."""
        models = ModelRegistry.list_physical_models()
        assert isinstance(models, list)

    def test_list_physical_models_has_expected_count(self):
        """Test that we have the expected number of physical models."""
        models = ModelRegistry.list_physical_models()
        # We have BurgersModel, SmokeModel, and HeatModel
        assert len(models) >= 3

    def test_physical_model_instance_type(self):
        """Test that created models have correct type."""
        config = {
            "domain": {"size_x": 100, "size_y": 100},
            "resolution": {"x": 64, "y": 64},
            "dt": 0.01,
            "pde_params": {"batch_size": 1, "nu": 0.1},
        }

        model = ModelRegistry.get_physical_model("BurgersModel", config)
        # Check it has the expected methods
        assert callable(getattr(model, "step", None))
        assert callable(getattr(model, "get_initial_state", None))


class TestModelRegistrySynthetic:
    """Tests for synthetic model registration."""

    def test_unet_model_registered(self):
        """Test that UNet is registered."""
        assert "UNet" in ModelRegistry.list_synthetic_models()

    def test_get_unet_model(self):
        """Test getting UNet from registry."""
        config = {
            "input_specs": {"velocity": 2},
            "output_specs": {"velocity": 2},
            "architecture": {"levels": 4, "filters": 64, "batch_norm": True},
        }

        model = ModelRegistry.get_synthetic_model("UNet", config)
        assert model is not None
        assert hasattr(model, "forward")

    def test_get_unknown_synthetic_model(self):
        """Test that getting unknown model raises error."""
        with pytest.raises(ValueError, match="Synthetic model.*not found"):
            ModelRegistry.get_synthetic_model("UnknownModel", {})

    def test_list_synthetic_models_returns_list(self):
        """Test that list_synthetic_models returns a list."""
        models = ModelRegistry.list_synthetic_models()
        assert isinstance(models, list)

    def test_list_synthetic_models_has_unet(self):
        """Test that UNet is in the list."""
        models = ModelRegistry.list_synthetic_models()
        assert "UNet" in models

    def test_synthetic_model_is_torch_module(self):
        """Test that synthetic models are PyTorch modules."""
        import torch.nn as nn

        config = {
            "input_specs": {"velocity": 2},
            "output_specs": {"velocity": 2},
            "architecture": {"levels": 4, "filters": 64, "batch_norm": True},
        }

        model = ModelRegistry.get_synthetic_model("UNet", config)
        assert isinstance(model, nn.Module)


class TestModelRegistryIntegration:
    """Integration tests for the model registry."""

    def test_can_create_all_physical_models(self):
        """Test that all registered physical models can be instantiated."""
        configs = {
            "BurgersModel": {
                "domain": {"size_x": 100, "size_y": 100},
                "resolution": {"x": 64, "y": 64},
                "dt": 0.01,
                "pde_params": {"batch_size": 1, "nu": 0.1},
            },
            "SmokeModel": {
                "domain": {"size_x": 100, "size_y": 100},
                "resolution": {"x": 64, "y": 64},
                "dt": 0.01,
                "pde_params": {"batch_size": 1, "buoyancy_factor": 0.1},
            },
            "HeatModel": {
                "domain": {"size_x": 100, "size_y": 100},
                "resolution": {"x": 64, "y": 64},
                "dt": 0.01,
                "pde_params": {"batch_size": 1, "diffusivity": 0.1},
            },
        }

        for model_name in ModelRegistry.list_physical_models():
            if model_name in configs:
                config = configs[model_name]
                model = ModelRegistry.get_physical_model(model_name, config)
                assert model is not None

    def test_can_create_all_synthetic_models(self):
        """Test that all registered synthetic models can be instantiated."""
        config = {
            "input_specs": {"velocity": 2},
            "output_specs": {"velocity": 2},
            "architecture": {"levels": 4, "filters": 64, "batch_norm": True},
        }

        for model_name in ModelRegistry.list_synthetic_models():
            model = ModelRegistry.get_synthetic_model(model_name, config)
            assert model is not None

    def test_physical_models_have_consistent_interface(self):
        """Test that all physical models have the same interface."""
        configs = {
            "BurgersModel": {
                "domain": {"size_x": 100, "size_y": 100},
                "resolution": {"x": 64, "y": 64},
                "dt": 0.01,
                "pde_params": {"batch_size": 1, "nu": 0.1},
            },
            "SmokeModel": {
                "domain": {"size_x": 100, "size_y": 100},
                "resolution": {"x": 64, "y": 64},
                "dt": 0.01,
                "pde_params": {"batch_size": 1, "buoyancy_factor": 0.1},
            },
            "HeatModel": {
                "domain": {"size_x": 100, "size_y": 100},
                "resolution": {"x": 64, "y": 64},
                "dt": 0.01,
                "pde_params": {"batch_size": 1, "diffusivity": 0.1},
            },
        }

        required_methods = ["step", "get_initial_state", "__call__"]

        for model_name, config in configs.items():
            model = ModelRegistry.get_physical_model(model_name, config)
            for method in required_methods:
                assert hasattr(model, method), f"{model_name} missing {method}"
                assert callable(
                    getattr(model, method)
                ), f"{model_name}.{method} not callable"

    def test_registry_is_case_sensitive(self):
        """Test that model names are case-sensitive."""
        config = {
            "domain": {"size_x": 100, "size_y": 100},
            "resolution": {"x": 64, "y": 64},
            "dt": 0.01,
            "pde_params": {"batch_size": 1, "nu": 0.1},
        }

        # Correct case should work
        model = ModelRegistry.get_physical_model("BurgersModel", config)
        assert model is not None

        # Wrong case should fail
        with pytest.raises(ValueError):
            ModelRegistry.get_physical_model("burgersmodel", config)

        with pytest.raises(ValueError):
            ModelRegistry.get_physical_model("BURGERSMODEL", config)

    def test_models_created_independently(self):
        """Test that multiple model instances are independent."""
        config = {
            "domain": {"size_x": 100, "size_y": 100},
            "resolution": {"x": 64, "y": 64},
            "dt": 0.01,
            "pde_params": {"batch_size": 1, "nu": 0.1},
        }

        model1 = ModelRegistry.get_physical_model("BurgersModel", config)
        model2 = ModelRegistry.get_physical_model("BurgersModel", config)

        # Should be different instances
        assert model1 is not model2

        # Modifying one should not affect the other
        model1.nu = 0.5
        assert model2.nu == 0.1
