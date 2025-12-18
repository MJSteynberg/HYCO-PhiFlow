"""Unit tests for ModelRegistry."""

import pytest
from src.models.registry import ModelRegistry


class TestModelRegistry:
    """Test suite for ModelRegistry class."""

    def test_register_physical_model(self):
        """Test registering a physical model."""
        # Clear registry first
        ModelRegistry._physical_models = {}

        @ModelRegistry.register_physical("TestPhysical")
        class TestPhysicalModel:
            pass

        assert "TestPhysical" in ModelRegistry._physical_models
        assert ModelRegistry._physical_models["TestPhysical"] == TestPhysicalModel

    def test_register_synthetic_model(self):
        """Test registering a synthetic model."""
        # Clear registry first
        ModelRegistry._synthetic_models = {}

        @ModelRegistry.register_synthetic("TestSynthetic")
        class TestSyntheticModel:
            pass

        assert "TestSynthetic" in ModelRegistry._synthetic_models
        assert ModelRegistry._synthetic_models["TestSynthetic"] == TestSyntheticModel

    def test_list_physical_models(self):
        """Test listing physical models."""
        ModelRegistry._physical_models = {"ModelA": None, "ModelB": None}
        result = ModelRegistry.list_physical_models()
        assert result == ["ModelA", "ModelB"]

    def test_list_synthetic_models(self):
        """Test listing synthetic models."""
        ModelRegistry._synthetic_models = {"UNet": None, "ResNet": None}
        result = ModelRegistry.list_synthetic_models()
        assert result == ["ResNet", "UNet"]  # Sorted alphabetically

    def test_is_physical_model_registered(self):
        """Test checking if physical model is registered."""
        ModelRegistry._physical_models = {"BurgersModel": None}
        assert ModelRegistry.is_physical_model_registered("BurgersModel") is True
        assert ModelRegistry.is_physical_model_registered("NonExistent") is False

    def test_is_synthetic_model_registered(self):
        """Test checking if synthetic model is registered."""
        ModelRegistry._synthetic_models = {"UNet": None}
        assert ModelRegistry.is_synthetic_model_registered("UNet") is True
        assert ModelRegistry.is_synthetic_model_registered("NonExistent") is False

    def test_clear_registry(self):
        """Test clearing the registry."""
        ModelRegistry._physical_models = {"Model": None}
        ModelRegistry._synthetic_models = {"Model": None}

        ModelRegistry.clear_registry()

        assert len(ModelRegistry._physical_models) == 0
        assert len(ModelRegistry._synthetic_models) == 0

    def test_get_physical_model_not_found(self):
        """Test that getting non-existent physical model raises ValueError."""
        ModelRegistry._physical_models = {}
        config = {"model": {"physical": {"name": "NonExistent"}}}

        with pytest.raises(ValueError, match="not found in registry"):
            ModelRegistry.get_physical_model(config)

    def test_get_synthetic_model_not_found(self):
        """Test that getting non-existent synthetic model raises ValueError."""
        ModelRegistry._synthetic_models = {}
        config = {"model": {"synthetic": {"name": "NonExistent"}}}

        with pytest.raises(ValueError, match="not found in registry"):
            ModelRegistry.get_synthetic_model(config)

    def test_overwrite_replaces_model(self):
        """Test that registering a model with same name replaces the previous."""
        ModelRegistry._physical_models = {}

        @ModelRegistry.register_physical("DuplicateModel")
        class FirstModel:
            pass

        @ModelRegistry.register_physical("DuplicateModel")
        class SecondModel:
            pass

        # Second model should have replaced first
        assert ModelRegistry._physical_models["DuplicateModel"] == SecondModel
        assert ModelRegistry._physical_models["DuplicateModel"] != FirstModel
