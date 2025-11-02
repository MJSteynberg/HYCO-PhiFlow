"""
Tests for SyntheticModel base class.
Tests abstract base class interface and common functionality.
"""

import pytest
import torch
import torch.nn as nn
from phi.flow import Box, CenteredGrid, StaggeredGrid, spatial

from src.models.synthetic.base import SyntheticModel


class ConcreteSyntheticModel(SyntheticModel):
    """Concrete implementation of SyntheticModel for testing."""

    def __init__(self, config):
        super().__init__(config)
        # Simple linear layer for testing
        in_channels = sum(self.INPUT_SPECS.values())
        out_channels = sum(self.OUTPUT_SPECS.values())
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, state, dt=0.0):
        """Simple forward pass for testing."""
        # Return a copy of the input state with slight modifications
        output = {}
        for field_name in self.OUTPUT_FIELDS:
            if field_name in state:
                output[field_name] = state[field_name]
        return output


class TestSyntheticModelBase:
    """Tests for SyntheticModel abstract base class."""

    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing."""
        return {
            "input_specs": {"velocity": 2, "density": 1},
            "output_specs": {"velocity": 2, "density": 1},
        }

    @pytest.fixture
    def model(self, basic_config):
        """Create a concrete model instance."""
        return ConcreteSyntheticModel(basic_config)

    def test_initialization(self, basic_config):
        """Test that model can be initialized."""
        model = ConcreteSyntheticModel(basic_config)
        assert model is not None
        assert isinstance(model, SyntheticModel)
        assert isinstance(model, nn.Module)

    def test_config_storage(self, model, basic_config):
        """Test that config is stored correctly."""
        assert model.config == basic_config
        assert model.INPUT_SPECS == basic_config["input_specs"]
        assert model.OUTPUT_SPECS == basic_config["output_specs"]

    def test_input_specs(self, model):
        """Test that INPUT_SPECS is set correctly."""
        assert "velocity" in model.INPUT_SPECS
        assert "density" in model.INPUT_SPECS
        assert model.INPUT_SPECS["velocity"] == 2
        assert model.INPUT_SPECS["density"] == 1

    def test_output_specs(self, model):
        """Test that OUTPUT_SPECS is set correctly."""
        assert "velocity" in model.OUTPUT_SPECS
        assert "density" in model.OUTPUT_SPECS
        assert model.OUTPUT_SPECS["velocity"] == 2
        assert model.OUTPUT_SPECS["density"] == 1

    def test_input_fields(self, model):
        """Test that INPUT_FIELDS list is derived correctly."""
        assert len(model.INPUT_FIELDS) == 2
        assert "velocity" in model.INPUT_FIELDS
        assert "density" in model.INPUT_FIELDS

    def test_output_fields(self, model):
        """Test that OUTPUT_FIELDS list is derived correctly."""
        assert len(model.OUTPUT_FIELDS) == 2
        assert "velocity" in model.OUTPUT_FIELDS
        assert "density" in model.OUTPUT_FIELDS

    def test_empty_config(self):
        """Test initialization with empty config."""
        config = {}
        model = ConcreteSyntheticModel(config)
        assert model.INPUT_SPECS == {}
        assert model.OUTPUT_SPECS == {}
        assert model.INPUT_FIELDS == []
        assert model.OUTPUT_FIELDS == []

    def test_default_specs(self):
        """Test that missing specs default to empty dict."""
        config = {"some_other_key": "value"}
        model = ConcreteSyntheticModel(config)
        assert model.INPUT_SPECS == {}
        assert model.OUTPUT_SPECS == {}

    def test_abstract_forward_method(self):
        """Test that SyntheticModel enforces forward method."""
        # Cannot instantiate abstract class directly
        with pytest.raises(TypeError):
            SyntheticModel({"input_specs": {}, "output_specs": {}})

    def test_forward_method_exists(self, model):
        """Test that concrete model has forward method."""
        assert hasattr(model, "forward")
        assert callable(model.forward)

    def test_forward_signature(self, model):
        """Test forward method signature."""
        import inspect

        sig = inspect.signature(model.forward)
        assert "state" in sig.parameters
        assert "dt" in sig.parameters
        # dt should have default value
        assert sig.parameters["dt"].default == 0.0

    def test_inheritance(self, model):
        """Test that model inherits from both nn.Module and ABC."""
        assert isinstance(model, nn.Module)
        assert isinstance(model, SyntheticModel)

    def test_pytorch_module_methods(self, model):
        """Test that PyTorch Module methods are available."""
        assert hasattr(model, "parameters")
        assert hasattr(model, "train")
        assert hasattr(model, "eval")
        assert hasattr(model, "state_dict")
        assert hasattr(model, "load_state_dict")

    def test_different_input_output_specs(self):
        """Test model with different input and output specs."""
        config = {
            "input_specs": {"velocity": 2, "density": 1, "temp": 1},
            "output_specs": {"velocity": 2, "density": 1},
        }
        model = ConcreteSyntheticModel(config)
        assert len(model.INPUT_FIELDS) == 3
        assert len(model.OUTPUT_FIELDS) == 2
        assert "temp" in model.INPUT_FIELDS
        assert "temp" not in model.OUTPUT_FIELDS

    def test_multi_channel_fields(self):
        """Test model with fields having different channel counts."""
        config = {
            "input_specs": {"velocity": 3, "tensor": 9, "scalar": 1},
            "output_specs": {"velocity": 3, "scalar": 1},
        }
        model = ConcreteSyntheticModel(config)
        assert model.INPUT_SPECS["velocity"] == 3
        assert model.INPUT_SPECS["tensor"] == 9
        assert model.INPUT_SPECS["scalar"] == 1

    def test_config_with_additional_keys(self):
        """Test that additional config keys don't interfere."""
        config = {
            "input_specs": {"velocity": 2},
            "output_specs": {"velocity": 2},
            "architecture": {"levels": 4},
            "training": {"lr": 0.001},
            "other_param": "value",
        }
        model = ConcreteSyntheticModel(config)
        assert model.config == config
        assert "architecture" in model.config
        assert "training" in model.config

    def test_field_order_preservation(self):
        """Test that field order is preserved from specs."""
        config = {
            "input_specs": {"density": 1, "velocity": 2, "temp": 1},
            "output_specs": {"velocity": 2, "temp": 1, "density": 1},
        }
        model = ConcreteSyntheticModel(config)
        # In Python 3.7+, dict order is preserved
        assert list(model.INPUT_SPECS.keys()) == ["density", "velocity", "temp"]
        assert list(model.OUTPUT_SPECS.keys()) == ["velocity", "temp", "density"]
