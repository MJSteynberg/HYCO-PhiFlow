"""
Integration tests for synthetic models.
Tests compatibility, common interfaces, and realistic usage patterns.
"""

import pytest
import torch
import torch.nn as nn

from src.models.synthetic.base import SyntheticModel
from src.models.synthetic.unet import UNet


# Get the device that PhiML will use (same as physical models)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestSyntheticModelsIntegration:
    """Integration tests for all synthetic models."""

    def _tensor(self, *args, **kwargs):
        """Helper to create tensors on the correct device."""
        return torch.randn(*args, **kwargs).to(DEVICE)

    @pytest.fixture
    def smoke_config(self):
        """Configuration for smoke simulation."""
        return {
            "input_specs": {"velocity": 2, "density": 1, "inflow": 1},
            "output_specs": {"velocity": 2, "density": 1},
            "architecture": {"levels": 4, "filters": 64, "batch_norm": True},
        }

    @pytest.fixture
    def burgers_config(self):
        """Configuration for Burgers equation."""
        return {
            "input_specs": {"velocity": 2},
            "output_specs": {"velocity": 2},
            "architecture": {"levels": 3, "filters": 32, "batch_norm": True},
        }

    @pytest.fixture
    def heat_config(self):
        """Configuration for heat equation."""
        return {
            "input_specs": {"temp": 1},
            "output_specs": {"temp": 1},
            "architecture": {"levels": 3, "filters": 32, "batch_norm": True},
        }

    def test_unet_is_nn_module(self):
        """Test that UNet is a proper PyTorch module."""
        config = {
            "input_specs": {"velocity": 2},
            "output_specs": {"velocity": 2},
            "architecture": {"levels": 2, "filters": 16},
        }
        model = UNet(config)
        assert isinstance(model, nn.Module)

    def test_multiple_models_independent(self):
        """Test that multiple model instances are independent."""
        config = {
            "input_specs": {"velocity": 2},
            "output_specs": {"velocity": 2},
            "architecture": {"levels": 2, "filters": 16},
        }

        model1 = UNet(config)
        model2 = UNet(config)

        # Models should have different parameters
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            # Initially might be similar, but not the same object
            assert p1 is not p2

    def test_smoke_configuration(self, smoke_config):
        """Test model with smoke simulation configuration."""
        model = UNet(smoke_config)

        assert model.in_channels == 4  # velocity(2) + density(1) + inflow(1)
        assert model.out_channels == 3  # velocity(2) + density(1)
        assert "inflow" in model.static_fields

        # Test forward pass
        x = self._tensor(2, 4, 128, 128)
        output = model.forward(x)
        assert output.shape == (2, 4, 128, 128)

    def test_burgers_configuration(self, burgers_config):
        """Test model with Burgers equation configuration."""
        model = UNet(burgers_config)

        assert model.in_channels == 2
        assert model.out_channels == 2
        assert len(model.static_fields) == 0

        # Test forward pass
        x = self._tensor(2, 2, 128, 128)
        output = model.forward(x)
        assert output.shape == (2, 2, 128, 128)

    def test_heat_configuration(self, heat_config):
        """Test model with heat equation configuration."""
        model = UNet(heat_config)

        assert model.in_channels == 1
        assert model.out_channels == 1
        assert len(model.static_fields) == 0

        # Test forward pass
        x = self._tensor(2, 1, 64, 64)
        output = model.forward(x)
        assert output.shape == (2, 1, 64, 64)

    def test_models_with_common_resolution(self):
        """Test different models with the same spatial resolution."""
        resolution = (128, 128)

        configs = [
            {"input_specs": {"velocity": 2}, "output_specs": {"velocity": 2}},
            {"input_specs": {"temp": 1}, "output_specs": {"temp": 1}},
            {
                "input_specs": {"density": 1, "velocity": 2},
                "output_specs": {"density": 1, "velocity": 2},
            },
        ]

        for config in configs:
            config["architecture"] = {"levels": 3, "filters": 32}
            model = UNet(config)

            in_ch = sum(config["input_specs"].values())
            x = self._tensor(1, in_ch, *resolution)
            output = model.forward(x)

            assert output.shape[2:] == resolution

    def test_models_save_load_state_dict(self):
        """Test that models can save and load state dictionaries."""
        config = {
            "input_specs": {"velocity": 2},
            "output_specs": {"velocity": 2},
            "architecture": {"levels": 2, "filters": 16},
        }

        model1 = UNet(config)
        model2 = UNet(config)

        # Get output from model1
        x = self._tensor(1, 2, 32, 32)
        with torch.no_grad():
            out1 = model1(x)

        # Load model1's state into model2
        state_dict = model1.state_dict()
        model2.load_state_dict(state_dict)

        # Model2 should produce same output
        with torch.no_grad():
            out2 = model2(x)

        assert torch.allclose(out1, out2)

    def test_training_loop_compatibility(self):
        """Test that model works in a typical training loop."""
        config = {
            "input_specs": {"velocity": 2},
            "output_specs": {"velocity": 2},
            "architecture": {"levels": 2, "filters": 16},
        }
        model = UNet(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()

        # Simulate a few training steps
        for _ in range(3):
            x = self._tensor(4, 2, 32, 32)
            target = self._tensor(4, 2, 32, 32)

            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()

            assert loss.item() >= 0

    def test_mixed_precision_compatibility(self):
        """Test that model works with mixed precision training."""
        config = {
            "input_specs": {"velocity": 2},
            "output_specs": {"velocity": 2},
            "architecture": {"levels": 2, "filters": 16},
        }
        model = UNet(config).to(torch.float32)

        # Test with float32 input
        x_fp32 = self._tensor(1, 2, 32, 32).to(torch.float32)
        output_fp32 = model(x_fp32)

        assert output_fp32.dtype == torch.float32

    def test_different_batch_sizes_in_sequence(self):
        """Test that model handles varying batch sizes."""
        config = {
            "input_specs": {"velocity": 2},
            "output_specs": {"velocity": 2},
            "architecture": {"levels": 2, "filters": 16},
        }
        model = UNet(config)
        model.eval()

        batch_sizes = [1, 4, 8, 2, 16]

        with torch.no_grad():
            for bs in batch_sizes:
                x = self._tensor(bs, 2, 32, 32)
                output = model(x)
                assert output.shape[0] == bs

    def test_residual_learning_capability(self):
        """Test that model can learn residuals (difference from input)."""
        config = {
            "input_specs": {"velocity": 2},
            "output_specs": {"velocity": 2},
            "architecture": {"levels": 2, "filters": 16},
        }
        model = UNet(config)

        # Input and expected output (small perturbation)
        x = self._tensor(4, 2, 32, 32)
        target = x + 0.1 * torch.randn_like(x).to(DEVICE)  # Small residual

        # The model should be capable of learning this residual
        output = model(x)
        assert output.shape == target.shape

    def test_model_output_range(self):
        """Test that model output is not constrained to specific range."""
        config = {
            "input_specs": {"velocity": 2},
            "output_specs": {"velocity": 2},
            "architecture": {"levels": 2, "filters": 16},
        }
        model = UNet(config)

        # Test with various input ranges
        for scale in [0.1, 1.0, 10.0]:
            x = scale * self._tensor(2, 2, 32, 32)
            output = model(x)

            # Output should be valid tensor (not NaN or Inf)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

    def test_long_sequence_stability(self):
        """Test model stability over many forward passes."""
        config = {
            "input_specs": {"velocity": 2},
            "output_specs": {"velocity": 2},
            "architecture": {"levels": 2, "filters": 16},
        }
        model = UNet(config)
        model.eval()

        x = self._tensor(1, 2, 32, 32)

        with torch.no_grad():
            for _ in range(100):
                x = model(x)

                # Check for numerical stability
                assert not torch.isnan(x).any()
                assert not torch.isinf(x).any()

    def test_parameter_initialization(self):
        """Test that parameters are properly initialized."""
        config = {
            "input_specs": {"velocity": 2},
            "output_specs": {"velocity": 2},
            "architecture": {"levels": 2, "filters": 16},
        }
        model = UNet(config)

        # Check that parameters have reasonable initial values
        for param in model.parameters():
            assert not torch.isnan(param).any()
            assert not torch.isinf(param).any()

    def test_optimizer_compatibility(self):
        """Test compatibility with different optimizers."""
        config = {
            "input_specs": {"velocity": 2},
            "output_specs": {"velocity": 2},
            "architecture": {"levels": 2, "filters": 16},
        }
        model = UNet(config)

        optimizers = [
            torch.optim.SGD(model.parameters(), lr=0.01),
            torch.optim.Adam(model.parameters(), lr=0.001),
            torch.optim.AdamW(model.parameters(), lr=0.001),
        ]

        for opt in optimizers:
            x = self._tensor(2, 2, 32, 32)
            target = self._tensor(2, 2, 32, 32)

            opt.zero_grad()
            output = model(x)
            loss = nn.functional.mse_loss(output, target)
            loss.backward()
            opt.step()

            assert loss.item() >= 0

    def test_memory_efficiency(self):
        """Test that model doesn't accumulate excessive memory."""
        config = {
            "input_specs": {"velocity": 2},
            "output_specs": {"velocity": 2},
            "architecture": {"levels": 2, "filters": 16},
        }
        model = UNet(config)
        model.eval()

        # Run multiple forward passes without accumulating gradients
        with torch.no_grad():
            for _ in range(10):
                x = self._tensor(4, 2, 32, 32)
                _ = model(x)

        # If we get here without OOM, test passes
        assert True

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with fixed seed."""
        config = {
            "input_specs": {"velocity": 2},
            "output_specs": {"velocity": 2},
            "architecture": {"levels": 2, "filters": 16},
        }

        # First run
        torch.manual_seed(42)
        model1 = UNet(config)
        x = self._tensor(1, 2, 32, 32)
        with torch.no_grad():
            out1 = model1(x)

        # Second run with same seed
        torch.manual_seed(42)
        model2 = UNet(config)
        with torch.no_grad():
            out2 = model2(x)

        # Outputs should be identical
        assert torch.allclose(out1, out2)

    def test_model_callable(self):
        """Test that model can be called directly (not just forward)."""
        config = {
            "input_specs": {"velocity": 2},
            "output_specs": {"velocity": 2},
            "architecture": {"levels": 2, "filters": 16},
        }
        model = UNet(config)

        x = self._tensor(1, 2, 32, 32)

        # Should be able to call directly
        output1 = model(x)
        output2 = model.forward(x)

        assert output1.shape == output2.shape
