"""
Tests for UNet synthetic model.
Tests architecture, forward pass, channel handling, and static/dynamic field separation.
"""

import pytest
import torch
import torch.nn as nn

from src.models.synthetic.unet import UNet


# Get the device that PhiML will use (same as physical models)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestUNet:
    """Tests for UNet model."""
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing."""
        return {
            'input_specs': {'velocity': 2, 'density': 1},
            'output_specs': {'velocity': 2, 'density': 1},
            'architecture': {
                'levels': 3,
                'filters': 32,
                'batch_norm': True
            }
        }
    
    @pytest.fixture
    def model(self, basic_config):
        """Create a UNet model instance."""
        return UNet(basic_config)
    
    def _tensor(self, *args, **kwargs):
        """Helper to create tensors on the correct device."""
        return torch.randn(*args, **kwargs).to(DEVICE)
    
    def test_initialization(self, basic_config):
        """Test that UNet can be initialized."""
        model = UNet(basic_config)
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_config_storage(self, model, basic_config):
        """Test that config is stored correctly."""
        assert model.config == basic_config
        assert model.input_specs == basic_config['input_specs']
        assert model.output_specs == basic_config['output_specs']
    
    def test_channel_calculation(self, model):
        """Test that input/output channels are calculated correctly."""
        assert model.in_channels == 3  # 2 for velocity + 1 for density
        assert model.out_channels == 3
    
    def test_no_static_fields(self, model):
        """Test identification of static fields when all fields are dynamic."""
        assert len(model.static_fields) == 0
        assert len(model.dynamic_fields) == 2
        assert 'velocity' in model.dynamic_fields
        assert 'density' in model.dynamic_fields
    
    def test_static_field_identification(self):
        """Test correct identification of static fields."""
        config = {
            'input_specs': {'velocity': 2, 'density': 1, 'inflow': 1},
            'output_specs': {'velocity': 2, 'density': 1},
            'architecture': {'levels': 2, 'filters': 16}
        }
        model = UNet(config)
        assert 'inflow' in model.static_fields
        assert 'inflow' not in model.dynamic_fields
        assert len(model.static_fields) == 1
        assert len(model.dynamic_fields) == 2
    
    def test_channel_indices_building(self, model):
        """Test that channel indices are built correctly."""
        assert hasattr(model, 'input_channel_map')
        assert 'velocity' in model.input_channel_map
        assert 'density' in model.input_channel_map
        
        vel_start, vel_end = model.input_channel_map['velocity']
        dens_start, dens_end = model.input_channel_map['density']
        
        # Check ranges are correct
        assert vel_end - vel_start == 2
        assert dens_end - dens_start == 1
        
        # Check no overlap
        assert vel_end <= dens_start or dens_end <= vel_start
    
    def test_unet_attribute(self, model):
        """Test that unet attribute exists."""
        assert hasattr(model, 'unet')
        assert model.unet is not None
    
    def test_forward_shape_preservation(self, model):
        """Test that forward pass preserves tensor shape."""
        batch_size = 2
        height, width = 32, 32
        x = self._tensor(batch_size, model.in_channels, height, width)
        
        output = model.forward(x)
        
        assert output.shape == x.shape
        assert output.shape[0] == batch_size
        assert output.shape[1] == model.in_channels
        assert output.shape[2] == height
        assert output.shape[3] == width
    
    def test_forward_no_static_fields(self, model):
        """Test forward pass when there are no static fields."""
        x = self._tensor(1, 3, 32, 32)
        output = model.forward(x)
        
        assert output.shape == x.shape
        assert torch.is_tensor(output)
    
    def test_forward_with_static_fields(self):
        """Test forward pass preserves static fields."""
        config = {
            'input_specs': {'velocity': 2, 'density': 1, 'inflow': 1},
            'output_specs': {'velocity': 2, 'density': 1},
            'architecture': {'levels': 2, 'filters': 16}
        }
        model = UNet(config)
        
        # Create input with known static field values
        x = self._tensor(2, 4, 32, 32)
        # Set inflow to specific values
        inflow_start, inflow_end = model.input_channel_map['inflow']
        x[:, inflow_start:inflow_end, :, :] = 5.0
        
        output = model.forward(x)
        
        # Check that static field is preserved
        assert torch.allclose(output[:, inflow_start:inflow_end, :, :], 
                            x[:, inflow_start:inflow_end, :, :])
    
    def test_different_resolutions(self, model):
        """Test forward pass with different spatial resolutions."""
        resolutions = [(16, 16), (32, 32), (64, 64), (128, 128)]
        
        for h, w in resolutions:
            x = self._tensor(1, model.in_channels, h, w)
            output = model.forward(x)
            assert output.shape == (1, model.in_channels, h, w)
    
    def test_batch_sizes(self, model):
        """Test forward pass with different batch sizes."""
        batch_sizes = [1, 2, 4, 8, 16]
        
        for bs in batch_sizes:
            x = self._tensor(bs, model.in_channels, 32, 32)
            output = model.forward(x)
            assert output.shape[0] == bs
    
    def test_gradient_flow(self, model):
        """Test that gradients flow through the model."""
        model.train()
        x = self._tensor(2, model.in_channels, 32, 32)
        x.requires_grad = True
        
        output = model.forward(x)
        loss = output.sum()
        loss.backward()
        
        # Check that input has gradients
        assert x.grad is not None
        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_architecture_levels(self):
        """Test different architecture levels."""
        levels_list = [2, 3, 4, 5]
        
        for levels in levels_list:
            config = {
                'input_specs': {'velocity': 2},
                'output_specs': {'velocity': 2},
                'architecture': {'levels': levels, 'filters': 16}
            }
            model = UNet(config)
            x = self._tensor(1, 2, 64, 64)
            output = model.forward(x)
            assert output.shape == x.shape
    
    def test_architecture_filters(self):
        """Test different filter counts."""
        filters_list = [16, 32, 64, 128]
        
        for filters in filters_list:
            config = {
                'input_specs': {'velocity': 2},
                'output_specs': {'velocity': 2},
                'architecture': {'levels': 2, 'filters': filters}
            }
            model = UNet(config)
            x = self._tensor(1, 2, 32, 32)
            output = model.forward(x)
            assert output.shape == x.shape
    
    def test_batch_norm_option(self):
        """Test batch normalization on/off."""
        for batch_norm in [True, False]:
            config = {
                'input_specs': {'velocity': 2},
                'output_specs': {'velocity': 2},
                'architecture': {'levels': 2, 'filters': 16, 'batch_norm': batch_norm}
            }
            model = UNet(config)
            x = self._tensor(2, 2, 32, 32)
            output = model.forward(x)
            assert output.shape == x.shape
    
    def test_default_architecture(self):
        """Test that default architecture values work."""
        config = {
            'input_specs': {'velocity': 2},
            'output_specs': {'velocity': 2}
        }
        model = UNet(config)
        x = self._tensor(1, 2, 64, 64)
        output = model.forward(x)
        assert output.shape == x.shape
    
    def test_single_channel_input(self):
        """Test with single channel input/output."""
        config = {
            'input_specs': {'density': 1},
            'output_specs': {'density': 1},
            'architecture': {'levels': 2, 'filters': 16}
        }
        model = UNet(config)
        assert model.in_channels == 1
        assert model.out_channels == 1
        
        x = self._tensor(1, 1, 32, 32)
        output = model.forward(x)
        assert output.shape == x.shape
    
    def test_multi_channel_fields(self):
        """Test with multi-channel vector fields."""
        config = {
            'input_specs': {'velocity': 3, 'tensor': 9},  # 3D velocity, 3x3 tensor
            'output_specs': {'velocity': 3, 'tensor': 9},
            'architecture': {'levels': 2, 'filters': 16}
        }
        model = UNet(config)
        assert model.in_channels == 12
        assert model.out_channels == 12
        
        x = self._tensor(1, 12, 32, 32)
        output = model.forward(x)
        assert output.shape == x.shape
    
    def test_multiple_static_fields(self):
        """Test with multiple static fields."""
        config = {
            'input_specs': {'velocity': 2, 'density': 1, 'inflow': 1, 'obstacle': 1},
            'output_specs': {'velocity': 2, 'density': 1},
            'architecture': {'levels': 2, 'filters': 16}
        }
        model = UNet(config)
        assert len(model.static_fields) == 2
        assert 'inflow' in model.static_fields
        assert 'obstacle' in model.static_fields
        
        x = self._tensor(2, 5, 32, 32)
        output = model.forward(x)
        assert output.shape == x.shape
    
    def test_output_channel_order(self):
        """Test that output channels maintain input field order."""
        config = {
            'input_specs': {'a': 1, 'b': 2, 'c': 1},
            'output_specs': {'a': 1, 'b': 2, 'c': 1},
            'architecture': {'levels': 2, 'filters': 8}
        }
        model = UNet(config)
        
        # Create input with distinct values per field
        x = torch.zeros(1, 4, 16, 16, device=DEVICE)
        x[:, 0:1, :, :] = 1.0   # field 'a'
        x[:, 1:3, :, :] = 2.0   # field 'b'
        x[:, 3:4, :, :] = 3.0   # field 'c'
        
        output = model.forward(x)
        
        # Output should have same shape
        assert output.shape == (1, 4, 16, 16)
    
    def test_train_eval_modes(self, model):
        """Test switching between train and eval modes."""
        model.train()
        assert model.training
        
        model.eval()
        assert not model.training
        
        # Test that forward works in both modes
        x = self._tensor(1, model.in_channels, 32, 32)
        
        model.train()
        out_train = model.forward(x)
        
        model.eval()
        out_eval = model.forward(x)
        
        # Both should produce valid outputs
        assert out_train.shape == out_eval.shape
    
    def test_parameter_count(self, model):
        """Test that model has learnable parameters."""
        params = list(model.parameters())
        assert len(params) > 0
        
        total_params = sum(p.numel() for p in params)
        assert total_params > 0
    
    def test_device_compatibility(self, model):
        """Test that model is on the expected device."""
        # Check that model parameters are on the expected device
        for param in model.parameters():
            assert param.device.type == DEVICE.type
    
    def test_deterministic_output(self, model):
        """Test that model produces deterministic output in eval mode."""
        model.eval()
        x = self._tensor(1, model.in_channels, 32, 32)
        
        with torch.no_grad():
            out1 = model.forward(x)
            out2 = model.forward(x)
        
        assert torch.allclose(out1, out2)
    
    def test_static_field_order_preservation(self):
        """Test that static fields are reconstructed in correct order."""
        config = {
            'input_specs': {'density': 1, 'inflow': 1, 'velocity': 2, 'obstacle': 1},
            'output_specs': {'velocity': 2, 'density': 1},
            'architecture': {'levels': 2, 'filters': 8}
        }
        model = UNet(config)
        
        # Create input with identifiable values
        x = self._tensor(1, 5, 16, 16)
        inflow_vals = x[:, 1:2, :, :].clone()
        obstacle_vals = x[:, 4:5, :, :].clone()
        
        output = model.forward(x)
        
        # Check that static fields are in the correct positions
        assert torch.allclose(output[:, 1:2, :, :], inflow_vals)
        assert torch.allclose(output[:, 4:5, :, :], obstacle_vals)
