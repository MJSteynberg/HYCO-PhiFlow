"""Test script to verify input noise injection works correctly."""

import torch
from phi.torch.flow import *
from phi.math import math
from phiml.math import channel

# Minimal config for testing
config = {
    'model': {
        'synthetic': {
            'input_noise': {
                'enabled': True,
                'scale': 0.1,  # Use larger scale for testing
                'type': 'gaussian'
            }
        }
    }
}

# Import after config is set
from src.models.synthetic.base import SyntheticModel
from phiml import nn

# Create a simple test model
class TestSyntheticModel(SyntheticModel):
    """Minimal synthetic model for testing."""

    def __init__(self, config):
        super().__init__(config, num_channels=2, static_fields=[])
        # Simple 1D convolution network
        self._network = nn.conv_net(
            in_channels=2,
            out_channels=2,
            layers=[8, 8],
            batch_norm=False,
            activation='ReLU'
        )

def test_noise_injection():
    """Test that noise is only applied during training mode."""
    print("Testing input noise injection...\n")

    # Create model
    model = TestSyntheticModel(config)

    # Create test input: batch of 4, spatial size 32x32, 2 channels
    test_input = math.random_normal(batch(batch=4) & spatial(x=32, y=32) & channel(field='u,v'))

    print(f"Input shape: {test_input.shape}")
    print(f"Input noise enabled: {model.input_noise_enabled}")
    print(f"Input noise scale: {model.input_noise_scale}")
    print()

    # Test 1: Inference mode (training=False)
    print("Test 1: Inference mode (training=False)")
    model.training = False

    # Run same input through model multiple times
    with torch.no_grad():
        output1 = model(test_input)
        output2 = model(test_input)

    # Outputs should be identical (no noise)
    diff = math.mean(abs(output1 - output2))
    print(f"  Difference between two forward passes: {diff}")

    # Test 2: Training mode (training=True)
    print("Test 2: Training mode (training=True)")
    model.training = True

    # Run same input through model multiple times
    # Note: We use no_grad just to avoid tracking gradients for this test
    with torch.no_grad():
        outputs = [model(test_input) for _ in range(10)]

    # Calculate pairwise differences
    diffs = []
    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            diff = math.mean(abs(outputs[i] - outputs[j]))
            diffs.append(diff)
    print(diffs)
    avg_diff = math.mean(math.stack(diffs, 'batch'))
    print(f"  Average difference between forward passes: {avg_diff:.6f}")


    # Test 3: Verify noise scale is reasonable
    print("Test 3: Verify noise scale")
    model.training = True

    with torch.no_grad():
        # Get original output without noise
        model.training = False
        output_no_noise = model(test_input)

        # Get output with noise
        model.training = True
        output_with_noise = model(test_input)



if __name__ == "__main__":
    test_noise_injection()
