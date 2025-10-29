# In tests/models/synthetic/test_unet_model.py

import pytest
from phiml.math import extrapolation, batch, spatial, channel
from phi.field import CenteredGrid, StaggeredGrid
from phi.geom import Box

from src.models.synthetic.base import SyntheticModel
from src.models.synthetic.unet import UNet

# --- Use your preferred constants ---
TEST_RES_X = 64
TEST_RES_Y = 80
TEST_RESOLUTION = spatial(x=TEST_RES_X, y=TEST_RES_Y)
TEST_DOMAIN = Box(x=100, y=100)
# ---

@pytest.fixture
def unet_config():
    """
    Provides a standard U-Net configuration for smoke simulation.
    """
    return {
        'name': 'UNet',
        'levels': 3,
        'filters': 16,
        'in_channels': 3,   # density (1) + velocity (2)
        'out_channels': 3,  # density (1) + velocity (2)
        
        'input_fields': ['density', 'velocity'],
        'input_channels': [1, 2],
        
        'output_fields': ['density', 'velocity'],
        'output_channels': [1, 2],
        
        'staggered_input_fields': ['velocity']
    }

@pytest.fixture
def unet_model(unet_config):
    """
    Provides an initialized UNet model instance.
    """
    return UNet(unet_config)

def test_unet_initialization(unet_model):
    """
    Tests if the UNet model is initialized correctly.
    """
    assert unet_model is not None
    assert isinstance(unet_model, UNet)
    assert isinstance(unet_model, SyntheticModel)
    
    # Check that the config was processed correctly
    assert unet_model.input_fields_list == ['density', 'velocity']
    assert unet_model.staggered_input_fields == ['velocity']
    
    # --- THIS IS THE FIX ---
    # Check the config dict on the model, not the model attributes
    assert unet_model.config['in_channels'] == 3
    assert unet_model.config['out_channels'] == 3

# ... (rest of the file, including test_unet_forward_pass, is unchanged) ...

def test_unet_forward_pass(unet_model):
    """
    Tests a single forward pass of the UNet model,
    checking input and output field types and shapes.
    """
    # 1. Create a sample input state
    
    # Create a StaggeredGrid for velocity
    velocity = StaggeredGrid(
        values=0.0,
        extrapolation=extrapolation.ZERO,
        domain=TEST_DOMAIN,
        resolution=TEST_RESOLUTION
    )
    
    # Create a CenteredGrid for density
    density = CenteredGrid(
        values=1.0,
        extrapolation=extrapolation.ZERO,
        domain=TEST_DOMAIN,
        resolution=TEST_RESOLUTION
    )
    
    state_in = {
        'density': density,
        'velocity': velocity
    }
    
    # 2. Perform the forward pass
    state_out = unet_model.forward(state_in, dt=1.0)
    
    # 3. Check the output state
    assert isinstance(state_out, dict)
    
    # Check that the correct output fields exist
    assert 'density' in state_out
    assert 'velocity' in state_out
    
    # Check that outputs are CenteredGrids (as designed)
    pred_density = state_out['density']
    pred_velocity = state_out['velocity']
    
    # Check that shapes are correct
    # Density: (x, y, 1 channel)
    assert pred_density.shape.spatial == TEST_RESOLUTION
    assert pred_density.shape.channel == channel(vector=1)
    
    # Velocity: (x, y, 2 channels)
    assert pred_velocity.shape.spatial == TEST_RESOLUTION
    assert pred_velocity.shape.channel == channel(vector=2)