# In tests/models/physical/test_burgers_model.py

import pytest
from phi.torch.flow import (
    Box,
    extrapolation,
    batch,
    spatial
)
from phi.field import StaggeredGrid 

# --- We must import the class we want to test ---
from src.models.physical.burgers import BurgersModel

# --- Test Parameters ---
TEST_RES_X = 64
TEST_RES_Y = 64
TEST_RESOLUTION = spatial(x=TEST_RES_X, y=TEST_RES_Y)
TEST_DOMAIN = Box(x=100, y=100) 
TEST_DT = 0.5                  
TEST_BATCH_SIZE = 4            
TEST_NU = 0.1                  

@pytest.fixture(scope="module")
def burgers_model() -> BurgersModel:
    """
    A pytest fixture to create a single, reusable BurgersModel
    instance for all tests in this file.
    """
    print("Setting up BurgersModel fixture...")
    
    model = BurgersModel(
        domain=TEST_DOMAIN,
        resolution=TEST_RESOLUTION,
        dt=TEST_DT,
        batch_size=TEST_BATCH_SIZE,
        nu=TEST_NU
    )
    return model

def test_burgers_model_init(burgers_model: BurgersModel):
    """
    Tests if the model was initialized with the correct parameters.
    """
    assert burgers_model.dt == TEST_DT
    assert burgers_model.nu == TEST_NU
    assert burgers_model.domain == TEST_DOMAIN
    assert burgers_model.resolution == TEST_RESOLUTION
    assert burgers_model.batch_size == TEST_BATCH_SIZE

def test_get_initial_state(burgers_model: BurgersModel):
    """
    Tests the 'get_initial_state' method.
    """
    # --- MODIFIED: Expect a dictionary ---
    state_0 = burgers_model.get_initial_state() 
    
    assert isinstance(state_0, dict)
    assert 'velocity' in state_0
    
    vel = state_0['velocity']
    
    # Check batch dimension
    assert 'batch' in vel.shape
    assert vel.shape['batch'].size == TEST_BATCH_SIZE
    
    # Check spatial resolution
    assert vel.shape.spatial == TEST_RESOLUTION
    
    # Check boundaries (specific to this model)
    assert vel.extrapolation == extrapolation.PERIODIC

def test_step_function(burgers_model: BurgersModel):
    """
    Tests the 'step' method for one iteration.
    """
    # 1. Get an initial state dictionary
    state_0_dict = burgers_model.get_initial_state()
    vel_0 = state_0_dict['velocity']

    print(f"Initial velocity shape: {vel_0}")
    
    # 2. Run one step by passing the field as a positional argument
    # --- MODIFIED: Unpack the dict before calling ---
    state_1_dict = burgers_model(vel_0)
    
    # 3. Check the returned dictionary
    assert isinstance(state_1_dict, dict)
    assert 'velocity' in state_1_dict
    
    vel_1 = state_1_dict['velocity']
    
    # The output shape must *exactly* match the input shape.
    assert vel_1.shape == vel_0.shape
    