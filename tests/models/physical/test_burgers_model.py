# tests/models/physical/test_burgers_model.py

import pytest
from phi.torch.flow import (
    Box,
    extrapolation,
    batch,
    spatial
)
from phi.field import StaggeredGrid # Only need StaggeredGrid

# --- We must import the class we want to test ---
from src.models.physical.burgers import BurgersModel

# --- Test Parameters ---
TEST_RES_X = 64
TEST_RES_Y = 64
TEST_RESOLUTION = spatial(x=TEST_RES_X, y=TEST_RES_Y)
TEST_DOMAIN = Box(x=100, y=100) # From burgers.yaml
TEST_DT = 0.5                  # From burgers.yaml
TEST_BATCH_SIZE = 4            # Use a multi-batch size for testing
TEST_NU = 0.1                  # Use a non-zero value for testing

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
    # BurgersModel only returns one state: velocity
    vel = burgers_model.get_initial_state() 
    
    
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
    # 1. Get an initial state
    vel_0 = burgers_model.get_initial_state()
    
    # 2. Run one step (model is callable, which routes to 'step')
    vel_1 = burgers_model(vel_0)
    
    # The output shape must *exactly* match the input shape.
    assert vel_1.shape == vel_0.shape
    
    # The grid type and extrapolation should also be preserved
    assert vel_1.extrapolation == vel_0.extrapolation