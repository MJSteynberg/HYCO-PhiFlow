# tests/models/physical/test_smoke_model.py

import pytest
from phi.torch.flow import (
    Box,
    extrapolation,
    math,
    batch,
    spatial,
    Solve # Import Solve
)
from phi.field import CenteredGrid, StaggeredGrid # Use direct imports
from phi.geom import Sphere # Import Sphere

# --- We must import the class we want to test ---
# (Make sure your file is named 'smoke_model.py')
from src.models.physical.smoke import SmokeModel

# --- Test Parameters ---
TEST_RES_X = 64
TEST_RES_Y = 80
TEST_RESOLUTION = spatial(x=TEST_RES_X, y=TEST_RES_Y)
TEST_DOMAIN = Box(x=80, y=100) # Physical size
TEST_DT = 1.0
TEST_BATCH_SIZE = 4
# --- Set NU to 0.0 to avoid the commented-out code ---
TEST_NU = 0.0 
TEST_INFLOW_CENTER = (40.0, 20.0) # A different center for testing
TEST_INFLOW_RADIUS = 10.0
TEST_INFLOW_RATE = 0.2


@pytest.fixture(scope="module")
def smoke_model() -> SmokeModel:
    """
    A pytest fixture to create a single, reusable SmokeModel
    instance for all tests in this file.
    """
    print("Setting up SmokeModel fixture...")
    
    model = SmokeModel(
        domain=TEST_DOMAIN,
        resolution=TEST_RESOLUTION,
        dt=TEST_DT,
        batch_size=TEST_BATCH_SIZE, # Pass new param
        nu=TEST_NU,
        buoyancy=1.0,
        inflow_center=TEST_INFLOW_CENTER, # Pass new param
        inflow_radius=TEST_INFLOW_RADIUS, # Pass new param
        inflow_rate=TEST_INFLOW_RATE      # Pass new param
    )
    return model


def test_smoke_model_init(smoke_model: SmokeModel):
    """
    Tests if the model was initialized with the correct parameters.
    """
    assert smoke_model.dt == TEST_DT
    assert smoke_model.nu == TEST_NU
    assert smoke_model.domain == TEST_DOMAIN
    assert smoke_model.resolution == TEST_RESOLUTION
    assert smoke_model.batch_size == TEST_BATCH_SIZE
    
    # Check that inflow was created correctly
    assert smoke_model.inflow is not None
    assert 'batch' in smoke_model.inflow.shape
    assert smoke_model.inflow.shape['batch'].size == TEST_BATCH_SIZE
    assert smoke_model.inflow.shape.spatial == TEST_RESOLUTION


def test_get_initial_state(smoke_model: SmokeModel):
    """
    Tests the 'get_initial_state' method.
    """
    # --- MODIFIED: No longer takes batch_size ---
    vel, den = smoke_model.get_initial_state() 
    
    # Check batch dimension
    assert 'batch' in vel.shape
    assert 'batch' in den.shape
    assert vel.shape['batch'].size == TEST_BATCH_SIZE
    assert den.shape['batch'].size == TEST_BATCH_SIZE
    
    # Check spatial resolution
    assert vel.shape.spatial == TEST_RESOLUTION
    assert den.shape.spatial == TEST_RESOLUTION
    


def test_step_function(smoke_model: SmokeModel):
    """
    Tests the 'step' method for one iteration.
    """
    # 1. Get an initial state
    vel_0, den_0 = smoke_model.get_initial_state()
    
    # 2. Run one step
    vel_1, den_1 = smoke_model(vel_0, den_0)
    
    # The output shapes must *exactly* match the input shapes.
    assert vel_1.shape == vel_0.shape
    assert den_1.shape == den_0.shape
    
    
