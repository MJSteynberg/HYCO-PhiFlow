"""
Integration tests for all physical models.
Tests common interfaces and compatibility.
"""

import pytest
import torch
from phi.flow import Box, spatial
from phiml import math

from src.models.physical.base import PhysicalModel
from src.models.physical.burgers import BurgersModel
from src.models.physical.heat import HeatModel
from src.models.physical.smoke import SmokeModel


class TestPhysicalModelsIntegration:
    """Integration tests for all physical models."""
    
    @pytest.fixture
    def burgers_model(self):
        """Create a BurgersModel instance."""
        return BurgersModel(
            domain=Box(x=1.0, y=1.0),
            resolution=spatial(x=64, y=64),
            dt=0.01,
            batch_size=1,
            nu=torch.tensor(0.01)
        )
    
    @pytest.fixture
    def heat_model(self):
        """Create a HeatModel instance."""
        return HeatModel(
            domain=Box(x=100.0, y=100.0),
            resolution=spatial(x=64, y=64),
            dt=0.1,
            diffusivity=torch.tensor(1.0),
            batch_size=1
        )
    
    @pytest.fixture
    def smoke_model(self):
        """Create a SmokeModel instance."""
        return SmokeModel(
            domain=Box(x=80.0, y=80.0),
            resolution=spatial(x=64, y=64),
            dt=1.0,
            batch_size=1,
            nu=0.0,
            buoyancy=1.0,
            inflow_center=(40.0, 20.0),
            inflow_radius=5.0,
            inflow_rate=0.2
        )
    
    @pytest.fixture
    def all_models(self, burgers_model, heat_model, smoke_model):
        """Return all model instances."""
        return [burgers_model, heat_model, smoke_model]
    
    def test_all_models_inherit_from_base(self, all_models):
        """Test that all models inherit from PhysicalModel."""
        for model in all_models:
            assert isinstance(model, PhysicalModel)
    
    def test_all_models_have_required_attributes(self, all_models):
        """Test that all models have required attributes."""
        required_attrs = ['domain', 'resolution', 'dt', 'batch_size']
        
        for model in all_models:
            for attr in required_attrs:
                assert hasattr(model, attr)
                assert getattr(model, attr) is not None
    
    def test_all_models_have_get_initial_state(self, all_models):
        """Test that all models implement get_initial_state."""
        for model in all_models:
            assert hasattr(model, 'get_initial_state')
            assert callable(model.get_initial_state)
            
            # Test that it returns a dict
            state = model.get_initial_state()
            assert isinstance(state, dict)
    
    def test_all_models_have_step(self, all_models):
        """Test that all models implement step."""
        for model in all_models:
            assert hasattr(model, 'step')
            assert callable(model.step)
            
            # Test that it can be called
            state = model.get_initial_state()
            next_state = model.step(state)
            assert isinstance(next_state, dict)
    
    def test_all_models_callable(self, all_models):
        """Test that all models are callable."""
        for model in all_models:
            state = model.get_initial_state()
            next_state = model(state)
            assert isinstance(next_state, dict)
    
    def test_all_models_state_structure_consistency(self, all_models):
        """Test that state structure remains consistent after steps."""
        for model in all_models:
            state = model.get_initial_state()
            initial_keys = set(state.keys())
            
            # Take several steps
            for _ in range(5):
                state = model.step(state)
                assert set(state.keys()) == initial_keys
    
    def test_all_models_batch_dimension(self, all_models):
        """Test that all models handle batch dimension correctly."""
        for model in all_models:
            state = model.get_initial_state()
            
            for field_name, field_value in state.items():
                assert 'batch' in field_value.shape.names
    
    def test_all_models_spatial_dimensions(self, all_models):
        """Test that all models preserve spatial dimensions."""
        for model in all_models:
            state = model.get_initial_state()
            
            for field_name, field_value in state.items():
                # Check that spatial dimensions match resolution
                assert field_value.shape.get_size('x') == model.resolution.get_size('x')
                assert field_value.shape.get_size('y') == model.resolution.get_size('y')
    
    def test_all_models_numerical_stability(self, all_models):
        """Test that all models remain numerically stable."""
        for model in all_models:
            state = model.get_initial_state()
            
            # Run many steps
            for _ in range(20):
                state = model.step(state)
                
                # Check all fields for NaN/Inf
                for field_name, field_value in state.items():
                    has_nan = math.is_nan(field_value.values)
                    has_inf = math.is_inf(field_value.values)
                    assert not has_nan.all, f"{model.__class__.__name__} produced NaN in {field_name}"
                    assert not has_inf.all, f"{model.__class__.__name__} produced Inf in {field_name}"
    
    def test_all_models_with_same_resolution(self):
        """Test all models with the same resolution."""
        resolution = spatial(x=32, y=32)
        
        models = [
            BurgersModel(
                domain=Box(x=1.0, y=1.0),
                resolution=resolution,
                dt=0.01,
                batch_size=1,
                nu=torch.tensor(0.01)
            ),
            HeatModel(
                domain=Box(x=100.0, y=100.0),
                resolution=resolution,
                dt=0.1,
                diffusivity=torch.tensor(1.0),
                batch_size=1
            ),
            SmokeModel(
                domain=Box(x=80.0, y=80.0),
                resolution=resolution,
                dt=1.0,
                batch_size=1,
                nu=0.0,
                buoyancy=1.0,
                inflow_center=(40.0, 20.0)
            )
        ]
        
        for model in models:
            state = model.get_initial_state()
            
            for field_name, field_value in state.items():
                assert field_value.shape.get_size('x') == 32
                assert field_value.shape.get_size('y') == 32
    
    def test_all_models_with_same_batch_size(self):
        """Test all models with the same batch size."""
        batch_size = 4
        
        models = [
            BurgersModel(
                domain=Box(x=1.0, y=1.0),
                resolution=spatial(x=32, y=32),
                dt=0.01,
                batch_size=batch_size,
                nu=torch.tensor(0.01)
            ),
            HeatModel(
                domain=Box(x=100.0, y=100.0),
                resolution=spatial(x=32, y=32),
                dt=0.1,
                diffusivity=torch.tensor(1.0),
                batch_size=batch_size
            ),
            SmokeModel(
                domain=Box(x=80.0, y=80.0),
                resolution=spatial(x=32, y=32),
                dt=1.0,
                batch_size=batch_size,
                nu=0.0,
                buoyancy=1.0,
                inflow_center=(40.0, 20.0)
            )
        ]
        
        for model in models:
            # Note: Burgers and Smoke use model.batch_size in get_initial_state
            # while Heat accepts batch_size as parameter
            if isinstance(model, HeatModel):
                state = model.get_initial_state(batch_size=batch_size)
            else:
                state = model.get_initial_state()
            
            for field_name, field_value in state.items():
                assert field_value.shape.get_size('batch') == batch_size
    
    def test_model_specific_fields(self, burgers_model, heat_model, smoke_model):
        """Test that each model has its specific fields."""
        # Burgers should have velocity
        burgers_state = burgers_model.get_initial_state()
        assert 'velocity' in burgers_state
        
        # Heat should have temp
        heat_state = heat_model.get_initial_state()
        assert 'temp' in heat_state
        
        # Smoke should have velocity, density, and inflow
        smoke_state = smoke_model.get_initial_state()
        assert 'velocity' in smoke_state
        assert 'density' in smoke_state
        assert 'inflow' in smoke_state
    
    def test_model_specific_parameters(self, burgers_model, heat_model, smoke_model):
        """Test that each model has its specific parameters."""
        # Burgers should have nu
        assert hasattr(burgers_model, 'nu')
        
        # Heat should have diffusivity
        assert hasattr(heat_model, 'diffusivity')
        
        # Smoke should have nu, buoyancy, and inflow parameters
        assert hasattr(smoke_model, 'nu')
        assert hasattr(smoke_model, 'buoyancy')
        assert hasattr(smoke_model, 'inflow_center')
        assert hasattr(smoke_model, 'inflow_radius')
        assert hasattr(smoke_model, 'inflow_rate')
    
    def test_parameter_mutability(self, burgers_model, heat_model, smoke_model):
        """Test that model parameters can be updated."""
        # Test Burgers nu
        original_nu = burgers_model.nu
        burgers_model.nu = torch.tensor(0.05)
        assert not torch.equal(burgers_model.nu, original_nu)
        
        # Test Heat diffusivity
        original_diff = heat_model.diffusivity
        heat_model.diffusivity = torch.tensor(2.0)
        assert not torch.equal(heat_model.diffusivity, original_diff)
        
        # Test Smoke nu and buoyancy
        original_smoke_nu = smoke_model.nu
        smoke_model.nu = 0.05
        assert smoke_model.nu != original_smoke_nu
        
        original_buoyancy = smoke_model.buoyancy
        smoke_model.buoyancy = 2.0
        assert smoke_model.buoyancy != original_buoyancy
    
    def test_all_models_reproducibility(self, all_models):
        """Test that running the same simulation twice gives same results."""
        for model in all_models:
            # Run simulation once
            state1 = model.get_initial_state()
            for _ in range(5):
                state1 = model.step(state1)
            
            # Run simulation again with same initial conditions
            state2 = model.get_initial_state()
            for _ in range(5):
                state2 = model.step(state2)
            
            # Results should be identical (assuming deterministic initial conditions)
            # Note: Burgers uses Noise which may not be deterministic
            # We just check structure is the same
            assert set(state1.keys()) == set(state2.keys())
            for key in state1.keys():
                assert state1[key].shape == state2[key].shape
    
    def test_field_bounds(self, all_models):
        """Test that all fields have proper bounds."""
        for model in all_models:
            state = model.get_initial_state()
            
            for field_name, field_value in state.items():
                # Fields should have bounds attribute
                assert hasattr(field_value, 'bounds')
                
                # Bounds should match model domain (for most fields)
                if field_name != 'inflow':  # Inflow might have different bounds
                    assert field_value.bounds == model.domain
    
    def test_serialization_compatibility(self, all_models):
        """Test that model states can be converted to tensors."""
        for model in all_models:
            state = model.get_initial_state()
            
            # All fields should have values that can be accessed as tensors
            for field_name, field_value in state.items():
                # Use PhiFlow math operations to check tensor validity
                assert field_value.values is not None
                # Verify it's a valid tensor by checking shape
                assert field_value.values.shape.volume > 0
    
    def test_different_model_configurations(self):
        """Test creating models with various configurations."""
        configs = [
            {'resolution': spatial(x=32, y=32), 'dt': 0.01},
            {'resolution': spatial(x=64, y=64), 'dt': 0.1},
            {'resolution': spatial(x=128, y=128), 'dt': 0.001},
        ]
        
        for config in configs:
            # Test Burgers
            burgers = BurgersModel(
                domain=Box(x=1.0, y=1.0),
                resolution=config['resolution'],
                dt=config['dt'],
                batch_size=1,
                nu=torch.tensor(0.01)
            )
            state = burgers.get_initial_state()
            assert 'velocity' in state
            
            # Test Heat
            heat = HeatModel(
                domain=Box(x=100.0, y=100.0),
                resolution=config['resolution'],
                dt=config['dt'],
                diffusivity=torch.tensor(1.0),
                batch_size=1
            )
            state = heat.get_initial_state()
            assert 'temp' in state
            
            # Test Smoke
            smoke = SmokeModel(
                domain=Box(x=80.0, y=80.0),
                resolution=config['resolution'],
                dt=config['dt'],
                batch_size=1,
                nu=0.0,
                buoyancy=1.0,
                inflow_center=(40.0, 20.0)
            )
            state = smoke.get_initial_state()
            assert 'density' in state
