"""
Tests for PhysicalModel base class functionality.
"""

import pytest
from phi.flow import Box, CenteredGrid, spatial
from phi.math import batch

from src.models.physical.base import PhysicalModel


class ConcretePhysicalModel(PhysicalModel):
    """Concrete implementation of PhysicalModel for testing."""
    
    def get_initial_state(self, batch_size: int = 1):
        """Generate a simple initial state with a temperature field."""
        from phi.math import math
        b = batch(batch=batch_size)
        temp_field = CenteredGrid(
            values=0.0,
            bounds=self.domain,
            resolution=self.resolution,
            extrapolation='boundary'
        )
        return {
            'temp': math.expand(temp_field, b)
        }
    
    def step(self, current_state):
        """Simple step that returns the same state."""
        return current_state


class TestPhysicalModelBase:
    """Test PhysicalModel base class functionality."""
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing."""
        return {
            'domain': Box(x=1.0, y=1.0),
            'resolution': spatial(x=64, y=64),
            'dt': 0.01,
            'batch_size': 1
        }
    
    @pytest.fixture
    def model(self, basic_config):
        """Create a concrete model instance."""
        return ConcretePhysicalModel(**basic_config)
    
    def test_initialization(self, basic_config):
        """Test that model initializes with correct attributes."""
        model = ConcretePhysicalModel(**basic_config)
        
        assert model.domain == basic_config['domain']
        assert model.resolution == basic_config['resolution']
        assert model.dt == basic_config['dt']
        assert model.batch_size == basic_config['batch_size']
    
    def test_initialization_with_pde_params(self, basic_config):
        """Test that PDE-specific parameters are stored as attributes."""
        pde_params = {'diffusivity': 0.1, 'viscosity': 0.01}
        model = ConcretePhysicalModel(**basic_config, **pde_params)
        
        assert hasattr(model, 'diffusivity')
        assert model.diffusivity == 0.1
        assert hasattr(model, 'viscosity')
        assert model.viscosity == 0.01
    
    def test_get_initial_state_returns_dict(self, model):
        """Test that get_initial_state returns a dictionary."""
        state = model.get_initial_state()
        
        assert isinstance(state, dict)
        assert len(state) > 0
    
    def test_get_initial_state_contains_fields(self, model):
        """Test that get_initial_state returns Field objects."""
        state = model.get_initial_state()
        
        for field_name, field_value in state.items():
            assert hasattr(field_value, 'shape')
            assert hasattr(field_value, 'values')
    
    def test_get_initial_state_batch_dimension(self, model):
        """Test that initial state has correct batch dimension."""
        batch_size = 4
        state = model.get_initial_state(batch_size=batch_size)
        
        for field_name, field_value in state.items():
            assert 'batch' in field_value.shape.names
            assert field_value.shape.get_size('batch') == batch_size
    
    def test_get_initial_state_default_batch_size(self, model):
        """Test that get_initial_state uses default batch_size=1."""
        state = model.get_initial_state()
        
        for field_name, field_value in state.items():
            assert 'batch' in field_value.shape.names
            assert field_value.shape.get_size('batch') == 1
    
    def test_get_initial_state_spatial_resolution(self, model):
        """Test that initial state fields have correct spatial resolution."""
        state = model.get_initial_state()
        
        for field_name, field_value in state.items():
            # Check spatial dimensions match resolution
            if 'x' in model.resolution.names:
                assert field_value.shape.get_size('x') == model.resolution.get_size('x')
            if 'y' in model.resolution.names:
                assert field_value.shape.get_size('y') == model.resolution.get_size('y')
    
    def test_get_initial_state_multiple_batch_sizes(self, model):
        """Test get_initial_state with various batch sizes."""
        batch_sizes = [1, 2, 8, 16]
        
        for batch_size in batch_sizes:
            state = model.get_initial_state(batch_size=batch_size)
            
            for field_name, field_value in state.items():
                assert field_value.shape.get_size('batch') == batch_size
    
    def test_step_method_exists(self, model):
        """Test that step method is implemented."""
        state = model.get_initial_state()
        next_state = model.step(state)
        
        assert isinstance(next_state, dict)
    
    def test_step_preserves_field_names(self, model):
        """Test that step preserves the dictionary keys."""
        state = model.get_initial_state()
        next_state = model.step(state)
        
        assert set(state.keys()) == set(next_state.keys())
    
    def test_call_method_wraps_step(self, model):
        """Test that __call__ wraps the step method."""
        state = model.get_initial_state()
        next_state_call = model(state)
        next_state_step = model.step(state)
        
        assert type(next_state_call) == type(next_state_step)
    
    def test_abstract_methods_enforced(self):
        """Test that PhysicalModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            PhysicalModel(
                domain=Box(x=1.0, y=1.0),
                resolution=spatial(x=64, y=64),
                dt=0.01
            )
    
    def test_multiple_resolutions(self, basic_config):
        """Test initialization with different resolutions."""
        resolutions = [
            spatial(x=32, y=32),
            spatial(x=64, y=64),
            spatial(x=128, y=128),
            spatial(x=64, y=128)
        ]
        
        for res in resolutions:
            config = basic_config.copy()
            config['resolution'] = res
            model = ConcretePhysicalModel(**config)
            
            assert model.resolution == res
            state = model.get_initial_state()
            
            for field in state.values():
                assert field.shape.get_size('x') == res.get_size('x')
                assert field.shape.get_size('y') == res.get_size('y')
    
    def test_different_domains(self, basic_config):
        """Test initialization with different domain sizes."""
        domains = [
            Box(x=1.0, y=1.0),
            Box(x=2.0, y=2.0),
            Box(x=1.0, y=2.0),
            Box(x=(0.5, 1.5), y=(0.0, 1.0))
        ]
        
        for domain in domains:
            config = basic_config.copy()
            config['domain'] = domain
            model = ConcretePhysicalModel(**config)
            
            assert model.domain == domain
    
    def test_different_dt_values(self, basic_config):
        """Test initialization with different time step sizes."""
        dt_values = [0.001, 0.01, 0.1, 1.0]
        
        for dt in dt_values:
            config = basic_config.copy()
            config['dt'] = dt
            model = ConcretePhysicalModel(**config)
            
            assert model.dt == dt
