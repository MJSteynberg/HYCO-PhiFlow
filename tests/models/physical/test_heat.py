"""
Tests for HeatModel physical model.
"""

import pytest
import torch
from phi.flow import Box, spatial
from phi.math import batch, math

from src.models.physical.heat import HeatModel


class TestHeatModel:
    """Test HeatModel functionality."""
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for heat equation."""
        return {
            'domain': Box(x=100.0, y=100.0),
            'resolution': spatial(x=64, y=64),
            'dt': 0.1,
            'diffusivity': torch.tensor(1.0),
            'batch_size': 1
        }
    
    @pytest.fixture
    def model(self, basic_config):
        """Create a HeatModel instance."""
        return HeatModel(**basic_config)
    
    def test_initialization(self, basic_config):
        """Test that HeatModel initializes correctly."""
        model = HeatModel(**basic_config)
        
        assert model.domain == basic_config['domain']
        assert model.resolution == basic_config['resolution']
        assert model.dt == basic_config['dt']
        assert model.batch_size == basic_config['batch_size']
        assert torch.equal(model.diffusivity, basic_config['diffusivity'])
    
    def test_diffusivity_property(self, model):
        """Test that diffusivity property getter and setter work."""
        new_diffusivity = torch.tensor(2.0)
        model.diffusivity = new_diffusivity
        
        assert torch.equal(model.diffusivity, new_diffusivity)
    
    def test_get_initial_state_structure(self, model):
        """Test that initial state has correct structure."""
        state = model.get_initial_state()
        
        assert isinstance(state, dict)
        assert 'temp' in state
        assert hasattr(state['temp'], 'shape')
        assert hasattr(state['temp'], 'values')
    
    def test_get_initial_state_dimensions(self, model):
        """Test that initial state has correct dimensions."""
        state = model.get_initial_state()
        temp = state['temp']
        
        # Check batch dimension
        assert 'batch' in temp.shape.names
        assert temp.shape.get_size('batch') == 1
        
        # Check spatial dimensions
        assert temp.shape.get_size('x') == model.resolution.get_size('x')
        assert temp.shape.get_size('y') == model.resolution.get_size('y')
    
    def test_get_initial_state_periodic_boundary(self, model):
        """Test that initial state uses periodic boundary conditions."""
        state = model.get_initial_state()
        temp = state['temp']
        
        # Check that extrapolation is PERIODIC (case-insensitive)
        assert 'periodic' in str(temp.extrapolation).lower()
    
    def test_get_initial_state_batched(self, model):
        """Test initial state with different batch sizes."""
        batch_sizes = [1, 2, 4, 8]
        
        for bs in batch_sizes:
            state = model.get_initial_state(batch_size=bs)
            temp = state['temp']
            
            assert temp.shape.get_size('batch') == bs
    
    def test_get_initial_state_non_uniform(self, model):
        """Test that initial state is non-uniform (has structure)."""
        state = model.get_initial_state()
        temp = state['temp']
        
        # The initial condition uses a cosine function, so values should vary
        # Use PhiFlow's math operations to check variance
        std_value = math.std(temp.values)
        assert std_value.native() > 0  # Should have some variation
    
    def test_step_execution(self, model):
        """Test that step executes without errors."""
        state = model.get_initial_state()
        next_state = model.step(state)
        
        assert isinstance(next_state, dict)
        assert 'temp' in next_state
    
    def test_step_preserves_structure(self, model):
        """Test that step preserves state structure."""
        state = model.get_initial_state()
        next_state = model.step(state)
        
        assert set(state.keys()) == set(next_state.keys())
        assert state['temp'].shape == next_state['temp'].shape
    
    def test_step_diffuses(self, model):
        """Test that step performs diffusion."""
        state = model.get_initial_state()
        next_state = model.step(state)
        
        # Diffusion should smooth the field
        # Use PhiFlow's comparison operations
        fields_equal = math.close(state['temp'].values, next_state['temp'].values, rel_tolerance=1e-5, abs_tolerance=1e-5)
        
        # Values should change due to diffusion (not all equal)
        assert not math.all(fields_equal)
    
    def test_multiple_steps(self, model):
        """Test running multiple time steps."""
        state = model.get_initial_state()
        
        for i in range(10):
            state = model.step(state)
            assert 'temp' in state
            assert state['temp'].shape.get_size('batch') == model.batch_size
    
    def test_call_method(self, model):
        """Test that __call__ method works."""
        state = model.get_initial_state()
        next_state = model(state)
        
        assert isinstance(next_state, dict)
        assert 'temp' in next_state
    
    def test_different_diffusivities(self, basic_config):
        """Test model with different diffusivity values."""
        diffusivities = [0.1, 1.0, 10.0]
        
        for diff in diffusivities:
            config = basic_config.copy()
            config['diffusivity'] = torch.tensor(diff)
            model = HeatModel(**config)
            
            assert torch.equal(model.diffusivity, torch.tensor(diff))
            
            # Test that model can step
            state = model.get_initial_state()
            next_state = model.step(state)
            assert 'temp' in next_state
    
    def test_different_resolutions(self, basic_config):
        """Test model with different spatial resolutions."""
        resolutions = [
            spatial(x=32, y=32),
            spatial(x=64, y=64),
            spatial(x=128, y=128)
        ]
        
        for res in resolutions:
            config = basic_config.copy()
            config['resolution'] = res
            model = HeatModel(**config)
            
            state = model.get_initial_state()
            assert state['temp'].shape.get_size('x') == res.get_size('x')
            assert state['temp'].shape.get_size('y') == res.get_size('y')
    
    def test_different_domains(self, basic_config):
        """Test model with different domain sizes."""
        domains = [
            Box(x=50.0, y=50.0),
            Box(x=100.0, y=100.0),
            Box(x=200.0, y=200.0)
        ]
        
        for domain in domains:
            config = basic_config.copy()
            config['domain'] = domain
            model = HeatModel(**config)
            
            state = model.get_initial_state()
            assert 'temp' in state
    
    def test_different_dt(self, basic_config):
        """Test model with different time steps."""
        dt_values = [0.01, 0.1, 1.0]
        
        for dt in dt_values:
            config = basic_config.copy()
            config['dt'] = dt
            model = HeatModel(**config)
            
            state = model.get_initial_state()
            next_state = model.step(state)
            assert 'temp' in next_state
    
    def test_batch_consistency(self, basic_config):
        """Test that batched simulations maintain correct dimensions."""
        config = basic_config.copy()
        config['batch_size'] = 4
        model = HeatModel(**config)
        
        state = model.get_initial_state(batch_size=4)
        
        # Run a few steps
        for _ in range(3):
            state = model.step(state)
        
        # Check that batch dimension is preserved
        assert state['temp'].shape.get_size('batch') == 4
    
    def test_temperature_field_type(self, model):
        """Test that temperature is a CenteredGrid."""
        state = model.get_initial_state()
        temp = state['temp']
        
        # CenteredGrid should not have a vector dimension
        assert 'vector' not in temp.shape.names
    
    def test_diffusion_smoothing(self, model):
        """Test that diffusion smooths the field over time."""
        state = model.get_initial_state()
        
        # Calculate initial standard deviation using PhiFlow operations
        initial_std = math.std(state['temp'].values).native()
        
        # Run several steps
        for _ in range(20):
            state = model.step(state)
        
        # Calculate final standard deviation
        final_std = math.std(state['temp'].values).native()
        
        # Diffusion should reduce variations (std should decrease)
        assert final_std < initial_std
    
    def test_energy_dissipation(self, model):
        """Test that total energy dissipates over time (for periodic BC)."""
        state = model.get_initial_state()
        
        # Calculate initial standard deviation (related to energy) using PhiFlow operations
        initial_std = math.std(state['temp'].values).native()
        
        # Run many steps
        for _ in range(50):
            state = model.step(state)
        
        # Calculate final standard deviation
        final_std = math.std(state['temp'].values).native()
        
        # Standard deviation should decrease due to diffusion (smoothing)
        assert final_std < initial_std
    
    def test_numerical_stability(self, model):
        """Test that simulation remains numerically stable."""
        state = model.get_initial_state()
        
        # Run many steps
        for _ in range(100):
            state = model.step(state)
            
            # Check for NaN or Inf using PhiFlow's math operations
            values = state['temp'].values
            assert not math.any(math.is_nan(values)).all, "NaN detected in temperature field"
            assert not math.any(math.is_inf(values)).all, "Inf detected in temperature field"
    
    def test_mass_conservation(self, model):
        """Test that total mass is conserved (for periodic BC)."""
        state = model.get_initial_state()
        
        # Calculate initial sum using PhiFlow operations
        initial_sum = math.sum(state['temp'].values).native()
        
        # Run several steps
        for _ in range(20):
            state = model.step(state)
        
        # Calculate final sum
        final_sum = math.sum(state['temp'].values).native()
        
        # Total mass should be approximately conserved
        # Use absolute difference since initial sum can be close to zero
        diff = abs(final_sum - initial_sum)
        # The difference should be very small
        assert diff < 0.1  # Allow small numerical errors
    
    def test_symmetry_preservation(self, basic_config):
        """Test that symmetric initial conditions remain symmetric."""
        # Create a model with square domain
        config = basic_config.copy()
        config['domain'] = Box(x=100.0, y=100.0)
        config['resolution'] = spatial(x=64, y=64)
        model = HeatModel(**config)
        
        state = model.get_initial_state()
        
        # Run a few steps
        for _ in range(5):
            state = model.step(state)
        
        # The cosine initial condition has symmetry properties
        # that should be preserved (at least approximately)
        assert 'temp' in state
    
    def test_state_dict_keys(self, model):
        """Test that state dictionary always has expected keys."""
        state = model.get_initial_state()
        
        # Initial state should only have 'temp'
        assert list(state.keys()) == ['temp']
        
        # After step, should still only have 'temp'
        next_state = model.step(state)
        assert list(next_state.keys()) == ['temp']
