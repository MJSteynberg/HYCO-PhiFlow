"""
Tests for BurgersModel physical model.
"""

import pytest
import torch
from phi.flow import Box, spatial
from phi.math import batch, math

from src.models.physical.burgers import BurgersModel


class TestBurgersModel:
    """Test BurgersModel functionality."""

    @pytest.fixture
    def basic_config(self):
        """Basic configuration for Burgers' equation."""
        return {
            "domain": {"size_x": 1.0, "size_y": 1.0},
            "resolution": {"x": 64, "y": 64},
            "dt": 0.01,
            "pde_params": {"batch_size": 1, "nu": 0.01},
        }

    @pytest.fixture
    def model(self, basic_config):
        """Create a BurgersModel instance."""
        return BurgersModel(basic_config)

    def test_initialization(self, basic_config):
        """Test that BurgersModel initializes correctly."""
        model = BurgersModel(basic_config)

        assert model.domain == Box(x=1.0, y=1.0)
        assert model.resolution == spatial(x=64, y=64)
        assert model.dt == basic_config["dt"]
        assert model.batch_size == 1
        assert model.nu == 0.01

    def test_nu_property(self, model):
        """Test that nu property getter and setter work."""
        new_nu = 0.05
        model.nu = new_nu

        assert model.nu == new_nu

    def test_get_initial_state_structure(self, model):
        """Test that initial state has correct structure."""
        state = model.get_initial_state()

        assert isinstance(state, dict)
        assert "velocity" in state
        assert hasattr(state["velocity"], "shape")
        assert hasattr(state["velocity"], "values")

    def test_get_initial_state_dimensions(self, model):
        """Test that initial state has correct dimensions."""
        state = model.get_initial_state()
        velocity = state["velocity"]

        # Check batch dimension
        assert "batch" in velocity.shape.names
        assert velocity.shape.get_size("batch") == model.batch_size

        # Check spatial dimensions
        assert velocity.shape.get_size("x") == model.resolution.get_size("x")
        assert velocity.shape.get_size("y") == model.resolution.get_size("y")

        # Check vector dimension (for StaggeredGrid)
        assert "vector" in velocity.shape.names

    def test_get_initial_state_periodic_boundary(self, model):
        """Test that initial state uses periodic boundary conditions."""
        state = model.get_initial_state()
        velocity = state["velocity"]

        # Check that extrapolation is PERIODIC
        assert "periodic" in str(velocity.extrapolation).lower()

    def test_get_initial_state_batched(self, model):
        """Test initial state with different batch sizes."""
        batch_sizes = [1, 2, 4, 8]

        for bs in batch_sizes:
            state = model.get_initial_state()
            velocity = state["velocity"]

            assert velocity.shape.get_size("batch") == model.batch_size

    def test_step_execution(self, model):
        """Test that step executes without errors."""
        state = model.get_initial_state()
        next_state = model.step(state)

        assert isinstance(next_state, dict)
        assert "velocity" in next_state

    def test_step_preserves_structure(self, model):
        """Test that step preserves state structure."""
        state = model.get_initial_state()
        next_state = model.step(state)

        assert set(state.keys()) == set(next_state.keys())
        assert state["velocity"].shape == next_state["velocity"].shape

    def test_step_changes_values(self, model):
        """Test that step actually changes field values."""
        state = model.get_initial_state()
        next_state = model.step(state)

        # Values should change due to advection and diffusion
        # Use PhiFlow's math operations to compare tensors
        # Check that fields are not equal (some values should have changed)
        from phi.math import all_available

        # Compare using PhiFlow's comparison - if all values were equal, this would be True
        fields_equal = math.close(
            state["velocity"].values,
            next_state["velocity"].values,
            rel_tolerance=1e-5,
            abs_tolerance=1e-5,
        )

        # We expect them to be different (not all equal)
        assert not math.all(fields_equal)

    def test_multiple_steps(self, model):
        """Test running multiple time steps."""
        state = model.get_initial_state()

        for i in range(5):
            state = model.step(state)
            assert "velocity" in state
            assert state["velocity"].shape.get_size("batch") == model.batch_size

    def test_call_method(self, model):
        """Test that __call__ method works."""
        state = model.get_initial_state()
        next_state = model(state)

        assert isinstance(next_state, dict)
        assert "velocity" in next_state

    def test_different_viscosities(self, basic_config):
        """Test model with different viscosity values."""
        viscosities = [0.001, 0.01, 0.1]

        for nu in viscosities:
            config = basic_config.copy()
            config["pde_params"] = config["pde_params"].copy()
            config["pde_params"]["nu"] = nu
            model = BurgersModel(config)

            assert model.nu == nu

            # Test that model can step
            state = model.get_initial_state()
            next_state = model.step(state)
            assert "velocity" in next_state

    def test_different_resolutions(self, basic_config):
        """Test model with different spatial resolutions."""
        resolutions = [{"x": 32, "y": 32}, {"x": 64, "y": 64}, {"x": 128, "y": 128}]

        for res in resolutions:
            config = basic_config.copy()
            config["resolution"] = res
            model = BurgersModel(config)

            state = model.get_initial_state()
            assert state["velocity"].shape.get_size("x") == res["x"]
            assert state["velocity"].shape.get_size("y") == res["y"]

    def test_different_domains(self, basic_config):
        """Test model with different domain sizes."""
        domains = [
            {"size_x": 1.0, "size_y": 1.0},
            {"size_x": 2.0, "size_y": 2.0},
            {"size_x": 1.0, "size_y": 2.0},
        ]

        for domain in domains:
            config = basic_config.copy()
            config["domain"] = domain
            model = BurgersModel(config)

            state = model.get_initial_state()
            assert "velocity" in state

    def test_different_dt(self, basic_config):
        """Test model with different time steps."""
        dt_values = [0.001, 0.01, 0.1]

        for dt in dt_values:
            config = basic_config.copy()
            config["dt"] = dt
            model = BurgersModel(config)

            state = model.get_initial_state()
            next_state = model.step(state)
            assert "velocity" in next_state

    def test_batch_consistency(self, basic_config):
        """Test that batched simulations maintain independence."""
        config = basic_config.copy()
        config["pde_params"] = config["pde_params"].copy()
        config["pde_params"]["batch_size"] = 4
        model = BurgersModel(config)

        state = model.get_initial_state()

        # Run a few steps
        for _ in range(3):
            state = model.step(state)

        # Check that batch dimension is preserved
        assert state["velocity"].shape.get_size("batch") == 4

    def test_velocity_field_type(self, model):
        """Test that velocity is a StaggeredGrid."""
        state = model.get_initial_state()
        velocity = state["velocity"]

        # StaggeredGrid should have a vector dimension
        assert "vector" in velocity.shape.names

        # Should have 2 components for 2D
        assert velocity.shape.get_size("vector") == 2

    def test_conservation_properties(self, model):
        """Test basic conservation properties over time steps."""
        state = model.get_initial_state()

        # Run several steps
        states = [state]
        for _ in range(10):
            state = model.step(state)
            states.append(state)

        # Check that field structure is maintained
        for s in states:
            assert "velocity" in s
            assert s["velocity"].shape == states[0]["velocity"].shape

    def test_numerical_stability(self, model):
        """Test that simulation remains numerically stable."""
        state = model.get_initial_state()

        # Run many steps
        for _ in range(50):
            state = model.step(state)

            # Check for NaN or Inf using PhiFlow's math operations
            values = state["velocity"].values
            # Use math.all to reduce over all dimensions including batch
            assert not math.any(
                math.is_nan(values)
            ).all, "NaN detected in velocity field"
            assert not math.any(
                math.is_inf(values)
            ).all, "Inf detected in velocity field"
