import unittest
import torch
from typing import Dict, Any

from phi.torch.flow import (
    Box,
    CenteredGrid,
    extrapolation,
    advect,
    math,
    batch,
    channel,
    jit_compile,
    Tensor,
    Noise,
    Field,
)

from src.models.physical.advection import AdvectionModel, _advection_step


class TestAdvectionModel(unittest.TestCase):
    """Unit tests for the AdvectionModel."""

    def setUp(self):
        """Set up a standard config and model instance for tests."""
        self.config = {
            "domain": {"size_x": 32, "size_y": 32},
            "resolution": {"x": 32, "y": 32},
            "dt": 0.1,
            "pde_params": {
                "batch_size": 2,
                "advection_coeff": 1.5,
            },
        }
        self.model = AdvectionModel(self.config)

    def test_initialization(self):
        """Test if the model initializes correctly."""
        self.assertIsInstance(self.model, AdvectionModel)
        self.assertEqual(self.model.dt, 0.1)
        self.assertEqual(self.model.batch_size, 2)
        self.assertEqual(self.model.advection_coeff, 1.5)

    def test_get_initial_state(self):
        """Test the structure and shape of the initial state."""
        initial_state = self.model.get_initial_state()

        self.assertIn("density", initial_state)
        self.assertIn("velocity", initial_state)

        density = initial_state["density"]
        velocity = initial_state["velocity"]

        self.assertIsInstance(density, Field)
        self.assertIsInstance(velocity, Field)

        # Check shapes
        self.assertEqual(density.shape.get_size("batch"), 2)
        self.assertEqual(velocity.shape.get_size("batch"), 2)
        self.assertEqual(density.shape.spatial, math.spatial(x=32, y=32))
        self.assertEqual(velocity.shape.spatial, math.spatial(x=32, y=32))
        self.assertEqual(velocity.shape.get_size("vector"), 2)

    def test_get_random_state(self):
        """Test the structure and shape of the random state."""
        random_state = self.model.get_random_state()

        self.assertIn("density", random_state)
        self.assertIn("velocity", random_state)

        density = random_state["density"]
        velocity = random_state["velocity"]

        self.assertIsInstance(density, Field)
        self.assertIsInstance(velocity, Field)

        # Check shapes (random state is not batched by default in the implementation)
        self.assertNotIn("batch", density.shape)
        self.assertNotIn("batch", velocity.shape)
        self.assertEqual(density.shape.spatial, math.spatial(x=32, y=32))
        self.assertEqual(velocity.shape.spatial, math.spatial(x=32, y=32))
        self.assertEqual(velocity.shape.get_size("vector"), 2)

    def test_forward_pass(self):
        """Test a single forward pass of the model."""
        initial_state = self.model.get_initial_state()
        next_state = self.model.forward(initial_state)

        self.assertIn("density", next_state)
        self.assertIn("velocity", next_state)

        # Check that velocity is static and passed through
        torch.testing.assert_close(
            initial_state["velocity"].values.native('x,y,vector,batch'),
            next_state["velocity"].values.native('x,y,vector,batch'),
        )

        # Check that density has changed
        density_changed = not torch.allclose(
            initial_state["density"].values.native('x,y,batch'),
            next_state["density"].values.native('x,y,batch'),
        )
        self.assertTrue(density_changed, "Density field should change after advection.")

        # Check shapes of the output
        self.assertEqual(next_state["density"].shape, initial_state["density"].shape)
        self.assertEqual(next_state["velocity"].shape, initial_state["velocity"].shape)

    def test_advection_step_function(self):
        """Test the _advection_step JIT function in isolation."""
        domain = Box(x=10, y=10)
        # A simple density blob in the center
        density = CenteredGrid(
            Noise(scale=2, smoothness=2),
            extrapolation.ZERO,
            x=10,
            y=10,
            bounds=domain,
        )
        # Constant velocity moving right and up
        velocity = CenteredGrid(
            (1, 1), extrapolation.ZERO, x=10, y=10, bounds=domain
        )

        dt = 1.0
        advection_coeff = math.tensor(1.0)

        # Advect the density
        new_density = _advection_step(density, velocity, advection_coeff, dt)



if __name__ == "__main__":
    unittest.main()
