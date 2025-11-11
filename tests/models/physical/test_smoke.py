import unittest
import torch
from typing import Dict, Any
import numpy as np
from unittest.mock import patch
import random

from phi.torch.flow import (
    Box,
    Field,
    CenteredGrid,
    StaggeredGrid,
    extrapolation,
    math,
    batch,
    channel,
    jit_compile,
    Tensor,
    Sphere,
)

from src.models.physical.smoke import SmokeModel, _smoke_physics_step


class TestSmokeModel(unittest.TestCase):
    """Unit tests for the SmokeModel."""

    def setUp(self):
        """Set up a standard config and model instance for tests."""
        self.config = {
            "domain": {"size_x": 32, "size_y": 32},
            "resolution": {"x": 32, "y": 32},
            "dt": 0.1,
            "pde_params": {
                "batch_size": 1,
                "nu": 0.01,
                "buoyancy": 0.5,
                "inflow_radius": 5.0,
                "inflow_rate": 0.2,
                "inflow_rand_x_range": [0.4, 0.6],
                "inflow_rand_y_range": [0.1, 0.2],
            },
        }
        self.model = SmokeModel(self.config)

    def test_initialization(self):
        """Test if the model initializes correctly."""
        self.assertIsInstance(self.model, SmokeModel)
        self.assertEqual(float(self.model.dt), 0.1)
        self.assertEqual(self.model.batch_size, 1)
        self.assertEqual(float(self.model.nu), 0.01)
        self.assertEqual(float(self.model.buoyancy), 0.5)
        self.assertEqual(float(self.model.inflow_radius), 5.0)
        self.assertEqual(float(self.model.inflow_rate), 0.2)
        self.assertEqual(self.model._inflow_rand_x_range, [0.4, 0.6])
        self.assertEqual(self.model._inflow_rand_y_range, [0.1, 0.2])

    def test_get_inflow_center(self):
        """Test that inflow center is generated within specified ranges."""
        # Patch random.random to return predictable values
        with patch.object(random, "random", return_value=0.5):
            inflow_center = self.model._get_inflow_center()
            self.assertIsInstance(inflow_center, Tensor)
            self.assertEqual(inflow_center.shape.get_size("vector"), 2)

            # Expected calculation:
            # rand_x = 32 * (0.4 + 0.6 * 0.5) = 32 * (0.4 + 0.3) = 32 * 0.7 = 22.4
            # rand_y = 32 * (0.1 + 0.2 * 0.5) = 32 * (0.1 + 0.1) = 32 * 0.2 = 6.4
            self.assertAlmostEqual(float(inflow_center.vector[0]), 22.4)
            self.assertAlmostEqual(float(inflow_center.vector[1]), 6.4)

    def test_get_initial_state(self):
        """Test the structure and shape of the initial state."""
        # Patch random.random for deterministic inflow center
        with patch.object(random, "random", return_value=0.5):
            initial_state = self.model.get_initial_state()

        self.assertIn("velocity", initial_state)
        self.assertIn("density", initial_state)
        self.assertIn("inflow", initial_state)

        velocity = initial_state["velocity"]
        density = initial_state["density"]
        inflow = initial_state["inflow"]

        self.assertIsInstance(velocity, Field)
        self.assertIsInstance(density, Field)
        self.assertIsInstance(inflow, Field)

        # Check shapes
        self.assertEqual(velocity.shape.get_size("batch"), 1)
        self.assertEqual(density.shape.get_size("batch"), 1)
        self.assertEqual(inflow.shape.get_size("batch"), 1)

        self.assertEqual(velocity.shape.spatial, math.spatial(x=32, y=32))
        self.assertEqual(density.shape.spatial, math.spatial(x=32, y=32))
        self.assertEqual(inflow.shape.spatial, math.spatial(x=32, y=32))

        self.assertEqual(velocity.shape.get_size("vector"), 2)
        
        # Initial values should be zero for velocity and density
        self.assertAlmostEqual(float(math.max(velocity.values)), 0.0)
        self.assertAlmostEqual(float(math.max(density.values)), 0.0)
        # Inflow should have non-zero values within the sphere
        self.assertGreater(float(math.max(inflow.values)), 0.0)

    def test_get_random_state(self):
        """Test the structure and shape of the random state (not batched)."""
        # Patch random.random for deterministic inflow center
        with patch.object(random, "random", return_value=0.5):
            random_state = self.model.get_random_state()

        self.assertIn("velocity", random_state)
        self.assertIn("density", random_state)
        self.assertIn("inflow", random_state)

        velocity = random_state["velocity"]
        density = random_state["density"]
        inflow = random_state["inflow"]

        self.assertIsInstance(velocity, Field)
        self.assertIsInstance(density, Field)
        self.assertIsInstance(inflow, Field)

        # Check shapes (random state is not batched by default in the implementation)
        self.assertNotIn("batch", velocity.shape)
        self.assertNotIn("batch", density.shape)
        self.assertNotIn("batch", inflow.shape)

        self.assertEqual(velocity.shape.spatial, math.spatial(x=32, y=32))
        self.assertEqual(density.shape.spatial, math.spatial(x=32, y=32))
        self.assertEqual(inflow.shape.spatial, math.spatial(x=32, y=32))

        self.assertEqual(velocity.shape.get_size("vector"), 2)

    def test_forward_pass(self):
        """Test a single forward pass of the model."""
        # Patch random.random for deterministic inflow center
        with patch.object(random, "random", return_value=0.5):
            initial_state = self.model.get_initial_state()
        next_state = self.model.forward(initial_state)

        self.assertIn("velocity", next_state)
        self.assertIn("density", next_state)
        self.assertIn("inflow", next_state)

        # Check that inflow is static and passed through
        torch.testing.assert_close(
            initial_state["inflow"].values.native("x,y,batch"),
            next_state["inflow"].values.native("x,y,batch"),
        )

        # Check that density and velocity have changed
        density_changed = not torch.allclose(
            initial_state["density"].values.native("x,y,batch"),
            next_state["density"].values.native("x,y,batch"),
        )
        self.assertTrue(density_changed, "Density field should change after a step.")
        print(next_state["velocity"], initial_state["velocity"])
        print(next_state["velocity"].values.native("x,y,vector,batch"), initial_state["velocity"].values.native("x,y,vector,batch"))
        velocity_changed = not torch.allclose(
            initial_state["velocity"].values.native("x,y,vector,batch"),
            next_state["velocity"].values.native("x,y,vector,batch"),
        )
        self.assertTrue(
            velocity_changed, "Velocity field should change after a step."
        )

        # Check shapes of the output
        self.assertEqual(next_state["velocity"].shape, initial_state["velocity"].shape)
        self.assertEqual(next_state["density"].shape, initial_state["density"].shape)
        self.assertEqual(next_state["inflow"].shape, initial_state["inflow"].shape)

    def test_smoke_physics_step_function(self):
        """Test the _smoke_physics_step JIT function in isolation."""
        domain = Box(x=16, y=16)
        velocity_in = CenteredGrid(
            (0, 0), extrapolation.ZERO, x=16, y=16, bounds=domain
        )
        density_in = CenteredGrid(
            0.1, extrapolation.BOUNDARY, x=16, y=16, bounds=domain
        )
        inflow_in = CenteredGrid(
            Sphere(center=math.tensor((8,4), channel(vector="x,y")), radius=2),
            extrapolation.BOUNDARY,
            x=16,
            y=16,
            bounds=domain,
        ) * 1.0  # Constant inflow

        dt = 0.1
        buoyancy_factor = 1.0
        nu = 0.01

        new_velocity, new_density = _smoke_physics_step(
            velocity_in, density_in, inflow_in, domain, dt, buoyancy_factor, nu
        )

        self.assertIsInstance(new_velocity, Field)
        self.assertIsInstance(new_density, Field)

        # Check that density has increased due to inflow
        # The inflow region should have higher density than initial 0.1
        self.assertGreater(float(math.max(new_density.values)), 0.1)

        # Check that velocity has changed due to buoyancy (upwards in y-direction)
        # Initial velocity is zero, so it should become positive in y-direction
        self.assertGreater(float(math.max(new_velocity.values)), 0.0)

        # Check that velocity has changed from initial zero
        velocity_changed = not torch.allclose(
            velocity_in.values.native("x,y,vector"),
            new_velocity.values.native("x,y,vector"),
        )
        self.assertTrue(velocity_changed, "Velocity should change due to physics.")


if __name__ == "__main__":
    unittest.main()