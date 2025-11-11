import unittest
import torch

from phi.torch.flow import (
    Box,
    Field,
    Noise,
    extrapolation,
    math,
    CenteredGrid,
    batch,
    StaggeredGrid,
    jit_compile,
)

from src.models.physical.burgers import BurgersModel, _burgers_physics_step


class TestBurgersModel(unittest.TestCase):
    """Unit tests for the BurgersModel."""

    def setUp(self):
        """Set up a standard config and model instance for tests."""
        self.config = {
            "domain": {"size_x": 32, "size_y": 32},
            "resolution": {"x": 32, "y": 32},
            "dt": 0.1,
            "pde_params": {
                "batch_size": 2,
                "nu": 0.05,
            },
        }
        self.model = BurgersModel(self.config)

    def test_initialization(self):
        """Test if the model initializes correctly."""
        self.assertIsInstance(self.model, BurgersModel)
        self.assertEqual(self.model.dt, 0.1)
        self.assertEqual(self.model.batch_size, 2)
        self.assertEqual(self.model.nu, 0.05)

    def test_initialization_default_nu(self):
        """Test if the model initializes with the default viscosity."""
        config = self.config.copy()
        del config["pde_params"]["nu"]
        model = BurgersModel(config)
        self.assertEqual(model.nu, 0.01)  # Check against default value

    def test_get_initial_state(self):
        """Test the structure and shape of the initial state."""
        initial_state = self.model.get_initial_state()

        self.assertIn("velocity", initial_state)
        velocity = initial_state["velocity"]

        self.assertIsInstance(velocity, Field)
        self.assertEqual(velocity.extrapolation, extrapolation.PERIODIC)

        # Check shapes
        self.assertEqual(velocity.shape.get_size("batch"), 2)
        self.assertEqual(velocity.shape.spatial, math.spatial(x=32, y=32))

    def test_get_random_state(self):
        """Test the structure and shape of the random state."""
        random_state = self.model.get_random_state()

        self.assertIn("velocity", random_state)
        velocity = random_state["velocity"]

        self.assertIsInstance(velocity, Field)
        self.assertEqual(velocity.extrapolation, extrapolation.PERIODIC)

        # Check shapes (random state is not batched)
        self.assertNotIn("batch", velocity.shape)
        self.assertEqual(velocity.shape.spatial, math.spatial(x=32, y=32))

    def test_forward_pass(self):
        """Test a single forward pass of the model."""
        initial_state = self.model.get_initial_state()
        next_state = self.model.forward(initial_state)

        self.assertIn("velocity", next_state)
        self.assertIsInstance(next_state["velocity"], Field)

        # Check that velocity has changed
        velocity_changed = not torch.allclose(
            initial_state["velocity"].values.native('x,y,vector,batch'),
            next_state["velocity"].values.native('x,y,vector,batch'),
        )
        self.assertTrue(
            velocity_changed, "Velocity field should change after a physics step."
        )

        # Check shapes of the output
        self.assertEqual(next_state["velocity"].shape, initial_state["velocity"].shape)

    def test_burgers_physics_step_function(self):
        """Test the _burgers_physics_step JIT function in isolation."""
        b = batch(batch=1)

        domain = Box(x=10,y=10)

        temp = StaggeredGrid(
            Noise(scale=20),  # Initialize with noise
            extrapolation.PERIODIC,  # Use periodic boundaries
            x=10,
            y=10,
            bounds=domain,
        )

        velocity = CenteredGrid(
            temp,
            extrapolation.PERIODIC,  # Use periodic boundaries
            x=10,
            y=10,
            bounds=domain,
        )
        # The diffusion term should reduce the total variance of the field
        initial_variance = math.std(velocity.values)

        new_velocity = _burgers_physics_step(velocity, dt=0.1, nu=math.tensor(0.1))

        final_variance = math.std(new_velocity.values)
        self.assertIsInstance(new_velocity, Field)
        self.assertLess(
            float(final_variance),
            float(initial_variance),
            "Variance should decrease due to diffusion.",
        )


if __name__ == "__main__":
    unittest.main()