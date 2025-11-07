import unittest
from unittest.mock import MagicMock, patch, call
from typing import Dict, Any

from phi.flow import *
from phi.math import Shape, spatial, batch

from src.models.physical.base import PhysicalModel


# --- Concrete Implementation for Testing ---


class MockPhysicalModel(PhysicalModel):
    """A concrete implementation of PhysicalModel for testing purposes."""

    PDE_PARAMETERS = {
        "param1": {"type": float, "default": 1.0},
        "param2": {"type": int, "default": 10},
    }

    def get_initial_state(self) -> Dict[str, Field]:
        """Returns a mock initial state."""
        b_dim = batch(batch=self.batch_size)
        grid = CenteredGrid(
            1,
            extrapolation.ZERO,
            x=self.resolution.get_size("x"),
            y=self.resolution.get_size("y"),
            bounds=self.domain,
        )
        grid = math.expand(grid, b_dim)
        return {"field1": grid}

    def get_random_state(self) -> Dict[str, Field]:
        """Returns a mock random state."""
        grid = CenteredGrid(
            99,
            extrapolation.ZERO,
            x=self.resolution.get_size("x"),
            y=self.resolution.get_size("y"),
            bounds=self.domain,
        )
        return {"field1": grid}

    def forward(self, current_state: Dict[str, Field]) -> Dict[str, Field]:
        """Mock forward step that increments the field value."""
        new_field = current_state["field1"] + self.param1
        return {"field1": new_field}


# --- Unit Test Class ---


class TestPhysicalModel(unittest.TestCase):
    """Unit tests for the PhysicalModel abstract base class."""

    def setUp(self):
        """Set up a standard config and model instance for tests."""
        self.config = {
            "domain": {"size_x": 64, "size_y": 64},
            "resolution": {"x": 64, "y": 64},
            "dt": 0.1,
            "pde_params": {
                "batch_size": 4,
                "param1": 2.5,
                # param2 will use default
            },
        }
        self.model = MockPhysicalModel(self.config)

    def test_initialization(self):
        """Test if the model initializes correctly with given config."""
        self.assertIsInstance(self.model.domain, Box)
        self.assertEqual(tuple(self.model.domain.size), (64, 64))

        self.assertIsInstance(self.model.resolution, Shape)
        self.assertEqual(self.model.resolution.sizes, (64, 64))

        self.assertEqual(self.model.dt, 0.1)
        self.assertEqual(self.model.batch_size, 4)

    def test_parse_pde_parameters(self):
        """Test parsing of PDE-specific parameters."""
        # Test parameter provided in config
        self.assertEqual(self.model.param1, 2.5)
        self.assertIsInstance(self.model.param1, float)

        # Test parameter using default value
        self.assertEqual(self.model.param2, 10)
        self.assertIsInstance(self.model.param2, int)

    def test_dynamic_property_setter(self):
        """Test if dynamically created properties can be set."""
        self.assertEqual(self.model.param1, 2.5)
        self.model.param1 = 5.0
        self.assertEqual(self.model.param1, 5.0)

    def test_get_initial_state(self):
        """Test the structure and batch size of the initial state."""
        initial_state = self.model.get_initial_state()
        self.assertIn("field1", initial_state)
        self.assertIsInstance(initial_state["field1"], Field)
        self.assertEqual(initial_state["field1"].shape.get_size("batch"), 4)

    def test_forward_call(self):
        """Test the forward pass and __call__ wrapper."""
        initial_state = self.model.get_initial_state()

        # Mock the forward method to check if __call__ uses it
        with patch.object(self.model, "forward", wraps=self.model.forward) as mock_forward:
            next_state = self.model(initial_state)
            mock_forward.assert_called_once_with(initial_state)

        # Check the logic of the mock forward pass
        self.assertAlmostEqual(
            float(next_state["field1"].values.mean),
            1.0 + self.model.param1,
            places=6,
        )

    def test_select_proportional_indices(self):
        """Test the logic for selecting proportional indices."""
        # Select all
        indices = self.model._select_proportional_indices(100, 100)
        self.assertEqual(len(indices), 100)
        self.assertEqual(indices, list(range(100)))

        # Select more than available
        indices = self.model._select_proportional_indices(100, 120)
        self.assertEqual(len(indices), 100)

        # Select a proportion
        indices = self.model._select_proportional_indices(100, 10)
        self.assertEqual(len(indices), 10)
        self.assertEqual(indices, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

        # Select zero
        indices = self.model._select_proportional_indices(100, 0)
        self.assertEqual(len(indices), 0)

    def test_perform_rollout(self):
        """Test if the rollout performs the correct number of forward steps."""
        initial_state = self.model.get_random_state()
        num_steps = 5

        with patch.object(self.model, "forward", wraps=self.model.forward) as mock_forward:
            rollout = self.model._perform_rollout(initial_state, num_steps)

            self.assertEqual(mock_forward.call_count, num_steps)
            self.assertEqual(len(rollout), num_steps)

            # Check that the value increases with each step
            self.assertAlmostEqual(
                float(rollout[0]["field1"].values.mean), 99.0 + self.model.param1
            )
            self.assertAlmostEqual(
                float(rollout[4]["field1"].values.mean), 99.0 + 5 * self.model.param1
            )

    def test_generate_predictions(self):
        """Test the high-level prediction generation for data augmentation."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        alpha = 0.2
        num_rollout_steps = 3
        num_generate = int(len(mock_dataset) * alpha)

        with patch.object(
            self.model, "_perform_rollout", wraps=self.model._perform_rollout
        ) as mock_rollout, patch.object(
            self.model, "get_random_state", wraps=self.model.get_random_state
        ) as mock_get_random:
            initial_list, target_list = self.model.generate_predictions(
                real_dataset=mock_dataset,
                alpha=alpha,
                num_rollout_steps=num_rollout_steps,
            )

            self.assertEqual(len(initial_list), num_generate)
            self.assertEqual(len(target_list), num_generate)

            self.assertEqual(mock_get_random.call_count, num_generate)
            self.assertEqual(mock_rollout.call_count, num_generate)

            # Check structure of the output
            self.assertIsInstance(initial_list[0], dict)
            self.assertIn("field1", initial_list[0])

            self.assertIsInstance(target_list[0], list)
            self.assertEqual(len(target_list[0]), num_rollout_steps)
            self.assertIsInstance(target_list[0][0], dict)
            self.assertIn("field1", target_list[0][0])

    def test_generate_predictions_zero_alpha(self):
        """Test that generate_predictions returns empty lists for alpha=0."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100

        initial_list, target_list = self.model.generate_predictions(
            real_dataset=mock_dataset, alpha=0.0
        )

        self.assertEqual(initial_list, [])
        self.assertEqual(target_list, [])


if __name__ == "__main__":
    unittest.main()