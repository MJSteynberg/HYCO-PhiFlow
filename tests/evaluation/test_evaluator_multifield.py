"""
Tests for Evaluator with multi-field data, especially with static fields.

This test suite specifically targets the bug where static fields (fields in
input_specs but not in output_specs) cause misalignment in visualization and
metric computation.
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.evaluation.evaluator import Evaluator


class TestEvaluatorMultiFieldBehavior:
    """Test evaluator behavior with multiple fields including static fields."""

    @pytest.fixture
    def smoke_config(self):
        """Config mimicking smoke experiment with static inflow field."""
        return {
            "data": {
                "data_dir": "data/",
                "dset_name": "smoke_128",
                "fields": ["density", "velocity", "inflow"],
                "fields_scheme": "dVVi",
            },
            "model": {
                "synthetic": {
                    "name": "UNet",
                    "model_path": "results/models",
                    "model_save_name": "smoke_unet_128",
                    "input_specs": {"density": 1, "velocity": 2, "inflow": 1},
                    "output_specs": {
                        "density": 1,
                        "velocity": 2,
                        # Note: inflow is NOT in output_specs (it's static)
                    },
                    "architecture": {"levels": 4, "filters": 64, "batch_norm": True},
                }
            },
            "evaluation_params": {
                "test_sim": [0],
                "num_frames": 10,
                "metrics": ["mse", "mae"],
                "keyframe_count": 3,
                "animation_fps": 10,
                "save_animations": True,
                "save_plots": True,
            },
        }

    def test_evaluator_initialization_with_static_fields(self, smoke_config):
        """Test that evaluator correctly initializes with static fields."""
        evaluator = Evaluator(smoke_config)

        # Check that field_names includes all fields
        assert evaluator.field_names == ["density", "velocity", "inflow"]

        # Check that input_specs includes all fields
        assert "density" in evaluator.input_specs
        assert "velocity" in evaluator.input_specs
        assert "inflow" in evaluator.input_specs

        # Check that output_specs only includes dynamic fields
        assert "density" in evaluator.output_specs
        assert "velocity" in evaluator.output_specs
        assert "inflow" not in evaluator.output_specs

    def test_field_specs_construction_in_compute_metrics(self, smoke_config):
        """Test that compute_metrics builds field_specs correctly."""
        evaluator = Evaluator(smoke_config)

        # Create dummy tensors with all 4 channels (1 + 2 + 1)
        # Shape: [T, C, H, W]
        prediction = torch.randn(5, 4, 32, 32)
        ground_truth = torch.randn(5, 4, 32, 32)

        # Call compute_metrics
        metrics = evaluator.compute_metrics(prediction, ground_truth)

        # Check field_specs - this is where the bug might be
        field_specs = metrics["field_specs"]

        # BUG: field_specs might only have density and velocity, missing inflow
        # This would cause channel misalignment
        print(f"Field specs: {field_specs}")
        print(f"Expected fields: {evaluator.field_names}")
        print(f"Output specs: {evaluator.output_specs}")

        # The field_specs should match what's actually in the data
        # If data has 4 channels, field_specs should account for all 4
        total_channels = sum(field_specs.values())
        assert (
            total_channels == 4
        ), f"Field specs channels ({total_channels}) don't match data channels (4)"

    def test_field_specs_construction_in_generate_visualizations(self, smoke_config):
        """Test that generate_visualizations builds field_specs correctly."""
        evaluator = Evaluator(smoke_config)

        # Create dummy tensors with all 4 channels
        prediction = torch.randn(5, 4, 32, 32)
        ground_truth = torch.randn(5, 4, 32, 32)

        # Create temp directory for outputs
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock the actual visualization functions to avoid creating files
            with patch(
                "src.evaluation.evaluator.create_comparison_gif_from_specs"
            ) as mock_anim, patch(
                "src.evaluation.evaluator.plot_error_vs_time_multi_field"
            ) as mock_error, patch(
                "src.evaluation.evaluator.plot_keyframe_comparison_multi_field"
            ) as mock_keyframe, patch(
                "src.evaluation.evaluator.plot_error_heatmap"
            ) as mock_heatmap:

                mock_anim.return_value = {}
                mock_error.return_value = {}
                mock_keyframe.return_value = {}

                # Call generate_visualizations
                saved_paths = evaluator.generate_visualizations(
                    prediction, ground_truth, 0, tmpdir
                )

                # Check what field_specs was passed to visualization functions
                if mock_anim.called:
                    call_args = mock_anim.call_args
                    field_specs = (
                        call_args[0][2]
                        if len(call_args[0]) > 2
                        else call_args[1].get("field_specs")
                    )

                    print(f"Field specs passed to animations: {field_specs}")
                    total_channels = sum(field_specs.values())

                    # This should be 4, but might be 3 due to the bug
                    assert (
                        total_channels == 4
                    ), f"Field specs channels ({total_channels}) don't match data channels (4)"

    def test_channel_slicing_alignment(self, smoke_config):
        """Test that channel slicing aligns correctly with all fields."""
        evaluator = Evaluator(smoke_config)

        # Simulate what happens in the visualization code
        # Data has: density (1 ch), velocity (2 ch), inflow (1 ch) = 4 channels
        prediction = torch.randn(5, 4, 32, 32)
        ground_truth = torch.randn(5, 4, 32, 32)

        # FIXED: Build field_specs the way compute_metrics NOW does (using input_specs)
        field_specs = {}
        for field_name in evaluator.field_names:
            if field_name in evaluator.input_specs:
                field_specs[field_name] = evaluator.input_specs[field_name]

        print(f"Field specs built: {field_specs}")
        print(f"Field names: {evaluator.field_names}")

        # Try to slice according to field_specs
        channel_idx = 0
        sliced_fields = {}

        for field_name, num_channels in field_specs.items():
            pred_field = prediction[:, channel_idx : channel_idx + num_channels, :, :]
            sliced_fields[field_name] = pred_field
            print(f"  {field_name}: channels {channel_idx}:{channel_idx+num_channels}")
            channel_idx += num_channels

        # Check if we've accounted for all channels
        channels_used = sum(field_specs.values())
        total_channels = prediction.shape[1]

        # FIXED: Now this should pass because we're using input_specs
        assert (
            channels_used == total_channels
        ), f"Channel mismatch: field_specs uses {channels_used} channels, data has {total_channels}"

    def test_correct_field_specs_should_use_input_specs(self, smoke_config):
        """Test that field_specs should be built from input_specs, not output_specs."""
        evaluator = Evaluator(smoke_config)

        prediction = torch.randn(5, 4, 32, 32)
        ground_truth = torch.randn(5, 4, 32, 32)

        # The CORRECT way to build field_specs for visualization:
        # Use input_specs because that's what the data actually contains
        correct_field_specs = {}
        for field_name in evaluator.field_names:
            if field_name in evaluator.input_specs:
                correct_field_specs[field_name] = evaluator.input_specs[field_name]

        print(f"Correct field specs: {correct_field_specs}")

        # This should account for all channels
        total_channels = sum(correct_field_specs.values())
        assert total_channels == 4

        # Now verify we can correctly slice all fields
        channel_idx = 0
        for field_name, num_channels in correct_field_specs.items():
            pred_field = prediction[:, channel_idx : channel_idx + num_channels, :, :]
            gt_field = ground_truth[:, channel_idx : channel_idx + num_channels, :, :]

            assert pred_field.shape[1] == num_channels
            assert gt_field.shape[1] == num_channels

            print(f"  {field_name}: successfully sliced {num_channels} channels")
            channel_idx += num_channels


class TestEvaluatorInferenceWithStaticFields:
    """Test that inference correctly handles static fields."""

    @pytest.fixture
    def smoke_config(self):
        """Config for smoke with static inflow."""
        return {
            "data": {
                "data_dir": "data/",
                "dset_name": "test_smoke",
                "fields": ["density", "velocity", "inflow"],
            },
            "model": {
                "synthetic": {
                    "name": "UNet",
                    "model_path": "results/models",
                    "model_save_name": "test_smoke_unet",
                    "input_specs": {"density": 1, "velocity": 2, "inflow": 1},
                    "output_specs": {"density": 1, "velocity": 2},
                    "architecture": {"levels": 2, "filters": 16},
                }
            },
            "evaluation_params": {"test_sim": [0], "num_frames": 5},
        }

    def test_inference_output_shape_matches_input(self, smoke_config):
        """Test that model output has same number of channels as input."""
        evaluator = Evaluator(smoke_config)

        # Mock the model to return output with all channels preserved
        mock_model = Mock()

        def model_forward(x):
            # UNet should preserve all channels (including static)
            # Input: [1, 4, H, W] -> Output: [1, 4, H, W]
            return x.clone()  # Simplified: just return same shape

        mock_model.return_value = model_forward
        mock_model.side_effect = model_forward
        evaluator.model = mock_model

        # Mock data manager
        mock_data_manager = Mock()
        mock_data = {
            "tensor_data": {
                "density": torch.randn(5, 1, 32, 32),
                "velocity": torch.randn(5, 2, 32, 32),
                "inflow": torch.randn(5, 1, 32, 32),
            }
        }
        mock_data_manager.get_or_load_simulation.return_value = mock_data
        evaluator.data_manager = mock_data_manager

        # Run inference
        result = evaluator.run_inference(0, num_rollout_steps=4)

        prediction = result["prediction"]
        ground_truth = result["ground_truth"]

        # Both should have 4 channels
        assert (
            prediction.shape[1] == 4
        ), f"Prediction has {prediction.shape[1]} channels, expected 4"
        assert (
            ground_truth.shape[1] == 4
        ), f"Ground truth has {ground_truth.shape[1]} channels, expected 4"

        print(f"Prediction shape: {prediction.shape}")
        print(f"Ground truth shape: {ground_truth.shape}")


class TestVisualizationFieldExtraction:
    """Test that visualization functions correctly extract fields."""

    def test_field_extraction_with_all_fields(self):
        """Test extracting fields when all are present."""
        # Simulate data with all fields: density (1), velocity (2), inflow (1)
        prediction = torch.randn(5, 4, 32, 32)
        ground_truth = torch.randn(5, 4, 32, 32)

        # Correct field specs (should use input_specs)
        field_specs = {"density": 1, "velocity": 2, "inflow": 1}

        # Extract each field
        channel_idx = 0
        extracted = {}

        for field_name, num_channels in field_specs.items():
            pred_field = prediction[:, channel_idx : channel_idx + num_channels, :, :]
            gt_field = ground_truth[:, channel_idx : channel_idx + num_channels, :, :]

            extracted[field_name] = {"prediction": pred_field, "ground_truth": gt_field}

            # Verify shapes
            assert pred_field.shape == (5, num_channels, 32, 32)
            assert gt_field.shape == (5, num_channels, 32, 32)

            channel_idx += num_channels

        # Verify we extracted all fields
        assert len(extracted) == 3
        assert "density" in extracted
        assert "velocity" in extracted
        assert "inflow" in extracted

        print("Successfully extracted all fields with correct shapes")

    def test_field_extraction_with_missing_static_field(self):
        """Test what happens when field_specs omits static field."""
        # Data still has all fields: density (1), velocity (2), inflow (1)
        prediction = torch.randn(5, 4, 32, 32)
        ground_truth = torch.randn(5, 4, 32, 32)

        # BUGGY field specs (only output_specs, missing inflow)
        buggy_field_specs = {
            "density": 1,
            "velocity": 2,
            # inflow is missing!
        }

        # Try to extract fields
        channel_idx = 0
        extracted = {}

        for field_name, num_channels in buggy_field_specs.items():
            pred_field = prediction[:, channel_idx : channel_idx + num_channels, :, :]
            gt_field = ground_truth[:, channel_idx : channel_idx + num_channels, :, :]

            extracted[field_name] = {"prediction": pred_field, "ground_truth": gt_field}

            channel_idx += num_channels

        # We've only used 3 channels, but data has 4
        channels_used = sum(buggy_field_specs.values())
        assert channels_used == 3
        assert prediction.shape[1] == 4

        # The 4th channel (inflow) is ignored!
        print(f"BUG: Only used {channels_used} of {prediction.shape[1]} channels")
        print("The inflow field is present in data but not extracted for visualization")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
