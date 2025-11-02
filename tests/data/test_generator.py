"""
Comprehensive tests for data generator.
Tests model instantiation, scene creation, and simulation generation.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import yaml

from src.data.generator import get_physical_model, run_generation
from src.models.physical import HeatModel, BurgersModel, SmokeModel
from phi.geom import Box
from phiml.math import spatial


class TestGetPhysicalModel:
    """Tests for get_physical_model function."""

    @pytest.fixture
    def heat_config(self):
        """Create config for heat model."""
        return {
            "model": {
                "physical": {
                    "name": "HeatModel",
                    "domain": {"size_x": 100.0, "size_y": 100.0},
                    "resolution": {"x": 64, "y": 64},
                    "dt": 0.1,
                    "pde_params": {"diffusivity": 1.0},
                }
            }
        }

    @pytest.fixture
    def burgers_config(self):
        """Create config for burgers model."""
        return {
            "model": {
                "physical": {
                    "name": "BurgersModel",
                    "domain": {"size_x": 1.0, "size_y": 1.0},
                    "resolution": {"x": 128, "y": 128},
                    "dt": 0.01,
                    "pde_params": {"nu": 0.01, "batch_size": 1},
                }
            }
        }

    @pytest.fixture
    def smoke_config(self):
        """Create config for smoke model."""
        return {
            "model": {
                "physical": {
                    "name": "SmokeModel",
                    "domain": {"size_x": 100.0, "size_y": 100.0},
                    "resolution": {"x": 128, "y": 128},
                    "dt": 0.1,
                    "pde_params": {"batch_size": 1},
                }
            }
        }

    def test_get_heat_model(self, heat_config):
        """Test getting HeatModel from config."""
        model = get_physical_model(heat_config)

        assert model is not None
        assert isinstance(model, HeatModel)

    def test_get_burgers_model(self, burgers_config):
        """Test getting BurgersModel from config."""
        model = get_physical_model(burgers_config)

        assert model is not None
        assert isinstance(model, BurgersModel)

    def test_get_smoke_model(self, smoke_config):
        """Test getting SmokeModel from config."""
        model = get_physical_model(smoke_config)

        assert model is not None
        assert isinstance(model, SmokeModel)

    def test_model_has_correct_resolution(self, heat_config):
        """Test that model has correct resolution."""
        model = get_physical_model(heat_config)

        # Check resolution dimensions using PhiFlow Shape API
        # Resolution is a Shape object with spatial dimensions
        assert model.resolution.spatial.volume == 64 * 64

    def test_model_has_correct_dt(self, heat_config):
        """Test that model has correct time step."""
        model = get_physical_model(heat_config)

        assert model.dt == 0.1

    def test_model_has_pde_params(self, heat_config):
        """Test that model has PDE parameters."""
        model = get_physical_model(heat_config)

        assert hasattr(model, "diffusivity")
        assert model.diffusivity == 1.0

    def test_invalid_model_name_raises_error(self):
        """Test that invalid model name raises error."""
        config = {
            "model": {
                "physical": {
                    "name": "InvalidModel",
                    "domain": {"size_x": 1.0, "size_y": 1.0},
                    "resolution": {"x": 64, "y": 64},
                    "dt": 0.1,
                    "pde_params": {},
                }
            }
        }

        with pytest.raises(ImportError, match="Model .* not found"):
            get_physical_model(config)


class TestRunGenerationSetup:
    """Tests for run_generation function setup."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def minimal_heat_config(self, temp_output_dir):
        """Create minimal config for heat generation."""
        return {
            "project_root": temp_output_dir,
            "data": {"data_dir": "data", "dset_name": "test_heat", "fields": ["temp"]},
            "model": {
                "physical": {
                    "name": "HeatModel",
                    "domain": {"size_x": 100.0, "size_y": 100.0},
                    "resolution": {"x": 32, "y": 32},
                    "dt": 0.1,
                    "pde_params": {"diffusivity": 1.0},
                }
            },
            "generation_params": {
                "num_simulations": 1,
                "total_steps": 5,
                "save_interval": 1,
            },
        }

    def test_output_directory_created(self, minimal_heat_config):
        """Test that output directory is created."""
        run_generation(minimal_heat_config)

        project_root = minimal_heat_config["project_root"]
        data_dir = minimal_heat_config["data"]["data_dir"]
        dset_name = minimal_heat_config["data"]["dset_name"]

        output_path = Path(project_root) / data_dir / dset_name

        assert output_path.exists()

    def test_scene_directory_created(self, minimal_heat_config):
        """Test that scene directory is created."""
        run_generation(minimal_heat_config)

        project_root = minimal_heat_config["project_root"]
        data_dir = minimal_heat_config["data"]["data_dir"]
        dset_name = minimal_heat_config["data"]["dset_name"]

        output_path = Path(project_root) / data_dir / dset_name
        scene_dirs = list(output_path.glob("sim_*"))

        assert len(scene_dirs) >= 1

    def test_description_json_created(self, minimal_heat_config):
        """Test that description.json is created."""
        run_generation(minimal_heat_config)

        project_root = minimal_heat_config["project_root"]
        data_dir = minimal_heat_config["data"]["data_dir"]
        dset_name = minimal_heat_config["data"]["dset_name"]

        output_path = Path(project_root) / data_dir / dset_name
        scene_dirs = list(output_path.glob("sim_*"))

        desc_path = scene_dirs[0] / "description.json"
        assert desc_path.exists()

    def test_initial_frame_saved(self, minimal_heat_config):
        """Test that initial frame (frame 0) is saved."""
        run_generation(minimal_heat_config)

        project_root = minimal_heat_config["project_root"]
        data_dir = minimal_heat_config["data"]["data_dir"]
        dset_name = minimal_heat_config["data"]["dset_name"]

        output_path = Path(project_root) / data_dir / dset_name
        scene_dirs = list(output_path.glob("sim_*"))

        # Check for initial frame file
        frame_files = list(scene_dirs[0].glob("*.npz"))
        assert len(frame_files) > 0

    def test_multiple_frames_saved(self, minimal_heat_config):
        """Test that multiple frames are saved."""
        minimal_heat_config["generation_params"]["total_steps"] = 10
        minimal_heat_config["generation_params"]["save_interval"] = 2

        run_generation(minimal_heat_config)

        project_root = minimal_heat_config["project_root"]
        data_dir = minimal_heat_config["data"]["data_dir"]
        dset_name = minimal_heat_config["data"]["dset_name"]

        output_path = Path(project_root) / data_dir / dset_name
        scene_dirs = list(output_path.glob("sim_*"))

        # Should have frame 0 + 5 saved frames (steps 2, 4, 6, 8, 10)
        frame_files = list(scene_dirs[0].glob("*.npz"))

        # At least 2 frames (initial + one saved)
        assert len(frame_files) >= 2

    def test_multiple_simulations_created(self, minimal_heat_config):
        """Test that multiple simulations are created."""
        minimal_heat_config["generation_params"]["num_simulations"] = 3

        run_generation(minimal_heat_config)

        project_root = minimal_heat_config["project_root"]
        data_dir = minimal_heat_config["data"]["data_dir"]
        dset_name = minimal_heat_config["data"]["dset_name"]

        output_path = Path(project_root) / data_dir / dset_name
        scene_dirs = list(output_path.glob("sim_*"))

        assert len(scene_dirs) == 3


class TestRunGenerationDifferentModels:
    """Tests for run_generation with different physical models."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_burgers_generation(self, temp_output_dir):
        """Test generation with Burgers model."""
        config = {
            "project_root": temp_output_dir,
            "data": {
                "data_dir": "data",
                "dset_name": "test_burgers",
                "fields": ["velocity"],
            },
            "model": {
                "physical": {
                    "name": "BurgersModel",
                    "domain": {"size_x": 1.0, "size_y": 1.0},
                    "resolution": {"x": 32, "y": 32},
                    "dt": 0.01,
                    "pde_params": {"nu": 0.01, "batch_size": 1},
                }
            },
            "generation_params": {
                "num_simulations": 1,
                "total_steps": 5,
                "save_interval": 1,
            },
        }

        run_generation(config)

        output_path = Path(temp_output_dir) / "data" / "test_burgers"
        assert output_path.exists()

        scene_dirs = list(output_path.glob("sim_*"))
        assert len(scene_dirs) >= 1

    def test_smoke_generation(self, temp_output_dir):
        """Test generation with Smoke model."""
        config = {
            "project_root": temp_output_dir,
            "data": {
                "data_dir": "data",
                "dset_name": "test_smoke",
                "fields": ["velocity", "density"],
            },
            "model": {
                "physical": {
                    "name": "SmokeModel",
                    "domain": {"size_x": 100.0, "size_y": 100.0},
                    "resolution": {"x": 32, "y": 32},
                    "dt": 0.1,
                    "pde_params": {"batch_size": 1},
                }
            },
            "generation_params": {
                "num_simulations": 1,
                "total_steps": 5,
                "save_interval": 1,
            },
        }

        run_generation(config)

        output_path = Path(temp_output_dir) / "data" / "test_smoke"
        assert output_path.exists()

        scene_dirs = list(output_path.glob("sim_*"))
        assert len(scene_dirs) >= 1


class TestRunGenerationMetadata:
    """Tests for metadata generation."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def config_with_metadata(self, temp_output_dir):
        """Create config for metadata testing."""
        return {
            "project_root": temp_output_dir,
            "data": {
                "data_dir": "data",
                "dset_name": "test_metadata",
                "fields": ["temp"],
                "fields_scheme": "T",
            },
            "model": {
                "physical": {
                    "name": "HeatModel",
                    "domain": {"size_x": 100.0, "size_y": 100.0},
                    "resolution": {"x": 32, "y": 32},
                    "dt": 0.1,
                    "pde_params": {"diffusivity": 2.0},
                }
            },
            "generation_params": {
                "num_simulations": 1,
                "total_steps": 5,
                "save_interval": 2,
            },
        }

    def test_metadata_contains_pde_name(self, config_with_metadata):
        """Test that metadata contains PDE name."""
        run_generation(config_with_metadata)

        from phi.torch.flow import Scene

        output_path = (
            Path(config_with_metadata["project_root"]) / "data" / "test_metadata"
        )
        scene_dirs = list(output_path.glob("sim_*"))
        scene = Scene.at(str(scene_dirs[0]))

        props = scene.properties
        assert "PDE" in props
        assert props["PDE"] == "HeatModel"

    def test_metadata_contains_fields(self, config_with_metadata):
        """Test that metadata contains fields list."""
        run_generation(config_with_metadata)

        from phi.torch.flow import Scene

        output_path = (
            Path(config_with_metadata["project_root"]) / "data" / "test_metadata"
        )
        scene_dirs = list(output_path.glob("sim_*"))
        scene = Scene.at(str(scene_dirs[0]))

        props = scene.properties
        assert "Fields" in props
        assert props["Fields"] == ["temp"]

    def test_metadata_contains_dt(self, config_with_metadata):
        """Test that metadata contains time step."""
        run_generation(config_with_metadata)

        from phi.torch.flow import Scene

        output_path = (
            Path(config_with_metadata["project_root"]) / "data" / "test_metadata"
        )
        scene_dirs = list(output_path.glob("sim_*"))
        scene = Scene.at(str(scene_dirs[0]))

        props = scene.properties
        assert "Dt" in props
        # Dt should be model dt * save_interval
        assert props["Dt"] == 0.1 * 2

    def test_metadata_contains_resolution(self, config_with_metadata):
        """Test that metadata contains resolution."""
        run_generation(config_with_metadata)

        from phi.torch.flow import Scene

        output_path = (
            Path(config_with_metadata["project_root"]) / "data" / "test_metadata"
        )
        scene_dirs = list(output_path.glob("sim_*"))
        scene = Scene.at(str(scene_dirs[0]))

        props = scene.properties
        assert "Resolution" in props
        assert props["Resolution"]["x"] == 32
        assert props["Resolution"]["y"] == 32

    def test_metadata_contains_pde_params(self, config_with_metadata):
        """Test that metadata contains PDE parameters."""
        run_generation(config_with_metadata)

        from phi.torch.flow import Scene

        output_path = (
            Path(config_with_metadata["project_root"]) / "data" / "test_metadata"
        )
        scene_dirs = list(output_path.glob("sim_*"))
        scene = Scene.at(str(scene_dirs[0]))

        props = scene.properties
        assert "PDE_Params" in props
        assert props["PDE_Params"]["diffusivity"] == 2.0


class TestRunGenerationSaveIntervals:
    """Tests for different save intervals."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_save_interval_1(self, temp_output_dir):
        """Test with save interval of 1 (save every step)."""
        config = {
            "project_root": temp_output_dir,
            "data": {
                "data_dir": "data",
                "dset_name": "test_interval_1",
                "fields": ["temp"],
            },
            "model": {
                "physical": {
                    "name": "HeatModel",
                    "domain": {"size_x": 100.0, "size_y": 100.0},
                    "resolution": {"x": 32, "y": 32},
                    "dt": 0.1,
                    "pde_params": {"diffusivity": 1.0},
                }
            },
            "generation_params": {
                "num_simulations": 1,
                "total_steps": 5,
                "save_interval": 1,
            },
        }

        run_generation(config)

        output_path = Path(temp_output_dir) / "data" / "test_interval_1"
        scene_dirs = list(output_path.glob("sim_*"))
        frame_files = list(scene_dirs[0].glob("*.npz"))

        # Should have 6 frames (0 + 5 steps)
        assert len(frame_files) == 6

    def test_save_interval_5(self, temp_output_dir):
        """Test with save interval of 5."""
        config = {
            "project_root": temp_output_dir,
            "data": {
                "data_dir": "data",
                "dset_name": "test_interval_5",
                "fields": ["temp"],
            },
            "model": {
                "physical": {
                    "name": "HeatModel",
                    "domain": {"size_x": 100.0, "size_y": 100.0},
                    "resolution": {"x": 32, "y": 32},
                    "dt": 0.1,
                    "pde_params": {"diffusivity": 1.0},
                }
            },
            "generation_params": {
                "num_simulations": 1,
                "total_steps": 10,
                "save_interval": 5,
            },
        }

        run_generation(config)

        output_path = Path(temp_output_dir) / "data" / "test_interval_5"
        scene_dirs = list(output_path.glob("sim_*"))
        frame_files = list(scene_dirs[0].glob("*.npz"))

        # Should have 3 frames (0, 5, 10)
        assert len(frame_files) == 3


class TestRunGenerationFieldsScheme:
    """Tests for different field schemes."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_single_field_scheme(self, temp_output_dir):
        """Test with single field."""
        config = {
            "project_root": temp_output_dir,
            "data": {
                "data_dir": "data",
                "dset_name": "test_single_field",
                "fields": ["temp"],
                "fields_scheme": "T",
            },
            "model": {
                "physical": {
                    "name": "HeatModel",
                    "domain": {"size_x": 100.0, "size_y": 100.0},
                    "resolution": {"x": 32, "y": 32},
                    "dt": 0.1,
                    "pde_params": {"diffusivity": 1.0},
                }
            },
            "generation_params": {
                "num_simulations": 1,
                "total_steps": 3,
                "save_interval": 1,
            },
        }

        run_generation(config)

        output_path = Path(temp_output_dir) / "data" / "test_single_field"
        assert output_path.exists()

    def test_multiple_fields_scheme(self, temp_output_dir):
        """Test with multiple fields."""
        config = {
            "project_root": temp_output_dir,
            "data": {
                "data_dir": "data",
                "dset_name": "test_multi_field",
                "fields": ["velocity", "density"],
                "fields_scheme": "VD",
            },
            "model": {
                "physical": {
                    "name": "SmokeModel",
                    "domain": {"size_x": 100.0, "size_y": 100.0},
                    "resolution": {"x": 32, "y": 32},
                    "dt": 0.1,
                    "pde_params": {"batch_size": 1},
                }
            },
            "generation_params": {
                "num_simulations": 1,
                "total_steps": 3,
                "save_interval": 1,
            },
        }

        run_generation(config)

        output_path = Path(temp_output_dir) / "data" / "test_multi_field"
        assert output_path.exists()
