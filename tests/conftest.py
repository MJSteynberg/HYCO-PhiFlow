"""
Pytest configuration and shared fixtures for HYCO-PhiFlow tests.

This module provides:
- Shared fixtures for configuration loading
- Model fixtures for testing
- Dataset fixtures
- Utility functions for test setup/teardown
"""

import sys
from pathlib import Path
import shutil
import pytest
import yaml
from copy import deepcopy

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def project_root():
    """Return project root path."""
    return PROJECT_ROOT


@pytest.fixture
def base_burgers_1d_config():
    """Load base Burgers 1D configuration."""
    config_path = PROJECT_ROOT / "conf" / "burgers_1d.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def base_burgers_2d_config():
    """Load base Burgers 2D configuration."""
    config_path = PROJECT_ROOT / "conf" / "burgers_2d.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def base_advection_config():
    """Load base advection configuration."""
    config_path = PROJECT_ROOT / "conf" / "advection.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def base_navier_stokes_config():
    """Load base Navier-Stokes configuration."""
    config_path = PROJECT_ROOT / "conf" / "navier_stokes_2d.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def quick_test_config(base_burgers_1d_config):
    """
    Create a minimal config for quick unit tests.

    Uses small values for all parameters to ensure fast execution.
    """
    config = deepcopy(base_burgers_1d_config)

    # Minimal data settings
    config['data']['num_simulations'] = 2
    config['data']['trajectory_length'] = 10
    config['data']['data_dir'] = 'data/test_quick'

    # Minimal training settings
    config['trainer']['batch_size'] = 4
    config['trainer']['rollout_steps'] = 2
    config['trainer']['train_sim'] = [0]
    config['trainer']['synthetic']['epochs'] = 1
    config['trainer']['physical']['epochs'] = 1
    config['trainer']['physical']['downsample_factor'] = 0  # Disable downsampling for faster tests
    config['trainer']['hybrid']['cycles'] = 1
    config['trainer']['hybrid']['warmup'] = 0

    # Use test model paths
    config['model']['synthetic']['model_path'] = 'results/test_models'
    config['model']['physical']['model_path'] = 'results/test_models'

    return config


@pytest.fixture
def synthetic_config(quick_test_config):
    """Config for synthetic training tests."""
    config = deepcopy(quick_test_config)
    config['general']['mode'] = 'synthetic'
    return config


@pytest.fixture
def physical_config(quick_test_config):
    """Config for physical training tests."""
    config = deepcopy(quick_test_config)
    config['general']['mode'] = 'physical'
    return config


@pytest.fixture
def hybrid_config(quick_test_config):
    """Config for hybrid training tests."""
    config = deepcopy(quick_test_config)
    config['general']['mode'] = 'hybrid'
    return config


# =============================================================================
# Data Fixtures
# =============================================================================

@pytest.fixture
def test_data_dir(quick_test_config):
    """
    Create test data directory and clean up after tests.

    Generates minimal simulation data for testing.
    """
    from src.data.data_generator import DataGenerator

    data_dir = Path(quick_test_config['data']['data_dir'])

    # Generate data if not exists
    if not data_dir.exists() or not list(data_dir.glob("sim_*.npz")):
        data_gen = DataGenerator(quick_test_config)
        data_gen.generate_data()

    yield data_dir

    # Cleanup after tests (optional - comment out to keep data for debugging)
    # if data_dir.exists():
    #     shutil.rmtree(data_dir)


@pytest.fixture
def sample_dataset(quick_test_config, test_data_dir):
    """Create a sample dataset for testing."""
    from src.data.dataset import Dataset

    return Dataset(
        config=quick_test_config,
        train_sim=quick_test_config['trainer']['train_sim'],
        rollout_steps=quick_test_config['trainer']['rollout_steps']
    )


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def physical_model(quick_test_config):
    """Create a physical model instance."""
    from src.factories.model_factory import ModelFactory
    return ModelFactory.create_physical_model(quick_test_config)


@pytest.fixture
def synthetic_model(quick_test_config, sample_dataset):
    """Create a synthetic model instance."""
    from src.factories.model_factory import ModelFactory
    return ModelFactory.create_synthetic_model(
        quick_test_config,
        num_channels=sample_dataset.num_channels
    )


# =============================================================================
# Trainer Fixtures
# =============================================================================

@pytest.fixture
def synthetic_trainer(synthetic_config, sample_dataset):
    """Create a synthetic trainer instance."""
    from src.factories.trainer_factory import TrainerFactory
    return TrainerFactory.create_trainer(
        synthetic_config,
        num_channels=sample_dataset.num_channels
    )


@pytest.fixture
def physical_trainer(physical_config, test_data_dir):
    """Create a physical trainer instance."""
    from src.factories.trainer_factory import TrainerFactory
    return TrainerFactory.create_trainer(physical_config)


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for test artifacts."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture(autouse=True)
def clear_model_registry():
    """Clear model registry before each test to ensure isolation."""
    from src.models.registry import ModelRegistry

    # Store current state
    physical_models = dict(ModelRegistry._physical_models)
    synthetic_models = dict(ModelRegistry._synthetic_models)

    yield

    # Restore state (in case test modified registry)
    ModelRegistry._physical_models = physical_models
    ModelRegistry._synthetic_models = synthetic_models


# =============================================================================
# Markers
# =============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
