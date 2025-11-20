# PhiML Migration Plan: PyTorch → Pure PhiFlow/PhiML

**Objective**: Migrate HYCO-PhiFlow from hybrid PyTorch-PhiFlow to pure PhiML implementation
**Strategy**: Gradual rollout with testing at each stage to prevent breakage
**Expected Benefits**:
- 🚀 15-20% performance improvement (eliminate conversion overhead)
- 🧹 ~30% code reduction (simpler data pipeline)
- 💪 10-100x speedup with JIT compilation
- 🎯 Better type safety with named dimensions
- 🔧 Easier maintenance and debugging

---

## Table of Contents

1. [Migration Overview](#1-migration-overview)
2. [Phase 0: Pre-Migration Preparation](#phase-0-pre-migration-preparation)
3. [Phase 1: Data Pipeline Migration](#phase-1-data-pipeline-migration)
4. [Phase 2: Model Migration](#phase-2-model-migration)
5. [Phase 3: Training Infrastructure Migration](#phase-3-training-infrastructure-migration)
6. [Phase 4: Hybrid Trainer Migration](#phase-4-hybrid-trainer-migration)
7. [Phase 5: Optimization and Cleanup](#phase-5-optimization-and-cleanup)
8. [Testing Strategy](#testing-strategy)
9. [Rollback Plan](#rollback-plan)
10. [Success Metrics](#success-metrics)

---

## 1. Migration Overview

### 1.1 Current State Analysis

**PyTorch Dependencies** (21 files):
- Data pipeline: DataManager, TensorDataset, FieldDataset (heavy conversion logic)
- Synthetic models: All use `nn.Module` base class
- Training: torch.optim, DataLoader, AMP
- Utilities: Memory profiling, device management

**Migration Complexity**:
```
Low Complexity:    Physical models (already mostly PhiFlow)
Medium Complexity: Synthetic models (phiml.nn available)
High Complexity:   Data pipeline (extensive conversion code)
High Complexity:   Hybrid trainer (orchestrates conversions)
```

### 1.2 Migration Phases

```
Phase 0: Preparation (1-2 days)
   ↓
Phase 1: Data Pipeline (3-5 days)
   ↓
Phase 2: Models (2-3 days)
   ↓
Phase 3: Training Infrastructure (3-4 days)
   ↓
Phase 4: Hybrid Trainer (2-3 days)
   ↓
Phase 5: Optimization & Cleanup (2-3 days)
   ↓
Total: ~2-3 weeks
```

### 1.3 Backward Compatibility Strategy

Each phase will:
1. ✅ Create new PhiML implementation alongside old code
2. ✅ Add feature flag to switch between old/new
3. ✅ Test extensively with both implementations
4. ✅ Only remove old code after full validation
5. ✅ Keep migration reversible until Phase 5

---

## Phase 0: Pre-Migration Preparation

**Duration**: 1-2 days
**Risk**: Low
**Goal**: Fix critical bugs and establish testing baseline

### 0.1 Fix Critical Bugs

**Priority 1: Config Access Bug**

File: [src/models/synthetic/resnet.py](src/models/synthetic/resnet.py#L39)
```python
# Current (BROKEN):
arch_config = config["synthetic"]["architecture"]

# Fix to:
arch_config = config["model"]["synthetic"]["architecture"]
```

File: [src/models/synthetic/convnet.py](src/models/synthetic/convnet.py#L39)
```python
# Same fix as above
arch_config = config["model"]["synthetic"]["architecture"]
```

**Priority 2: Replace Unsafe eval()**

Files: [src/models/physical/advection.py](src/models/physical/advection.py#L54), [src/models/physical/burgers.py](src/models/physical/burgers.py#L69)

```python
# Install numexpr
# pip install numexpr

# Replace:
value = eval(param_value, {'x': x, 'y': y, 'size_x': size_x, 'size_y': size_y})

# With:
import numexpr
value = numexpr.evaluate(param_value, local_dict={
    'x': x, 'y': y, 'size_x': size_x, 'size_y': size_y
})
```

**Priority 3: Fix Device Hardcoding**

File: [src/data/field_dataset.py](src/data/field_dataset.py#L229)

```python
# Add to __init__:
self.device = torch.device(config["trainer"]["device"])

# Replace hardcoded "cuda":
tensor_value = tensor_value.to(self.device)
```

**Priority 4: Replace Asserts**

File: [src/data/field_dataset.py](src/data/field_dataset.py#L464)

```python
# Replace:
assert cache_meta["creation_timestamp"] == timestamp, f"..."

# With:
if cache_meta["creation_timestamp"] != timestamp:
    raise ValueError(f"...")
```

### 0.2 Establish Testing Baseline

**Create test directory structure**:
```bash
mkdir -p tests/{unit,integration,fixtures}
touch tests/__init__.py
touch tests/conftest.py
```

**Create basic fixtures** (`tests/fixtures/sample_data.py`):
```python
from phiml import math
from phiml.math import batch, spatial, channel
from phi.flow import CenteredGrid, Box, ZERO_GRADIENT

def create_sample_field(resolution=64):
    """Create a simple test field"""
    return CenteredGrid(
        lambda x, y: math.sin(2 * math.pi * x / 100),
        ZERO_GRADIENT,
        Box(x=100, y=100),
        x=resolution, y=resolution
    )

def create_sample_tensor_data(batch_size=32, resolution=64):
    """Create sample tensor data"""
    return math.random_normal(
        batch(examples=batch_size),
        spatial(x=resolution, y=resolution),
        channel(c=3)
    )
```

**Create smoke tests** (`tests/integration/test_current_system.py`):
```python
import pytest
from src.factories.model_factory import ModelFactory
from src.factories.trainer_factory import TrainerFactory

def test_advection_model_creation():
    """Ensure advection model creates without errors"""
    config = load_test_config('advection')
    model = ModelFactory.create_physical_model(config)
    assert model is not None

def test_burgers_model_creation():
    """Ensure burgers model creates without errors"""
    config = load_test_config('burgers')
    model = ModelFactory.create_physical_model(config)
    assert model is not None

# Add tests for all current functionality
```

**Run baseline tests**:
```bash
pytest tests/ -v --tb=short
# Save output as baseline
pytest tests/ --json-report --json-report-file=baseline_results.json
```

### 0.3 Create Migration Feature Flag

Create `src/config/migration_flags.py`:
```python
from enum import Enum

class MigrationMode(Enum):
    LEGACY = "legacy"  # Pure PyTorch
    HYBRID = "hybrid"  # Mixed (for testing)
    PHIML = "phiml"    # Pure PhiML

# Global flag (will be configurable via YAML)
MIGRATION_MODE = MigrationMode.LEGACY
```

Add to config YAML:
```yaml
migration:
  mode: "legacy"  # Options: legacy, hybrid, phiml
```

### 0.4 Documentation

Create migration log:
```bash
touch MIGRATION_LOG.md
```

**Deliverables**:
- ✅ All critical bugs fixed
- ✅ Baseline test suite passing
- ✅ Feature flag system in place
- ✅ Migration log started

---

## Phase 1: Data Pipeline Migration

**Duration**: 3-5 days
**Risk**: High (most complex part)
**Goal**: Eliminate Field ↔ Tensor conversions, use PhiML tensors throughout

### 1.1 Understanding Current Data Flow

```
Current (Complex):
┌──────────────┐
│ Scene (Disk) │ PhiFlow Fields
└──────┬───────┘
       │
       v
┌──────────────────┐
│  DataManager     │ Converts Field → Tensor
│  .load()         │ Saves to cache (*.pth)
└──────┬───────────┘
       │
       ├─────────────────────┬──────────────────────┐
       v                     v                      v
┌─────────────┐    ┌──────────────┐      ┌─────────────────┐
│TensorDataset│    │ FieldDataset │      │AugmentationMgr  │
│(for synth)  │    │(for physical)│      │(converts too)   │
│             │    │              │      │                 │
│Converts     │    │Converts      │      │                 │
│back to Field│    │Tensor→Field  │      │                 │
└─────────────┘    └──────────────┘      └─────────────────┘
```

```
Target (Simple):
┌──────────────┐
│ Scene (Disk) │ PhiFlow Fields
└──────┬───────┘
       │
       v
┌──────────────────┐
│  DataManager     │ Loads as PhiML tensors directly
│  .load()         │ Saves to cache (*.npz or native)
└──────┬───────────┘
       │
       └────────────────────┐
                            v
                   ┌─────────────────┐
                   │ UnifiedDataset  │ PhiML tensors
                   │                 │ (no conversions!)
                   └─────────────────┘
```

### 1.2 Create New PhiML Data Manager

Create `src/data/phiml_data_manager.py`:

```python
from phiml import math
from phiml.math import batch, spatial, channel
from phi.flow import Scene, CenteredGrid, StaggeredGrid
from pathlib import Path
import logging

class PhiMLDataManager:
    """
    Data manager that works entirely with PhiML native tensors.
    No conversion to/from PyTorch tensors.
    """

    def __init__(self, config: dict):
        self.config = config
        self.data_dir = Path(config["data"]["data_dir"])
        self.cache_dir = Path(config["data"].get("cache_dir", "./cache"))
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def load_simulation(self, sim_index: int) -> 'Tensor':
        """
        Load a simulation and return as PhiML tensor.

        Returns:
            Tensor with shape (time=T, x=X, y=Y, channels=C)
            All dimensions are named!
        """
        # Check cache first
        cache_path = self.cache_dir / f"sim_{sim_index:04d}.npz"
        if cache_path.exists():
            return self._load_from_cache(cache_path)

        # Load from Scene
        scene_path = self.data_dir / f"sim_{sim_index:04d}"
        scene = Scene.at(scene_path)

        # Get all timesteps
        timesteps = list(range(len(scene)))
        fields_list = []

        for t in timesteps:
            frame = scene.read(frame=t)

            # Extract fields based on config
            field_data = []
            for field_name in self.config["data"]["fields"]:
                field = frame[field_name]

                # Convert Field to PhiML tensor
                # Field.values is already a PhiML tensor!
                tensor = field.values

                # Ensure it's a native tensor (not Field wrapper)
                if hasattr(tensor, 'native'):
                    # Already a phiml tensor, just extract values
                    pass

                field_data.append(tensor)

            # Stack fields along channel dimension
            frame_tensor = math.concat(field_data, channel('fields'))
            fields_list.append(frame_tensor)

        # Stack along time dimension
        simulation = math.stack(fields_list, batch('time'))

        # Cache for future use
        self._save_to_cache(simulation, cache_path)

        return simulation

    def _save_to_cache(self, tensor: 'Tensor', path: Path):
        """Save PhiML tensor to disk"""
        # Use NumPy format for compatibility
        import numpy as np

        data = {
            'values': tensor.native(),  # Get native array
            'shape_names': list(tensor.shape.names),
            'shape_sizes': list(tensor.shape.sizes),
            'shape_types': [str(t) for t in tensor.shape.types]
        }

        np.savez_compressed(path, **data)
        self.logger.info(f"Cached simulation to {path}")

    def _load_from_cache(self, path: Path) -> 'Tensor':
        """Load PhiML tensor from disk"""
        import numpy as np

        data = np.load(path)

        # Reconstruct shape
        from phiml.math import Shape
        shape = Shape(
            sizes=tuple(data['shape_sizes']),
            names=tuple(data['shape_names']),
            types=tuple(data['shape_types'])
        )

        # Wrap native array with shape
        tensor = math.wrap(data['values'], shape)

        self.logger.info(f"Loaded simulation from cache: {path}")
        return tensor

    def load_all_simulations(self) -> 'Tensor':
        """
        Load all simulations and stack them.

        Returns:
            Tensor with shape (simulations=N, time=T, x=X, y=Y, channels=C)
        """
        num_sims = self.config["data"]["num_simulations"]

        sims = []
        for i in range(num_sims):
            sim = self.load_simulation(i)
            sims.append(sim)

        # Stack along new 'simulations' batch dimension
        all_data = math.stack(sims, batch('simulations'))

        return all_data
```

### 1.3 Create Unified PhiML Dataset

Create `src/data/phiml_dataset.py`:

```python
from phiml import math
from phiml.math import batch
from typing import Optional

class PhiMLDataset:
    """
    Unified dataset that works with PhiML tensors.
    Replaces both TensorDataset and FieldDataset.
    """

    def __init__(
        self,
        data: 'Tensor',  # Shape: (simulations, time, ...spatial..., channels)
        rollout_steps: int,
        augmented_data: Optional['Tensor'] = None
    ):
        self.data = data
        self.rollout_steps = rollout_steps
        self.augmented_data = augmented_data

        # Pre-compute valid window indices
        self._build_index_map()

    def _build_index_map(self):
        """Pre-compute all valid (sim, time_start) pairs"""
        num_sims = self.data.shape.get_size('simulations')
        traj_length = self.data.shape.get_size('time')

        self.index_map = []
        for sim_idx in range(num_sims):
            # Each simulation contributes (traj_length - rollout_steps + 1) windows
            num_windows = traj_length - self.rollout_steps + 1
            for t_start in range(num_windows):
                self.index_map.append((sim_idx, t_start))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int) -> 'Tensor':
        """
        Get a windowed trajectory.

        Returns:
            Tensor with shape (time=rollout_steps, ...spatial..., channels)
        """
        sim_idx, t_start = self.index_map[idx]

        # Extract window using named dimensions
        window = self.data[{
            'simulations': sim_idx,
            'time': slice(t_start, t_start + self.rollout_steps)
        }]

        return window

    def get_batch(self, indices: list) -> 'Tensor':
        """
        Get multiple windows as a batch.

        Returns:
            Tensor with shape (batch=len(indices), time=rollout_steps, ...)
        """
        windows = [self[i] for i in indices]
        return math.stack(windows, batch('batch'))

    def enable_augmentation(self, augmented_data: 'Tensor'):
        """Add augmented data to the dataset"""
        self.augmented_data = augmented_data
        self._build_index_map()  # Rebuild with augmented data

    def disable_augmentation(self):
        """Remove augmented data"""
        self.augmented_data = None
        self._build_index_map()
```

### 1.4 Integration with Training

Create adapter in `src/factories/dataloader_factory.py`:

```python
# Add new method
def create_phiml_dataset(config: dict, mode: str = 'train'):
    """
    Create PhiML-based dataset (no PyTorch dependency)
    """
    from src.data.phiml_data_manager import PhiMLDataManager
    from src.data.phiml_dataset import PhiMLDataset

    # Load data
    manager = PhiMLDataManager(config)

    if mode == 'train':
        sim_indices = config["trainer"]["train_sim"]
    else:
        sim_indices = config["evaluation"]["test_sim"]

    # Load simulations
    all_data = []
    for sim_idx in sim_indices:
        sim = manager.load_simulation(sim_idx)
        all_data.append(sim)

    data = math.stack(all_data, batch('simulations'))

    # Create dataset
    dataset = PhiMLDataset(
        data=data,
        rollout_steps=config["trainer"]["rollout_steps"]
    )

    return dataset
```

### 1.5 Testing Phase 1

Create `tests/integration/test_phiml_data_pipeline.py`:

```python
import pytest
from src.data.phiml_data_manager import PhiMLDataManager
from src.data.phiml_dataset import PhiMLDataset

def test_phiml_data_manager_loads():
    config = load_test_config('burgers')
    manager = PhiMLDataManager(config)

    sim = manager.load_simulation(0)

    # Check it's a PhiML tensor
    assert hasattr(sim, 'shape')
    assert 'time' in sim.shape
    assert 'x' in sim.shape or 'y' in sim.shape

def test_phiml_dataset_windowing():
    # Create dummy data
    data = math.random_normal(
        batch(simulations=5, time=100),
        spatial(x=64, y=64),
        channel(c=3)
    )

    dataset = PhiMLDataset(data, rollout_steps=10)

    # Should have 5 * (100 - 10 + 1) = 455 windows
    assert len(dataset) == 455

    # Get a sample
    sample = dataset[0]
    assert sample.shape.get_size('time') == 10

def test_compare_old_vs_new():
    """Compare old TensorDataset vs new PhiMLDataset outputs"""
    config = load_test_config('burgers')

    # Old way
    from src.data.data_manager import DataManager
    from src.data.tensor_dataset import TensorDataset
    old_manager = DataManager(config)
    old_dataset = TensorDataset(old_manager, config, mode='train')

    # New way
    from src.data.phiml_data_manager import PhiMLDataManager
    from src.data.phiml_dataset import PhiMLDataset
    new_manager = PhiMLDataManager(config)
    new_dataset = create_phiml_dataset(config, mode='train')

    # Compare lengths
    assert len(old_dataset) == len(new_dataset)

    # Compare first sample (values should be identical)
    old_sample = old_dataset[0]
    new_sample = new_dataset[0]

    # Convert to numpy for comparison
    old_np = old_sample.detach().cpu().numpy()
    new_np = new_sample.native()

    np.testing.assert_allclose(old_np, new_np, rtol=1e-5)
```

Run tests:
```bash
pytest tests/integration/test_phiml_data_pipeline.py -v
```

**Deliverables**:
- ✅ PhiMLDataManager implemented
- ✅ PhiMLDataset implemented
- ✅ Tests pass showing equivalence to old system
- ✅ Feature flag allows switching between old/new

---

## Phase 2: Model Migration

**Duration**: 2-3 days
**Risk**: Medium
**Goal**: Convert synthetic models to pure phiml.nn, keep physical models as-is

### 2.1 Current Synthetic Model Structure

All synthetic models inherit from:
```python
class SyntheticModel(nn.Module):  # PyTorch
    def __init__(self, ...):
        super().__init__()
        # Uses phiml.nn internally but wrapped in nn.Module
```

### 2.2 Create PhiML-Native Synthetic Model Base

Create `src/models/synthetic/phiml_base.py`:

```python
from phiml import nn, math
from phiml.math import channel
import logging

class PhiMLSyntheticModel:
    """
    Base class for synthetic models using pure PhiML.
    No PyTorch dependency.
    """

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Get channel info
        self.num_dynamic_fields = len(config["data"]["fields_scheme"]["dynamic"])
        self.num_static_fields = len(config["data"]["fields_scheme"]["static"])
        self.total_fields = self.num_dynamic_fields + self.num_static_fields

        # Network will be created by subclasses
        self.network = None

    def __call__(self, input_tensor: 'Tensor') -> 'Tensor':
        """
        Forward pass with residual learning.

        Args:
            input_tensor: Shape (time=rollout_steps, ...spatial..., channels=C)

        Returns:
            predictions: Same shape as input
        """
        # Split dynamic and static channels
        dynamic = input_tensor.c[:self.num_dynamic_fields]
        static = input_tensor.c[self.num_dynamic_fields:]

        # Network predicts residual for dynamic fields
        residual = math.native_call(self.network, dynamic)

        # Add residual to input (residual learning)
        predicted_dynamic = dynamic + residual

        # Concatenate with static fields (unchanged)
        output = math.concat([predicted_dynamic, static], 'c')

        return output

    def predict_trajectory(
        self,
        initial_state: 'Tensor',
        num_steps: int
    ) -> 'Tensor':
        """
        Autoregressively predict a trajectory.

        Args:
            initial_state: Shape (...spatial..., channels=C)
            num_steps: Number of steps to predict

        Returns:
            trajectory: Shape (time=num_steps, ...spatial..., channels=C)
        """
        from phiml.math import batch

        states = [initial_state]

        for _ in range(num_steps - 1):
            # Current state
            current = states[-1]

            # Predict next
            next_state = self(current)

            states.append(next_state)

        # Stack along time dimension
        trajectory = math.stack(states, batch('time'))

        return trajectory

    def save(self, path: str):
        """Save model to disk"""
        math.save(self.network, path)
        self.logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk"""
        self.network = math.load(path)
        self.logger.info(f"Model loaded from {path}")
```

### 2.3 Migrate UNet Model

Create `src/models/synthetic/phiml_unet.py`:

```python
from phiml import nn
from src.models.synthetic.phiml_base import PhiMLSyntheticModel

class PhiMLUNet(PhiMLSyntheticModel):
    """UNet model using pure PhiML"""

    def __init__(self, config: dict):
        super().__init__(config)

        arch_config = config["model"]["synthetic"]["architecture"]

        # Create network using phiml.nn
        self.network = nn.u_net(
            in_channels=self.num_dynamic_fields,
            out_channels=self.num_dynamic_fields,
            levels=arch_config.get("levels", 4),
            filters=arch_config.get("filters", 32),
            batch_norm=arch_config.get("batch_norm", True),
            activation=arch_config.get("activation", "ReLU"),
            in_spatial=2  # 2D
        )

        self.logger.info(
            f"Created UNet: {self.num_dynamic_fields} channels, "
            f"{arch_config.get('levels', 4)} levels"
        )
```

### 2.4 Migrate ResNet Model

Create `src/models/synthetic/phiml_resnet.py`:

```python
from phiml import nn
from src.models.synthetic.phiml_base import PhiMLSyntheticModel

class PhiMLResNet(PhiMLSyntheticModel):
    """ResNet model using pure PhiML"""

    def __init__(self, config: dict):
        super().__init__(config)

        arch_config = config["model"]["synthetic"]["architecture"]

        self.network = nn.res_net(
            in_channels=self.num_dynamic_fields,
            out_channels=self.num_dynamic_fields,
            layers=arch_config.get("layers", [16, 32, 64]),
            batch_norm=arch_config.get("batch_norm", True),
            activation=arch_config.get("activation", "ReLU")
        )

        self.logger.info(f"Created ResNet: {arch_config.get('layers', [16, 32, 64])}")
```

### 2.5 Update Model Registry

Update `src/models/registry.py`:

```python
# Add PhiML versions
from src.models.synthetic.phiml_unet import PhiMLUNet
from src.models.synthetic.phiml_resnet import PhiMLResNet
from src.models.synthetic.phiml_convnet import PhiMLConvNet

class PhiMLModelRegistry:
    """Registry for PhiML-native models"""

    _synthetic_models = {
        'unet': PhiMLUNet,
        'resnet': PhiMLResNet,
        'convnet': PhiMLConvNet,
    }

    @classmethod
    def get_synthetic_model(cls, model_name: str):
        if model_name not in cls._synthetic_models:
            raise ValueError(f"Unknown model: {model_name}")
        return cls._synthetic_models[model_name]
```

### 2.6 Update Model Factory

Update `src/factories/model_factory.py`:

```python
@staticmethod
def create_synthetic_model(config: dict):
    """Create synthetic model (checks migration mode)"""
    from src.config.migration_flags import MIGRATION_MODE, MigrationMode

    model_name = config["model"]["synthetic"]["name"]

    if MIGRATION_MODE == MigrationMode.PHIML:
        # Use PhiML-native models
        from src.models.registry import PhiMLModelRegistry
        model_class = PhiMLModelRegistry.get_synthetic_model(model_name)
        return model_class(config)
    else:
        # Use old PyTorch models
        from src.models.registry import ModelRegistry
        model_class = ModelRegistry.get_synthetic_model(model_name)
        return model_class(config)
```

### 2.7 Testing Phase 2

Create `tests/unit/test_phiml_models.py`:

```python
import pytest
from phiml import math
from phiml.math import batch, spatial, channel

def test_phiml_unet_forward():
    from src.models.synthetic.phiml_unet import PhiMLUNet

    config = create_test_config_unet()
    model = PhiMLUNet(config)

    # Create input
    input_data = math.random_normal(
        batch(time=10),
        spatial(x=64, y=64),
        channel(c=3)
    )

    # Forward pass
    output = model(input_data)

    # Check shape
    assert output.shape == input_data.shape

def test_phiml_resnet_forward():
    from src.models.synthetic.phiml_resnet import PhiMLResNet

    config = create_test_config_resnet()
    model = PhiMLResNet(config)

    input_data = math.random_normal(
        batch(time=10),
        spatial(x=64, y=64),
        channel(c=3)
    )

    output = model(input_data)
    assert output.shape == input_data.shape

def test_autoregressive_prediction():
    from src.models.synthetic.phiml_unet import PhiMLUNet

    config = create_test_config_unet()
    model = PhiMLUNet(config)

    initial = math.random_normal(spatial(x=64, y=64), channel(c=3))

    trajectory = model.predict_trajectory(initial, num_steps=20)

    assert trajectory.shape.get_size('time') == 20

def test_model_save_load():
    from src.models.synthetic.phiml_unet import PhiMLUNet
    import tempfile

    config = create_test_config_unet()
    model = PhiMLUNet(config)

    with tempfile.NamedTemporaryFile(suffix='.phiml') as f:
        model.save(f.name)

        model2 = PhiMLUNet(config)
        model2.load(f.name)

        # Test they produce same output
        input_data = math.random_normal(spatial(x=64, y=64), channel(c=3))
        out1 = model(input_data)
        out2 = model2(input_data)

        np.testing.assert_allclose(out1.native(), out2.native())
```

**Deliverables**:
- ✅ PhiML synthetic model base class
- ✅ UNet, ResNet, ConvNet migrated
- ✅ Model factory updated with feature flag
- ✅ Tests pass for all models

---

## Phase 3: Training Infrastructure Migration

**Duration**: 3-4 days
**Risk**: Medium
**Goal**: Replace torch.optim with phiml.nn training

### 3.1 Create PhiML Synthetic Trainer

Create `src/training/synthetic/phiml_trainer.py`:

```python
from phiml import nn, math
from phiml.math import batch
import logging
from pathlib import Path

class PhiMLSyntheticTrainer:
    """
    Trainer for synthetic models using pure PhiML.
    No PyTorch dependency.
    """

    def __init__(self, model, config: dict):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Training config
        trainer_config = config["trainer"]["synthetic"]
        self.num_epochs = trainer_config["epochs"]
        self.learning_rate = trainer_config["learning_rate"]
        self.batch_size = config["trainer"]["batch_size"]
        self.rollout_steps = config["trainer"]["rollout_steps"]

        # Create optimizer
        self.optimizer = nn.adam(
            self.model.network,
            learning_rate=self.learning_rate
        )

        # Checkpointing
        self.checkpoint_dir = Path(config["general"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.best_loss = float('inf')

    def loss_function(self, prediction: 'Tensor', target: 'Tensor') -> 'Tensor':
        """
        Compute autoregressive rollout loss.

        Args:
            prediction: Shape (batch, time, ...spatial..., channels)
            target: Same shape

        Returns:
            Scalar loss
        """
        # L2 loss over all dimensions
        return math.l2_loss(prediction - target)

    def train_epoch(self, dataset) -> float:
        """Train for one epoch"""
        from phiml.math import batch as batch_dim
        import random

        # Shuffle indices
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        epoch_losses = []

        # Process in mini-batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]

            # Get batch
            batch_data = dataset.get_batch(batch_indices)
            # Shape: (batch=B, time=T, ...spatial..., channels=C)

            # Split into input and target
            input_seq = batch_data.time[0]  # First timestep
            target_seq = batch_data.time[1:]  # Remaining timesteps

            # Define loss for this batch
            def batch_loss_fn(input_data, target_data):
                # Predict trajectory
                pred = self.model.predict_trajectory(
                    input_data,
                    num_steps=self.rollout_steps - 1
                )
                return self.loss_function(pred, target_data)

            # Update weights (one line!)
            loss = nn.update_weights(
                self.model.network,
                self.optimizer,
                batch_loss_fn,
                input_seq,
                target_seq
            )

            epoch_losses.append(float(loss))

        return sum(epoch_losses) / len(epoch_losses)

    def validate(self, dataset) -> float:
        """Validate on validation set"""
        # Similar to train_epoch but without weight updates
        # Get a few batches
        val_losses = []

        for i in range(0, min(100, len(dataset)), self.batch_size):
            batch_data = dataset.get_batch(list(range(i, i+self.batch_size)))

            input_seq = batch_data.time[0]
            target_seq = batch_data.time[1:]

            # Forward pass only
            pred = self.model.predict_trajectory(input_seq, self.rollout_steps - 1)
            loss = self.loss_function(pred, target_seq)

            val_losses.append(float(loss))

        return sum(val_losses) / len(val_losses)

    def train(self, train_dataset, val_dataset=None):
        """Main training loop"""
        self.logger.info(f"Starting training for {self.num_epochs} epochs")

        for epoch in range(self.num_epochs):
            # Learning rate scheduling (cosine annealing)
            current_lr = self.learning_rate * 0.5 * (
                1 + math.cos(math.pi * epoch / self.num_epochs)
            )
            self.optimizer = nn.adam(self.model.network, learning_rate=current_lr)

            # Train
            train_loss = self.train_epoch(train_dataset)

            # Validate
            val_loss = self.validate(val_dataset) if val_dataset else train_loss

            self.logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"LR: {current_lr:.6f}"
            )

            # Checkpointing
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint('best_model.phiml')
                self.logger.info(f"Saved best model (val_loss={val_loss:.6f})")

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.phiml')

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        path = self.checkpoint_dir / filename
        self.model.save(str(path))

    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = self.checkpoint_dir / filename
        self.model.load(str(path))
```

### 3.2 Create PhiML Physical Trainer

Create `src/training/physical/phiml_trainer.py`:

```python
from phiml import math
from phi.field import Field
import logging

class PhiMLPhysicalTrainer:
    """
    Trainer for physical models using PhiML optimization.
    (This is similar to current implementation but cleaner)
    """

    def __init__(self, model, config: dict):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)

        trainer_config = config["trainer"]["physical"]
        self.num_epochs = trainer_config["epochs"]
        self.learnable_params = trainer_config["learnable_parameters"]

        # Solver configuration
        solve_config = trainer_config.get("solve", {})
        self.solve = math.Solve(
            method=solve_config.get("method", "L-BFGS-B"),
            rel_tol=solve_config.get("rel_tol", 1e-5),
            max_iterations=solve_config.get("max_iterations", 100)
        )

    def loss_function(self, prediction: Field, target: Field) -> 'Tensor':
        """Compute loss between predicted and target fields"""
        return math.l2_loss(prediction.values - target.values)

    def train(self, dataset):
        """
        Train physical model parameters.

        Uses math.minimize for optimization.
        """
        self.logger.info(f"Training physical model for {self.num_epochs} epochs")

        for epoch in range(self.num_epochs):
            epoch_losses = []

            for sample in dataset:
                # sample is a PhiML tensor or Field

                # Define loss for this sample
                def sample_loss():
                    # Simulate with current parameters
                    prediction = self.model.simulate(
                        initial_state=sample.time[0],
                        num_steps=sample.shape.get_size('time')
                    )

                    # Compute loss
                    loss = self.loss_function(prediction, sample)
                    return loss

                # Optimize learnable parameters
                result = math.minimize(
                    sample_loss,
                    solve=self.solve,
                    x0=self.model.get_parameters()
                )

                # Update model parameters
                self.model.set_parameters(result.x)

                epoch_losses.append(float(result.f))

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {avg_loss:.6f}")

    def save_checkpoint(self, filename: str):
        """Save learnable parameters"""
        params = self.model.get_parameters()
        math.save(params, filename)
```

### 3.3 Update Trainer Factory

Update `src/factories/trainer_factory.py`:

```python
@staticmethod
def create_synthetic_trainer(model, config: dict):
    """Create trainer based on migration mode"""
    from src.config.migration_flags import MIGRATION_MODE, MigrationMode

    if MIGRATION_MODE == MigrationMode.PHIML:
        from src.training.synthetic.phiml_trainer import PhiMLSyntheticTrainer
        return PhiMLSyntheticTrainer(model, config)
    else:
        from src.training.synthetic.trainer import SyntheticTrainer
        return SyntheticTrainer(model, config)

@staticmethod
def create_physical_trainer(model, config: dict):
    """Physical trainer (minimal changes needed)"""
    from src.config.migration_flags import MIGRATION_MODE, MigrationMode

    if MIGRATION_MODE == MigrationMode.PHIML:
        from src.training.physical.phiml_trainer import PhiMLPhysicalTrainer
        return PhiMLPhysicalTrainer(model, config)
    else:
        from src.training.physical.trainer import PhysicalTrainer
        return PhysicalTrainer(model, config)
```

### 3.4 Testing Phase 3

Create `tests/integration/test_phiml_training.py`:

```python
def test_synthetic_trainer():
    """Test PhiML synthetic trainer"""
    config = load_test_config('burgers')

    # Set migration mode
    from src.config.migration_flags import MIGRATION_MODE, MigrationMode
    MIGRATION_MODE = MigrationMode.PHIML

    # Create model
    from src.factories.model_factory import ModelFactory
    model = ModelFactory.create_synthetic_model(config)

    # Create dataset
    from src.factories.dataloader_factory import create_phiml_dataset
    dataset = create_phiml_dataset(config, mode='train')

    # Create trainer
    from src.factories.trainer_factory import TrainerFactory
    trainer = TrainerFactory.create_synthetic_trainer(model, config)

    # Train for 2 epochs (quick test)
    config["trainer"]["synthetic"]["epochs"] = 2
    trainer.train(dataset)

    # Should complete without errors
    assert True

def test_physical_trainer():
    """Test PhiML physical trainer"""
    # Similar to above but for physical model
    pass
```

**Deliverables**:
- ✅ PhiML synthetic trainer implemented
- ✅ PhiML physical trainer implemented
- ✅ Trainer factory updated
- ✅ Tests pass for training loop

---

## Phase 4: Hybrid Trainer Migration

**Duration**: 2-3 days
**Risk**: High
**Goal**: Simplify hybrid trainer by eliminating conversions

### 4.1 Current Hybrid Trainer Complexity

```python
# Current flow:
1. Physical model generates Fields
2. Convert Fields → Tensors
3. Train synthetic on tensors
4. Synthetic generates tensor predictions
5. Convert Tensors → Fields
6. Train physical on fields
```

### 4.2 New PhiML Hybrid Trainer

Create `src/training/hybrid/phiml_trainer.py`:

```python
from phiml import math
from phiml.math import batch
import logging

class PhiMLHybridTrainer:
    """
    Hybrid trainer using pure PhiML.
    No conversions needed - everything is PhiML tensors!
    """

    def __init__(self, physical_model, synthetic_model, config: dict):
        self.physical_model = physical_model
        self.synthetic_model = synthetic_model
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Create trainers
        from src.training.synthetic.phiml_trainer import PhiMLSyntheticTrainer
        from src.training.physical.phiml_trainer import PhiMLPhysicalTrainer

        self.synthetic_trainer = PhiMLSyntheticTrainer(synthetic_model, config)
        self.physical_trainer = PhiMLPhysicalTrainer(physical_model, config)

        # Hybrid config
        hybrid_config = config["trainer"]["hybrid"]
        self.num_cycles = hybrid_config["cycles"]
        self.warmup_epochs = hybrid_config.get("warmup", 0)

    def generate_physical_data(self, dataset, num_samples: int):
        """
        Generate synthetic data using physical model.

        Args:
            dataset: Dataset with real data
            num_samples: Number of trajectories to generate

        Returns:
            PhiML tensor with shape (samples, time, ...spatial..., channels)
        """
        self.logger.info(f"Generating {num_samples} trajectories with physical model")

        trajectories = []

        for i in range(num_samples):
            # Get random initial condition from dataset
            ic = dataset[i].time[0]  # First timestep

            # Simulate with physical model
            trajectory = self.physical_model.simulate(
                initial_state=ic,
                num_steps=self.config["trainer"]["rollout_steps"]
            )

            trajectories.append(trajectory)

        # Stack along new batch dimension
        return math.stack(trajectories, batch('generated'))

    def generate_synthetic_data(self, dataset, num_samples: int):
        """
        Generate data using synthetic model.

        Returns:
            PhiML tensor (no conversion needed!)
        """
        self.logger.info(f"Generating {num_samples} trajectories with synthetic model")

        trajectories = []

        for i in range(num_samples):
            ic = dataset[i].time[0]

            # Predict with neural network
            trajectory = self.synthetic_model.predict_trajectory(
                ic,
                num_steps=self.config["trainer"]["rollout_steps"]
            )

            trajectories.append(trajectory)

        return math.stack(trajectories, batch('generated'))

    def train(self, real_dataset):
        """
        Main hybrid training loop.

        MUCH SIMPLER than old version - no conversions!
        """
        self.logger.info("Starting hybrid training")

        # Warmup: Train synthetic on real data only
        if self.warmup_epochs > 0:
            self.logger.info(f"Warmup: {self.warmup_epochs} epochs")
            self.config["trainer"]["synthetic"]["epochs"] = self.warmup_epochs
            self.synthetic_trainer.train(real_dataset)

        # Main hybrid loop
        for cycle in range(self.num_cycles):
            self.logger.info(f"\n=== Cycle {cycle+1}/{self.num_cycles} ===")

            # 1. Generate data with physical model
            physical_data = self.generate_physical_data(
                real_dataset,
                num_samples=len(real_dataset) // 2
            )

            # 2. Train synthetic on real + physical data
            # Combine datasets (both are PhiML tensors!)
            combined_synth_data = math.concat(
                [real_dataset.data, physical_data],
                'simulations'
            )

            from src.data.phiml_dataset import PhiMLDataset
            synth_dataset = PhiMLDataset(
                combined_synth_data,
                rollout_steps=self.config["trainer"]["rollout_steps"]
            )

            self.logger.info("Training synthetic model on real + physical data")
            self.synthetic_trainer.train(synth_dataset)

            # 3. Generate data with synthetic model
            synthetic_data = self.generate_synthetic_data(
                real_dataset,
                num_samples=len(real_dataset) // 2
            )

            # 4. Train physical on real + synthetic data
            combined_phys_data = math.concat(
                [real_dataset.data, synthetic_data],
                'simulations'
            )

            phys_dataset = PhiMLDataset(
                combined_phys_data,
                rollout_steps=self.config["trainer"]["rollout_steps"]
            )

            self.logger.info("Training physical model on real + synthetic data")
            self.physical_trainer.train(phys_dataset)

        self.logger.info("Hybrid training complete!")
```

**Key Improvements**:
- 🎯 **No conversions** - Everything is PhiML tensors
- 📉 **~50% less code** - Removed all conversion logic
- 🚀 **Faster** - No CPU ↔ GPU transfers for conversions
- 🐛 **Fewer bugs** - No dimension mismatch issues

### 4.3 Update Main Entry Point

Update `run.py`:

```python
from src.config.migration_flags import MIGRATION_MODE, MigrationMode

def main():
    config = load_config()

    # Check migration mode from config
    if "migration" in config and config["migration"]["mode"] == "phiml":
        MIGRATION_MODE = MigrationMode.PHIML
    else:
        MIGRATION_MODE = MigrationMode.LEGACY

    # Rest of code uses factories, which respect migration mode
    if config["general"]["mode"] == "hybrid":
        # Create models
        physical_model = ModelFactory.create_physical_model(config)
        synthetic_model = ModelFactory.create_synthetic_model(config)

        # Create dataset
        if MIGRATION_MODE == MigrationMode.PHIML:
            from src.factories.dataloader_factory import create_phiml_dataset
            dataset = create_phiml_dataset(config, mode='train')
        else:
            dataset = DataLoaderFactory.create_dataloader(config, mode='train')

        # Create trainer
        trainer = TrainerFactory.create_hybrid_trainer(
            physical_model,
            synthetic_model,
            config
        )

        # Train
        trainer.train(dataset)
```

### 4.4 Testing Phase 4

Create `tests/integration/test_phiml_hybrid.py`:

```python
def test_hybrid_trainer_runs():
    """Test that hybrid trainer completes without errors"""
    config = load_test_config('burgers_hybrid')

    # Reduce epochs for testing
    config["trainer"]["hybrid"]["cycles"] = 2
    config["trainer"]["synthetic"]["epochs"] = 2
    config["trainer"]["physical"]["epochs"] = 2

    # Set migration mode
    from src.config.migration_flags import MIGRATION_MODE, MigrationMode
    MIGRATION_MODE = MigrationMode.PHIML

    # Create models
    from src.factories.model_factory import ModelFactory
    physical = ModelFactory.create_physical_model(config)
    synthetic = ModelFactory.create_synthetic_model(config)

    # Create dataset
    from src.factories.dataloader_factory import create_phiml_dataset
    dataset = create_phiml_dataset(config, mode='train')

    # Create trainer
    from src.training.hybrid.phiml_trainer import PhiMLHybridTrainer
    trainer = PhiMLHybridTrainer(physical, synthetic, config)

    # Train
    trainer.train(dataset)

    assert True  # Completed without errors

def test_compare_hybrid_outputs():
    """Compare old vs new hybrid trainer outputs"""
    # This would be a comprehensive test comparing:
    # 1. Loss curves
    # 2. Final model outputs
    # 3. Physical parameters learned
    pass
```

**Deliverables**:
- ✅ PhiML hybrid trainer implemented
- ✅ Integration with main script
- ✅ Tests pass
- ✅ Side-by-side comparison with old trainer

---

## Phase 5: Optimization and Cleanup

**Duration**: 2-3 days
**Risk**: Low
**Goal**: Add JIT compilation, remove old code, optimize

### 5.1 Add JIT Compilation

Update physical models to use JIT:

```python
# In src/models/physical/burgers.py
from phiml import math

class BurgersModel(PhysicalModel):

    @math.jit_compile  # Add this!
    def step(self, state: Field, dt: float) -> Field:
        """Single timestep (now JIT compiled)"""
        # ... existing code ...
        return new_state

    def simulate(self, initial_state: Field, num_steps: int) -> Field:
        """Use iterate for automatic stacking"""
        from phi.flow import iterate
        from phiml.math import batch

        # JIT compiled iteration (FAST!)
        trajectory = iterate(
            self.step,
            batch(time=num_steps),
            initial_state,
            dt=self.dt
        )

        return trajectory
```

### 5.2 Optimize Data Loading with Caching

Update `PhiMLDataManager` to use parallel caching:

```python
from phiml.dataclasses import parallel_compute, cached_property
from dataclasses import dataclass

@dataclass
class SimulationLoader:
    """Wrapper for parallel loading with caching"""
    sim_index: int
    config: dict

    @cached_property
    def data(self):
        """This gets cached to disk automatically"""
        # Load simulation logic here
        return load_simulation_data(self.sim_index, self.config)

class PhiMLDataManager:
    def load_all_simulations(self):
        """Load all simulations in parallel with disk caching"""

        # Create loader tasks
        loaders = [
            SimulationLoader(i, self.config)
            for i in range(self.config["data"]["num_simulations"])
        ]

        # Compute in parallel with caching
        parallel_compute(
            loaders,
            [SimulationLoader.data],
            batch_dim='simulations',
            cache_dir=str(self.cache_dir),
            memory_limit=4096  # MB
        )

        # Collect results
        sims = [loader.data for loader in loaders]
        return math.stack(sims, batch('simulations'))
```

### 5.3 Remove Old PyTorch Code

Once all tests pass with `migration_mode: phiml`:

```bash
# Remove old implementations
rm src/data/data_manager.py
rm src/data/tensor_dataset.py
rm src/data/field_dataset.py
rm src/models/synthetic/base.py
rm src/models/synthetic/unet.py
rm src/models/synthetic/resnet.py
rm src/models/synthetic/convnet.py
rm src/training/synthetic/trainer.py
rm src/training/physical/trainer.py
rm src/training/hybrid/trainer.py

# Rename new implementations (remove "phiml_" prefix)
mv src/data/phiml_data_manager.py src/data/data_manager.py
mv src/data/phiml_dataset.py src/data/dataset.py
mv src/models/synthetic/phiml_base.py src/models/synthetic/base.py
# ... etc
```

### 5.4 Update Dependencies

Update `requirements.txt`:

```txt
# BEFORE (many PyTorch dependencies):
torch>=2.0.0
torchvision>=0.15.0
tensorboard>=2.12.0
phiflow>=2.3.0

# AFTER (minimal):
phiflow>=2.3.0
numexpr>=2.8.0  # For safe eval
pyyaml>=6.0
numpy>=1.24.0
matplotlib>=3.7.0
pytest>=7.3.0
```

### 5.5 Update Documentation

Update [README.md](README.md):

```markdown
# HYCO-PhiFlow

Hybrid Continuous-Discrete Physics-ML Framework using **pure PhiFlow/PhiML**.

## Features

- 🚀 **Pure PhiML** - No PyTorch dependency, cleaner code
- ⚡ **JIT Compiled** - 10-100x faster simulations
- 📊 **Named Dimensions** - No dimension order confusion
- 🔄 **Hybrid Training** - Combines physics and neural models
- 💾 **Smart Caching** - Parallel computation with disk caching

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python run.py --config conf/burgers.yaml
```

## Migration from v1.0

The codebase has been migrated from PyTorch to pure PhiML. If you have old checkpoints:

```python
# Convert old PyTorch checkpoint to PhiML
from src.utils.convert_checkpoint import convert_torch_to_phiml
convert_torch_to_phiml('old_model.pth', 'new_model.phiml')
```
```

### 5.6 Performance Benchmarking

Create `benchmark.py`:

```python
import time
from phiml import math

def benchmark_simulation():
    """Compare old vs new implementation"""

    # Setup
    config = load_config('burgers')

    # Benchmark old (if still available)
    start = time.time()
    old_trainer = OldHybridTrainer(...)
    old_trainer.train(dataset)
    old_time = time.time() - start

    # Benchmark new
    start = time.time()
    new_trainer = PhiMLHybridTrainer(...)
    new_trainer.train(dataset)
    new_time = time.time() - start

    print(f"Old: {old_time:.2f}s")
    print(f"New: {new_time:.2f}s")
    print(f"Speedup: {old_time/new_time:.2f}x")

if __name__ == "__main__":
    benchmark_simulation()
```

**Deliverables**:
- ✅ JIT compilation added
- ✅ Parallel caching implemented
- ✅ Old code removed
- ✅ Dependencies updated
- ✅ Documentation updated
- ✅ Performance benchmarks showing improvement

---

## Testing Strategy

### Regression Testing

For each phase, ensure:

1. **Functional equivalence**: New code produces same outputs as old
2. **Performance improvement**: New code is faster
3. **No breaking changes**: Existing configs still work

### Test Pyramid

```
         /\
        /  \    E2E Tests (5%)
       /    \   - Full training runs
      /------\  - Compare old vs new
     /        \
    /          \ Integration Tests (20%)
   /            \ - Multi-component tests
  /              \ - Dataset + Model + Trainer
 /----------------\
/                  \ Unit Tests (75%)
--------------------  - Individual components
                      - Fast, isolated
```

### Continuous Testing

After each phase:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Compare outputs
python scripts/compare_outputs.py --old old_results/ --new new_results/

# Performance benchmark
python benchmark.py
```

---

## Rollback Plan

If issues arise:

### Quick Rollback (Feature Flag)

```yaml
# In config
migration:
  mode: "legacy"  # Switch back to PyTorch
```

### Full Rollback (Git)

```bash
# Each phase is a separate branch
git checkout phase-2-complete  # Go back to last working phase
```

### Incremental Rollback

```python
# Can mix old and new implementations
MIGRATION_MODE = MigrationMode.HYBRID

# Use new data pipeline, old models
# Or vice versa
```

---

## Success Metrics

### Performance Metrics

| Metric | Baseline (PyTorch) | Target (PhiML) | Measurement |
|--------|-------------------|----------------|-------------|
| Training time | 100% | 80-85% | Wall-clock time |
| Memory usage | 100% | 90-95% | Peak GPU memory |
| Data loading | 100% | 70-80% | Time to load dataset |
| Simulation speed | 100% | 10-20% (10x faster) | With JIT |

### Code Quality Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Lines of code | ~4,816 | ~3,500 | -27% |
| Files | 42 | 35 | -17% |
| Dependencies | 8+ | 4 | -50% |
| Test coverage | ~0% | >60% | +60% |

### Stability Metrics

| Metric | Target |
|--------|--------|
| Test pass rate | 100% |
| Regression tests | 100% match |
| Documentation coverage | >80% |

---

## Timeline Summary

| Phase | Duration | Cumulative | Risk | Team Size |
|-------|----------|-----------|------|-----------|
| Phase 0: Preparation | 1-2 days | 2 days | Low | 1 person |
| Phase 1: Data Pipeline | 3-5 days | 7 days | High | 1-2 people |
| Phase 2: Models | 2-3 days | 10 days | Medium | 1 person |
| Phase 3: Training | 3-4 days | 14 days | Medium | 1-2 people |
| Phase 4: Hybrid | 2-3 days | 17 days | High | 1-2 people |
| Phase 5: Optimization | 2-3 days | 20 days | Low | 1 person |
| **Total** | **13-20 days** | **~3 weeks** | | |

---

## Post-Migration Checklist

After completing all phases:

- [ ] All tests passing (unit, integration, E2E)
- [ ] Performance benchmarks show improvement
- [ ] Documentation updated
- [ ] Example configs work
- [ ] Old checkpoints can be converted
- [ ] CI/CD pipeline updated
- [ ] Team trained on new system
- [ ] Migration guide published

---

## Conclusion

This migration plan provides a **safe, gradual path** from the current PyTorch-PhiFlow hybrid to a pure PhiML implementation. The phased approach with feature flags allows:

- ✅ **Testing at each stage** - Catch issues early
- ✅ **Easy rollback** - Switch back if needed
- ✅ **Parallel development** - Old system keeps working
- ✅ **Knowledge transfer** - Team learns incrementally

**Expected outcome**: Simpler, faster, more maintainable codebase with ~15-20% performance improvement and ~30% code reduction.

**Next steps**: Review this plan, gather feedback, and begin Phase 0!
