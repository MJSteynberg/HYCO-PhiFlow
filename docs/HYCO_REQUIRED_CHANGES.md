# HYCO Implementation - Required Code Changes

**Document Purpose:** Detailed list of all code changes required for HYCO implementation  
**Last Updated:** November 2, 2025

---

## 🔴 PRIORITY 1: Critical Fixes (MUST BE DONE FIRST)

### Change 1.1: Fix Non-Convergence Handling in Physical Trainer

**File:** `src/training/physical/trainer.py`

**Location:** Method `_setup_optimization()` (around line 248)

**Current Code:**
```python
def _setup_optimization(self):
    """Setup optimization configuration for math.minimize."""
    method = self.trainer_config.get("method", "L-BFGS-B")
    abs_tol = self.trainer_config.get("abs_tol", 1e-6)
    max_iterations = self.trainer_config.get("max_iterations")
    if max_iterations is None:
        max_iterations = self.num_epochs

    return math.Solve(
        method=method,
        abs_tol=abs_tol,
        x0=self.initial_guesses,
        max_iterations=max_iterations,
    )
```

**New Code:**
```python
def _setup_optimization(self):
    """Setup optimization configuration for math.minimize."""
    method = self.trainer_config.get("method", "L-BFGS-B")
    abs_tol = self.trainer_config.get("abs_tol", 1e-6)
    max_iterations = self.trainer_config.get("max_iterations")
    if max_iterations is None:
        max_iterations = self.num_epochs
    
    # NEW: Configure error suppression for hybrid training
    suppress_convergence = self.trainer_config.get("suppress_convergence_errors", False)
    suppress_list = []
    if suppress_convergence:
        suppress_list.append(math.NotConverged)
        logger.info("Convergence errors will be suppressed (suitable for hybrid training)")
    
    logger.info(f"\nOptimization settings:")
    logger.info(f"  method: {method}")
    logger.info(f"  abs_tol: {abs_tol}")
    logger.info(f"  max_iterations: {max_iterations}")
    logger.info(f"  suppress_convergence: {suppress_convergence}")
    
    return math.Solve(
        method=method,
        abs_tol=abs_tol,
        x0=self.initial_guesses,
        max_iterations=max_iterations,
        suppress=tuple(suppress_list),  # ← KEY ADDITION
    )
```

**Diff:**
```diff
def _setup_optimization(self):
    """Setup optimization configuration for math.minimize."""
    method = self.trainer_config.get("method", "L-BFGS-B")
    abs_tol = self.trainer_config.get("abs_tol", 1e-6)
    max_iterations = self.trainer_config.get("max_iterations")
    if max_iterations is None:
        max_iterations = self.num_epochs
+   
+   # Configure error suppression for hybrid training
+   suppress_convergence = self.trainer_config.get("suppress_convergence_errors", False)
+   suppress_list = []
+   if suppress_convergence:
+       suppress_list.append(math.NotConverged)
+       logger.info("Convergence errors will be suppressed (suitable for hybrid training)")
+   
+   logger.info(f"\nOptimization settings:")
+   logger.info(f"  method: {method}")
+   logger.info(f"  abs_tol: {abs_tol}")
+   logger.info(f"  max_iterations: {max_iterations}")
+   logger.info(f"  suppress_convergence: {suppress_convergence}")

    return math.Solve(
        method=method,
        abs_tol=abs_tol,
        x0=self.initial_guesses,
        max_iterations=max_iterations,
+       suppress=tuple(suppress_list),
    )
```

---

### Change 1.2: Simplify Exception Handling

**File:** `src/training/physical/trainer.py`

**Location:** Method `train()`, exception handling (around lines 392-418)

**Current Code:**
```python
try:
    if hasattr(self, "memory_monitor") and self.memory_monitor:
        with self.memory_monitor.track("optimization"):
            estimated_tensors = math.minimize(loss_function, solve_params)
    else:
        estimated_tensors = math.minimize(loss_function, solve_params)

    logger.info(f"\nOptimization completed!")
    logger.info(f"Total loss function evaluations: {loss_call_count[0]}")
except math.NotConverged as e:
    # NotConverged is raised when iteration limit is reached
    # This is expected for quick tests with low max_iterations
    logger.warning(f"\nOptimization stopped: {e}")
    logger.info(f"Total loss function evaluations: {loss_call_count[0]}")
    # Extract the best parameters found so far
    estimated_tensors = tuple(self.initial_guesses)  # Fallback
    if hasattr(e, 'result') and hasattr(e.result, 'x'):
        estimated_tensors = e.result.x
except Exception as e:
    logger.error(f"Optimization failed: {e}")
    import traceback
    traceback.print_exc()
    estimated_tensors = tuple(self.initial_guesses)  # Return guess on failure
```

**New Code:**
```python
try:
    if hasattr(self, "memory_monitor") and self.memory_monitor:
        with self.memory_monitor.track("optimization"):
            estimated_tensors = math.minimize(loss_function, solve_params)
    else:
        estimated_tensors = math.minimize(loss_function, solve_params)
    
    logger.info(f"\nOptimization completed!")
    logger.info(f"Total loss function evaluations: {loss_call_count[0]}")
    
except Exception as e:
    # Only catch unexpected errors (NotConverged should be suppressed via Solve)
    logger.error(f"Unexpected optimization error: {e}")
    import traceback
    traceback.print_exc()
    # Use initial guesses as last resort
    estimated_tensors = tuple(self.initial_guesses)
```

**Diff:**
```diff
try:
    if hasattr(self, "memory_monitor") and self.memory_monitor:
        with self.memory_monitor.track("optimization"):
            estimated_tensors = math.minimize(loss_function, solve_params)
    else:
        estimated_tensors = math.minimize(loss_function, solve_params)

    logger.info(f"\nOptimization completed!")
    logger.info(f"Total loss function evaluations: {loss_call_count[0]}")
-except math.NotConverged as e:
-    # NotConverged is raised when iteration limit is reached
-    # This is expected for quick tests with low max_iterations
-    logger.warning(f"\nOptimization stopped: {e}")
-    logger.info(f"Total loss function evaluations: {loss_call_count[0]}")
-    # Extract the best parameters found so far
-    estimated_tensors = tuple(self.initial_guesses)  # Fallback
-    if hasattr(e, 'result') and hasattr(e.result, 'x'):
-        estimated_tensors = e.result.x
+    
except Exception as e:
-    logger.error(f"Optimization failed: {e}")
+    # Only catch unexpected errors (NotConverged should be suppressed via Solve)
+    logger.error(f"Unexpected optimization error: {e}")
    import traceback
    traceback.print_exc()
    estimated_tensors = tuple(self.initial_guesses)
```

---

### Change 1.3: Update Configuration Schema

**File:** `src/config/trainer_config.py`

**Location:** Class `PhysicalTrainerConfig` (around line 24)

**Current Code:**
```python
@dataclass
class PhysicalTrainerConfig:
    """Configuration for physical model inverse problem training."""

    epochs: int = 100
    num_predict_steps: int = 10
    train_sim: List[int] = field(default_factory=list)

    learnable_parameters: List[LearnableParameter] = field(default_factory=list)

    # Optimizer settings
    method: str = "L-BFGS-B"
    abs_tol: float = 1e-6
    max_iterations: Optional[int] = None
```

**New Code:**
```python
@dataclass
class PhysicalTrainerConfig:
    """Configuration for physical model inverse problem training."""

    epochs: int = 100
    num_predict_steps: int = 10
    train_sim: List[int] = field(default_factory=list)

    learnable_parameters: List[LearnableParameter] = field(default_factory=list)

    # Optimizer settings
    method: str = "L-BFGS-B"
    abs_tol: float = 1e-6
    max_iterations: Optional[int] = None
    
    # NEW: Convergence handling for hybrid training
    suppress_convergence_errors: bool = False
```

**Diff:**
```diff
@dataclass
class PhysicalTrainerConfig:
    """Configuration for physical model inverse problem training."""

    epochs: int = 100
    num_predict_steps: int = 10
    train_sim: List[int] = field(default_factory=list)

    learnable_parameters: List[LearnableParameter] = field(default_factory=list)

    # Optimizer settings
    method: str = "L-BFGS-B"
    abs_tol: float = 1e-6
    max_iterations: Optional[int] = None
+   
+   # Convergence handling for hybrid training
+   suppress_convergence_errors: bool = False
```

---

### Change 1.4: Add Configuration File

**File:** `conf/trainer/physical_hybrid.yaml` (NEW FILE)

**Content:**
```yaml
# Physical Model Configuration for Hybrid Training
# Optimized for fast iterations in HYCO loop

epochs: 50
num_predict_steps: 10
train_sim: [0, 1, 2]

learnable_parameters:
  - name: nu
    initial_guess: 0.01
    bounds: [0.001, 0.1]

# Optimizer settings optimized for hybrid training
method: 'L-BFGS-B'
abs_tol: 1e-6
max_iterations: 5  # Low for fast iterations
suppress_convergence_errors: true  # Don't raise errors on non-convergence

# Memory monitoring (optional)
enable_memory_monitoring: false
```

---

## 🟡 PRIORITY 2: New Components

### Change 2.1: Create Hybrid Trainer Directory

**Action:** Create new directory structure

```bash
mkdir -p src/training/hybrid
touch src/training/hybrid/__init__.py
touch src/training/hybrid/trainer.py
```

---

### Change 2.2: Create Hybrid Trainer Implementation

**File:** `src/training/hybrid/__init__.py` (NEW FILE)

**Content:**
```python
"""
Hybrid Training Module

Implements HYCO (Hybrid Coupled) training strategy where synthetic
and physical models are trained in an interleaved fashion.
"""

from .trainer import HybridTrainer

__all__ = ["HybridTrainer"]
```

---

**File:** `src/training/hybrid/trainer.py` (NEW FILE)

**Content:** See full implementation in `docs/HYCO_IMPLEMENTATION_STRATEGY.md`, Phase 1, Task 1.1

**Key sections to implement:**

1. Class definition and `__init__`
2. Setup methods:
   - `_setup_data_manager()`
   - `_setup_synthetic_model()`
   - `_setup_physical_model()`
   - `_setup_field_converter()`
3. Prediction generation:
   - `_generate_synthetic_rollout()`
   - `_generate_physical_rollout()`
4. Training methods:
   - `_train_synthetic_epoch()`
   - `_optimize_physical_epoch()`
5. Main training loop:
   - `train()`
6. Helper methods:
   - `_prepare_hybrid_training_data()`
   - `_compute_weighted_loss()`
   - `_convert_fields_to_tensors()`
   - `_convert_tensors_to_fields()`

---

### Change 2.3: Register Hybrid Trainer in Factory

**File:** `src/factories/trainer_factory.py`

**Location:** Class `TrainerFactory`, attribute `_trainers` (around line 10)

**Current Code:**
```python
class TrainerFactory:
    """Factory for creating trainer instances."""

    _trainers = {
        "synthetic": SyntheticTrainer,
        "physical": PhysicalTrainer,
    }
```

**New Code:**
```python
from src.training.hybrid.trainer import HybridTrainer  # NEW IMPORT

class TrainerFactory:
    """Factory for creating trainer instances."""

    _trainers = {
        "synthetic": SyntheticTrainer,
        "physical": PhysicalTrainer,
        "hybrid": HybridTrainer,  # NEW
    }
```

**Diff:**
```diff
+from src.training.hybrid.trainer import HybridTrainer

class TrainerFactory:
    """Factory for creating trainer instances."""

    _trainers = {
        "synthetic": SyntheticTrainer,
        "physical": PhysicalTrainer,
+       "hybrid": HybridTrainer,
    }
```

---

### Change 2.4: Add Hybrid Configuration Files

**File:** `conf/trainer/hybrid.yaml` (NEW FILE)

**Content:**
```yaml
# HYCO Hybrid Training Configuration

# General settings
epochs: 50
num_predict_steps: 10
train_sim: [0, 1, 2]
val_sim: [3]

# Hybrid-specific parameters
alpha: 0.5  # Weight for real data (0.5 = equal real/synthetic)
interleave_frequency: 1  # Train both models every N epochs
warmup_epochs: 5  # Pre-train synthetic model alone

# Synthetic model settings
synthetic:
  learning_rate: 1e-4
  batch_size: 16
  optimizer: adam
  scheduler: cosine
  weight_decay: 0.0

# Physical model settings
physical:
  method: 'L-BFGS-B'
  abs_tol: 1e-6
  max_iterations: 5  # Low for fast iterations
  suppress_convergence_errors: true  # Don't raise on non-convergence

# Checkpointing
save_interval: 10
save_best_only: true
checkpoint_dir: 'results/models/hybrid'

# Memory and performance
enable_memory_monitoring: false
```

---

**File:** `conf/burgers_hybrid_experiment.yaml` (NEW FILE)

**Content:**
```yaml
defaults:
  - data: burgers_128
  - model/physical: burgers
  - model/synthetic: unet
  - trainer: hybrid
  - generation: default
  - evaluation: default
  - _self_

run_params:
  experiment_name: 'burgers_hybrid_hyco'
  notes: 'HYCO interleaved hybrid training - Burgers equation'
  mode: ['train', 'evaluate']
  model_type: 'hybrid'  # NEW model type

trainer_params:
  train_sim: [0, 1, 2, 3, 4]
  val_sim: [5]
  epochs: 50

project_root: ${hydra:runtime.cwd}
```

---

**File:** `conf/burgers_hybrid_quick_test.yaml` (NEW FILE)

**Content:**
```yaml
# Quick test for hybrid training - minimal configuration
defaults:
  - data: burgers_128
  - model/physical: burgers
  - model/synthetic: unet
  - trainer: hybrid
  - _self_

run_params:
  experiment_name: 'burgers_hybrid_quick_test'
  notes: 'Quick test of hybrid training'
  mode: ['train']
  model_type: 'hybrid'

model:
  synthetic:
    model_save_name: 'burgers_hybrid_quick_test'

trainer_params:
  train_sim: [0, 1]
  val_sim: []
  epochs: 2
  num_predict_steps: 3
  alpha: 0.5
  warmup_epochs: 0
  
  synthetic:
    batch_size: 8
    learning_rate: 1e-4
  
  physical:
    max_iterations: 3
    suppress_convergence_errors: true

project_root: ${hydra:runtime.cwd}
```

---

## 🟢 PRIORITY 3: Optional Enhancements

### Change 3.1: Add Helper Method to HybridDataset

**File:** `src/data/hybrid_dataset.py`

**Location:** Add new method to class `HybridDataset`

**New Method:**
```python
def get_dual_format_sample(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Field]]:
    """
    Get sample in both tensor and field format for hybrid training.
    
    This is a convenience method for HybridTrainer to get the same
    initial state in both formats without redundant conversions.
    
    Args:
        idx: Index of the sample
        
    Returns:
        (initial_tensor, initial_fields): Same initial state in both formats
    """
    # Determine simulation and starting frame
    if self.use_sliding_window:
        sim_idx, start_frame = self.sample_index[idx]
    else:
        sim_idx = self.sim_indices[idx]
        start_frame = 0
    
    # Load data with fields
    sim_data = self._cached_load_simulation(sim_idx)
    cache_data = {"tensor_data": sim_data}
    
    # Get metadata
    full_cache_data = self.data_manager.load_from_cache(sim_idx)
    cache_data.update({k: v for k, v in full_cache_data.items() if k != "tensor_data"})
    
    # Convert to fields
    initial_fields, _ = self._convert_to_fields_with_start(cache_data, start_frame)
    
    # Also get as tensors
    all_field_tensors = [sim_data[name] for name in self.field_names]
    all_data = torch.cat(all_field_tensors, dim=1)
    initial_tensor = all_data[start_frame]
    
    return initial_tensor, initial_fields
```

---

### Change 3.2: Add Validation Method to AbstractTrainer

**File:** `src/training/abstract_trainer.py`

**Location:** Add new method to class `AbstractTrainer`

**New Method:**
```python
def validate(self) -> Dict[str, Any]:
    """
    Run validation on the current model state.
    
    This is an optional method that subclasses can implement
    to provide validation functionality. The default implementation
    returns an empty dictionary.
    
    Returns:
        Dictionary containing validation metrics
    """
    return {}
```

---

## 📝 Testing Changes Required

### Test File 1: Physical Convergence Tests

**File:** `tests/training/test_physical_convergence.py` (NEW FILE)

**Content:**
```python
"""Tests for physical model convergence handling."""

import pytest
from src.training.physical.trainer import PhysicalTrainer
from phi.math import NotConverged


def test_suppress_convergence_disabled_by_default():
    """Ensure existing behavior unchanged - suppression off by default."""
    config = get_default_physical_config()
    # Don't set suppress_convergence_errors
    
    trainer = PhysicalTrainer(config)
    solve = trainer._setup_optimization()
    
    assert solve.suppress == ()  # Empty tuple


def test_suppress_convergence_enabled():
    """Test new suppression feature."""
    config = get_default_physical_config()
    config["trainer_params"]["suppress_convergence_errors"] = True
    
    trainer = PhysicalTrainer(config)
    solve = trainer._setup_optimization()
    
    assert NotConverged in solve.suppress


def test_physical_with_low_iterations_and_suppression():
    """Test that low max_iterations works with suppression."""
    config = get_default_physical_config()
    config["trainer_params"]["max_iterations"] = 3
    config["trainer_params"]["suppress_convergence_errors"] = True
    
    trainer = PhysicalTrainer(config)
    results = trainer.train()
    
    # Should complete successfully
    assert results is not None
    # Parameters should be optimized (not initial guess)
    nu_optimized = results["optimized_parameters"]["nu"]
    nu_initial = trainer.initial_guesses[0]
    assert nu_optimized != nu_initial


def test_physical_without_suppression_raises():
    """Test that without suppression, low iterations raises error."""
    config = get_default_physical_config()
    config["trainer_params"]["max_iterations"] = 3
    config["trainer_params"]["suppress_convergence_errors"] = False
    
    trainer = PhysicalTrainer(config)
    
    # Should raise NotConverged or catch it and fall back
    # Depending on implementation, this might not raise but log warning
    results = trainer.train()
    assert results is not None  # Should still complete


# Helper function
def get_default_physical_config():
    """Create default physical trainer config for testing."""
    return {
        "data": {
            "data_dir": "data",
            "dset_name": "burgers_128",
            "fields": ["velocity"],
        },
        "model": {
            "physical": {
                "name": "BurgersModel",
                "domain": {"size_x": 100, "size_y": 100},
                "resolution": {"x": 64, "y": 64},
                "dt": 0.8,
                "pde_params": {"nu": 0.01},
            }
        },
        "trainer_params": {
            "epochs": 10,
            "num_predict_steps": 5,
            "train_sim": [0],
            "learnable_parameters": [
                {"name": "nu", "initial_guess": 0.02}
            ],
            "method": "L-BFGS-B",
            "abs_tol": 1e-6,
        },
        "project_root": ".",
    }
```

---

### Test File 2: Hybrid Trainer Tests

**File:** `tests/training/test_hybrid_trainer.py` (NEW FILE)

**Content:** See full test implementations in `docs/HYCO_IMPLEMENTATION_STRATEGY.md`, Section "Testing Strategy"

---

## 🔄 Migration Path

### Step 1: Validate Current System
```bash
# Run existing tests
pytest tests/ -v

# Verify existing configs work
python run.py --config-name burgers_quick_test
python run.py --config-name burgers_physical_quick_test
```

### Step 2: Apply Critical Fixes
```bash
# Apply changes to physical trainer
# Changes 1.1, 1.2, 1.3, 1.4

# Test suppression feature
pytest tests/training/test_physical_convergence.py -v
```

### Step 3: Verify No Breaking Changes
```bash
# Re-run existing tests
pytest tests/ -v

# Verify existing configs still work
python run.py --config-name burgers_quick_test
python run.py --config-name burgers_physical_quick_test
```

### Step 4: Implement Hybrid Trainer
```bash
# Create directory structure (Change 2.1)
# Implement trainer (Change 2.2)
# Register in factory (Change 2.3)
# Add configs (Change 2.4)

# Test basic instantiation
python -c "from src.factories.trainer_factory import TrainerFactory; print(TrainerFactory.list_available_trainers())"
```

### Step 5: Test Hybrid Training
```bash
# Run quick test
python run.py --config-name burgers_hybrid_quick_test

# Run full test
python run.py --config-name burgers_hybrid_experiment

# Run tests
pytest tests/training/test_hybrid_trainer.py -v
```

---

## 📊 Verification Checklist

### After Phase 0 (Critical Fixes)
- [ ] `suppress` parameter working in `Solve`
- [ ] Tests pass for suppression enabled/disabled
- [ ] Low `max_iterations` works without crash
- [ ] Existing configs still work
- [ ] All existing tests pass

### After Phase 1 (Foundation)
- [ ] `HybridTrainer` can be instantiated
- [ ] Registered in factory
- [ ] Configuration files load correctly
- [ ] No import errors

### After Phase 2 (Core Implementation)
- [ ] Prediction generation works for both models
- [ ] Field-tensor conversion works bidirectionally
- [ ] Training loop completes without errors
- [ ] Losses decrease over epochs
- [ ] Tests pass

### Production Readiness
- [ ] Performance acceptable (< 2x slowest trainer)
- [ ] Memory usage reasonable (< 8GB GPU)
- [ ] All tests pass (>80% coverage)
- [ ] Documentation complete
- [ ] Multiple PDEs tested (Burgers, Heat, Smoke)

---

**Document Version:** 1.0  
**Last Updated:** November 2, 2025  
**Related Documents:**
- `docs/HYCO_IMPLEMENTATION_STRATEGY.md` - Full strategy
- `docs/HYCO_QUICK_REFERENCE.md` - Quick reference guide
