# HYCO Hybrid Training Implementation Strategy

**Document Version:** 1.0  
**Date:** November 2, 2025  
**Status:** Planning Phase

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Analysis](#architecture-analysis)
3. [Critical Issues & Solutions](#critical-issues--solutions)
4. [Implementation Strategy](#implementation-strategy)
5. [Breaking Changes Mitigation](#breaking-changes-mitigation)
6. [Testing Strategy](#testing-strategy)
7. [Risk Assessment](#risk-assessment)
8. [Timeline & Milestones](#timeline--milestones)

---

## Executive Summary

### Objective

Implement HYCO (Hybrid Coupled) training strategy where synthetic and physical models are trained in an interleaved fashion:

- **Epoch N:** 
  1. Synthetic model trains on: real data + physical model predictions (as ground truth)
  2. Physical model optimizes on: real data + synthetic model predictions (as ground truth)

### Current Readiness: 70%

**Strengths:**
- ✅ Clean trainer hierarchy (`AbstractTrainer` → `TensorTrainer`/`FieldTrainer`)
- ✅ Field-tensor conversion infrastructure exists
- ✅ Model registry for easy instantiation
- ✅ Efficient data caching and loading

**Critical Gaps:**
- 🔴 No `HybridTrainer` class
- 🔴 No prediction generation from both models
- 🔴 **CRITICAL: Improper handling of non-convergence in physical optimization**
- 🟡 No augmented training data structures

---

## Architecture Analysis

### Current Trainer Hierarchy

```
AbstractTrainer (base interface)
├── TensorTrainer (PyTorch-based, epoch training)
│   └── SyntheticTrainer
└── FieldTrainer (PhiFlow-based, optimization)
    └── PhysicalTrainer
```

### Proposed Extension

```
AbstractTrainer
├── TensorTrainer
│   └── SyntheticTrainer
├── FieldTrainer
│   └── PhysicalTrainer
└── HybridTrainer ← NEW
    ├── Uses both SyntheticTrainer and PhysicalTrainer internally
    └── Orchestrates interleaved training
```

### Key Components

| Component | Purpose | Status |
|-----------|---------|--------|
| `HybridTrainer` | Orchestrate interleaved training | ❌ To be created |
| Field-Tensor Conversion | Bridge PhiFlow ↔ PyTorch | ✅ Exists (`src/utils/field_conversion/`) |
| Model Registry | Instantiate both models | ✅ Working |
| Data Pipeline | Load synchronized data | ✅ Working |
| Prediction Generation | Generate rollouts from both models | ❌ To be implemented |
| Convergence Handling | Proper non-convergence management | 🔴 **Needs fix** |

---

## Critical Issues & Solutions

### 🔴 CRITICAL ISSUE 1: Improper Non-Convergence Handling

#### Current Implementation Problem

**Location:** `src/training/physical/trainer.py`, lines 402-410

```python
except math.NotConverged as e:
    logger.warning(f"\nOptimization stopped: {e}")
    # Extract the best parameters found so far
    estimated_tensors = tuple(self.initial_guesses)  # ❌ PROBLEM: Falls back to initial guess!
    if hasattr(e, 'result') and hasattr(e.result, 'x'):
        estimated_tensors = e.result.x
```

**Problems:**
1. **Fallback to initial guesses** if `e.result` doesn't have `.x` attribute
2. **Doesn't properly extract intermediate results** from optimization
3. **No configuration to suppress convergence errors** proactively

#### Why This Matters for HYCO

In HYCO training:
- We want the physical model to take **a few steps in the right direction** each epoch
- We DON'T want full convergence (too slow)
- We NEED the best parameters found so far, not initial guesses
- Non-convergence is **expected and acceptable** behavior

#### ✅ Correct Solution: Use `suppress` Parameter in `Solve`

**PhiML Documentation:**

```python
Solve(
    method: str = 'auto',
    rel_tol: float = None,
    abs_tol: float = None,
    x0: Any = None,
    max_iterations: int = 1000,
    suppress: Union[tuple, list] = (),  # ← KEY PARAMETER
    ...
)
```

**Proper Implementation:**

```python
# Configuration (in YAML)
trainer_params:
  physical:
    method: 'L-BFGS-B'
    abs_tol: 1e-6
    max_iterations: 5  # Intentionally low for HYCO
    suppress_convergence_errors: true  # NEW: Enable suppression

# Implementation (in trainer code)
def _setup_optimization(self) -> Solve:
    method = self.trainer_config.get("method", "L-BFGS-B")
    abs_tol = self.trainer_config.get("abs_tol", 1e-6)
    max_iterations = self.trainer_config.get("max_iterations", 100)
    
    # NEW: Configure error suppression
    suppress_convergence = self.trainer_config.get("suppress_convergence_errors", False)
    suppress_list = []
    if suppress_convergence:
        suppress_list.append(math.NotConverged)
    
    return math.Solve(
        method=method,
        abs_tol=abs_tol,
        x0=self.initial_guesses,
        max_iterations=max_iterations,
        suppress=tuple(suppress_list)  # ← Prevents NotConverged from being raised!
    )
```

**Benefits:**
1. ✅ No exception raised when iteration limit reached
2. ✅ `math.minimize` returns best parameters found so far
3. ✅ Clean, intentional behavior (not error handling)
4. ✅ Can still detect true divergence (if needed)

**Example Usage:**

```python
# For HYCO hybrid training
solve = Solve(
    method="L-BFGS-B",
    abs_tol=1e-6,
    max_iterations=5,
    suppress=(math.NotConverged,)  # Suppress convergence warnings
)

# math.minimize will NOT raise NotConverged
# Returns best parameters found within 5 iterations
estimated_params = math.minimize(loss_fn, solve)
```

---

### 🔴 CRITICAL ISSUE 2: No Prediction Generation

#### Problem

Neither trainer can generate predictions for the other model's training.

#### Solution: Implement Prediction Generation Methods

```python
class HybridTrainer(AbstractTrainer):
    
    def _generate_synthetic_rollout(
        self, 
        initial_state: Dict[str, Field], 
        num_steps: int
    ) -> torch.Tensor:
        """
        Generate autoregressive rollout using synthetic model.
        
        Args:
            initial_state: Dictionary of PhiFlow Fields at t=0
            num_steps: Number of timesteps to predict
            
        Returns:
            Predictions as tensor [B, T, C, H, W]
        """
        # Convert Fields to Tensor
        initial_tensor = self.field_converter.fields_to_tensor_batch(initial_state)
        initial_tensor = initial_tensor.to(self.device)
        
        predictions = []
        current_state = initial_tensor
        
        with torch.no_grad():
            for t in range(num_steps):
                # Predict next state
                pred = self.synthetic_model(current_state)
                predictions.append(pred)
                
                # Use prediction as next input
                current_state = pred
        
        # Stack along time dimension
        rollout = torch.stack(predictions, dim=1)  # [B, T, C, H, W]
        return rollout
    
    def _generate_physical_rollout(
        self,
        initial_state: Dict[str, Field],
        num_steps: int
    ) -> Dict[str, Field]:
        """
        Generate rollout using physical model.
        
        Args:
            initial_state: Dictionary of PhiFlow Fields at t=0
            num_steps: Number of timesteps to simulate
            
        Returns:
            Predictions as dictionary of Fields with time dimension
        """
        from phi.field import stack
        from phi.math import batch
        
        # Initialize prediction storage
        predictions = {name: [initial_state[name]] for name in initial_state.keys()}
        current_state = initial_state
        
        # Simulate forward
        for t in range(num_steps):
            current_state = self.physical_model.step(current_state)
            
            # Store predictions
            for name, field in current_state.items():
                predictions[name].append(field)
        
        # Stack along time dimension
        stacked_predictions = {}
        for name, field_list in predictions.items():
            stacked_predictions[name] = stack(field_list, batch("time"))
        
        return stacked_predictions
```

---

### 🟡 ISSUE 3: Data Augmentation for Training

#### Problem

Current trainers only use real ground truth data. HYCO needs to train on real + model predictions.

#### Solution: Augmented Training Approach

```python
def _prepare_hybrid_training_data(
    self,
    real_data: torch.Tensor,
    model_predictions: torch.Tensor,
    alpha: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare augmented training data combining real and synthetic targets.
    
    Args:
        real_data: Ground truth data [B, T, C, H, W]
        model_predictions: Predictions from other model [B, T, C, H, W]
        alpha: Weight for real data (0.5 = equal weight)
        
    Returns:
        (combined_data, sample_weights)
    """
    # Concatenate along batch dimension
    combined_data = torch.cat([real_data, model_predictions], dim=0)
    
    # Create sample weights
    batch_size = real_data.shape[0]
    real_weights = torch.full((batch_size,), alpha, device=real_data.device)
    pred_weights = torch.full((batch_size,), 1.0 - alpha, device=real_data.device)
    sample_weights = torch.cat([real_weights, pred_weights], dim=0)
    
    return combined_data, sample_weights

def _compute_weighted_loss(
    self,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor
) -> torch.Tensor:
    """
    Compute weighted MSE loss for hybrid training.
    
    Args:
        predictions: Model predictions
        targets: Combined real + synthetic targets
        weights: Per-sample weights
        
    Returns:
        Weighted loss
    """
    # Compute per-sample loss
    loss_per_sample = F.mse_loss(predictions, targets, reduction='none')
    loss_per_sample = loss_per_sample.mean(dim=[1, 2, 3, 4])  # Average over T, C, H, W
    
    # Apply weights
    weighted_loss = (loss_per_sample * weights).mean()
    
    return weighted_loss
```

---

## Implementation Strategy

### Phase 0: Fix Critical Issues (2-3 days)

**Priority: CRITICAL - Must be done first**

#### Task 0.1: Fix Non-Convergence Handling

**Files to modify:**
- `src/training/physical/trainer.py`
- `src/config/trainer_config.py`
- `conf/trainer/physical.yaml`

**Changes:**

1. **Add configuration option:**

```python
# src/config/trainer_config.py

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
    
    # NEW: Convergence handling
    suppress_convergence_errors: bool = False  # For hybrid training
```

2. **Update Solve construction:**

```python
# src/training/physical/trainer.py

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
        suppress=tuple(suppress_list)  # ← KEY FIX
    )
```

3. **Simplify exception handling:**

```python
# src/training/physical/trainer.py

try:
    if hasattr(self, "memory_monitor") and self.memory_monitor:
        with self.memory_monitor.track("optimization"):
            estimated_tensors = math.minimize(loss_function, solve_params)
    else:
        estimated_tensors = math.minimize(loss_function, solve_params)
    
    logger.info(f"\nOptimization completed!")
    logger.info(f"Total loss function evaluations: {loss_call_count[0]}")
    
except Exception as e:
    # Only catch unexpected errors (not NotConverged if suppressed)
    logger.error(f"Unexpected optimization error: {e}")
    import traceback
    traceback.print_exc()
    # Use initial guesses as last resort
    estimated_tensors = tuple(self.initial_guesses)
```

4. **Add configuration files:**

```yaml
# conf/trainer/physical_hybrid.yaml
# Configuration for physical model in hybrid training

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
max_iterations: 5  # Low for fast iterations in hybrid loop
suppress_convergence_errors: true  # Don't raise errors on non-convergence
```

**Testing:**

```python
# tests/training/test_physical_convergence.py

def test_physical_with_suppressed_convergence():
    """Test that physical trainer handles suppressed convergence correctly."""
    config = {
        # ... config setup
        "trainer_params": {
            "method": "L-BFGS-B",
            "max_iterations": 3,  # Intentionally low
            "suppress_convergence_errors": True
        }
    }
    
    trainer = PhysicalTrainer(config)
    results = trainer.train()
    
    # Should complete without raising NotConverged
    assert results is not None
    assert "optimized_parameters" in results
    # Parameters should not be initial guesses
    assert results["optimized_parameters"]["nu"] != 0.01
```

---

### Phase 1: Foundation (3-4 days)

#### Task 1.1: Create HybridTrainer Skeleton

**New files:**
- `src/training/hybrid/__init__.py`
- `src/training/hybrid/trainer.py`

**Implementation:**

```python
# src/training/hybrid/trainer.py

"""
Hybrid Trainer for HYCO Strategy

Implements interleaved training where:
1. Synthetic model trains on: real data + physical predictions
2. Physical model optimizes on: real data + synthetic predictions

Both models evolve from the same initial state and produce
predictions of the same length.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from phi.torch.flow import *
from phi.field import Field, stack
from phi.math import batch

from src.training.abstract_trainer import AbstractTrainer
from src.data import DataManager, HybridDataset
from src.models import ModelRegistry
from src.utils.field_conversion import (
    make_batch_converter,
    FieldMetadata,
    create_field_metadata_from_model,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridTrainer(AbstractTrainer):
    """
    Hybrid trainer implementing HYCO interleaved training strategy.
    
    Training loop (per epoch):
    1. Load real data with initial states
    2. Generate physical model predictions from same initial states
    3. Train synthetic model on: real data + physical predictions
    4. Generate synthetic model predictions from same initial states
    5. Optimize physical model on: real data + synthetic predictions
    
    Attributes:
        config: Full configuration dictionary
        synthetic_model: PyTorch neural network
        physical_model: PhiFlow PDE model
        field_converter: Converter between Fields and Tensors
        device: PyTorch device (CPU/CUDA)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize hybrid trainer."""
        super().__init__(config)
        
        logger.info(f"\n{'='*60}")
        logger.info("INITIALIZING HYBRID TRAINER (HYCO)")
        logger.info(f"{'='*60}\n")
        
        # Configuration
        self.data_config = config["data"]
        self.synthetic_config = config["model"]["synthetic"]
        self.physical_config = config["model"]["physical"]
        self.trainer_config = config["trainer_params"]
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")
        
        # Hybrid-specific parameters
        self.epochs = self.trainer_config["epochs"]
        self.num_predict_steps = self.trainer_config["num_predict_steps"]
        self.train_sim = self.trainer_config["train_sim"]
        self.alpha = self.trainer_config.get("alpha", 0.5)  # Weight for real data
        self.interleave_frequency = self.trainer_config.get("interleave_frequency", 1)
        self.warmup_epochs = self.trainer_config.get("warmup_epochs", 0)
        
        # Initialize components (will be set in _setup methods)
        self.synthetic_model: Optional[nn.Module] = None
        self.physical_model: Optional[Any] = None
        self.data_manager: Optional[DataManager] = None
        self.field_converter = None
        
        # Setup
        self._setup_data_manager()
        self._setup_synthetic_model()
        self._setup_physical_model()
        self._setup_field_converter()
        
        logger.info(f"Hybrid trainer initialized successfully")
        logger.info(f"  Epochs: {self.epochs}")
        logger.info(f"  Prediction steps: {self.num_predict_steps}")
        logger.info(f"  Alpha (real data weight): {self.alpha}")
        logger.info(f"  Warmup epochs: {self.warmup_epochs}")
        logger.info(f"{'='*60}\n")
    
    def _setup_data_manager(self):
        """Create DataManager for loading data."""
        # Implementation
        pass
    
    def _setup_synthetic_model(self):
        """Create and load synthetic model."""
        # Implementation
        pass
    
    def _setup_physical_model(self):
        """Create physical model."""
        # Implementation
        pass
    
    def _setup_field_converter(self):
        """Setup Field-Tensor conversion."""
        # Implementation
        pass
    
    def train(self) -> Dict[str, Any]:
        """
        Execute HYCO hybrid training.
        
        Returns:
            Dictionary with training results
        """
        logger.info("Starting HYCO hybrid training...")
        
        results = {
            "synthetic_losses": [],
            "physical_losses": [],
            "epochs": [],
        }
        
        # Warmup: Train synthetic model alone
        if self.warmup_epochs > 0:
            logger.info(f"\nWarmup phase: Training synthetic model for {self.warmup_epochs} epochs...")
            self._warmup_synthetic()
        
        # Main hybrid training loop
        for epoch in range(self.epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            logger.info(f"{'='*60}\n")
            
            epoch_start = time.time()
            
            # Load real data
            real_data = self._load_real_data()
            
            # Generate predictions from both models
            synthetic_preds = self._generate_synthetic_predictions(real_data["initial_states"])
            physical_preds = self._generate_physical_predictions(real_data["initial_states"])
            
            # Train synthetic model
            synthetic_loss = self._train_synthetic_epoch(
                real_data["targets"], 
                physical_preds
            )
            
            # Optimize physical model (if at interleave frequency)
            physical_loss = None
            if (epoch + 1) % self.interleave_frequency == 0:
                physical_loss = self._optimize_physical_epoch(
                    real_data["targets"],
                    synthetic_preds
                )
            
            # Record results
            results["synthetic_losses"].append(synthetic_loss)
            results["physical_losses"].append(physical_loss)
            results["epochs"].append(epoch + 1)
            
            epoch_time = time.time() - epoch_start
            logger.info(f"\nEpoch {epoch + 1} complete in {epoch_time:.2f}s")
            logger.info(f"  Synthetic loss: {synthetic_loss:.6f}")
            if physical_loss is not None:
                logger.info(f"  Physical loss: {physical_loss:.6f}")
        
        logger.info(f"\n{'='*60}")
        logger.info("HYCO training complete!")
        logger.info(f"{'='*60}\n")
        
        return results
    
    # Placeholder methods (to be implemented)
    def _warmup_synthetic(self):
        pass
    
    def _load_real_data(self):
        pass
    
    def _generate_synthetic_predictions(self, initial_states):
        pass
    
    def _generate_physical_predictions(self, initial_states):
        pass
    
    def _train_synthetic_epoch(self, real_targets, physical_preds):
        pass
    
    def _optimize_physical_epoch(self, real_targets, synthetic_preds):
        pass
```

#### Task 1.2: Register Hybrid Trainer

```python
# src/factories/trainer_factory.py

class TrainerFactory:
    """Factory for creating trainer instances."""

    _trainers = {
        "synthetic": SyntheticTrainer,
        "physical": PhysicalTrainer,
        "hybrid": HybridTrainer,  # NEW
    }
```

#### Task 1.3: Add Configuration Schema

```yaml
# conf/trainer/hybrid.yaml

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

```yaml
# conf/burgers_hybrid_experiment.yaml

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

### Phase 2: Core Implementation (5-6 days)

#### Task 2.1: Implement Prediction Generation

See detailed implementations in [Critical Issue 2](#critical-issue-2-no-prediction-generation).

#### Task 2.2: Implement Data Augmentation

See detailed implementations in [Issue 3](#issue-3-data-augmentation-for-training).

#### Task 2.3: Implement Training Methods

```python
def _train_synthetic_epoch(
    self,
    real_targets: torch.Tensor,
    physical_preds: Dict[str, Field]
) -> float:
    """
    Train synthetic model for one epoch on real + physical predictions.
    
    Args:
        real_targets: Real ground truth [B, T, C, H, W]
        physical_preds: Predictions from physical model (Fields)
        
    Returns:
        Average loss for the epoch
    """
    self.synthetic_model.train()
    
    # Convert physical predictions to tensors
    physical_targets = self._convert_physical_preds_to_tensors(physical_preds)
    
    # Prepare augmented training data
    combined_targets, sample_weights = self._prepare_hybrid_training_data(
        real_targets, physical_targets, self.alpha
    )
    
    # Training loop
    total_loss = 0.0
    num_batches = 0
    
    # ... implement training loop with augmented data
    
    return total_loss / num_batches

def _optimize_physical_epoch(
    self,
    real_targets: Dict[str, Field],
    synthetic_preds: torch.Tensor
) -> float:
    """
    Optimize physical model for one epoch on real + synthetic predictions.
    
    Args:
        real_targets: Real ground truth (Fields)
        synthetic_preds: Predictions from synthetic model [B, T, C, H, W]
        
    Returns:
        Final loss value
    """
    # Convert synthetic predictions to Fields
    synthetic_targets = self._convert_synthetic_preds_to_fields(synthetic_preds)
    
    # Combine real and synthetic targets
    combined_targets = self._combine_field_targets(
        real_targets, synthetic_targets, self.alpha
    )
    
    # Define loss function
    def loss_fn(*params):
        # Update model parameters
        self._update_physical_parameters(params)
        
        # Run simulation
        predictions = self._run_physical_simulation(combined_targets["initial_state"])
        
        # Compute loss
        loss = self._compute_field_loss(predictions, combined_targets)
        
        return loss
    
    # Run optimization with proper Solve configuration
    solve_config = self._get_physical_solve_config()
    optimized_params = math.minimize(loss_fn, solve_config)
    
    # Update model with optimized parameters
    self._update_physical_parameters(optimized_params)
    
    # Compute final loss
    final_loss = loss_fn(*optimized_params)
    
    return float(final_loss)
```

---

## Breaking Changes Mitigation

### Strategy: Non-Breaking Extension

**Principle:** Add new functionality without modifying existing code paths.

### 1. No Changes to Existing Trainers ✅

- `SyntheticTrainer` remains unchanged
- `PhysicalTrainer` receives ONLY backward-compatible additions
- Both trainers continue to work independently

### 2. Configuration Compatibility ✅

**Existing configs continue to work:**

```yaml
# Existing synthetic training - NO CHANGES NEEDED
run_params:
  model_type: 'synthetic'  # Still works

# Existing physical training - NO CHANGES NEEDED
run_params:
  model_type: 'physical'  # Still works

# New hybrid training - NEW OPTION
run_params:
  model_type: 'hybrid'  # New option
```

### 3. Backward-Compatible Physical Trainer Changes

**Changes to `PhysicalTrainer`:**

```python
# BEFORE (no changes to existing behavior)
def _setup_optimization(self):
    return math.Solve(
        method=method,
        abs_tol=abs_tol,
        x0=self.initial_guesses,
        max_iterations=max_iterations,
    )

# AFTER (backward compatible)
def _setup_optimization(self):
    # NEW: Optional suppression (default False)
    suppress_convergence = self.trainer_config.get("suppress_convergence_errors", False)
    suppress_list = []
    if suppress_convergence:
        suppress_list.append(math.NotConverged)
    
    return math.Solve(
        method=method,
        abs_tol=abs_tol,
        x0=self.initial_guesses,
        max_iterations=max_iterations,
        suppress=tuple(suppress_list)  # Empty tuple by default (no change)
    )
```

**Impact:** 
- ✅ Existing configs: `suppress_convergence_errors` not present → `False` → no change
- ✅ Hybrid configs: `suppress_convergence_errors: true` → enables suppression

### 4. Field Conversion Module - No Changes ✅

- Existing field conversion utilities work as-is
- `HybridTrainer` uses existing `FieldTensorConverter` class
- No modifications to conversion logic needed

### 5. Data Pipeline - Extension Only

**New method in `HybridDataset`:**

```python
# NEW method (doesn't affect existing functionality)
def get_dual_format_sample(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Field]]:
    """
    Get sample in both tensor and field format.
    NEW method for hybrid training only.
    """
    pass
```

**Impact:**
- ✅ Existing `__getitem__` unchanged
- ✅ New method is optional, used only by `HybridTrainer`

### 6. Model Registry - Extension Only

```python
# No changes to existing methods
# HybridTrainer calls existing methods:
synthetic_model = ModelRegistry.get_synthetic_model(name, config)
physical_model = ModelRegistry.get_physical_model(name, config)
```

### 7. Testing Isolation

**Test organization:**

```
tests/
├── training/
│   ├── test_synthetic_trainer.py  # Existing - no changes
│   ├── test_physical_trainer.py   # Existing - minor additions
│   └── test_hybrid_trainer.py     # NEW - isolated tests
```

**Strategy:**
- Existing tests continue to pass without modification
- New hybrid tests are completely isolated
- Regression testing ensures no breakage

---

## Testing Strategy

### Unit Tests

#### Test 1: Non-Convergence Handling

```python
# tests/training/test_physical_convergence.py

def test_suppress_convergence_disabled_by_default():
    """Ensure existing behavior unchanged."""
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
    
    assert math.NotConverged in solve.suppress

def test_physical_with_low_iterations():
    """Test that low max_iterations doesn't crash with suppression."""
    config = get_default_physical_config()
    config["trainer_params"]["max_iterations"] = 3
    config["trainer_params"]["suppress_convergence_errors"] = True
    
    trainer = PhysicalTrainer(config)
    results = trainer.train()
    
    # Should complete successfully
    assert results is not None
    # Parameters should be optimized (not initial guess)
    assert results["optimized_parameters"]["nu"] != trainer.initial_guesses[0]
```

#### Test 2: Prediction Generation

```python
# tests/training/test_hybrid_trainer.py

def test_generate_synthetic_predictions():
    """Test synthetic prediction generation."""
    trainer = create_test_hybrid_trainer()
    
    # Create mock initial state
    initial_state = create_mock_field_dict()
    
    # Generate predictions
    preds = trainer._generate_synthetic_rollout(initial_state, num_steps=5)
    
    # Check shape
    assert preds.shape == (batch_size, 5, channels, height, width)
    # Check predictions are different from initial
    assert not torch.allclose(preds[:, 0], preds[:, -1])

def test_generate_physical_predictions():
    """Test physical prediction generation."""
    trainer = create_test_hybrid_trainer()
    
    # Create mock initial state
    initial_state = create_mock_field_dict()
    
    # Generate predictions
    preds = trainer._generate_physical_rollout(initial_state, num_steps=5)
    
    # Check structure
    assert isinstance(preds, dict)
    assert "velocity" in preds
    # Check has time dimension
    assert "time" in preds["velocity"].shape.names
```

#### Test 3: Field-Tensor Round Trip

```python
def test_field_tensor_conversion_preserves_values():
    """Ensure conversion is lossless."""
    # Create Field
    original_field = create_test_field()
    
    # Convert to tensor
    converter = make_converter(original_field)
    tensor = converter.field_to_tensor(original_field)
    
    # Convert back to Field
    metadata = FieldMetadata.from_field(original_field)
    reconstructed_field = converter.tensor_to_field(tensor, metadata)
    
    # Check values match
    assert fields_close(original_field, reconstructed_field, tol=1e-6)
```

### Integration Tests

#### Test 4: Hybrid Training Loop

```python
def test_hybrid_training_one_epoch():
    """Test one complete hybrid epoch."""
    config = get_test_hybrid_config()
    config["trainer_params"]["epochs"] = 1
    
    trainer = HybridTrainer(config)
    results = trainer.train()
    
    # Check results structure
    assert "synthetic_losses" in results
    assert "physical_losses" in results
    assert len(results["synthetic_losses"]) == 1

def test_hybrid_training_warmup():
    """Test warmup phase works."""
    config = get_test_hybrid_config()
    config["trainer_params"]["epochs"] = 5
    config["trainer_params"]["warmup_epochs"] = 2
    
    trainer = HybridTrainer(config)
    results = trainer.train()
    
    # First 2 epochs should have no physical loss
    assert results["physical_losses"][0] is None
    assert results["physical_losses"][1] is None
    # Later epochs should have physical loss
    assert results["physical_losses"][2] is not None
```

### System Tests

#### Test 5: End-to-End HYCO Training

```python
def test_burgers_hybrid_training_end_to_end():
    """Full end-to-end test on Burgers equation."""
    # Run full pipeline
    config = load_config("burgers_hybrid_experiment.yaml")
    
    # Generate data
    run_generation(config)
    
    # Train hybrid model
    trainer = TrainerFactory.create_trainer(config)
    results = trainer.train()
    
    # Check training completed
    assert results is not None
    assert len(results["epochs"]) == config["trainer_params"]["epochs"]
    
    # Check losses decreased
    assert results["synthetic_losses"][-1] < results["synthetic_losses"][0]
```

---

## Risk Assessment

### High Risk Items

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| Field-Tensor conversion performance bottleneck | High | Medium | Profile early, optimize conversion code, add caching |
| Memory overflow with dual models | High | Medium | Use gradient checkpointing, mixed precision, monitor memory |
| Unstable training dynamics | High | Medium | Careful alpha tuning, warmup period, gradient clipping |
| Physical model too slow per epoch | Medium | High | Reduce max_iterations to 3-5, use coarser resolution |

### Medium Risk Items

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| Configuration complexity | Medium | Low | Good documentation, example configs, validation |
| Breaking existing tests | Medium | Low | Run full regression suite, isolated test development |
| Different convergence rates | Medium | Medium | Adaptive interleave_frequency, separate learning rates |

### Low Risk Items

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| Code maintainability | Low | Low | Clean architecture, good documentation |
| User adoption | Low | Low | Backward compatibility, clear examples |

---

## Timeline & Milestones

### Week 1: Foundation & Critical Fixes

**Days 1-2: Fix Non-Convergence Handling (CRITICAL)**
- [ ] Add `suppress_convergence_errors` config option
- [ ] Update `_setup_optimization()` in `PhysicalTrainer`
- [ ] Add tests for suppression behavior
- [ ] Test with low `max_iterations`
- [ ] Document new configuration option

**Days 3-4: Create HybridTrainer Skeleton**
- [ ] Create `src/training/hybrid/` directory
- [ ] Implement `HybridTrainer` class skeleton
- [ ] Add to `TrainerFactory`
- [ ] Create `conf/trainer/hybrid.yaml`
- [ ] Create `conf/burgers_hybrid_experiment.yaml`

**Day 5: Testing & Review**
- [ ] Unit tests for convergence handling
- [ ] Integration tests for skeleton
- [ ] Code review
- [ ] Documentation review

**Milestone 1:** ✅ Foundation complete, no breaking changes

---

### Week 2: Core Implementation

**Days 6-7: Prediction Generation**
- [ ] Implement `_generate_synthetic_rollout()`
- [ ] Implement `_generate_physical_rollout()`
- [ ] Add field-tensor conversion integration
- [ ] Unit tests for prediction generation
- [ ] Performance profiling

**Days 8-9: Data Augmentation & Training**
- [ ] Implement `_prepare_hybrid_training_data()`
- [ ] Implement `_compute_weighted_loss()`
- [ ] Implement `_train_synthetic_epoch()`
- [ ] Implement `_optimize_physical_epoch()`
- [ ] Unit tests for training methods

**Day 10: Integration & Testing**
- [ ] Implement `train()` main loop
- [ ] Implement warmup phase
- [ ] Integration tests
- [ ] Test with `burgers_quick_test` equivalent

**Milestone 2:** ✅ Core functionality working

---

### Week 3: Optimization & Validation

**Days 11-12: Performance Optimization**
- [ ] Profile field-tensor conversions
- [ ] Optimize memory usage
- [ ] Add gradient checkpointing if needed
- [ ] Benchmark training speed

**Days 13-14: Comprehensive Testing**
- [ ] Full regression suite
- [ ] End-to-end system tests
- [ ] Test on multiple PDEs (Burgers, Heat, Smoke)
- [ ] Validation on test data

**Day 15: Documentation & Polish**
- [ ] Complete code documentation
- [ ] Write user guide
- [ ] Example notebooks
- [ ] Performance tuning guide

**Milestone 3:** ✅ Production-ready implementation

---

## Success Criteria

### Functional Requirements

- [x] ✅ Physical model handles non-convergence gracefully (with `suppress`)
- [ ] ✅ `HybridTrainer` trains both models in interleaved fashion
- [ ] ✅ Both models receive same initial states
- [ ] ✅ Predictions from both models have same length
- [ ] ✅ Field-Tensor conversion works bidirectionally
- [ ] ✅ Configuration backward compatible

### Performance Requirements

- [ ] Training time < 2x slowest individual trainer
- [ ] Memory usage < 1.5x combined individual trainers
- [ ] Field-Tensor conversion < 10% of epoch time

### Quality Requirements

- [ ] All existing tests pass
- [ ] New code has >80% test coverage
- [ ] No breaking changes to existing APIs
- [ ] Documentation complete

---

## Appendix A: Configuration Examples

### Minimal Hybrid Configuration

```yaml
# conf/minimal_hybrid.yaml
defaults:
  - data: burgers_128
  - model/physical: burgers
  - model/synthetic: unet
  - trainer: hybrid
  - _self_

run_params:
  experiment_name: 'minimal_hybrid'
  mode: ['train']
  model_type: 'hybrid'

trainer_params:
  train_sim: [0]
  epochs: 10
  num_predict_steps: 5
```

### Full Hybrid Configuration

```yaml
# conf/full_hybrid.yaml
defaults:
  - data: burgers_128
  - model/physical: burgers
  - model/synthetic: unet
  - trainer: hybrid
  - generation: default
  - evaluation: default
  - _self_

run_params:
  experiment_name: 'burgers_hybrid_full'
  notes: 'Full HYCO hybrid training with all features'
  mode: ['train', 'evaluate']
  model_type: 'hybrid'

trainer_params:
  # Data
  train_sim: [0, 1, 2, 3, 4]
  val_sim: [5, 6]
  
  # Training
  epochs: 100
  num_predict_steps: 20
  
  # Hybrid parameters
  alpha: 0.6  # 60% real, 40% synthetic
  interleave_frequency: 2  # Update physical every 2 epochs
  warmup_epochs: 10
  
  # Synthetic settings
  synthetic:
    learning_rate: 1e-4
    batch_size: 16
    optimizer: adam
    scheduler: cosine
    weight_decay: 1e-5
    use_sliding_window: false
  
  # Physical settings
  physical:
    method: 'L-BFGS-B'
    abs_tol: 1e-6
    max_iterations: 5
    suppress_convergence_errors: true
  
  # Checkpointing
  save_interval: 10
  save_best_only: true
  
  # Validation
  validate_every: 5
  validation_rollout: true

project_root: ${hydra:runtime.cwd}
```

---

## Appendix B: Quick Start Guide

### 1. Fix Non-Convergence First

```bash
# Test current behavior
conda activate torch-env
python run.py --config-name burgers_physical_quick_test

# Apply fixes from Phase 0
# Edit src/training/physical/trainer.py
# Add suppress parameter to Solve()

# Test fixed behavior
python run.py --config-name burgers_physical_quick_test trainer_params.suppress_convergence_errors=true trainer_params.max_iterations=3
```

### 2. Create Hybrid Skeleton

```bash
# Create directory
mkdir -p src/training/hybrid

# Create files
touch src/training/hybrid/__init__.py
touch src/training/hybrid/trainer.py

# Add configuration
touch conf/trainer/hybrid.yaml
touch conf/burgers_hybrid_quick_test.yaml
```

### 3. Test Skeleton

```bash
# Register in factory
# Edit src/factories/trainer_factory.py

# Test loading
python -c "from src.factories.trainer_factory import TrainerFactory; print(TrainerFactory.list_available_trainers())"

# Should output: ['synthetic', 'physical', 'hybrid']
```

### 4. Implement Core Functionality

Follow Phase 2 tasks sequentially.

### 5. Run First Hybrid Training

```bash
python run.py --config-name burgers_hybrid_quick_test
```

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-02 | Initial | Complete implementation strategy |

**Status:** Ready for implementation  
**Next Review:** After Phase 0 completion
