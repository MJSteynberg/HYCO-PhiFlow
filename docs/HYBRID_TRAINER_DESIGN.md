# Hybrid Trainer Design Document

**Date:** November 3, 2025  
**Status:** Design Phase - Updated Architecture  
**Branch:** feature/hyco-trainer

---

## Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [Architecture](#architecture)
4. [Configuration](#configuration)
5. [Data Flow](#data-flow)
6. [Implementation Details](#implementation-details)
7. [Efficiency Analysis](#efficiency-analysis)
8. [Open Questions](#open-questions)

---

## Overview

### Objective

Implement a HybridTrainer that orchestrates interleaved training of synthetic (neural network) and physical (PDE) models using the HYCO (Hybrid Coupled) strategy.

### Key Concept

**Composition over Direct Implementation:**
- HybridTrainer does NOT directly manage models
- HybridTrainer COMPOSES existing SyntheticTrainer and PhysicalTrainer
- Sub-trainers remain unchanged and fully functional standalone
- HybridTrainer handles: data augmentation, conversion, orchestration

---

## Design Principles

### 1. **Explicit Data Passing Architecture**

**Core Principle:** Trainers do NOT manage data internally. Data is explicitly passed to the `train()` method.

**Rationale:**
- ✅ Clear separation: trainer handles optimization, caller handles data
- ✅ Maximum flexibility: same trainer can be used with different data sources
- ✅ No trainer recreation: single trainer instance persists across training cycles
- ✅ Explicit and transparent: caller always knows what data is being used

**Design:**
```python
class TensorTrainer(AbstractTrainer):
    def __init__(self, config, model):
        """Initialize trainer with model and hyperparameters. NO data."""
        self.model = model
        self.optimizer = self._create_optimizer()
    
    def train(self, data_source, num_epochs):
        """
        Train on provided data.
        
        Args:
            data_source: Iterable providing (input, target, weight) batches
            num_epochs: Number of epochs to train
        """
        for epoch in range(num_epochs):
            for batch in data_source:
                # Training logic
                pass
```

**Key Changes:**
- ✅ Remove `data_manager` from trainer `__init__`
- ✅ Add `data_source` parameter to `train()` method
- ✅ Data source provides `(input, target, weight)` tuples
- ✅ Trainer persists across multiple `train()` calls

**Backward Compatibility:**
```python
# For existing code, create convenience factory method
@classmethod
def from_config(cls, config):
    """Factory for backward compatibility."""
    model = ModelRegistry.get_model(config["model"])
    data_source = create_dataloader(config["data"])
    
    trainer = cls(config["trainer_params"], model)
    # Can still call: trainer.train(data_source, config["epochs"])
    return trainer
```

### 2. **Composition Architecture**

```
HybridTrainer (orchestrator)
├── SyntheticTrainer (black box)
│   └── Trains neural network
└── PhysicalTrainer (black box)
    └── Optimizes PDE parameters

HybridTrainer responsibilities:
- Generate predictions from both models
- Convert between Fields ↔ Tensors
- Augment datasets with generated data
- Orchestrate alternating training cycles
```

### 3. **Persistent Trainers, Dynamic Data**

**Key Insight:** Create trainers ONCE, pass different data to each training cycle.

**Benefits:**
- 🚀 **Performance**: No trainer recreation overhead
- 💾 **Memory**: Model stays in GPU memory, optimizer state preserved
- 📈 **Convergence**: Learning rate schedules and momentum continue across cycles
- 🧹 **Clean**: Explicit data passing makes code easy to understand

**Usage Pattern:**
```python
# Create trainer once
trainer = SyntheticTrainer(config, model)

# Use many times with different data
for cycle in range(num_cycles):
    new_data = prepare_data_for_cycle(cycle)
    trainer.train(data_source=new_data, num_epochs=5)
    # Trainer persists, optimizer state preserved!
```

### 4. **Sub-trainers Are Black Boxes**

Sub-trainers know NOTHING about:
- Where data comes from (real vs generated)
- The other model's existence
- Hybrid training strategy

They just:
- Accept configuration (including epochs/iterations)
- Train on provided data
- Return results

### 5. **Weighted Loss via Data Contract**

Mathematical formulation:
```
Loss = LossReal + alpha * LossGenerated

Where:
- LossReal: Loss on real simulation data
- LossGenerated: Loss on predictions from other model
- alpha: Weighting factor (0 to 1)
```

**Implementation:** Weights are part of the data contract.

Each batch provides: `(input, target, weight)` tuples

```python
# Real data batches have weight = 1.0
real_batch = (input, target, 1.0)

# Generated data batches have weight = alpha
gen_batch = (input, target, alpha)

# Trainer naturally handles weights
for batch in data_source:
    inputs, targets, weights = batch
    loss = criterion(model(inputs), targets)
    weighted_loss = (loss * weights).mean()
    weighted_loss.backward()
```

**No special trainer logic needed!** Weights are just data.

---

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    HybridTrainer                        │
├─────────────────────────────────────────────────────────┤
│  Responsibilities:                                      │
│  - Orchestrate training cycles                          │
│  - Generate predictions from both models                │
│  - Create augmented datasets (real + generated)         │
│  - Convert between Fields ↔ Tensors                     │
│  - Manage alpha weighting                               │
│                                                         │
│  ┌──────────────┐         ┌──────────────┐            │
│  │   Synthetic  │         │   Physical   │            │
│  │   Trainer    │         │   Trainer    │            │
│  │  (persistent)│         │  (persistent)│            │
│  └──────┬───────┘         └──────┬───────┘            │
│         │                        │                      │
│    train(data, epochs)      train(data, iters)         │
│         ▲                        ▲                      │
│         │                        │                      │
│    ┌────┴──────────┐       ┌────┴──────────┐          │
│    │  Augmented    │       │  Augmented    │          │
│    │  DataLoader   │       │  DataSource   │          │
│    │ (Tensors)     │       │  (Fields)     │          │
│    └───────────────┘       └───────────────┘          │
│         ▲                        ▲                      │
│         └────────┬───────────────┘                     │
│                  │                                      │
│         ┌────────┴─────────┐                          │
│         │  Data Creation   │                          │
│         │  - Load real     │                          │
│         │  - Generate preds│                          │
│         │  - Add weights   │                          │
│         │  - Field↔Tensor  │                          │
│         └──────────────────┘                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Trainer Initialization

```python
class HybridTrainer(AbstractTrainer):
    def __init__(self, config):
        """Initialize hybrid trainer with persistent sub-trainers."""
        
        # Hybrid-specific parameters
        self.alpha = config["trainer_params"]["alpha"]
        self.synthetic_epochs_per_cycle = config["trainer_params"]["synthetic_epochs_per_cycle"]
        self.physical_epochs_per_cycle = config["trainer_params"]["physical_epochs_per_cycle"]
        self.num_cycles = config["trainer_params"]["epochs"]
        
        # Create models
        self.synthetic_model = ModelRegistry.get_synthetic_model(
            config["model"]["synthetic"]["name"],
            config["model"]["synthetic"]
        )
        self.physical_model = ModelRegistry.get_physical_model(
            config["model"]["physical"]["name"],
            config["model"]["physical"]
        )
        
        # Create trainers ONCE (persistent, no data yet)
        self.synthetic_trainer = SyntheticTrainer(
            config=config["trainer_params"]["synthetic"],
            model=self.synthetic_model
        )
        
        self.physical_trainer = PhysicalTrainer(
            config=config["trainer_params"]["physical"],
            model=self.physical_model,
            learnable_params=config["trainer_params"]["learnable_parameters"]
        )
        
        # Data loading and conversion utilities
        self.real_data_manager = DataManager(config["data"])
        self.field_converter = FieldTensorConverter(...)
        
        logger.info("HybridTrainer initialized with persistent sub-trainers")
```

**Key Points:**
- Trainers created ONCE during initialization
- No data passed to trainers at creation
- Same trainer instances used throughout all cycles
- Data passed explicitly to each `train()` call

---

## Configuration

### Hybrid Trainer Configuration

```yaml
# conf/trainer/hybrid.yaml

# Total training cycles
epochs: 20

# Sub-trainer epochs per cycle
synthetic_epochs_per_cycle: 5  # How many epochs synthetic trains per cycle
physical_epochs_per_cycle: 3   # How many iterations physical optimizes per cycle

# Training data
train_sim: [0, 1, 2]
val_sim: [3]

# Hybrid-specific parameters
alpha: 0.5  # Weight for generated data (Loss = LossReal + alpha * LossGen)
num_predict_steps: 10  # How many steps to predict when generating data

# Sub-trainer configurations
synthetic:
  learning_rate: 1e-4
  batch_size: 16
  optimizer: adam
  scheduler: cosine
  weight_decay: 0.0
  # Note: epochs will be set by synthetic_epochs_per_cycle

physical:
  method: 'L-BFGS-B'
  abs_tol: 1e-6
  # Note: max_iterations will be set by physical_epochs_per_cycle
  suppress_convergence_errors: true

# Model specifications (for factory loading)
model_types:
  synthetic: 'unet'  # Can change to 'transformer', etc.
  physical: 'burgers'  # Can change to 'heat', 'smoke', etc.
```

### Full Experiment Configuration

```yaml
# conf/burgers_hybrid_experiment.yaml

defaults:
  - data: burgers_128
  - model/physical: burgers
  - model/synthetic: unet
  - trainer: hybrid
  - _self_

run_params:
  experiment_name: 'burgers_hybrid_hyco'
  notes: 'HYCO interleaved hybrid training'
  mode: ['train', 'evaluate']
  model_type: 'hybrid'

trainer_params:
  epochs: 20
  synthetic_epochs_per_cycle: 5
  physical_epochs_per_cycle: 3
  train_sim: [0, 1, 2, 3, 4]
  val_sim: [5]
  alpha: 0.5
  num_predict_steps: 10

project_root: ${hydra:runtime.cwd}
```

---

## Data Flow

### Training Cycle Pseudocode

```python
### Training Cycle Pseudocode

```python
def train(self):
    """Main hybrid training loop with persistent trainers."""
    
    for cycle in range(self.num_cycles):
        logger.info(f"\n{'='*60}")
        logger.info(f"Cycle {cycle + 1}/{self.num_cycles}")
        logger.info(f"{'='*60}\n")
        
        # ─────────────────────────────────────────────────
        # PHASE 1: Train Synthetic Model
        # ─────────────────────────────────────────────────
        
        # 1.1: Load real data as tensors
        real_data_tensor = self._load_real_data_as_tensors()
        
        # 1.2: Generate physical model predictions (in Fields)
        physical_predictions_fields = self._generate_physical_predictions()
        
        # 1.3: Convert physical predictions to tensors
        physical_predictions_tensor = self._convert_fields_to_tensors(
            physical_predictions_fields
        )
        
        # 1.4: Create augmented dataloader (real + generated with weights)
        synthetic_dataloader = self._create_augmented_dataloader(
            real_data=real_data_tensor,
            generated_data=physical_predictions_tensor,
            alpha=self.alpha
        )
        
        # 1.5: Train synthetic model (trainer persists!)
        synthetic_results = self.synthetic_trainer.train(
            data_source=synthetic_dataloader,
            num_epochs=self.synthetic_epochs_per_cycle
        )
        
        logger.info(f"Synthetic training complete: {synthetic_results}")
        
        # ─────────────────────────────────────────────────
        # PHASE 2: Optimize Physical Model
        # ─────────────────────────────────────────────────
        
        # 2.1: Load real data as fields
        real_data_fields = self._load_real_data_as_fields()
        
        # 2.2: Generate synthetic model predictions (in Tensors)
        synthetic_predictions_tensor = self._generate_synthetic_predictions()
        
        # 2.3: Convert synthetic predictions to fields
        synthetic_predictions_fields = self._convert_tensors_to_fields(
            synthetic_predictions_tensor
        )
        
        # 2.4: Create augmented data source (real + generated with weights)
        physical_data_source = self._create_augmented_field_source(
            real_data=real_data_fields,
            generated_data=synthetic_predictions_fields,
            alpha=self.alpha
        )
        
        # 2.5: Optimize physical model (trainer persists!)
        physical_results = self.physical_trainer.train(
            data_source=physical_data_source,
            num_epochs=self.physical_epochs_per_cycle
        )
        
        logger.info(f"Physical optimization complete: {physical_results}")
        
        # ─────────────────────────────────────────────────
        # Log and save results
        # ─────────────────────────────────────────────────
        self._log_cycle_results(cycle, synthetic_results, physical_results)
    
    return self._compile_final_results()
```

### Key Points

1. **Trainers created once** in `__init__`, reused in every cycle
2. **Data passed explicitly** to each `train()` call
3. **No trainer recreation** - optimizer state, learning rate, momentum all preserved
4. **Clear data flow** - easy to see what data is used in each phase
```

### Data Format Flow

```
Real Data:
    Disk → DataManager → Tensors (for synthetic) / Fields (for physical)

Physical Predictions:
    Fields (native) → Convert → Tensors (for synthetic trainer)

Synthetic Predictions:
    Tensors (native) → Convert → Fields (for physical trainer)

Augmented Data:
    Real + Generated (weighted by alpha)
```

---

## Implementation Details

### Required Changes to Trainer Base Classes

#### TensorTrainer Modifications

```python
# src/training/tensor_trainer.py

class TensorTrainer(AbstractTrainer):
    """
    Base class for PyTorch tensor-based trainers.
    
    Key Design: Trainer manages optimization, caller provides data.
    """
    
    def __init__(self, config, model):
        """
        Initialize tensor trainer.
        
        Args:
            config: Training configuration (lr, optimizer, scheduler, etc.)
            model: PyTorch neural network to train
        """
        super().__init__(config)
        
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup training components (persistent)
        self.optimizer = self._create_optimizer()
        self.criterion = self._create_loss_function()
        self.scheduler = self._create_scheduler()
        
        # Training state (persistent across train() calls)
        self.global_epoch = 0
        self.training_history = []
        self.best_loss = float('inf')
    
    def train(self, data_source, num_epochs):
        """
        Train on provided data for specified epochs.
        
        Args:
            data_source: Iterable providing (input, target, weight) batches
            num_epochs: Number of epochs to train
            
        Returns:
            Training metrics dictionary
        """
        epoch_losses = []
        
        for local_epoch in range(num_epochs):
            epoch_loss = self._train_epoch(data_source)
            
            # Update persistent state
            self.global_epoch += 1
            epoch_losses.append(epoch_loss)
            self.training_history.append(epoch_loss)
            
            # Track best model
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
            
            # Scheduler step
            self.scheduler.step()
            
            logger.info(
                f"Epoch {self.global_epoch} (local {local_epoch+1}/{num_epochs}): "
                f"loss = {epoch_loss:.6f}"
            )
        
        return {
            'epoch_losses': epoch_losses,
            'global_epoch': self.global_epoch,
            'best_loss': self.best_loss
        }
    
    def _train_epoch(self, data_source):
        """Train for one epoch on provided data."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in data_source:
            inputs, targets, weights = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            weights = weights.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute loss
            loss = self.criterion(outputs, targets)
            
            # Apply weights (per-sample or per-batch)
            weighted_loss = (loss * weights).mean()
            
            # Backward pass
            self.optimizer.zero_grad()
            weighted_loss.backward()
            self.optimizer.step()
            
            total_loss += weighted_loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
```

**Key Changes:**
- ❌ Removed: `data_manager` from `__init__`
- ❌ Removed: Internal dataset creation
- ✅ Added: `data_source` parameter to `train()`
- ✅ Added: Persistent training state (`global_epoch`, `training_history`)
- ✅ Added: Weight handling in loss computation

#### FieldTrainer Modifications

```python
# src/training/field_trainer.py

class FieldTrainer(AbstractTrainer):
    """
    Base class for PhiFlow field-based trainers.
    
    Key Design: Trainer manages optimization, caller provides data.
    """
    
    def __init__(self, config, model, learnable_params):
        """
        Initialize field trainer.
        
        Args:
            config: Training configuration (method, tolerance, etc.)
            model: PhiFlow physical model
            learnable_params: List of parameter names to optimize
        """
        super().__init__(config)
        
        self.model = model
        self.learnable_params = learnable_params
        
        # Persistent optimization state
        self.iteration_count = 0
        self.optimization_history = []
        self.best_params = None
    
    def train(self, data_source, num_epochs):
        """
        Optimize parameters on provided data.
        
        Args:
            data_source: Iterable providing (initial_state, target, weight) tuples
            num_epochs: Maximum optimization iterations (called num_epochs for consistency)
            
        Returns:
            Optimization results dictionary
        """
        def loss_fn(*params):
            total_loss = 0.0
            
            for initial_state, target_trajectory, weight in data_source:
                # Update model parameters
                self.model.update_parameters(params)
                
                # Simulate
                prediction = self.model.simulate(initial_state)
                
                # Compute weighted loss
                loss = self._compute_loss(prediction, target_trajectory)
                total_loss += loss * weight
            
            return total_loss
        
        # Setup optimization
        solve_config = self._create_solve_config(num_epochs)
        
        # Optimize
        optimized_params = math.minimize(loss_fn, solve_config)
        
        # Update persistent state
        self.iteration_count += num_epochs
        final_loss = loss_fn(*optimized_params)
        self.optimization_history.append(final_loss)
        self.best_params = optimized_params
        
        logger.info(
            f"Optimization complete: {num_epochs} iterations, "
            f"total iterations: {self.iteration_count}, "
            f"final loss: {final_loss:.6f}"
        )
        
        return {
            'optimized_params': optimized_params,
            'final_loss': final_loss,
            'total_iterations': self.iteration_count
        }
```

**Key Changes:**
- ❌ Removed: `data_manager` from `__init__`
- ❌ Removed: Internal data loading
- ✅ Added: `data_source` parameter to `train()`
- ✅ Added: Persistent optimization state
- ✅ Added: Weight handling in loss function
            weighted_loss = loss * self.loss_weight_fn()
            
            # Backward pass
            self.optimizer.zero_grad()
            weighted_loss.backward()
            self.optimizer.step()
            
            total_loss += weighted_loss.item()
        
        return total_loss / len(self.train_loader)
```

#### FieldTrainer Modifications

```python
# src/training/field_trainer.py

class FieldTrainer(AbstractTrainer):
    """Base class for PhiFlow field-based trainers."""
    
    def __init__(self, config, data_manager=None, loss_weight_fn=None):
        """
        Initialize field trainer.
        
        Args:
            config: Configuration dictionary
            data_manager: Optional custom data manager (for hybrid training)
            loss_weight_fn: Optional function returning loss weight scalar
        """
        super().__init__(config)
        
        # Use provided data manager or create default
        if data_manager is not None:
            self.data_manager = data_manager
        else:
            self._setup_data_manager()  # Existing method
        
        # Use provided weight function or default to 1.0
        self.loss_weight_fn = loss_weight_fn if loss_weight_fn is not None else (lambda: 1.0)
        
        # Rest of initialization...
    
    def _compute_loss(self, predictions, targets):
        """
        Compute loss between predictions and targets.
        
        Args:
            predictions: Model predictions (Fields)
            targets: Ground truth (Fields)
            
        Returns:
            Weighted loss value
        """
        # Compute base loss
        loss = field_l2_loss(predictions, targets)
        
        # Apply weight from weight function
        weighted_loss = loss * self.loss_weight_fn()
        
        return weighted_loss
```

### HybridTrainer Key Methods

#### Prediction Generation

```python
def _generate_physical_predictions(self) -> Dict[int, Dict[str, Field]]:
    """
    Generate predictions from physical model for all training simulations.
    
    Returns:
        Dictionary mapping sim_idx to dict of field predictions
        Format: {sim_idx: {"velocity": Field, "density": Field, ...}}
    """
    predictions = {}
    
    for sim_idx in self.train_sim:
        # Load initial state
        initial_state = self._load_initial_state_as_fields(sim_idx)
        
        # Run physical model forward
        trajectory = []
        current_state = initial_state
        
        for step in range(self.num_predict_steps):
            next_state = self.physical_model.step(current_state)
            trajectory.append(next_state)
            current_state = next_state
        
        predictions[sim_idx] = trajectory
    
    return predictions


def _generate_synthetic_predictions(self) -> Dict[int, torch.Tensor]:
    """
    Generate predictions from synthetic model for all training simulations.
    
    Returns:
        Dictionary mapping sim_idx to tensor predictions
        Format: {sim_idx: Tensor[T, C, H, W]}
    """
    predictions = {}
    
    self.synthetic_model.eval()
    with torch.no_grad():
        for sim_idx in self.train_sim:
            # Load initial state as tensor
            initial_state = self._load_initial_state_as_tensor(sim_idx)
            
            # Run synthetic model forward
            trajectory = self.synthetic_model.predict(
                initial_state, 
                num_steps=self.num_predict_steps
            )
            
            predictions[sim_idx] = trajectory
    
    return predictions
```

#### Data Conversion

```python
def _convert_fields_to_tensors(
    self, 
    fields_dict: Dict[int, Dict[str, Field]]
) -> Dict[int, torch.Tensor]:
    """
    Convert PhiFlow Fields to PyTorch tensors.
    
    Args:
        fields_dict: {sim_idx: {field_name: Field}}
        
    Returns:
        {sim_idx: Tensor[T, C, H, W]}
    """
    tensors = {}
    
    for sim_idx, fields in fields_dict.items():
        # Use field converter
        tensor = self.field_converter.fields_to_tensor(fields)
        tensors[sim_idx] = tensor
    
    return tensors


def _convert_tensors_to_fields(
    self,
    tensors_dict: Dict[int, torch.Tensor]
) -> Dict[int, Dict[str, Field]]:
    """
    Convert PyTorch tensors to PhiFlow Fields.
    
    Args:
        tensors_dict: {sim_idx: Tensor[T, C, H, W]}
        
    Returns:
        {sim_idx: {field_name: Field}}
    """
    fields = {}
    
    for sim_idx, tensor in tensors_dict.items():
        # Use field converter
        field_dict = self.field_converter.tensor_to_fields(tensor)
        fields[sim_idx] = field_dict
    
    return fields
```

#### Data Augmentation

```python
def _create_augmented_data_manager(
    self,
    real_data: Dict[int, Any],
    generated_data: Dict[int, Any],
    alpha: float
) -> DataManager:
    """
    Create a data manager that serves both real and generated data.
    
    The returned data manager will provide samples with appropriate
    weights for loss computation.
    
    Args:
        real_data: Real simulation data
        generated_data: Generated predictions from other model
        alpha: Weight for generated data
        
    Returns:
        Augmented data manager
    """
    # Implementation depends on final design choice
    # Could be CombinedDataManager or modified existing manager
    pass


def _create_weight_function(self, alpha: float):
    """
    Create a weight function for loss computation.
    
    Args:
        alpha: Weight for generated data
        
    Returns:
        Function that returns appropriate weight for current batch
    """
    # Simple version: constant weight
    def weight_fn():
        return alpha
    
    # Advanced version: could track which samples are generated
    # and return different weights accordingly
    
    return weight_fn
```

#### Configuration Preparation

```python
def _prepare_synthetic_config(self) -> Dict[str, Any]:
    """
    Prepare configuration for synthetic trainer.
    
    Extracts relevant sections and sets appropriate epochs.
    
    Returns:
        Complete config for SyntheticTrainer
    """
    config = {
        "data": self.config["data"],
        "model": {"synthetic": self.config["model"]["synthetic"]},
        "trainer_params": {
            **self.config["trainer_params"].get("synthetic", {}),
            "epochs": self.synthetic_epochs_per_cycle,
            "train_sim": self.train_sim,
        },
        "project_root": self.project_root,
    }
    return config


def _prepare_physical_config(self) -> Dict[str, Any]:
    """
    Prepare configuration for physical trainer.
    
    Extracts relevant sections and sets appropriate iterations.
    
    Returns:
        Complete config for PhysicalTrainer
    """
    config = {
        "data": self.config["data"],
        "model": {"physical": self.config["model"]["physical"]},
        "trainer_params": {
            **self.config["trainer_params"].get("physical", {}),
            "max_iterations": self.physical_epochs_per_cycle,
            "train_sim": self.train_sim,
        },
        "project_root": self.project_root,
    }
    return config
```

---

## Efficiency Analysis

### Computational Efficiency

**All approaches have similar computational cost:**
- The actual loss computation is identical: `Loss = L_real + alpha * L_gen`
- Difference is only in bookkeeping overhead

**Winner:** Option B (loss_weight_fn) - minimal overhead

### Memory Efficiency

**Option A (CombinedDataManager):**
- Risk of data duplication if implemented naively
- O(N_real + N_gen) if copies data
- O(1) if just references existing data

**Option B (loss_weight_fn):**
- No data duplication
- Original datasets stay in place
- Only passes scalar weights
- **Memory: O(1) extra**

**Winner:** Option B (loss_weight_fn)

### Implementation Efficiency

**Lines of code required:**
- Option A: ~100-150 lines (new CombinedDataManager class)
- Option B: ~10 lines (add weight parameter + multiplication)

**Winner:** Option B (loss_weight_fn)

### Maintenance & Extensibility

**Option B is more flexible:**
```python
# Can implement ANY weighting scheme
# Curriculum learning
def adaptive_weight(epoch):
    return min(1.0, 0.1 + 0.9 * epoch / total_epochs)

# Per-simulation weighting
def sim_specific_weight(sim_id):
    return weight_map[sim_id]

# Dynamic weighting based on loss
def loss_based_weight(current_loss, target_loss):
    return alpha * (current_loss / target_loss)
```

**Winner:** Option B (loss_weight_fn)

### Overall Ranking

| Criterion | CombinedDataManager | loss_weight_fn |
|-----------|---------------------|----------------|
| Computation | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Memory | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Implementation | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Maintenance | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Total** | **3.3/5** | **4.7/5** |

**Recommendation: loss_weight_fn approach**

---

## Open Questions

### 1. Data Manager Strategy

**Issue:** Even with loss_weight_fn, we still need to provide combined data (real + generated) to trainers.

**Options:**

**A) Recreate trainers each cycle with new data manager**
```python
for cycle in range(num_cycles):
    # Create augmented data manager
    augmented_dm = self._create_augmented_dm(real_data, gen_data)
    
    # Create new trainer with augmented data
    trainer = SyntheticTrainer(config, data_manager=augmented_dm)
    trainer.train()
```

**Pros:** Clean, simple, trainers fully encapsulated  
**Cons:** Overhead of trainer recreation each cycle

**B) Modify trainer's data manager in-place**
```python
for cycle in range(num_cycles):
    # Update trainer's data manager
    self.synthetic_trainer.data_manager = augmented_dm
    self.synthetic_trainer.train()
```

**Pros:** Less overhead  
**Cons:** Feels hacky, relies on trainer internals

**C) Create persistent combined data manager**
```python
# Create once with update capability
self.combined_dm = CombinedDataManager(real_dm)

for cycle in range(num_cycles):
    # Update generated data
    self.combined_dm.update_generated_data(new_gen_data)
    self.synthetic_trainer.train()
```

**Pros:** Clean, efficient  
**Cons:** Requires CombinedDataManager implementation

**Question:** Which approach do you prefer?

### 2. Loss Weighting Details

**Mathematical formulation:**

```
Option A: Weight individual samples
Total_Loss = (1/N) * Σ [weight_i * loss_i]
where weight_i = 1.0 for real, alpha for generated

Option B: Weight aggregated losses
Total_Loss = (1/N_real) * Σ loss_real + alpha * (1/N_gen) * Σ loss_gen
```

**Question:** Should alpha weight:
- Individual samples? (Option A - simpler implementation)
- Aggregated losses? (Option B - clearer semantics)

### 3. Dataset Sampling Strategy

**How to sample from real + generated data?**

**Option A: Proportional sampling**
- Sample real and generated in fixed ratio each batch
- Ensures consistent weighting per batch

**Option B: Random sampling**
- Pool all data together, sample randomly
- Statistical weighting over many batches

**Option C: Separate batches**
- Some batches are all real, some all generated
- Clear separation, easier debugging

**Question:** Which sampling strategy?

### 4. Prediction Generation Timing

**When to generate predictions?**

**Option A: Beginning of each cycle**
```python
for cycle in range(num_cycles):
    gen_data = generate_predictions()
    train_on_augmented_data(real + gen_data)
```

**Option B: After each sub-trainer**
```python
for cycle in range(num_cycles):
    physical_preds = generate_physical()
    train_synthetic(real + physical_preds)
    
    synthetic_preds = generate_synthetic()
    train_physical(real + synthetic_preds)
```

**Question:** When to generate? (Option B is current design)

### 5. Model State Management

**After training, we need updated models:**

**For Synthetic Model:**
```python
# Option A: Extract from trainer
synthetic_model = self.synthetic_trainer.get_model()

# Option B: Trainer saves model, we reload
model_path = self.synthetic_trainer.save_model()
synthetic_model = load_model(model_path)

# Option C: HybridTrainer maintains reference
# (If trainer modifies model in-place)
```

**For Physical Model:**
```python
# Option A: Extract parameters
params = self.physical_trainer.get_optimized_parameters()
self.physical_model.update_parameters(params)

# Option B: Trainer updates model directly
# (If model is passed by reference)
```

**Question:** How to handle model state updates?

### 6. Validation Strategy

**When and how to validate?**

- After each cycle?
- After all cycles?
- Validate both models separately?
- Validate combined performance?

**Question:** What validation strategy?

---

## Next Steps

### Phase 1: Refactor Trainer Base Classes
- [ ] Modify `TensorTrainer.__init__` - remove data_manager, add model parameter
- [ ] Modify `TensorTrainer.train` - add data_source parameter
- [ ] Modify `FieldTrainer.__init__` - remove data_manager, add model and learnable_params
- [ ] Modify `FieldTrainer.train` - add data_source parameter
- [ ] Update `SyntheticTrainer` to match new base class signature
- [ ] Update `PhysicalTrainer` to match new base class signature
- [ ] Ensure backward compatibility (add factory methods if needed)

### Phase 2: Data Source Infrastructure
- [ ] Create `AugmentedDataset` class (combines real + generated with weights)
- [ ] Implement `_create_augmented_dataloader()` in HybridTrainer
- [ ] Create `AugmentedFieldDataSource` class (for physical trainer)
- [ ] Implement `_create_augmented_field_source()` in HybridTrainer
- [ ] Test data sources independently

### Phase 3: HybridTrainer Core Implementation
- [ ] Implement `_generate_physical_predictions()`
- [ ] Implement `_generate_synthetic_predictions()`
- [ ] Implement `_convert_fields_to_tensors()`
- [ ] Implement `_convert_tensors_to_fields()`
- [ ] Implement complete `train()` loop
- [ ] Add logging and metrics tracking

### Phase 4: Testing & Validation
- [ ] Unit tests for augmented datasets
- [ ] Unit tests for prediction generation
- [ ] Unit tests for field-tensor conversion
- [ ] Integration test: single cycle training
- [ ] Integration test: full multi-cycle training
- [ ] Validate against standalone trainers

### Phase 5: Optimization & Documentation
- [ ] Profile performance bottlenecks
- [ ] Optimize data loading and conversion
- [ ] Add progress bars and detailed logging
- [ ] Document usage examples
- [ ] Create tutorial notebook

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| Nov 2, 2025 | Use composition architecture | More modular, reusable, less code duplication |
| Nov 2, 2025 | HybridTrainer manages data augmentation | Sub-trainers remain agnostic to data source |
| Nov 2, 2025 | Weighted loss: L = L_real + alpha*L_gen | Clear mathematical formulation |
| Nov 2, 2025 | Sub-trainer epochs configurable in YAML | Flexibility for different training regimes |
| Nov 2, 2025 | HybridTrainer handles field-tensor conversion | Clean separation, sub-trainers unaware |
| Nov 3, 2025 | **Explicit data passing to train()** | No internal data management, maximum flexibility |
| Nov 3, 2025 | **Persistent trainers with dynamic data** | Create once, reuse with different data sources |
| Nov 3, 2025 | **Weights as part of data contract** | Natural handling, no special trainer logic |
| Nov 3, 2025 | **Remove data_manager from trainer init** | Cleaner separation of concerns |

---

## References

- [HYCO Implementation Strategy](./HYCO_IMPLEMENTATION_STRATEGY.md)
- [HYCO Required Changes](./HYCO_REQUIRED_CHANGES.md)
- [Field Conversion Documentation](./FIELD_CONVERSION.md)

---

**Document Status:** Design Phase - Ready for Implementation  
**Last Updated:** November 3, 2025
