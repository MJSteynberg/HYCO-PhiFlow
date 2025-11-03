# Phase 1 Migration - COMPLETE ✅

## Summary

Successfully migrated both `SyntheticTrainer` and `PhysicalTrainer` to use the new Phase 1 API with explicit data passing and external model/parameter management.

**Status**: ✅ All base trainers and subclasses migrated
**Errors**: ✅ None - all files compile successfully

---

## Changes Made

### 1. SyntheticTrainer Migration

**File**: `src/training/synthetic/trainer.py`

#### Signature Changes

**Before**:
```python
def __init__(self, config: Dict[str, Any]):
    super().__init__(config)
    # Internal model creation
    self.model = self._create_model()
    # Internal data loader creation
    self._create_data_loaders()
```

**After**:
```python
def __init__(self, config: Dict[str, Any], model: nn.Module):
    super().__init__(config, model)
    # Model passed in externally
    # No internal data loader creation
```

#### Removed Methods

- ❌ `_create_data_loaders()` - Data now passed via `train(data_source, num_epochs)`
- ❌ `_create_model()` - Model now passed to constructor
- ❌ `_train_epoch()` - Replaced with `_train_epoch_with_data(data_source)`
- ❌ `_validate_epoch_rollout()` - Validation handled externally

#### Added Methods

- ✅ `_train_epoch_with_data(data_source)` - New signature accepting DataLoader
  ```python
  def _train_epoch_with_data(self, data_source):
      """Trains one epoch using provided data source."""
      self.model.train()
      total_loss = 0.0
      
      for batch_idx, batch in enumerate(data_source):
          self.optimizer.zero_grad()
          avg_rollout_loss = self._compute_batch_loss(batch)
          avg_rollout_loss.backward()
          self.optimizer.step()
          self.scheduler.step()
          total_loss += avg_rollout_loss.item()
      
      return total_loss / len(data_source)
  ```

#### Removed Configuration Attributes

These were removed from `__init__` as they're no longer needed:
- ❌ `self.checkpoint_path`
- ❌ `self.learning_rate`
- ❌ `self.epochs`
- ❌ `self.batch_size`
- ❌ `self.train_sim`
- ❌ `self.val_sim`
- ❌ `self.use_sliding_window`
- ❌ `self.num_frames`
- ❌ `self.train_loader`
- ❌ `self.val_loader`
- ❌ `self.optimizer` (now created in base class)
- ❌ `self.scheduler`

#### Retained Attributes

These are still needed for loss computation:
- ✅ `self.field_names`
- ✅ `self.dynamic_fields`
- ✅ `self.static_fields`
- ✅ `self.channel_map`
- ✅ `self.loss_fn`
- ✅ `self.memory_monitor` (optional)

---

### 2. PhysicalTrainer Migration

**File**: `src/training/physical/trainer.py`

#### Signature Changes

**Before**:
```python
def __init__(self, config: Dict[str, Any]):
    super().__init__(config)
    # Internal model creation
    self.model = self._setup_physical_model()
    # Internal data manager creation
    self.data_manager = self._create_data_manager()
    # Internal parameter initialization
    self.initial_guesses = self._get_initial_guesses()
```

**After**:
```python
def __init__(self, config: Dict[str, Any], model, learnable_params: List[Tensor]):
    super().__init__(config, model, learnable_params)
    # Model and params passed in externally
    # No internal data manager creation
```

#### Removed Methods

- ❌ `_create_data_manager()` - Data now passed via `train(data_source, num_epochs)`
- ❌ `_create_model()` - Model now passed to constructor
- ❌ `_setup_physical_model()` - Model setup handled externally
- ❌ `_get_initial_guesses()` - Parameters now passed to constructor
- ❌ `_load_ground_truth_data(sim_index)` - Data loading handled externally
- ❌ `_setup_optimization()` - Optimization setup moved to `_train_sample`
- ❌ `train()` - Replaced with base class `train(data_source, num_epochs)`

#### Added Methods

- ✅ `_train_sample(initial_fields, target_fields)` - New signature for single sample training
  ```python
  def _train_sample(self, initial_fields: Dict[str, Any], target_fields: Dict[str, Any]):
      """
      Trains on a single sample by optimizing learnable parameters.
      
      Returns:
          Tuple of (optimized_parameters, final_loss)
      """
      # Setup loss function
      def loss_function(*learnable_tensors):
          # Run forward simulation
          current_state = initial_fields
          for step in range(self.num_predict_steps):
              current_state = self.model.step(current_state)
              # Compute loss against targets
          return final_loss
      
      # Run optimization
      solve_params = math.Solve(...)
      estimated_tensors = math.minimize(loss_function, solve_params)
      
      return estimated_tensors, final_loss
  ```

#### Removed Configuration Attributes

- ❌ `self.train_sims`
- ❌ `self.num_epochs`
- ❌ `self.learnable_params_config`
- ❌ `self.data_manager`
- ❌ `self.initial_guesses`
- ❌ `self.true_pde_params`

#### Retained Attributes

These are still needed for optimization:
- ✅ `self.num_predict_steps`
- ✅ `self.gt_fields`
- ✅ `self.method`
- ✅ `self.abs_tol`
- ✅ `self.max_iterations`
- ✅ `self.suppress_convergence`
- ✅ `self.memory_monitor` (optional)

---

## New Usage Pattern

### Creating and Using SyntheticTrainer

```python
from src.models import ModelRegistry
from src.data import DataManager, HybridDataset
from src.training.synthetic import SyntheticTrainer
from torch.utils.data import DataLoader

# 1. Create model externally
model = ModelRegistry.get_synthetic_model("UNet", config=config["model"]["synthetic"])

# 2. Create trainer with model
trainer = SyntheticTrainer(config, model)

# 3. Create data externally
data_manager = DataManager(...)
dataset = HybridDataset(
    data_manager=data_manager,
    sim_indices=[0, 1, 2],
    field_names=config["data"]["fields"],
    num_frames=None,  # Load all frames
    num_predict_steps=config["trainer_params"]["num_predict_steps"],
    dynamic_fields=dynamic_fields,
    static_fields=static_fields,
    use_sliding_window=True,  # ALWAYS True for synthetic
)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. Train with explicit data
trainer.train(data_source=data_loader, num_epochs=100)

# 5. Reuse trainer with different data (e.g., augmented)
augmented_loader = DataLoader(augmented_dataset, batch_size=32, shuffle=True)
trainer.train(data_source=augmented_loader, num_epochs=50)
```

### Creating and Using PhysicalTrainer

```python
from src.models import ModelRegistry
from src.data import DataManager, HybridDataset
from src.training.physical import PhysicalTrainer
from phi.math import math

# 1. Create model and parameters externally
model = ModelRegistry.get_physical_model("BurgersPDE", config=config["model"]["physical"])

learnable_params = [
    math.tensor(0.01, name="viscosity"),  # Initial guess
]

# 2. Create trainer with model and params
trainer = PhysicalTrainer(config, model, learnable_params)

# 3. Create data externally (using HybridDataset with return_fields=True)
data_manager = DataManager(...)
dataset = HybridDataset(
    data_manager=data_manager,
    sim_indices=[0],
    field_names=config["data"]["fields"],
    num_frames=config["trainer_params"]["num_predict_steps"] + 1,
    num_predict_steps=config["trainer_params"]["num_predict_steps"],
    return_fields=True,  # Return PhiFlow fields, not tensors
    use_sliding_window=True,  # ALWAYS True for physical
)

# 4. Train with explicit data
trainer.train(data_source=dataset, num_epochs=1)  # Physical typically single epoch

# 5. Reuse trainer with different data (e.g., augmented)
augmented_dataset = HybridDataset(...)
trainer.train(data_source=augmented_dataset, num_epochs=1)
```

---

## Key Design Principles

### 1. Explicit Data Passing
- **Before**: Trainers created data loaders internally
- **After**: Data passed explicitly via `train(data_source, num_epochs)`
- **Benefit**: Maximum flexibility for hybrid training

### 2. External Model Management
- **Before**: Trainers created and managed models internally
- **After**: Models passed to constructor, managed externally
- **Benefit**: Shared models between trainers, easier checkpoint management

### 3. Persistent Trainers
- **Before**: Create new trainer instance per training run
- **After**: Create once, reuse with different data sources
- **Benefit**: Preserves optimizer state, enables progressive training

### 4. Sliding Window Always Enabled
- **Both trainers**: Now always use `use_sliding_window=True` for HybridDataset
- **Rationale**: Maximum data augmentation, consistent with hybrid design
- **Implementation**: Specified when creating HybridDataset externally

### 5. Simplified Responsibilities
- **SyntheticTrainer**: Focus on PyTorch training loop with autoregressive rollout
- **PhysicalTrainer**: Focus on PhiFlow optimization with math.minimize
- **Both**: No longer handle data loading, model creation, or validation

---

## Validation

### Compilation Status
✅ **SyntheticTrainer**: No errors
✅ **PhysicalTrainer**: No errors

### Base Classes
✅ **TensorTrainer**: Refactored and tested
✅ **FieldTrainer**: Refactored and tested
✅ **AbstractTrainer**: Unchanged, working

---

## Next Steps

### Immediate (Phase 2)
1. **Update Factory Methods** - Modify trainer creation in `src/factories/`
2. **Update run.py** - Adapt main execution script to new API
3. **Create DataLoader Utilities** - Helper functions for creating data sources
4. **Test Standalone Training** - Verify SyntheticTrainer and PhysicalTrainer work independently

### Future (Phase 3+)
1. **Implement AugmentedDataset** - Count-based data augmentation classes
2. **Implement CacheManager** - Cache organization and cleanup
3. **Implement HybridTrainer** - Orchestrate hybrid training workflow
4. **Integration Testing** - Full end-to-end hybrid training tests

---

## Migration Checklist

- [x] Phase 1: Refactor TensorTrainer base class
- [x] Phase 1: Refactor FieldTrainer base class
- [x] Phase 1: Migrate SyntheticTrainer
- [x] Phase 1: Migrate PhysicalTrainer
- [x] Phase 1: Verify no compilation errors
- [x] Phase 1: Document migration
- [ ] Phase 2: Update factory methods
- [ ] Phase 2: Update run.py
- [ ] Phase 2: Test standalone training
- [ ] Phase 3: Implement data augmentation
- [ ] Phase 3: Implement HybridTrainer
- [ ] Phase 4: Integration testing

---

## Notes

### Important Considerations

1. **Optimizer Creation**: Now handled in base class `_create_optimizer()`, but subclasses can override
2. **Scheduler**: SyntheticTrainer still creates its own scheduler - consider moving to base class in Phase 2
3. **Memory Monitoring**: Retained in both trainers, works with new API
4. **Loss Functions**: `_compute_batch_loss()` and `_train_sample()` handle domain-specific logic

### Backward Compatibility

⚠️ **Breaking Changes**: This migration is **NOT backward compatible**
- Old code calling `SyntheticTrainer(config)` will **fail**
- Old code expecting `trainer.train()` (no args) will **fail**
- Factory methods and run.py **must be updated** before this works

### Migration Path for External Code

If you have code outside this repo that uses these trainers:

```python
# OLD (broken after Phase 1):
trainer = SyntheticTrainer(config)
trainer.train()

# NEW (Phase 1):
model = ModelRegistry.get_synthetic_model("UNet", config=config["model"]["synthetic"])
trainer = SyntheticTrainer(config, model)
data_loader = create_data_loader(config)  # External utility
trainer.train(data_source=data_loader, num_epochs=config["trainer_params"]["epochs"])
```

---

## Conclusion

✅ Phase 1 migration is **COMPLETE**

Both trainers successfully refactored with:
- ✅ External model management
- ✅ Explicit data passing
- ✅ Persistent trainer architecture
- ✅ No compilation errors
- ✅ Documented usage patterns

**Ready to proceed to Phase 2**: Factory method and execution script updates.
