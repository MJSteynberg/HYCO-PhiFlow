# Phase 1 Migration Guide: Base Trainer Refactoring

**Date:** November 3, 2025  
**Status:** Base classes updated, subclasses need migration

---

## Changes Summary

### TensorTrainer Changes

**Old Signature:**
```python
class TensorTrainer(AbstractTrainer):
    def __init__(self, config: Dict[str, Any]):
        # Model created internally via _create_model()
        # Data loaders created internally via _create_data_loaders()
        pass
    
    def train(self) -> Dict:
        # Uses self.train_loader and self.val_loader
        pass
```

**New Signature:**
```python
class TensorTrainer(AbstractTrainer):
    def __init__(self, config: Dict[str, Any], model: nn.Module):
        # Model passed in, optimizer created automatically
        pass
    
    def train(self, data_source: DataLoader, num_epochs: int) -> Dict:
        # Data passed explicitly
        pass
```

**Methods Changed:**
- ❌ Removed: `_create_model()` - model now passed to `__init__`
- ❌ Removed: `_create_data_loaders()` - data now passed to `train()`
- ❌ Removed: `_train_epoch()` - replaced with `_train_epoch_with_data()`
- ✅ Added: `_create_optimizer()` - creates optimizer for model
- ✅ Added: `_train_epoch_with_data(data_source)` - new abstract method
- ❌ Removed: All validation methods (will be handled externally)

---

### FieldTrainer Changes

**Old Signature:**
```python
class FieldTrainer(AbstractTrainer):
    def __init__(self, config: Dict[str, Any]):
        # Model created internally via _create_model()
        # DataManager created internally via _create_data_manager()
        pass
    
    def train(self) -> Dict:
        # Uses self.data_manager for data
        # Uses math.minimize for optimization
        pass
```

**New Signature:**
```python
class FieldTrainer(AbstractTrainer):
    def __init__(
        self, 
        config: Dict[str, Any],
        model: Any,
        learnable_params: List[torch.nn.Parameter]
    ):
        # Model and params passed in, optimizer created automatically
        pass
    
    def train(self, data_source: Iterable, num_epochs: int) -> Dict:
        # Data passed explicitly
        # Uses sample-by-sample iteration, not math.minimize
        pass
```

**Methods Changed:**
- ❌ Removed: `_create_model()` - model now passed to `__init__`
- ❌ Removed: `_create_data_manager()` - data now passed to `train()`
- ❌ Removed: `_setup_optimization()` - simpler approach with epochs
- ✅ Added: `_create_optimizer()` - creates optimizer for parameters
- ✅ Added: `_train_sample(initial_fields, target_fields)` - new abstract method
- ❌ Removed: `_load_ground_truth()`, `_run_simulation()`, `_compute_loss()`, `_update_model_parameters()`

---

## Migration Steps for SyntheticTrainer

### Step 1: Update `__init__` Signature

**Before:**
```python
class SyntheticTrainer(TensorTrainer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Setup data
        self._create_data_loaders()
        
        # Setup model
        self.model = self._create_model()
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
```

**After:**
```python
class SyntheticTrainer(TensorTrainer):
    def __init__(self, config: Dict[str, Any], model: nn.Module):
        # Pass model to parent
        super().__init__(config, model)
        
        # Store config details
        self.data_config = config["data"]
        self.trainer_config = config["trainer_params"]
        
        # NO data loader creation here!
        # NO model creation here!
        # Optimizer already created by parent
        
        # Can still create loss function, scheduler, etc.
        self.loss_fn = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config["trainer_params"]["epochs"]  # Note: no len(train_loader)
        )
```

### Step 2: Remove Data Loader Creation

**Before:**
```python
def _create_data_loaders(self):
    """Creates DataManager and train/validation DataLoaders."""
    # ... lots of code to create data_manager, datasets, loaders ...
    self.train_loader = DataLoader(...)
    self.val_loader = DataLoader(...)
```

**After:**
```python
# DELETE THIS METHOD ENTIRELY
# Data loaders will be created externally and passed to train()
```

### Step 3: Remove Model Creation

**Before:**
```python
def _create_model(self) -> nn.Module:
    """Create and return the UNet model."""
    # ... model creation code ...
    return model
```

**After:**
```python
# DELETE THIS METHOD ENTIRELY
# Model is now passed to __init__
```

### Step 4: Update Training Method

**Before:**
```python
def _train_epoch(self) -> float:
    """Train for one epoch using self.train_loader."""
    self.model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    for batch in self.train_loader:  # Uses internal loader
        inputs, targets = batch
        # ... training logic ...
    
    return epoch_loss / num_batches
```

**After:**
```python
def _train_epoch_with_data(self, data_source: DataLoader) -> float:
    """Train for one epoch using provided data_source."""
    self.model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    for batch in data_source:  # Uses provided data_source
        inputs, targets = batch  # NO weights in tuple!
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        self.optimizer.zero_grad()
        predictions = self.model(inputs)
        loss = self.loss_fn(predictions, targets)
        loss.backward()
        self.optimizer.step()
        
        # Update scheduler (if using)
        if hasattr(self, 'scheduler'):
            self.scheduler.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    return epoch_loss / num_batches if num_batches > 0 else 0.0
```

### Step 5: Update Usage Pattern

**Before (in run.py or elsewhere):**
```python
# Old usage
trainer = SyntheticTrainer(config)  # Creates everything internally
results = trainer.train()  # Uses internal data
```

**After (in run.py or elsewhere):**
```python
# New usage

# 1. Create model
from src.models import ModelRegistry
model_registry = ModelRegistry(config)
synthetic_model = model_registry.get_synthetic_model()

# 2. Create data
from src.data import DataManager, HybridDataset
from torch.utils.data import DataLoader

data_manager = DataManager(
    raw_data_dir=config["data"]["data_dir"],
    cache_dir=config["data"]["cache_dir"],
    config=config
)

train_dataset = HybridDataset(
    data_manager=data_manager,
    sim_indices=config["trainer_params"]["train_sim"],
    field_names=config["data"]["fields"],
    num_frames=50,
    num_predict_steps=config["trainer_params"]["num_predict_steps"],
    use_sliding_window=True,
    return_fields=False  # Tensors for synthetic
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["trainer_params"]["batch_size"],
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# 3. Create trainer with model
trainer = SyntheticTrainer(config, synthetic_model)

# 4. Train with explicit data
results = trainer.train(train_loader, num_epochs=100)
```

---

## Migration Steps for PhysicalTrainer

### Step 1: Update `__init__` Signature

**Before:**
```python
class PhysicalTrainer(FieldTrainer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Setup data manager
        self.data_manager = self._create_data_manager()
        
        # Setup model
        self.model = self._setup_physical_model()
        
        # Setup learnable params
        self.initial_guesses = self._get_initial_guesses()
```

**After:**
```python
class PhysicalTrainer(FieldTrainer):
    def __init__(
        self, 
        config: Dict[str, Any],
        model: PhysicalModel,
        learnable_params: List[torch.nn.Parameter]
    ):
        # Pass model and params to parent
        super().__init__(config, model, learnable_params)
        
        # Store config details
        self.data_config = config["data"]
        self.model_config = config["model"]["physical"]
        self.trainer_config = config["trainer_params"]
        
        # NO data manager creation here!
        # NO model creation here!
        # Optimizer already created by parent
```

### Step 2: Remove Data Manager Creation

**Before:**
```python
def _create_data_manager(self) -> DataManager:
    """Create DataManager for loading cached field data."""
    # ... creation code ...
    return data_manager
```

**After:**
```python
# DELETE THIS METHOD ENTIRELY
# DataManager will be created externally
```

### Step 3: Remove Model Setup

**Before:**
```python
def _setup_physical_model(self) -> PhysicalModel:
    """Setup physical model with initial parameter guesses."""
    # ... setup code ...
    return model

def _get_initial_guesses(self) -> Dict:
    """Get initial parameter guesses."""
    # ... code ...
    return guesses
```

**After:**
```python
# DELETE THESE METHODS ENTIRELY
# Model and params are now passed to __init__
```

### Step 4: Implement Training Method

**Before:**
```python
# Used math.minimize with custom loss function
def train(self) -> Dict:
    # Complex optimization-based approach
    pass
```

**After:**
```python
def _train_sample(
    self, 
    initial_fields: Dict[str, Field], 
    target_fields: Dict[str, Field]
) -> float:
    """
    Train on a single sample.
    
    Args:
        initial_fields: Dict of Fields at t=0
        target_fields: Dict of Lists of Fields for t=1..T
    
    Returns:
        Loss value
    """
    self.optimizer.zero_grad()
    
    # Run simulation from initial state
    predicted_fields = self.model.predict_trajectory(
        initial_fields,
        steps=len(target_fields[next(iter(target_fields))]),
        dt=self.model_config["dt"]
    )
    
    # Compute loss
    from phi.field import l2_loss
    total_loss = 0.0
    
    for field_name in target_fields.keys():
        pred_list = predicted_fields[field_name]
        target_list = target_fields[field_name]
        
        for pred, target in zip(pred_list, target_list):
            total_loss = total_loss + l2_loss(pred - target)
    
    # Backward and optimize
    total_loss.backward()
    self.optimizer.step()
    
    return float(total_loss)
```

### Step 5: Update Usage Pattern

**Before:**
```python
# Old usage
trainer = PhysicalTrainer(config)  # Creates everything internally
results = trainer.train()  # Uses internal data
```

**After:**
```python
# New usage

# 1. Create model and learnable params
from src.models import ModelRegistry
model_registry = ModelRegistry(config)
physical_model = model_registry.get_physical_model()

# Get learnable parameters
learnable_params = []
for param_config in config["trainer_params"]["learnable_parameters"]:
    param_name = param_config["name"]
    param = getattr(physical_model, param_name)
    param.requires_grad = True
    learnable_params.append(param)

# 2. Create data
from src.data import DataManager, HybridDataset

data_manager = DataManager(
    raw_data_dir=config["data"]["data_dir"],
    cache_dir=config["data"]["cache_dir"],
    config=config
)

field_dataset = HybridDataset(
    data_manager=data_manager,
    sim_indices=config["trainer_params"]["train_sim"],
    field_names=config["data"]["fields"],
    num_frames=50,
    num_predict_steps=config["trainer_params"]["num_predict_steps"],
    use_sliding_window=True,  # NOW ALWAYS TRUE!
    return_fields=True  # Fields for physical
)

# 3. Create trainer with model and params
trainer = PhysicalTrainer(config, physical_model, learnable_params)

# 4. Train with explicit data
results = trainer.train(field_dataset, num_epochs=5)
```

---

## Testing Checklist

After migration, test each trainer:

### For SyntheticTrainer:
- [ ] Model is passed to `__init__` correctly
- [ ] Training runs without errors with external DataLoader
- [ ] Loss decreases over epochs
- [ ] Optimizer state is preserved across multiple `train()` calls
- [ ] No references to `self.train_loader` or `self.val_loader` remain
- [ ] Checkpoint saving/loading still works

### For PhysicalTrainer:
- [ ] Model and learnable_params passed to `__init__` correctly
- [ ] Training runs without errors with external dataset
- [ ] Loss decreases over epochs
- [ ] Gradients flow to learnable parameters
- [ ] Optimizer state is preserved
- [ ] No references to `self.data_manager` remain
- [ ] Works with sliding window (`use_sliding_window=True`)

### Integration Tests:
- [ ] Can create trainer once, call `train()` multiple times
- [ ] Optimizer state carries over between calls
- [ ] Can pass different datasets to same trainer
- [ ] Memory usage is reasonable

---

## Common Pitfalls

1. **Forgetting to remove data loader creation:**
   - ❌ Don't call `self._create_data_loaders()` in `__init__`
   - ✅ Data is created externally and passed to `train()`

2. **Expecting weights in batch:**
   - ❌ `inputs, targets, weights = batch`
   - ✅ `inputs, targets = batch` (NO weights!)

3. **Using old validation methods:**
   - ❌ Calling `self._validate_epoch()`
   - ✅ Validation will be handled externally or not at all initially

4. **Forgetting to move data to device:**
   - ✅ Always `inputs = inputs.to(self.device)`

5. **Creating optimizer incorrectly:**
   - ❌ Creating optimizer in `__init__` of subclass
   - ✅ Optimizer is created by parent's `_create_optimizer()

6. **Hardcoding epoch count:**
   - ❌ Using `self.epochs` from config
   - ✅ Using `num_epochs` parameter passed to `train()`

---

## Next Steps

1. ✅ Phase 1 Complete: Base classes refactored
2. ⏩ **Current Step:** Migrate SyntheticTrainer
3. ⏩ **Next:** Migrate PhysicalTrainer  
4. ⏩ **Then:** Update run.py and factories
5. ⏩ **Finally:** Test hybrid training workflow

---

**Status:** Base trainers updated, ready to migrate subclasses  
**Last Updated:** November 3, 2025
