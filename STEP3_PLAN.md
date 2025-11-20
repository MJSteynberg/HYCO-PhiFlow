# Step 3: PhiML Data Pipeline Migration - Detailed Plan

## Goal
Migrate the entire data pipeline to use **pure PhiML tensors** with no PyTorch dependency. Data should be cached and loaded as PhiML tensors using PhiML's native save/load mechanisms.

## Current State Analysis

### Current Pipeline (PyTorch-based)
```
PhiFlow Scene → Fields → torch.Tensor → torch.save() → TensorDataset → PyTorch DataLoader
```

### Issues with Current Approach
1. **PyTorch dependency**: Uses `torch.Tensor` and `torch.save()`
2. **Conversion overhead**: Must convert PhiML → PyTorch at model boundary
3. **Memory inefficient**: Loads entire tensors into memory
4. **Loss of metadata**: PhiML's named dimensions lost in conversion

## Target State

### New Pipeline (PhiML-native)
```
PhiFlow Scene → Fields → PhiML Tensor → phiml.math.save() → PhiMLDataset → PhiML iterator
```

### Benefits
1. **No conversions**: Direct PhiML tensor flow from data to model
2. **Named dimensions**: Preserved throughout pipeline
3. **Disk-backed tensors**: PhiML's automatic caching with memory limits
4. **Cleaner code**: No torch dependency in data layer

## Implementation Strategy

### Phase 1: Create PhiML DataManager
**File**: `src/data/phiml_data_manager.py`

Key changes from current `DataManager`:
- Replace `torch.save()` → `phiml.math.save()`
- Replace `torch.load()` → `phiml.math.load()`
- Store tensors with named dimensions
- Use PhiML's HDF5 backend for caching

```python
def save_phiml_cache(self, sim_index: int, field_data: Dict[str, Tensor]):
    """Save PhiML tensors with named dimensions."""
    cache_path = self.get_cached_path(sim_index)

    # PhiML's native save (preserves named dimensions)
    phimath.save(field_data, str(cache_path))

def load_phiml_cache(self, sim_index: int) -> Dict[str, Tensor]:
    """Load PhiML tensors with named dimensions."""
    cache_path = self.get_cached_path(sim_index)

    # PhiML's native load (restores named dimensions)
    return phimath.load(str(cache_path))
```

### Phase 2: Create PhiML Dataset
**File**: `src/data/phiml_dataset.py`

Replace PyTorch `Dataset` with PhiML's approach:
- No `__len__` and `__getitem__` (PyTorch specific)
- Use generator pattern or PhiML's parallel iteration
- Return PhiML tensors directly

```python
class PhiMLDataset:
    """
    Pure PhiML dataset that yields PhiML tensors.
    No PyTorch dependency.
    """

    def iterate_batches(self, batch_size: int):
        """
        Yield batches of PhiML tensors.

        Yields:
            Dict with 'initial_state' and 'targets' as PhiML tensors
        """
        for indices in self._batch_indices(batch_size):
            batch_data = self._load_batch(indices)
            yield batch_data

    def _load_batch(self, indices):
        """Load multiple samples and stack into batch dimension."""
        samples = [self._load_sample(idx) for idx in indices]

        # Stack using PhiML's batch dimension
        initial_states = phimath.stack(
            [s['initial'] for s in samples],
            batch_dim('batch')
        )
        targets = phimath.stack(
            [s['target'] for s in samples],
            batch_dim('batch', 'time')
        )

        return {'initial_state': initial_states, 'targets': targets}
```

### Phase 3: Update Trainer Integration
**File**: `src/training/synthetic/phiml_trainer.py`

Remove tensor conversion layer:
```python
# OLD (Step 2):
def _train_batch(self, batch):
    initial_state, targets = batch  # PyTorch tensors
    initial_phiml = torch_to_phiml(initial_state)  # Convert
    targets_phiml = torch_to_phiml(targets)  # Convert
    # ... train ...

# NEW (Step 3):
def _train_batch(self, batch):
    # Already PhiML tensors from dataset!
    initial_state = batch['initial_state']
    targets = batch['targets']
    # ... train ...
```

### Phase 4: Cache Format Migration

Current cache format (PyTorch):
```python
{
    'tensor_data': {
        'velocity': torch.Tensor[C, T, H, W]
    },
    'metadata': {...}
}
```

New cache format (PhiML):
```python
{
    'velocity': Tensor(time=T, spatial(x=H, y=W), vector=C),
    'metadata': {...}
}
```

## Implementation Details

### 1. Field → PhiML Tensor Conversion

```python
def field_to_phiml_tensor(field: Field) -> Tensor:
    """
    Convert PhiFlow Field to PhiML tensor with named dimensions.

    Args:
        field: PhiFlow Field object

    Returns:
        PhiML Tensor with proper dimension naming
    """
    # PhiFlow Fields already use PhiML tensors internally!
    # Just ensure proper dimension names
    values = field.values

    # Rename dimensions if needed to match our convention
    # (time, x, y, vector) for velocity fields
    return values.rename_dims({...})
```

### 2. Disk-Backed Tensors with Memory Limits

```python
from phiml.dataclasses import parallel_compute

# PhiML can automatically cache to disk with memory limits
# This is useful for large datasets
def create_cached_dataset(config):
    dataset = PhiMLDataset(config)

    # Let PhiML handle disk caching automatically
    parallel_compute(
        dataset,
        [PhiMLDataset.all_samples],
        'batch',
        cache_dir='data/cache',
        memory_limit=2048  # MB
    )

    return dataset
```

### 3. Batch Iteration

```python
def iterate_epochs(dataset, num_epochs, batch_size):
    """
    Iterate through dataset for training.

    No PyTorch DataLoader needed!
    """
    for epoch in range(num_epochs):
        for batch in dataset.iterate_batches(batch_size):
            # batch contains PhiML tensors
            yield batch
```

## Migration Steps

### Step 3.1: Create PhiML DataManager
- [ ] Create `phiml_data_manager.py`
- [ ] Implement `save_phiml_cache()` using `phiml.math.save()`
- [ ] Implement `load_phiml_cache()` using `phiml.math.load()`
- [ ] Test save/load preserves named dimensions

### Step 3.2: Create PhiML Dataset
- [ ] Create `phiml_dataset.py`
- [ ] Implement generator-based iteration
- [ ] Implement batch stacking with named dimensions
- [ ] Handle sliding windows for trajectories

### Step 3.3: Remove Tensor Conversions
- [ ] Update trainer to accept PhiML tensors directly
- [ ] Remove `torch_to_phiml()` and `phiml_to_torch()` functions
- [ ] Update tests

### Step 3.4: Update run.py
- [ ] Replace PyTorch DataLoader with PhiML iterator
- [ ] Update factory to create PhiML dataset
- [ ] Test end-to-end training

### Step 3.5: Cache Migration (Optional)
- [ ] Create script to convert existing PyTorch caches to PhiML format
- [ ] Or regenerate caches in PhiML format

## Testing Strategy

### Unit Tests
1. **PhiML DataManager**
   - Save PhiML tensor → Load → Verify dimensions match
   - Test with different field types
   - Test metadata preservation

2. **PhiML Dataset**
   - Iterate batches → Verify shapes
   - Test batch stacking
   - Test sliding window extraction

3. **Integration Test**
   - Full training loop with PhiML data
   - Compare results with PyTorch version

## Key PhiML APIs to Use

### Saving/Loading
```python
from phiml import math as phimath

# Save
phimath.save(tensor, 'path/to/file.hdf5')

# Load
tensor = phimath.load('path/to/file.hdf5')
```

### Stacking Batches
```python
from phiml.math import batch, stack

tensors = [tensor1, tensor2, tensor3]
batched = stack(tensors, batch('batch'))
```

### Named Dimension Access
```python
# Indexing by name
velocity_field.time[0]  # First timestep
velocity_field.x[10:20]  # Spatial slice

# Shape access
num_timesteps = velocity_field.shape.get_size('time')
```

## Potential Challenges

1. **PhiML save/load format**: May need to wrap in dict for compatibility
2. **Batch iteration**: PhiML doesn't have DataLoader equivalent, need custom iterator
3. **Augmentation**: Need to handle augmented data in PhiML format
4. **Memory management**: May need to implement lazy loading for large datasets

## Success Criteria

- [ ] No `torch.Tensor` or `torch.save()` in data layer
- [ ] No tensor conversions in training loop
- [ ] All data operations use PhiML named dimensions
- [ ] Training works end-to-end with PhiML pipeline
- [ ] Memory usage is reasonable (disk-backed if needed)
- [ ] Code is cleaner and more maintainable

## References

- PhiML Caching Example: `references/cached_parallel_example.py`
- PhiML Tensors Guide: `references/tensors.py`
- PhiML save/load: Check `phiml.math.save()` and `phiml.math.load()`
- Parallel compute: `phiml.dataclasses.parallel_compute()`
