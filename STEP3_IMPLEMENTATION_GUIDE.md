# Step 3 Implementation Guide: PhiML Data Pipeline

## Executive Summary

Based on testing, PhiML's `math.save()` and `math.load()` work perfectly for our use case:
- ✅ Preserves named dimensions (time, x, y, vector)
- ✅ Works with dicts of tensors
- ✅ Stores metadata
- ✅ Uses `.npz` format (NumPy archive)
- ✅ Efficient and fast

## Implementation Architecture

### 1. Data Flow Comparison

**OLD (Step 2 - Current)**:
```
PhiFlow Scene → Field → PyTorch Tensor → torch.save(.pt) → TensorDataset
   ↓
PyTorch DataLoader → torch.Tensor batch
   ↓
torch_to_phiml() conversion → PhiML Tensor → Model → Training
```

**NEW (Step 3 - Target)**:
```
PhiFlow Scene → Field → PhiML Tensor → phimath.save(.npz) → PhiMLDataset
   ↓
PhiML iterator → PhiML Tensor batch → Model → Training
                   (NO CONVERSION!)
```

### 2. File Structure

```
src/data/
├── phiml_data_manager.py      # NEW: PhiML-native caching
├── phiml_dataset.py            # NEW: PhiML-native dataset
├── data_manager.py             # OLD: Keep for backward compat (optional)
├── tensor_dataset.py           # OLD: Keep for backward compat (optional)
└── ...
```

### 3. Cache Format Changes

**Current Format** (`.pt` files):
```python
{
    'tensor_data': {
        'velocity': torch.Tensor([C, T, H, W])  # PyTorch format
    },
    'metadata': {...}
}
```

**New Format** (`.npz` files):
```python
{
    'velocity': Tensor(time=T, x=H, y=W, vector=C),  # PhiML format with named dims
    'metadata': {...}
}
```

## Key Implementation Details

### A. PhiML DataManager

**File**: `src/data/phiml_data_manager.py`

```python
class PhiMLDataManager:
    """
    Manages PhiML tensor caching for training data.
    """

    def get_cached_path(self, sim_index: int) -> Path:
        """Returns path with .npz extension."""
        return self.cache_dir / f"sim_{sim_index:06d}.npz"

    def save_cache(self, sim_index: int, data: Dict[str, Tensor]):
        """
        Save PhiML tensors to cache.

        Args:
            data: Dict with fields as PhiML tensors + metadata
                  {'velocity': Tensor(...), 'metadata': {...}}
        """
        cache_path = self.get_cached_path(sim_index)
        phimath.save(str(cache_path), data)

    def load_cache(self, sim_index: int) -> Dict[str, Tensor]:
        """
        Load PhiML tensors from cache.

        Returns:
            Dict with PhiML tensors (named dimensions preserved)
        """
        cache_path = self.get_cached_path(sim_index)
        return phimath.load(str(cache_path))

    def field_to_phiml_tensor(self, field: Field, field_name: str) -> Tensor:
        """
        Convert PhiFlow Field to PhiML Tensor with proper naming.

        PhiFlow Fields already use PhiML tensors internally,
        we just need to ensure proper dimension naming.
        """
        values = field.values  # Already a PhiML tensor!

        # Ensure dimension names match our convention:
        # - time: batch dimension for timesteps
        # - x, y: spatial dimensions
        # - vector: channel dimension for vector fields
        # NOTE: PhiFlow Fields may use different names, so rename if needed

        return values
```

### B. PhiML Dataset

**File**: `src/data/phiml_dataset.py`

Key design decisions:
1. **No PyTorch Dataset inheritance**: We don't need `__getitem__` and `__len__`
2. **Generator-based iteration**: Yield batches directly
3. **PhiML stacking**: Use `phimath.stack()` with `batch()` dimension
4. **Named dimensions**: All operations preserve dimension names

```python
from phiml import math as phimath
from phiml.math import batch, stack

class PhiMLDataset:
    """
    Pure PhiML dataset that yields PhiML tensor batches.
    No PyTorch dependency.
    """

    def __init__(self, config, data_manager: PhiMLDataManager, ...):
        self.data_manager = data_manager
        # ... setup ...

    def iterate_batches(self, batch_size: int, shuffle: bool = True):
        """
        Iterate through dataset yielding PhiML tensor batches.

        Yields:
            Dict with 'initial_state' and 'targets' as PhiML tensors:
            {
                'initial_state': Tensor(batch=B, x=H, y=W, vector=V),
                'targets': Tensor(batch=B, time=T, x=H, y=W, vector=V)
            }
        """
        indices = list(range(len(self)))
        if shuffle:
            import random
            random.shuffle(indices)

        # Yield batches
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            yield self._load_batch(batch_indices)

    def _load_batch(self, indices: List[int]) -> Dict[str, Tensor]:
        """
        Load multiple samples and stack into batch dimension.
        """
        samples = [self._load_sample(idx) for idx in indices]

        # Stack using PhiML's batch dimension
        initial_states = stack(
            [s['initial'] for s in samples],
            batch('batch')
        )
        targets = stack(
            [s['target'] for s in samples],
            batch('batch')
        )

        return {
            'initial_state': initial_states,
            'targets': targets
        }

    def _load_sample(self, idx: int) -> Dict[str, Tensor]:
        """
        Load a single sample (initial state + targets).

        Returns PhiML tensors with named dimensions.
        """
        # Load cached simulation data
        sim_idx, start_frame = self._compute_sim_and_frame(idx)
        sim_data = self.data_manager.load_cache(sim_idx)

        # Extract velocity field (already PhiML tensor!)
        velocity = sim_data['velocity']  # Shape: (time=T, x=H, y=W, vector=V)

        # Extract window
        initial = velocity.time[start_frame]  # (x=H, y=W, vector=V)
        targets = velocity.time[start_frame+1:start_frame+1+self.rollout_steps]  # (time=T, x=H, y=W, vector=V)

        return {'initial': initial, 'target': targets}
```

### C. Trainer Updates

**File**: `src/training/synthetic/phiml_trainer.py`

Remove conversion functions and simplify:

```python
# REMOVE these functions (no longer needed!):
# - torch_to_phiml()
# - phiml_to_torch()

class PhiMLSyntheticTrainer:
    def train(self, dataset: PhiMLDataset, num_epochs: int):
        """
        Train using PhiML dataset directly.
        """
        for epoch in range(num_epochs):
            epoch_loss = self._train_epoch(dataset)
            # ... rest of training ...

    def _train_epoch(self, dataset: PhiMLDataset) -> float:
        """
        Train for one epoch using PhiML iterator.
        """
        epoch_losses = []

        # Use PhiML dataset iterator (no DataLoader!)
        for batch in dataset.iterate_batches(self.batch_size, shuffle=True):
            loss = self._train_batch(batch)
            epoch_losses.append(float(loss))

        return sum(epoch_losses) / len(epoch_losses)

    def _train_batch(self, batch: Dict[str, Tensor]):
        """
        Train on a single batch (already PhiML tensors!).

        Args:
            batch: Dict with 'initial_state' and 'targets' as PhiML tensors
        """
        # NO CONVERSION NEEDED!
        initial_state = batch['initial_state']  # Already PhiML!
        targets = batch['targets']  # Already PhiML!

        def loss_function(init_state, targets):
            # ... same as before ...
            return total_loss / num_steps

        # PhiML training
        loss = phiml_nn.update_weights(
            self.model.get_network(),
            self.optimizer,
            loss_function,
            initial_state,
            targets
        )

        return loss
```

### D. Factory Updates

**File**: `src/factories/dataloader_factory.py`

```python
class DataLoaderFactory:
    @staticmethod
    def create_phiml(config, mode: str, sim_indices: List[int], ...):
        """
        Create PhiML dataset (no DataLoader wrapper).

        Returns:
            PhiMLDataset instance
        """
        data_manager = PhiMLDataManager(config)
        dataset = PhiMLDataset(
            config,
            data_manager,
            sim_indices=sim_indices,
            ...
        )
        return dataset
```

### E. run.py Updates

**File**: `run.py`

```python
# OLD:
dataloader = DataLoader(
    dataset,
    batch_size=config["trainer"]["batch_size"],
    shuffle=True
)
trainer.train(data_source=dataloader, num_epochs=num_epochs)

# NEW:
dataset = DataLoaderFactory.create_phiml(config, ...)
trainer.train(dataset=dataset, num_epochs=num_epochs)
```

## Migration Strategy

### Option 1: Clean Break (Recommended)
1. Create new PhiML files alongside old ones
2. Update config to choose which pipeline to use
3. Test PhiML pipeline thoroughly
4. Switch default to PhiML
5. Deprecate old pipeline

### Option 2: Gradual Migration
1. Keep both pipelines
2. Add `use_phiml: true/false` config flag
3. Run both in parallel for validation
4. Eventually remove old pipeline

## Testing Plan

### Test 1: PhiML DataManager
```bash
python test_phiml_data_manager.py
```
Tests:
- Save PhiML cache
- Load PhiML cache
- Verify named dimensions preserved
- Verify metadata preserved

### Test 2: PhiML Dataset
```bash
python test_phiml_dataset.py
```
Tests:
- Iterate batches
- Verify batch shapes
- Verify named dimensions
- Test sliding window extraction

### Test 3: End-to-End Training
```bash
python run.py --config-name=burgers.yaml use_phiml_data=true
```
Tests:
- Full training loop with PhiML data
- No conversion errors
- Loss decreases
- Compare with PyTorch pipeline results

## Implementation Checklist

- [ ] Create `phiml_data_manager.py`
  - [ ] Implement save_cache()
  - [ ] Implement load_cache()
  - [ ] Implement field_to_phiml_tensor()
  - [ ] Add validation logic

- [ ] Create `phiml_dataset.py`
  - [ ] Implement iterate_batches()
  - [ ] Implement _load_batch()
  - [ ] Implement _load_sample()
  - [ ] Handle sliding windows

- [ ] Update `phiml_trainer.py`
  - [ ] Remove torch_to_phiml()
  - [ ] Remove phiml_to_torch()
  - [ ] Update train() signature
  - [ ] Update _train_epoch()
  - [ ] Update _train_batch()

- [ ] Update factories
  - [ ] Add create_phiml() to DataLoaderFactory
  - [ ] Update TrainerFactory if needed

- [ ] Update run.py
  - [ ] Replace DataLoader with PhiML dataset
  - [ ] Test end-to-end

- [ ] Create tests
  - [ ] test_phiml_data_manager.py
  - [ ] test_phiml_dataset.py
  - [ ] test_phiml_training.py

- [ ] Documentation
  - [ ] Update STEP3_SUMMARY.md
  - [ ] Document cache format change
  - [ ] Add migration guide

## Expected Benefits

1. **Performance**
   - No tensor conversion overhead
   - Direct PhiML tensor flow
   - Potential for better memory usage with disk-backed tensors

2. **Code Quality**
   - Simpler trainer code
   - No conversion layer
   - Type-safe named dimensions

3. **Debugging**
   - Named dimensions make errors clearer
   - Easier to track data flow
   - Better error messages

4. **Future-Proof**
   - Pure PhiML ecosystem
   - Ready for PhiML optimizations
   - No PyTorch dependency in data layer

## Estimated Complexity

- **PhiML DataManager**: Medium (2-3 hours)
- **PhiML Dataset**: Medium-High (4-5 hours)
- **Trainer Updates**: Low (1-2 hours)
- **Testing**: Medium (2-3 hours)
- **Integration**: Low-Medium (2-3 hours)

**Total**: ~15 hours (2 days of focused work)

## Next Steps

1. Review this implementation guide
2. Start with PhiML DataManager
3. Create comprehensive tests as we go
4. Iterate on PhiML Dataset
5. Update trainer once data pipeline is solid
6. Full end-to-end testing

---

**Ready to proceed?** Let me know if you'd like me to start implementing, or if you have questions about the approach!
