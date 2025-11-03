# Hybrid Training: Final Design Specification

**Document Version:** 1.0  
**Date:** November 3, 2025  
**Status:** Approved - Ready for Implementation

---

## Executive Summary

This document consolidates all design decisions for the hybrid training system (HYCO). It serves as the definitive reference for implementation.

---

## Core Architecture

### 1. Trainer Structure

**Principle: Explicit Data Passing**
- Trainers DO NOT hold data references internally
- Data is passed to `train(data_source, num_epochs)` method
- Trainers are persistent across cycles (preserve optimizer state)

```python
# Unified trainer signature
class TensorTrainer:
    def __init__(self, config: Dict, model: nn.Module):
        # No data_manager parameter!
        pass
    
    def train(self, data_source: DataLoader, num_epochs: int) -> Dict:
        # Data passed explicitly
        pass

class FieldTrainer:
    def __init__(self, config: Dict, model: Any, learnable_params: List):
        # No data_manager parameter!
        pass
    
    def train(self, data_source: Iterable, num_epochs: int) -> Dict:
        # Data passed explicitly
        pass
```

### 2. HybridTrainer Responsibilities

The HybridTrainer orchestrates:
1. **Prediction generation** (physical → tensors, synthetic → fields)
2. **Data augmentation** (combining real + generated)
3. **Training coordination** (calling sub-trainers)
4. **Cache management** (cleanup, monitoring)

```python
class HybridTrainer:
    def __init__(
        self,
        config: Dict,
        model_registry: ModelRegistry,
        synthetic_trainer: SyntheticTrainer,
        physical_trainer: PhysicalTrainer,
        converter: BatchConcatenationConverter,
        data_manager: DataManager,  # For metadata access
    ):
        # Initialize cache manager
        self.cache_manager = CacheManager(config)
        
        # Determine augmentation strategy
        self._determine_augmentation_strategy()
```

---

## Data Augmentation Strategy

### Count-Based Approach (Single Scaling Point)

**Key Decision:** Apply alpha ONLY in sample count, NOT in loss weight

```python
# Generate proportional number of samples
num_real = len(real_dataset)
num_generated = int(num_real * alpha)  # e.g., 0.1 → 10% as many

# All samples have equal weight in loss
# NO weight parameter in dataset output!
dataset[i] → (input, target)  # 2-tuple, not 3-tuple
```

**Mathematical Verification:**
```
For alpha = 0.1:
- Real: 1000 samples × 1.0 weight = 1000 effective
- Generated: 100 samples × 1.0 weight = 100 effective
- Ratio: 100/1000 = 0.1 = 10% ✓
```

### Adaptive Strategy Selection

Based on memory requirements:

| Strategy | When | Storage | Speed |
|----------|------|---------|-------|
| **Memory** | Generated data < 1GB | RAM | Fast |
| **Cache** | Generated data 1-2GB | Disk + LRU | Medium |
| **On-the-fly** | Generated data > 2GB | None | Slow |

Configuration:
```yaml
trainer:
  alpha: 0.1
  memory_budget_gb: 2.0  # Configurable threshold
```

---

## Unified Data Access Pattern

### Both Trainers Use Sliding Window

**Key Decision:** Physical trainer now uses sliding window (like synthetic)

```python
# Synthetic training (tensor-based)
tensor_dataset = HybridDataset(
    data_manager=dm,
    sim_indices=[0, 1, 2, ...],
    field_names=["velocity", "density"],
    num_frames=50,
    num_predict_steps=3,
    use_sliding_window=True,  # ← Multiple samples per sim
    return_fields=False       # ← Returns tensors
)

# Physical training (field-based)  
field_dataset = HybridDataset(
    data_manager=dm,
    sim_indices=[0, 1, 2, ...],
    field_names=["velocity", "density"],
    num_frames=50,
    num_predict_steps=3,
    use_sliding_window=True,  # ← ALSO uses sliding window now!
    return_fields=True        # ← Returns PhiFlow Fields
)
```

**Benefits:**
- Consistent data access across both trainers
- More training samples for physical model
- Fair comparison between models

---

## Cache Organization

### Directory Structure

```
cache/
├── tensors/              # Preprocessed real data
│   ├── burgers_128/
│   │   ├── sim_000000.pt
│   │   └── ...
│   └── smoke_128/
│
├── generated/            # Temporary generated predictions
│   ├── synthetic_preds/
│   │   ├── cycle_000/
│   │   │   └── burgers_128/
│   │   └── cycle_001/
│   │
│   └── physical_preds/
│       ├── cycle_000/
│       │   └── burgers_128/
│       └── cycle_001/
│
└── evaluation/           # Evaluation cache
    └── burgers_128/
```

### Cache Management

```python
class CacheManager:
    """Centralized cache management."""
    
    def get_synthetic_pred_cache_dir(self, cycle: int) -> Path:
        """Get cache dir for synthetic predictions."""
        
    def get_physical_pred_cache_dir(self, cycle: int) -> Path:
        """Get cache dir for physical predictions."""
    
    def cleanup_cycle_cache(self, cycle: int):
        """Clean up cache after cycle completes."""
    
    def print_cache_summary(self):
        """Print cache usage statistics."""
```

**Lifecycle:**
1. Create cache dir at start of cycle
2. Save generated predictions to cache (if using cache strategy)
3. Train sub-trainers (load from cache as needed)
4. Clean up cache at end of cycle (in `finally` block)

---

## Training Loop

### High-Level Flow

```python
def train(self, num_cycles: int):
    for cycle in range(num_cycles):
        try:
            # Phase 1: Generate physical predictions (fields → tensors)
            field_dataset = self._create_base_field_dataset(use_sliding_window=True)
            generated_tensors = self._generate_physical_predictions(field_dataset)
            
            # Phase 2: Train synthetic model with augmented tensors
            real_tensor_dataset = self._create_base_tensor_dataset(use_sliding_window=True)
            augmented_loader = self._create_augmented_tensor_loader(
                real_tensor_dataset, generated_tensors
            )
            synthetic_metrics = self.synthetic_trainer.train(
                augmented_loader, self.synthetic_epochs_per_cycle
            )
            
            # Phase 3: Generate synthetic predictions (tensors → fields)
            generated_fields = self._generate_synthetic_predictions(real_tensor_dataset)
            
            # Phase 4: Train physical model with augmented fields
            augmented_source = self._create_augmented_field_source(
                field_dataset, generated_fields
            )
            physical_metrics = self.physical_trainer.train(
                augmented_source, self.physical_epochs_per_cycle
            )
            
        finally:
            # Always cleanup (even on error)
            if self.augmentation_strategy == "cache":
                self.cache_manager.cleanup_cycle_cache(cycle)
```

### Generation Details

**Proportional Sampling:**
```python
def _generate_physical_predictions(self, field_dataset: HybridDataset):
    # Calculate how many to generate (THIS IS WHERE WE APPLY ALPHA)
    num_to_generate = int(len(field_dataset) * self.alpha)
    
    logger.info(f"Real samples: {len(field_dataset)}")
    logger.info(f"Generating: {num_to_generate} ({self.alpha*100:.1f}%)")
    
    # Sample random indices
    indices = torch.randperm(len(field_dataset))[:num_to_generate]
    
    # Generate only for selected samples
    for idx in indices:
        # ... generate prediction ...
```

---

## Configuration Schema

### Trainer Config

```yaml
# conf/trainer/hybrid.yaml
_target_: src.training.hybrid.HybridTrainer

# Hybrid training parameters
alpha: 0.1                      # Proportion of generated samples (10%)
memory_budget_gb: 2.0           # Memory threshold for strategy selection
num_cycles: 10                  # Number of hybrid training cycles

# Sub-trainer epochs per cycle
synthetic_epochs_per_cycle: 10
physical_epochs_per_cycle: 5

# Generated data cache locations
synthetic_pred_cache: "${paths.generated_cache_dir}/synthetic_preds"
physical_pred_cache: "${paths.generated_cache_dir}/physical_preds"
```

### Paths Config

```yaml
# conf/config.yaml (base configuration)
paths:
  # Raw data
  raw_data_root: "data"
  
  # Cache hierarchy
  cache_root: "cache"
  tensor_cache_dir: "${paths.cache_root}/tensors"
  generated_cache_dir: "${paths.cache_root}/generated"
  eval_cache_dir: "${paths.cache_root}/evaluation"
  
  # Results
  results_root: "results"
  model_checkpoint_dir: "${paths.results_root}/models"
  evaluation_output_dir: "${paths.results_root}/evaluation"
  
  # Logs
  log_dir: "logs"
```

### Data Config

```yaml
# conf/data/burgers_128.yaml
dset_name: "burgers_128"
raw_data_dir: "${paths.raw_data_root}/${data.dset_name}"
cache_dir: "${paths.tensor_cache_dir}/${data.dset_name}"

# Simulation parameters
num_frames: 50
num_predict_steps: 3

# Train/val split
train_sim_indices: [0, 1, 2, ..., 29]  # 30 simulations
val_sim_indices: [30, 31, 32]          # 3 simulations

# Field configuration
fields_scheme:
  all: ["velocity", "density"]
  dynamic: ["velocity", "density"]
  static: []
```

---

## Implementation Checklist

### Phase 1: Refactor Base Trainers ✓
- [ ] Modify `TensorTrainer.__init__` - remove `data_manager`, add `model`
- [ ] Modify `TensorTrainer.train` - add `data_source` parameter, remove weighting logic
- [ ] Modify `FieldTrainer.__init__` - remove `data_manager`, add `model` and `learnable_params`
- [ ] Modify `FieldTrainer.train` - add `data_source` parameter, remove weighting logic
- [ ] Update `SyntheticTrainer` to use new base class signature
- [ ] Update `PhysicalTrainer` to use new base class signature and support sliding window

### Phase 2: Data Augmentation Infrastructure ✓
- [ ] Create `AugmentedTensorDataset` class (count-based, no weights)
- [ ] Create `AugmentedFieldDataset` class (count-based, no weights)
- [ ] Create `CachedAugmentedDataset` class (for disk caching strategy)
- [ ] Create on-the-fly DataLoader classes (for large datasets)
- [ ] Add validation function to check augmentation balance

### Phase 3: Cache Management ✓
- [ ] Create `CacheManager` class
- [ ] Implement cache directory creation
- [ ] Implement cycle cleanup logic
- [ ] Implement cache size monitoring
- [ ] Add cache summary logging
- [ ] Update config files with new path structure

### Phase 4: HybridTrainer Implementation ✓
- [ ] Implement `__init__` with cache manager
- [ ] Implement `_determine_augmentation_strategy()`
- [ ] Implement `_generate_physical_predictions()` with proportional sampling
- [ ] Implement `_generate_synthetic_predictions()` with proportional sampling
- [ ] Implement `_create_augmented_tensor_loader()`
- [ ] Implement `_create_augmented_field_source()`
- [ ] Implement main `train()` loop with proper cleanup
- [ ] Add comprehensive logging

### Phase 5: Testing & Validation ✓
- [ ] Unit test: `AugmentedTensorDataset` balance
- [ ] Unit test: `AugmentedFieldDataset` balance
- [ ] Unit test: Cache manager cleanup
- [ ] Integration test: Single cycle with memory strategy
- [ ] Integration test: Single cycle with cache strategy
- [ ] Integration test: Full multi-cycle training
- [ ] Validation: Compare standalone vs hybrid training
- [ ] Performance profiling

---

## Key Validation Points

### 1. No Double Scaling
```python
# Verify dataset returns 2-tuple, not 3-tuple
sample = augmented_dataset[0]
assert len(sample) == 2, "Dataset should return (input, target), not (input, target, weight)"

# Verify correct proportion
validate_augmentation_balance(real_dataset, generated_data, alpha)
```

### 2. Sliding Window Consistency
```python
# Both datasets should use sliding window
tensor_dataset = HybridDataset(..., use_sliding_window=True, return_fields=False)
field_dataset = HybridDataset(..., use_sliding_window=True, return_fields=True)

# Should have same number of samples (from same simulations)
assert len(tensor_dataset) == len(field_dataset)
```

### 3. Cache Cleanup
```python
# After cycle, generated cache should be empty
try:
    train_cycle()
finally:
    cache_manager.cleanup_cycle_cache(cycle)
    # Verify cleanup
    assert not list(cache_dir.glob("*.pt"))
```

---

## Performance Expectations

### Memory Strategy
- **Speed:** Fast (no I/O during training)
- **Memory:** ~1-2GB for generated data
- **Best for:** Small datasets (< 1000 samples)

### Cache Strategy  
- **Speed:** Medium (disk I/O for loading)
- **Memory:** ~100-200MB (LRU cache)
- **Best for:** Medium datasets (1000-10000 samples)

### On-the-fly Strategy
- **Speed:** Slow (generation during training)
- **Memory:** Minimal (one batch at a time)
- **Best for:** Large datasets (> 10000 samples)

---

## References

- [Hybrid Trainer Design](./HYBRID_TRAINER_DESIGN.md) - Original architecture document
- [Data Augmentation Practical Solutions](./DATA_AUGMENTATION_PRACTICAL_SOLUTIONS.md) - Detailed implementations
- [Loss Weighting Strategy](./LOSS_WEIGHTING_STRATEGY.md) - Critical: avoiding double scaling
- [Cache Organization](./CACHE_ORGANIZATION.md) - Cache hierarchy and management
- [HYCO Implementation Strategy](./HYCO_IMPLEMENTATION_STRATEGY.md) - Original HYCO paper concepts

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| Nov 2, 2025 | Composition architecture | Modularity, code reuse |
| Nov 3, 2025 | Explicit data passing | No hidden state, maximum flexibility |
| Nov 3, 2025 | Count-based weighting | Avoid double scaling, simpler implementation |
| Nov 3, 2025 | Adaptive strategy selection | Handle both small and large datasets |
| Nov 3, 2025 | Unified sliding window | Consistency between trainers |
| Nov 3, 2025 | Configurable memory budget | User control over memory/speed tradeoff |
| Nov 3, 2025 | Separate cache hierarchy | Clean organization, easy cleanup |

---

**Document Status:** APPROVED - This is the definitive design specification  
**Implementation Status:** Ready to begin Phase 1  
**Last Updated:** November 3, 2025
