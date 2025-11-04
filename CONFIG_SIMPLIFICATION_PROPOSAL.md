# Configuration Simplification Proposal

**Date:** November 4, 2025  
**Project:** HYCO-PhiFlow  
**Purpose:** Document all tunable configuration variables and propose simplification strategy

---

## Section 1: Current Configuration Variables

### 1.1 Root Configuration (`config.yaml`)

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `run_params.experiment_name` | string | ??? (required) | Experiment identifier |
| `run_params.notes` | string | "" | Experiment description |
| `run_params.mode` | list | [train] | Run modes: train/evaluate/generate |
| `run_params.model_type` | string | synthetic | Model type: synthetic/physical/hybrid |
| `cache.root` | string | "data/cache" | Cache directory location |
| `cache.auto_create` | bool | true | Auto-create cache directory |
| `cache.validation.check_on_load` | bool | true | Validate cache on load |
| `cache.validation.expected_count` | int/null | null | Expected cache item count |
| `cache.cleanup.on_start` | bool | false | Clear cache before training |
| `cache.cleanup.on_error` | bool | false | Clear cache on validation failure |
| `project_root` | string | ${hydra:runtime.cwd} | Project root directory |

### 1.2 Data Configuration (`data/*.yaml`)

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `data_dir` | string | 'data/' | Data directory location |
| `dset_name` | string | varies | Dataset name (burgers_128/advection_128/smoke_128) |
| `fields` | list | varies | Physical fields (velocity/density/inflow) |
| `fields_scheme` | string | varies | Field encoding scheme (VV/dVV/dVVi) |
| `cache_dir` | string | 'data/cache' | Cache directory (duplicates root config) |
| `validate_cache` | bool | true | Validate cache (duplicates root config) |
| `auto_clear_invalid` | bool | true | Auto-clear invalid cache (duplicates root config) |

### 1.3 Trainer Configuration - Synthetic (`trainer/synthetic.yaml`)

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `learning_rate` | float | 0.0001 | Learning rate |
| `batch_size` | int | 16 | Batch size |
| `epochs` | int | 100 | Training epochs |
| `num_predict_steps` | int | 4 | Prediction steps |
| `train_sim` | list | [] | Training simulation indices |
| `val_sim` | list | [] | Validation simulation indices |
| `use_sliding_window` | bool | true | Use sliding window data loading |
| `validate_every` | int | 1 | Validation frequency (epochs) |
| `validate_on_train` | bool | false | Compute train metrics during validation |
| `validation_rollout` | bool | true | Use full rollout for validation |
| `validation_rollout_steps` | int/null | 75 | Rollout steps for validation |
| `early_stopping.enabled` | bool | false | Enable early stopping |
| `early_stopping.patience` | int | 10 | Early stopping patience |
| `early_stopping.min_delta` | float | 1e-6 | Minimum improvement delta |
| `early_stopping.monitor` | string | val_loss | Metric to monitor |
| `optimizer` | string | 'adam' | Optimizer type |
| `scheduler` | string | 'cosine' | Learning rate scheduler |
| `weight_decay` | float | 0.0 | Weight decay |
| `save_interval` | int | 10 | Checkpoint save interval |
| `save_best_only` | bool | true | Save only best checkpoints |
| `checkpoint_freq` | int | 10 | Checkpoint frequency (duplicates save_interval) |
| `print_freq` | int | 10 | Progress print frequency |
| `memory_monitor_batches` | int | 5 | Batches to monitor memory |
| `augmentation.enabled` | bool | false | Enable data augmentation |
| `augmentation.alpha` | float | 0.1 | Augmentation ratio |
| `augmentation.strategy` | string | "cached" | Strategy: cached/on_the_fly |
| `augmentation.cache.experiment_name` | string | "${data.dset_name}" | Cache experiment name |
| `augmentation.cache.format` | string | "dict" | Cache format |
| `augmentation.cache.max_memory_samples` | int | 1000 | LRU cache size |
| `augmentation.cache.reuse_existing` | bool | true | Reuse existing cache |
| `augmentation.on_the_fly.generate_every` | int | 1 | Regeneration frequency |
| `augmentation.on_the_fly.batch_size` | int | 32 | Generation batch size |
| `augmentation.on_the_fly.rollout_steps` | int | 10 | Rollout steps for generation |
| `augmentation.device` | string | "cuda" | Device for generation |

### 1.4 Trainer Configuration - Physical (`trainer/physical.yaml`)

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `learning_rate` | float | 0.0001 | Learning rate (unused for physics) |
| `batch_size` | int | 1 | Batch size |
| `epochs` | int | 50 | Training epochs |
| `num_predict_steps` | int | 5 | Prediction steps |
| `train_sim` | list | [] | Training simulation indices |
| `learnable_parameters` | list | [] | Parameters for inverse problems |
| `method` | string | 'L-BFGS-B' | Optimization method |
| `abs_tol` | float | 1e-6 | Absolute tolerance |
| `max_iterations` | int/null | null | Maximum optimizer iterations |
| `print_freq` | int | 10 | Progress print frequency |
| `checkpoint_freq` | int | 10 | Checkpoint frequency |

### 1.5 Trainer Configuration - Hybrid (`trainer/hybrid.yaml`)

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `epochs` | int | 50 | Training epochs |
| `num_predict_steps` | int | 10 | Prediction steps |
| `train_sim` | list | [0,1,2] | Training simulation indices |
| `val_sim` | list | [3] | Validation simulation indices |
| `alpha` | float | 0.5 | Real data weight |
| `interleave_frequency` | int | 1 | Training interleave frequency |
| `warmup_epochs` | int | 5 | Synthetic model warmup epochs |
| `augmentation.*` | various | various | All augmentation settings (duplicates synthetic) |
| `synthetic.learning_rate` | float | 1e-4 | Synthetic learning rate |
| `synthetic.batch_size` | int | 16 | Synthetic batch size |
| `synthetic.optimizer` | string | adam | Synthetic optimizer |
| `synthetic.scheduler` | string | cosine | Synthetic scheduler |
| `synthetic.weight_decay` | float | 0.0 | Synthetic weight decay |
| `physical.method` | string | 'L-BFGS-B' | Physical optimization method |
| `physical.abs_tol` | float | 1e-6 | Physical absolute tolerance |
| `physical.max_iterations` | int | 5 | Physical max iterations |
| `physical.suppress_convergence_errors` | bool | true | Suppress convergence errors |
| `save_interval` | int | 10 | Checkpoint save interval |
| `save_best_only` | bool | true | Save only best checkpoints |
| `checkpoint_dir` | string | 'results/models/hybrid' | Checkpoint directory |
| `enable_memory_monitoring` | bool | false | Enable memory monitoring |

### 1.6 Model Configuration - Physical (`model/physical/*.yaml`)

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `name` | string | varies | Model name |
| `domain.size_x` | float | 100 | Domain size X |
| `domain.size_y` | float | 100 | Domain size Y |
| `resolution.x` | int | 128 | Resolution X |
| `resolution.y` | int | 128 | Resolution Y |
| `dt` | float | varies | Time step |
| `pde_params.batch_size` | int | 1 | PDE batch size (duplicates trainer) |
| `pde_params.nu` | float | varies | Viscosity (Burgers/Smoke) |
| `pde_params.advection_coeff` | float | 1.0 | Advection coefficient (Advection) |
| `pde_params.buoyancy` | float | 1.0 | Buoyancy (Smoke) |
| `pde_params.inflow_radius` | float | 5.0 | Inflow radius (Smoke) |
| `pde_params.inflow_rate` | float | 0.2 | Inflow rate (Smoke) |
| `pde_params.inflow_rand_x_range` | list | [0.2, 0.6] | Inflow X randomization (Smoke) |
| `pde_params.inflow_rand_y_range` | list | [0.15, 0.05] | Inflow Y randomization (Smoke) |

### 1.7 Model Configuration - Synthetic (`model/synthetic/*.yaml`)

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `name` | string | 'UNet' | Model architecture name |
| `model_path` | string | 'results/models' | Model save path |
| `model_save_name` | string | ??? (required) | Model filename |
| `input_specs.*` | int | varies | Input channel specifications |
| `output_specs.*` | int | varies | Output channel specifications |
| `architecture.levels` | int | 4 | UNet depth levels |
| `architecture.filters` | int | 64 | Base filter count |
| `architecture.batch_norm` | bool | true | Use batch normalization |

### 1.8 Generation Configuration (`generation/default.yaml`)

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `num_simulations` | int | 25 | Number of simulations to generate |
| `total_steps` | int | 100 | Total time steps |
| `save_interval` | int | 1 | Save frequency |
| `seed` | int/null | null | Random seed |

### 1.9 Evaluation Configuration (`evaluation/default.yaml`)

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `test_sim` | list | [] | Test simulation indices |
| `num_frames` | int | 51 | Number of frames to evaluate |
| `metrics` | list | ['mse','mae','rmse'] | Evaluation metrics |
| `keyframe_count` | int | 5 | Number of keyframes for plots |
| `animation_fps` | int | 10 | Animation FPS |
| `save_animations` | bool | true | Save animation outputs |
| `save_plots` | bool | true | Save plot outputs |
| `output_dir` | string | 'results/evaluation' | Output directory |

### 1.10 Logging Configuration (`logging/default.yaml`)

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `logging.root_level` | string | INFO | Root logging level |
| `logging.log_dir` | string | logs | Log directory |
| `logging.log_to_file` | bool | true | Enable file logging |
| `logging.log_to_console` | bool | true | Enable console logging |
| `logging.module_levels.*` | string | varies | Module-specific log levels |
| `logging.log_memory_usage` | bool | false | Log memory usage |
| `logging.log_gpu_stats` | bool | false | Log GPU statistics |
| `logging.max_file_size_mb` | int | 100 | Max log file size |
| `logging.backup_count` | int | 5 | Number of log backups |

### 1.11 Hydra Configuration (`hydra/default.yaml`)

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `run.dir` | string | outputs/${now:%Y-%m-%d}/... | Output directory pattern |
| `job_logging.root.level` | string | INFO | Job logging level |
| `hydra_logging.root.level` | string | WARNING | Hydra logging level |

---

## Section 2: Simplification Proposals

### 2.1 Variables to HARDCODE (Remove from Config)

#### 2.1.1 Root Config (`config.yaml`)
**Hardcode as `True` (always enabled):**
- 🔒 `cache.auto_create` → Always `true` (no configuration needed)
- 🔒 `cache.validation.check_on_load` → Always `true` (validation is critical)

**Remove entirely:**
- ❌ `cache.validation.expected_count` → Remove logic, auto-detect only

**Rationale:** Cache creation and validation are essential behaviors that should never be disabled.

#### 2.1.2 Data Config (`data/*.yaml`)
**Remove (duplicates from root config):**
- ❌ `cache_dir` → Use `cache.root` from root config
- ❌ `validate_cache` → Now hardcoded in root
- ❌ `auto_clear_invalid` → Use `cache.cleanup.on_error` from root

**Rationale:** Single source of truth in root configuration; no duplication.

#### 2.1.3 Synthetic Trainer (`trainer/synthetic.yaml`)
**Hardcode as `True` (always enabled):**
- 🔒 `use_sliding_window` → Always `true` for training (better memory efficiency)
- 🔒 `validation_rollout` → Always `true` (more accurate validation)
- 🔒 `save_best_only` → Always `true` (save disk space, keep best models)

**Hardcode as `False` (always disabled):**
- 🔒 `validate_on_train` → Always `false` (validate on val set only)

**Remove entirely (including all logic):**
- ❌ `early_stopping.*` → Remove entire early stopping section and implementation
- ❌ `save_interval` → Remove in favor of `checkpoint_freq` only
- ❌ `augmentation.on_the_fly.*` → Remove on-the-fly generation, keep cached only
- ❌ `augmentation.strategy` → Remove (only "cached" supported now)

**Rationale:** Simplify to one proven approach. Early stopping rarely used in physics ML; on-the-fly augmentation too complex.

#### 2.1.4 Physical Trainer (`trainer/physical.yaml`)
**Remove and restructure:**
- ❌ `max_iterations` → Remove; `epochs` now serves as max iterations
- ❌ `learning_rate` → Not used by L-BFGS-B optimizer

**New behavior:**
- `epochs` → Now controls maximum optimization iterations per simulation
- Each simulation optimizes for up to `epochs` iterations

**Rationale:** Physical models don't iterate over epochs like neural networks; they optimize parameters. Using `epochs` as iteration count is more intuitive.

#### 2.1.5 Hybrid Trainer (`trainer/hybrid.yaml`)
**Apply both synthetic and physical simplifications:**
- 🔒 `validation_rollout` → Always `true`
- ❌ `augmentation.on_the_fly.*` → Remove
- ❌ `augmentation.strategy` → Remove
- ❌ `synthetic.save_best_only` → Hardcode as `true`
- ❌ `physical.max_iterations` → Remove, use epochs

**Rationale:** Consistency with simplified synthetic and physical trainers.

#### 2.1.6 Generation Config (`generation/default.yaml`)
**Remove:**
- ❌ `seed` → Remove random seed configuration

**Rationale:** Reproducibility not critical for data generation; adds unnecessary complexity.

#### 2.1.7 Other Removals
**Duplicate/Redundant:**
- ❌ `checkpoint_freq` → Use `save_interval` consistently
- ❌ `pde_params.batch_size` → Already defined in trainer
- ❌ `trainer_params.memory_monitor_batches` → Use logging config
- ❌ `trainer_params.enable_memory_monitoring` → Use logging config

### 2.2 Variables to MERGE and Simplify

#### 2.2.1 Simplified Augmentation Configuration
**Current Issue:** Complex augmentation with cached/on-the-fly strategies duplicated across trainers.

**New simplified structure:**
```yaml
augmentation:
  enabled: bool
  alpha: float  # Ratio of synthetic to real data
  cache:
    experiment_name: string
    format: "dict"  # Hardcoded, remove as config
    max_memory_samples: int
    reuse_existing: bool
  device: "cuda" | "cpu"
```

**Removed:**
- Strategy selection (always cached)
- All on-the-fly generation options
- Cache format selection (always "dict")

**Rationale:** Cached approach is proven and performant. On-the-fly adds complexity without clear benefits.

#### 2.2.2 Unified Checkpoint Configuration
**Current Issue:** Checkpoint settings inconsistent across trainers.

**New structure:**
```yaml
checkpointing:
  checkpoint_freq: int  # Save every N epochs
  output_dir: string  # Override default
  # save_best_only: always true (hardcoded)
```

**Removed:**
- `save_interval` (use `checkpoint_freq` consistently)
- `save_best_only` (always true)
- `monitor_metric` (always val_loss when val_sim present)

**Rationale:** Simpler, consistent checkpointing with sensible defaults.

#### 2.2.3 Unified Optimizer Configuration
**Current Issue:** Optimizer settings scattered across trainer configs.

**New structure:**
```yaml
optimizer:
  # For synthetic/neural models
  neural:
    type: "adam" | "sgd" | "adamw"
    learning_rate: float
    weight_decay: float
    scheduler: "cosine" | "step" | "none"
  
  # For physical/inverse models
  physical:
    method: "L-BFGS-B" | "BFGS" | "CG"
    abs_tol: float
    suppress_convergence_errors: bool
    # max_iterations removed - use trainer.epochs instead
```

**Changes:**
- Removed `max_iterations` from physical optimizer
- Physical models now use `epochs` parameter as iteration count
- Clear separation between neural and physical optimization

**Rationale:** Physical optimization doesn't need separate iteration count; epochs serves this purpose.

#### 2.2.4 Simplified Validation Configuration
**Current Issue:** Validation settings scattered across trainer configs.

**New structure:**
```yaml
validation:
  val_sim: list  # Validation simulation indices
  validate_every: int  # Validate every N epochs
  rollout_steps: int  # Number of steps for validation rollout
  # validate_on_train: always false (hardcoded)
  # validation_rollout: always true (hardcoded)
```

**Removed:**
- `validate_on_train` (always false)
- `validation_rollout` (always true)
- `enabled` flag (enabled when val_sim is non-empty)

**Rationale:** Validation is essential and should always use rollout for accuracy.

#### 2.2.5 Domain/Resolution Settings (No change)
**Keep in physical model configs:**
```yaml
# In model/physical/*.yaml
domain:
  size_x: float
  size_y: float
resolution:
  x: int
  y: int
dt: float
```

**Rationale:** These are model-specific physical parameters, not data properties.

### 2.3 Variables to RENAME for Clarity

| Current Name | Proposed Name | Reason |
|--------------|---------------|--------|
| `train_sim` | `training_simulations` | More explicit |
| `val_sim` | `validation_simulations` | More explicit |
| `test_sim` | `test_simulations` | More explicit |
| `num_predict_steps` | `prediction_steps` | Shorter, clearer |
| `use_sliding_window` | `sliding_window_enabled` | Consistency |
| `fields_scheme` | `field_encoding` | More descriptive |
| `dset_name` | `dataset_name` | Standard naming |
| `abs_tol` | `absolute_tolerance` | Full form |
| `nu` | `viscosity` | Physical meaning |

### 2.4 Proposed Simplified Structure

```yaml
# Root config.yaml
defaults:
  - data: ???
  - model: ???
  - trainer: ???
  - _self_

experiment:
  name: ???
  notes: ""
  mode: [train]
  model_type: synthetic  # synthetic/physical/hybrid

paths:
  project_root: ${hydra:runtime.cwd}
  data_dir: data/
  cache_dir: data/cache
  models_dir: results/models
  logs_dir: logs
  outputs_dir: results/

cache:
  root: "data/cache"
  cleanup:
    on_start: false
    on_error: false
  # auto_create: true (hardcoded)
  # validate_on_load: true (hardcoded)

# Groups (imported from separate configs)
data: {}
model: {}
trainer: {}
checkpointing: {}
validation: {}
augmentation: {}
generation: {}
evaluation: {}
logging: {}
```

#### Simplified Trainer Configs

**Synthetic Trainer (`trainer/synthetic.yaml`):**
```yaml
learning_rate: float
batch_size: int
epochs: int
num_predict_steps: int
train_sim: []
val_sim: []
validate_every: int
validation_rollout_steps: int

optimizer: 'adam'
scheduler: 'cosine'
weight_decay: float

checkpoint_freq: int
print_freq: int

# Optional augmentation
augmentation:
  enabled: bool
  alpha: float
  cache:
    experiment_name: string
    max_memory_samples: int
    reuse_existing: bool
  device: "cuda"

# Hardcoded (removed from config):
# - use_sliding_window: true
# - validate_on_train: false
# - validation_rollout: true
# - save_best_only: true
# - early_stopping: removed
# - save_interval: removed (use checkpoint_freq)
```

**Physical Trainer (`trainer/physical.yaml`):**
```yaml
batch_size: int
epochs: int  # Now serves as max_iterations
num_predict_steps: int
train_sim: []

learnable_parameters: []

method: 'L-BFGS-B'
abs_tol: float

checkpoint_freq: int
print_freq: int

# Removed:
# - learning_rate: not used
# - max_iterations: use epochs instead
```

**Hybrid Trainer (`trainer/hybrid.yaml`):**
```yaml
epochs: int
num_predict_steps: int
train_sim: []
val_sim: []

alpha: float  # Real data weight
interleave_frequency: int
warmup_epochs: int

# Synthetic model settings
synthetic:
  learning_rate: float
  batch_size: int
  optimizer: 'adam'
  scheduler: 'cosine'
  weight_decay: float

# Physical model settings
physical:
  method: 'L-BFGS-B'
  abs_tol: float
  suppress_convergence_errors: bool

# Augmentation settings
augmentation:
  enabled: bool
  alpha: float
  cache:
    experiment_name: string
    max_memory_samples: int
    reuse_existing: bool
  device: "cuda"

checkpoint_freq: int
checkpoint_dir: string

# Hardcoded:
# - validation_rollout: true
# - save_best_only: true
```

**Generation Config (`generation/default.yaml`):**
```yaml
num_simulations: int
total_steps: int
save_interval: int

# Removed:
# - seed: removed
```

### 2.5 Summary Statistics

**Current Complexity:**
- Total unique variables: ~120+
- Config files: 30+
- Configurable boolean flags: ~20
- Redundant variables: ~15
- Strategy options: ~5

**Proposed Complexity:**
- Total unique variables: ~70
- Config files: 25
- Configurable boolean flags: ~5 (most hardcoded)
- Redundant variables: 0
- Strategy options: 0 (single approach per feature)

**Reduction:** 
- ~40% fewer variables overall
- ~75% fewer boolean toggles
- 100% elimination of redundancy
- Single proven approach (no strategy selection)

**Key Improvements:**
- ✅ Cache validation always enabled (critical for data integrity)
- ✅ Sliding window always used (better memory efficiency)
- ✅ Validation always uses rollout (accurate metrics)
- ✅ Only best models saved (saves disk space)
- ✅ Single augmentation approach (cached only - simpler, proven)
- ✅ Physical optimization uses epochs as iteration count (clearer semantics)
- ✅ No early stopping complexity (rarely needed in physics ML)
- ✅ No random seed management for generation (unnecessary complexity)

---

## Section 3: Implementation Guide

### 3.1 Code Changes Required

#### 3.1.1 Cache Management (Hardcode Behaviors)
**Files to modify:**
- `src/data/*.py` - Remove conditional checks for `auto_create` and `check_on_load`

**Changes:**
```python
# OLD:
if config.cache.auto_create:
    os.makedirs(cache_dir, exist_ok=True)

if config.cache.validation.check_on_load:
    validate_cache(cache_dir)

# NEW (always execute):
os.makedirs(cache_dir, exist_ok=True)
validate_cache(cache_dir)
```

#### 3.1.2 Sliding Window (Always Enabled)
**Files to modify:**
- `src/data/dataloader.py` or similar

**Changes:**
```python
# OLD:
if config.use_sliding_window:
    return SlidingWindowDataset(...)
else:
    return StandardDataset(...)

# NEW (always sliding window):
return SlidingWindowDataset(...)
```

#### 3.1.3 Validation Settings (Hardcode)
**Files to modify:**
- `src/training/trainer.py` or validation logic

**Changes:**
```python
# OLD:
if config.validate_on_train:
    train_metrics = compute_metrics(train_data)

if config.validation_rollout:
    val_metrics = rollout_validation(...)
else:
    val_metrics = single_step_validation(...)

# NEW (always rollout, never validate_on_train):
val_metrics = rollout_validation(...)
```

#### 3.1.4 Checkpointing (Always Save Best)
**Files to modify:**
- `src/training/checkpoint.py` or similar

**Changes:**
```python
# OLD:
if config.save_best_only:
    if val_loss < best_loss:
        save_checkpoint(...)
else:
    if epoch % config.save_interval == 0:
        save_checkpoint(...)

# NEW (always save best only):
if val_loss < best_loss:
    save_checkpoint(...)
```

#### 3.1.5 Remove Early Stopping
**Files to modify:**
- `src/training/trainer.py`

**Changes:**
- Remove `EarlyStoppingCallback` or similar classes
- Remove all early stopping logic from training loop
- Clean up imports and dependencies

#### 3.1.6 Augmentation Strategy (Cached Only)
**Files to modify:**
- `src/data/augmentation.py` or similar

**Changes:**
```python
# OLD:
if config.augmentation.strategy == "cached":
    augmenter = CachedAugmenter(...)
elif config.augmentation.strategy == "on_the_fly":
    augmenter = OnTheFlyAugmenter(...)

# NEW (cached only):
augmenter = CachedAugmenter(...)
```

#### 3.1.7 Physical Trainer (Epochs as Iterations)
**Files to modify:**
- `src/training/physical_trainer.py`

**Changes:**
```python
# OLD:
max_iter = config.max_iterations or float('inf')
for epoch in range(config.epochs):
    for sim in train_sims:
        optimize_simulation(sim, max_iter=max_iter)

# NEW (epochs is max_iterations per simulation):
for sim in train_sims:
    optimize_simulation(sim, max_iterations=config.epochs)
```

#### 3.1.8 Remove Seed from Generation
**Files to modify:**
- `src/generation/generator.py` or similar

**Changes:**
```python
# OLD:
if config.seed is not None:
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

# NEW (remove seed management):
# Let system use default random state
```

### 3.2 Config File Updates

#### 3.2.1 Update All Config Files
**Action items:**
1. Remove hardcoded variables from all `.yaml` files
2. Remove early_stopping sections
3. Remove on_the_fly augmentation settings
4. Rename `save_interval` → `checkpoint_freq` (or vice versa, be consistent)
5. Remove `max_iterations` from physical configs
6. Remove `seed` from generation configs
7. Remove duplicate cache settings from data configs

#### 3.2.2 Add Comments for Hardcoded Behaviors
```yaml
# Example in trainer/synthetic.yaml
# Note: The following behaviors are hardcoded:
# - use_sliding_window: always true
# - validate_on_train: always false
# - validation_rollout: always true
# - save_best_only: always true

learning_rate: 0.0001
batch_size: 16
...
```

### 3.3 Migration Priority

**Phase 1 (Quick Wins - 1-2 hours):**
1. ✅ Remove duplicate cache settings from `data/*.yaml`
2. ✅ Remove `seed` from generation config
3. ✅ Add comments documenting hardcoded behaviors
4. ✅ Remove early_stopping sections from configs

**Phase 2 (Core Changes - 2-4 hours):**
1. ✅ Hardcode cache behaviors in data loading code
2. ✅ Hardcode sliding window in dataloader
3. ✅ Remove early stopping implementation
4. ✅ Remove on_the_fly augmentation code
5. ✅ Standardize checkpoint naming (`checkpoint_freq`)

**Phase 3 (Physical Trainer - 2-3 hours):**
1. ✅ Refactor physical trainer to use epochs as iterations
2. ✅ Remove `max_iterations` parameter
3. ✅ Update all physical experiment configs
4. ✅ Test physical training with new semantics

**Phase 4 (Validation & Testing - 2-3 hours):**
1. ✅ Update all experiment configs
2. ✅ Run quick tests on all trainer types
3. ✅ Verify checkpointing works correctly
4. ✅ Validate cache behavior
5. ✅ Update documentation

**Total estimated time:** 8-12 hours of focused work

### 3.4 Backward Compatibility Strategy

**NOT RECOMMENDED:** These are breaking changes that significantly simplify the codebase. 

**Recommended Approach:**
1. Create new branch: `config-simplification`
2. Make all changes at once (atomic commit)
3. Update all experiment configs in same commit
4. No backward compatibility layer (clean break)
5. Document changes in CHANGELOG.md

**Migration for Users:**
- Update experiment configs to remove hardcoded variables
- Adjust physical training configs (epochs now = max_iterations)
- Remove early stopping if used
- Remove on_the_fly augmentation if used

### 3.5 Testing Strategy

**Unit Tests to Update:**
1. Cache management tests - remove toggle checks
2. Dataloader tests - remove non-sliding-window paths
3. Training tests - remove early stopping tests
4. Augmentation tests - remove on_the_fly tests
5. Physical trainer tests - update epoch semantics

**Integration Tests:**
1. ✅ Run `burgers_quick_test.yaml` (synthetic)
2. ✅ Run `burgers_physical_quick_test.yaml` (physical)
3. ✅ Run `burgers_hybrid_quick_test.yaml` (hybrid)
4. ✅ Verify checkpoints saved correctly
5. ✅ Verify cache validation works
6. ✅ Verify augmentation (cached) works

**Validation Checklist:**
- [ ] All hardcoded behaviors work as expected
- [ ] No runtime errors from removed config variables
- [ ] Checkpointing saves only best models
- [ ] Physical trainer iterations controlled by epochs
- [ ] Cache always created and validated
- [ ] Sliding window always used
- [ ] Validation always uses rollout
- [ ] No early stopping triggered
- [ ] Augmentation uses cached approach only

### 3.6 Documentation Updates

**Files to update:**
1. `README.md` - Update configuration examples
2. `CHANGELOG.md` - Document breaking changes
3. Config file comments - Add hardcoded behavior notes
4. Any tutorial notebooks - Update config examples

**Key points to document:**
- Cache validation is mandatory and automatic
- Sliding window is always enabled for efficiency
- Only best models are saved to disk
- Physical training: epochs parameter controls optimization iterations
- Augmentation only supports cached strategy
- Early stopping has been removed (train for fixed epochs)

## Section 4: Detailed Change Summary

### 4.1 Variables Being Hardcoded

| Variable | Old Behavior | New Behavior | Impact |
|----------|-------------|--------------|--------|
| `cache.auto_create` | Configurable bool | Always `true` | Safer, less config |
| `cache.validation.check_on_load` | Configurable bool | Always `true` | Data integrity guaranteed |
| `cache.validation.expected_count` | Optional check | Removed | Simpler validation |
| `use_sliding_window` | Toggle | Always `true` | Better memory efficiency |
| `validate_on_train` | Toggle | Always `false` | Standard ML practice |
| `validation_rollout` | Toggle | Always `true` | Accurate validation |
| `save_best_only` | Toggle | Always `true` | Saves disk space |

### 4.2 Features Being Removed

| Feature | Rationale | Alternative |
|---------|-----------|-------------|
| Early stopping | Rarely used in physics ML; adds complexity | Train for fixed epochs |
| On-the-fly augmentation | Complex, cached approach is sufficient | Use cached augmentation |
| Strategy selection | Single proven approach is better | Cached only |
| Random seed in generation | Unnecessary for data generation | System default randomness |
| `max_iterations` (physical) | Redundant with epochs | Use epochs parameter |
| Dual checkpoint naming | Confusing (`save_interval` vs `checkpoint_freq`) | Use `checkpoint_freq` only |

### 4.3 Semantic Changes

**Physical Trainer - Epochs Reinterpretation:**

**Before:**
```yaml
epochs: 50  # Iterate over all simulations 50 times
max_iterations: 100  # Each simulation optimizes up to 100 iterations
```
Training loop:
```python
for epoch in range(50):
    for sim in train_sims:
        optimize(sim, max_iter=100)
```

**After:**
```yaml
epochs: 100  # Each simulation optimizes up to 100 iterations
# max_iterations removed
```
Training loop:
```python
for sim in train_sims:
    optimize(sim, max_iterations=100)  # epochs parameter
```

**Impact:** More intuitive; epochs directly controls optimization iterations per simulation.

### 4.4 Files Requiring Changes

#### Python Code Files (estimated 15-20 files):
- `src/data/dataloader.py` - Remove sliding window toggle
- `src/data/cache_manager.py` - Hardcode cache validation
- `src/data/augmentation.py` - Remove on_the_fly strategy
- `src/training/trainer.py` - Remove early stopping
- `src/training/synthetic_trainer.py` - Simplify validation
- `src/training/physical_trainer.py` - Refactor epochs usage
- `src/training/hybrid_trainer.py` - Apply both sets of changes
- `src/training/checkpoint.py` - Always save best only
- `src/generation/generator.py` - Remove seed management
- `src/utils/config.py` - Update config validation

#### Config Files (estimated 25-30 files):
- `conf/config.yaml` - Remove cache validation toggles
- `conf/data/*.yaml` (3 files) - Remove duplicate cache settings
- `conf/trainer/*.yaml` (8 files) - Remove hardcoded variables
- `conf/generation/default.yaml` - Remove seed
- All experiment configs (15+ files) - Update to new structure

#### Test Files (estimated 10-15 files):
- Tests for cache management
- Tests for data loading
- Tests for training
- Tests for augmentation
- Tests for physical optimization

### 4.5 Breaking Changes for Users

**Must update experiment configs:**
1. Remove `early_stopping` section if present
2. Remove `use_sliding_window` (now implicit)
3. Remove `validate_on_train` (now implicit)
4. Remove `validation_rollout` (now implicit)
5. Remove `save_best_only` (now implicit)
6. Remove `save_interval`, use `checkpoint_freq`
7. Remove `max_iterations` from physical configs, adjust `epochs` value
8. Remove `seed` from generation configs
9. Remove `augmentation.strategy` (always cached)
10. Remove `augmentation.on_the_fly` section

**Example migration:**

**OLD config:**
```yaml
trainer_params:
  epochs: 50
  use_sliding_window: true
  validate_on_train: false
  validation_rollout: true
  save_best_only: true
  save_interval: 10
  early_stopping:
    enabled: false
    patience: 10
```

**NEW config:**
```yaml
trainer_params:
  epochs: 50
  checkpoint_freq: 10
  # Removed: use_sliding_window, validate_on_train, validation_rollout,
  #          save_best_only, save_interval, early_stopping
```

---

## Section 5: Next Steps

### 5.1 Immediate Actions

1. **Review this proposal** - Get team/advisor feedback
2. **Prioritize changes** - Decide which to implement first
3. **Create branch** - `git checkout -b config-simplification`
4. **Start with Phase 1** - Quick wins (remove duplicates, add comments)

### 5.2 Implementation Phases

**Week 1: Foundation (Phase 1 & 2)**
- Remove duplicate cache settings
- Hardcode cache validation behaviors
- Remove early stopping
- Remove on_the_fly augmentation
- Standardize checkpoint naming

**Week 2: Physical Trainer (Phase 3)**
- Refactor physical trainer epoch semantics
- Update physical model configs
- Test physical training thoroughly

**Week 3: Testing & Documentation (Phase 4)**
- Update all experiment configs
- Run comprehensive tests
- Update documentation
- Create migration guide

### 5.3 Success Criteria

✅ **Configuration Complexity:**
- 40% reduction in total variables
- 75% reduction in boolean toggles
- Zero redundant variables

✅ **Code Simplification:**
- Remove all conditional checks for hardcoded behaviors
- Remove early stopping implementation (~200-300 lines)
- Remove on_the_fly augmentation (~300-500 lines)
- Cleaner physical trainer logic

✅ **Maintainability:**
- Fewer code paths to test
- Single proven approach per feature
- Clearer semantics (epochs in physical trainer)

✅ **User Experience:**
- Simpler configs to write
- Fewer decisions to make
- Sensible defaults baked in

### 5.4 Risk Mitigation

**Risks:**
1. Breaking existing experiments
2. Changing physical trainer semantics might confuse users
3. Loss of flexibility for edge cases

**Mitigations:**
1. Comprehensive testing before merge
2. Clear documentation of breaking changes
3. Examples showing old vs new configs
4. Can add back features if truly needed (but unlikely)
