# Data Loading Architecture Documentation

**Project**: HYCO-PhiFlow  
**Document Version**: 1.0  
**Date**: November 3, 2025  
**Purpose**: Complete architectural overview of the data loading system

---

## Table of Contents

1. [Overview](#1-overview)
2. [Component Hierarchy](#2-component-hierarchy)
3. [Data Flow](#3-data-flow)
4. [Core Components](#4-core-components)
5. [Augmentation System](#5-augmentation-system)
6. [Factory Integration](#6-factory-integration)
7. [Usage Patterns](#7-usage-patterns)
8. [Complexity Analysis](#8-complexity-analysis)
9. [Pain Points & Redundancies](#9-pain-points--redundancies)

---

## 1. Overview

The data loading system is a **multi-layered architecture** designed to handle:
- ✅ PhiFlow Scene loading (raw simulation data)
- ✅ Field-to-Tensor conversion and caching
- ✅ PyTorch Dataset interface for training
- ✅ Data augmentation with generated predictions
- ✅ Multiple augmentation strategies (memory/cache/on-the-fly)
- ✅ Dual-mode operation (tensors for synthetic, fields for physical)

### Key Design Goals

1. **Avoid redundant Field conversions** (expensive operation)
2. **Support both synthetic and physical models** (different data formats)
3. **Enable data augmentation** for hybrid training
4. **Memory efficiency** for large datasets
5. **Flexibility** in augmentation strategies

### System Complexity

- **Total Classes**: 10+ data-related classes
- **Abstraction Layers**: 5 layers deep
- **Configuration Points**: 15+ parameters
- **Factory Methods**: 4+ creation methods

---

## 2. Component Hierarchy

### 2.1 Full Architecture Tree

```
┌─────────────────────────────────────────────────────────────────┐
│                     TrainerFactory                               │
│  ├─ create_data_loader_for_synthetic()                          │
│  └─ create_dataset_for_physical()                               │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────────┐
│              AdaptiveAugmentedDataLoader                         │
│  (Strategy selector: memory/cache/on_the_fly)                   │
│  ├─ _auto_select_strategy()                                     │
│  ├─ _validate_strategy()                                        │
│  └─ _create_dataset()                                           │
└──────────────────┬──────────────────────────────────────────────┘
                   │
         ┌─────────┴─────────┬─────────────┐
         ↓                   ↓             ↓
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ AugmentedTensor  │  │ CachedAugmented  │  │ On-the-fly       │
│ Dataset          │  │ Dataset          │  │ (Not Implemented)│
│ (memory)         │  │ (disk cache)     │  │                  │
└────────┬─────────┘  └────────┬─────────┘  └──────────────────┘
         │                     │
         └──────────┬──────────┘
                    ↓
         ┌──────────────────────┐
         │   HybridDataset      │
         │ (Base real dataset)  │
         └──────────┬───────────┘
                    ↓
         ┌──────────────────────┐
         │    DataManager       │
         │ (Caching & Loading)  │
         └──────────┬───────────┘
                    ↓
         ┌──────────────────────┐
         │  PhiFlow Scene Files │
         │  (Raw simulation)    │
         └──────────────────────┘
```

### 2.2 Layer Breakdown

| Layer | Component | Responsibility | Complexity |
|-------|-----------|----------------|------------|
| **L5** | `TrainerFactory` | Factory methods for data creation | Medium |
| **L4** | `AdaptiveAugmentedDataLoader` | Strategy selection & orchestration | High |
| **L3** | `AugmentedTensorDataset` / `CachedAugmentedDataset` | Augmentation implementation | Medium |
| **L2** | `HybridDataset` | PyTorch Dataset interface | High |
| **L1** | `DataManager` | Caching & Field conversion | High |
| **L0** | PhiFlow Scene | Raw data storage | N/A |

---

## 3. Data Flow

### 3.1 Training Data Flow (Synthetic Model)

```
┌────────────────────────────────────────────────────────────────┐
│ 1. Factory Request                                              │
│    TrainerFactory.create_data_loader_for_synthetic(config)      │
└────────────────────────────────┬───────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ 2. Create Base Dataset                                          │
│    HybridDataset(                                               │
│        data_manager=DataManager(...),                           │
│        sim_indices=[0,1,2,...],                                 │
│        return_fields=False  # ← Tensors for synthetic          │
│    )                                                            │
└────────────────────────────────┬───────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ 3. Check Augmentation Config                                   │
│    if augmentation_enabled:                                     │
│        ↓                                                        │
│    AdaptiveAugmentedDataLoader(                                 │
│        real_dataset=base_dataset,                               │
│        alpha=0.1,                                               │
│        strategy='auto'  # ← Auto-select                        │
│    )                                                            │
│    else:                                                        │
│        ↓                                                        │
│    Standard DataLoader(base_dataset)                            │
└────────────────────────────────┬───────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ 4. Strategy Selection (if augmentation enabled)                │
│    if generated_data provided:                                  │
│        → Memory strategy (AugmentedTensorDataset)               │
│    elif cache_dir exists with samples:                          │
│        → Cache strategy (CachedAugmentedDataset)                │
│    else:                                                        │
│        → On-the-fly strategy (NotImplementedError)              │
└────────────────────────────────┬───────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ 5. Return DataLoader                                            │
│    PyTorch DataLoader wrapping the augmented/base dataset       │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 Training Data Flow (Physical Model)

```
┌────────────────────────────────────────────────────────────────┐
│ 1. Factory Request                                              │
│    TrainerFactory.create_dataset_for_physical(config)           │
└────────────────────────────────┬───────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ 2. Create Field Dataset                                         │
│    HybridDataset(                                               │
│        data_manager=DataManager(...),                           │
│        sim_indices=[0,1,2,...],                                 │
│        return_fields=True  # ← Fields for physical             │
│    )                                                            │
└────────────────────────────────┬───────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ 3. Return Dataset (No DataLoader)                              │
│    Physical trainer iterates directly over dataset             │
│    (No batching - one sample at a time)                        │
└────────────────────────────────────────────────────────────────┘
```

### 3.3 Data Loading Flow (Individual Sample)

```
┌────────────────────────────────────────────────────────────────┐
│ 1. Request Sample: dataset[idx]                                 │
└────────────────────────────────┬───────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ 2. Route to Base or Generated                                   │
│    if idx < num_real:                                           │
│        → real_dataset[idx]                                      │
│    else:                                                        │
│        → generated_data[idx - num_real]                         │
└────────────────────────────────┬───────────────────────────────┘
                                 ↓ (if real sample)
┌────────────────────────────────────────────────────────────────┐
│ 3. HybridDataset Processing                                     │
│    ├─ Determine sim_idx and start_frame                        │
│    ├─ Load from cache: _cached_load_simulation(sim_idx)        │
│    │   └─ LRU cache (maxsize=5 simulations)                    │
│    └─ If not cached: _load_simulation_uncached()               │
└────────────────────────────────┬───────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ 4. DataManager Loading                                          │
│    ├─ Check if cached: is_cached(sim_idx)                      │
│    ├─ If cached: load_from_cache(sim_idx)                      │
│    │   └─ torch.load("cache/{dset}/sim_{idx:06d}.pt")          │
│    └─ If not cached: load_and_cache_simulation()               │
│        ├─ Load PhiFlow Scene                                   │
│        ├─ Convert Fields → Tensors                             │
│        ├─ Extract metadata                                     │
│        └─ Save to cache                                        │
└────────────────────────────────┬───────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────┐
│ 5. Format Conversion                                            │
│    if return_fields=False:                                      │
│        → Slice tensors, return (initial_state, targets)         │
│    else:                                                        │
│        → Convert tensors → Fields, return (initial, targets)    │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. Core Components

### 4.1 DataManager

**Purpose**: Single source of truth for cached data

**File**: `src/data/data_manager.py`

**Key Features**:
- ✅ One-time Field → Tensor conversion
- ✅ Comprehensive metadata storage
- ✅ Cache validation (PDE params, resolution, domain)
- ✅ Automatic cache invalidation
- ✅ Checksum-based integrity checks

**API**:
```python
class DataManager:
    def __init__(self, raw_data_dir, cache_dir, config, validate_cache, auto_clear_invalid)
    
    # Main methods
    def get_or_load_simulation(sim_index, field_names, num_frames) -> Dict
    def load_and_cache_simulation(sim_index, field_names, num_frames) -> Dict
    def load_from_cache(sim_index) -> Dict
    def is_cached(sim_index, field_names, num_frames) -> bool
    
    # Utilities
    def get_cached_path(sim_index) -> Path
```

**Data Structure**:
```python
{
    "tensor_data": {
        "velocity": torch.Tensor,  # [T, C, H, W]
        "density": torch.Tensor,   # [T, C, H, W]
        # ... more fields
    },
    "metadata": {
        "version": "1.0",
        "phiflow_version": "2.4.0",
        "created_at": "2025-11-03T...",
        "field_metadata": {
            "velocity": {
                "shape": "...",
                "spatial_dims": ["x", "y"],
                "channel_dims": ["vector"],
                "extrapolation": "PERIODIC",
                "bounds_lower": (0.0, 0.0),
                "bounds_upper": (100.0, 100.0),
                "field_type": "staggered"
            }
        },
        "generation_params": {
            "pde_name": "BurgersModel",
            "pde_params": {"nu": 0.01},
            "domain": {"size_x": 100, "size_y": 100},
            "resolution": {"x": 128, "y": 128},
            "dt": 0.8
        },
        "checksums": {
            "pde_params_hash": "abc123...",
            "resolution_hash": "def456...",
            "domain_hash": "ghi789..."
        }
    }
}
```

**Complexity**: ⚠️ **HIGH**
- Multiple validation layers
- Complex metadata structure
- Tight coupling with config structure

---

### 4.2 HybridDataset

**Purpose**: PyTorch Dataset interface with dual-mode operation

**File**: `src/data/hybrid_dataset.py`

**Key Features**:
- ✅ Implements PyTorch `Dataset` interface
- ✅ Lazy loading with LRU cache (maxsize=5 sims)
- ✅ Sliding window support (multiple samples per simulation)
- ✅ Static/dynamic field separation
- ✅ Dual return modes: tensors OR Fields
- ✅ Automatic Field reconstruction from cached tensors

**API**:
```python
class HybridDataset(Dataset):
    def __init__(
        self,
        data_manager: DataManager,
        sim_indices: List[int],
        field_names: List[str],
        num_frames: int,
        num_predict_steps: int,
        dynamic_fields: List[str],
        static_fields: List[str],
        use_sliding_window: bool,
        return_fields: bool,  # ← KEY: tensors or Fields?
        max_cached_sims: int = 5,
        pin_memory: bool = True
    )
    
    # PyTorch Dataset interface
    def __len__(self) -> int
    def __getitem__(self, idx) -> Union[
        Tuple[torch.Tensor, torch.Tensor],          # Tensor mode
        Tuple[Dict[str, Field], Dict[str, Field]]   # Field mode
    ]
    
    # Utilities
    def clear_cache(self)
    def _build_sliding_window_index(self)
    def _convert_to_fields_with_start(self, data, start_frame)
```

**Return Formats**:

**Tensor Mode** (`return_fields=False`):
```python
initial_state: torch.Tensor    # [C_all, H, W] - ALL fields
rollout_targets: torch.Tensor  # [T, C_dynamic, H, W] - DYNAMIC fields only
```

**Field Mode** (`return_fields=True`):
```python
initial_fields: Dict[str, Field]  # ALL fields at t=0
target_fields: Dict[str, List[Field]]  # DYNAMIC fields at t=1..T
```

**Complexity**: ⚠️ **HIGH**
- Dual-mode operation adds complexity
- LRU cache management
- Complex sliding window indexing
- Field reconstruction logic

---

### 4.3 AugmentedTensorDataset

**Purpose**: Combine real and generated tensors

**File**: `src/data/augmentation/augmented_dataset.py`

**Key Features**:
- ✅ Count-based augmentation (no weights)
- ✅ Simple concatenation of real + generated
- ✅ All samples have weight = 1.0
- ✅ Returns 2-tuple (no weights)

**API**:
```python
class AugmentedTensorDataset(Dataset):
    def __init__(
        self,
        real_dataset: HybridDataset,
        generated_data: List[Tuple[Tensor, Tensor]],
        alpha: float,
        validate_count: bool,
        device: str
    )
    
    def __len__(self) -> int
    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]
    def get_statistics(self) -> Dict
```

**Structure**:
```
Indices:  [0 ... num_real-1] [num_real ... num_real+num_gen-1]
          └─ Real samples ─┘ └─── Generated samples ────────┘
```

**Complexity**: ✅ **LOW**
- Simple concatenation logic
- Straightforward indexing

---

### 4.4 CachedAugmentedDataset

**Purpose**: Lazy-load generated predictions from disk

**File**: `src/data/augmentation/cached_dataset.py`

**Key Features**:
- ✅ Lazy loading with LRU cache
- ✅ Disk I/O minimization
- ✅ Memory-efficient for large augmentation datasets
- ✅ Cache hit/miss statistics

**API**:
```python
class CachedAugmentedDataset(Dataset):
    def __init__(
        self,
        real_dataset: Dataset,
        cache_dir: str,
        alpha: float,
        cache_size: int = 128,
        validate_count: bool = True
    )
    
    def __len__(self) -> int
    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]
    def clear_cache(self)
    def get_cache_info(self) -> Dict
```

**Cache Structure**:
```
cache_dir/
  ├── sample_000000.pt  # {'input': Tensor, 'target': Tensor}
  ├── sample_000001.pt
  ├── ...
  └── metadata.json
```

**Complexity**: ⚠️ **MEDIUM**
- LRU cache management
- Disk I/O handling
- Cache validation

---

### 4.5 AdaptiveAugmentedDataLoader

**Purpose**: Automatically select augmentation strategy

**File**: `src/data/augmentation/adaptive_loader.py`

**Key Features**:
- ✅ Automatic strategy selection
- ✅ Strategy validation
- ✅ Unified interface regardless of strategy
- ⚠️ On-the-fly not implemented

**API**:
```python
class AdaptiveAugmentedDataLoader:
    def __init__(
        self,
        real_dataset: Dataset,
        alpha: float,
        generated_data: Optional[Dataset],
        cache_dir: Optional[str],
        cache_size: int,
        strategy: Optional[Literal['memory', 'cache', 'on_the_fly']]
    )
    
    def get_loader(self, batch_size, shuffle, num_workers) -> DataLoader
    def get_dataset(self) -> Dataset
    def get_strategy(self) -> str
    def get_info(self) -> Dict
```

**Strategy Selection Logic**:
```python
if generated_data is not None:
    → 'memory' strategy (AugmentedTensorDataset)
elif cache_dir exists and has samples:
    → 'cache' strategy (CachedAugmentedDataset)
else:
    → 'on_the_fly' strategy (NotImplementedError)
```

**Complexity**: ⚠️ **HIGH**
- Multiple code paths
- Strategy validation
- Error-prone configuration

---

## 5. Augmentation System

### 5.1 Augmentation Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Augmentation Module                         │
│  src/data/augmentation/                                      │
└─────────────────────────────────────────────────────────────┘
        │
        ├─ augmented_dataset.py
        │   ├─ AugmentedTensorDataset (in-memory, tensors)
        │   └─ AugmentedFieldDataset (in-memory, Fields)
        │
        ├─ cached_dataset.py
        │   └─ CachedAugmentedDataset (disk-cached, tensors)
        │
        ├─ adaptive_loader.py
        │   └─ AdaptiveAugmentedDataLoader (strategy selector)
        │
        ├─ cache_manager.py
        │   └─ CacheManager (lifecycle management)
        │
        └─ generation_utils.py
            ├─ generate_synthetic_predictions()
            ├─ generate_physical_predictions()
            └─ generate_and_cache_predictions()
```

### 5.2 Generation Utilities

**File**: `src/data/augmentation/generation_utils.py`

**Functions**:

1. **`generate_synthetic_predictions(model, dataset, alpha, device, batch_size)`**
   - Uses synthetic model to generate predictions
   - Count-based: `num_gen = int(len(dataset) * alpha)`
   - Returns: `(inputs_list, targets_list)`

2. **`generate_physical_predictions(model, dataset, alpha, device, rollout_steps)`**
   - Uses physical model for rollout predictions
   - Count-based: `num_gen = int(len(dataset) * alpha)`
   - Returns: `(initial_fields_list, target_fields_list)`

3. **`generate_and_cache_predictions(model, dataset, cache_manager, ...)`**
   - Generates and saves to cache in one step
   - Convenience wrapper for cache population

**Complexity**: ✅ **MEDIUM**
- Clear separation of concerns
- Could be simplified

---

### 5.3 CacheManager

**Purpose**: Lifecycle management for augmentation caches

**File**: `src/data/augmentation/cache_manager.py`

**Key Features**:
- ✅ Create/organize cache directory structure
- ✅ Save/load cached predictions
- ✅ Track metadata (generation time, model version)
- ✅ Monitor disk space usage
- ✅ Clean up old/invalid caches
- ✅ Validate cache integrity

**API**:
```python
class CacheManager:
    def __init__(self, cache_root, experiment_name, auto_create)
    
    # Saving
    def save_sample(index, input_data, target_data, format) -> Path
    def save_batch(start_index, inputs, targets, format) -> List[Path]
    
    # Loading
    def load_sample(index) -> Tuple[Tensor, Tensor]
    
    # Management
    def count_samples() -> int
    def list_samples() -> List[Path]
    def clear_cache(confirm)
    
    # Metadata
    def update_metadata(metadata: Dict)
    def load_metadata() -> Dict
    
    # Utilities
    def get_disk_usage() -> Dict
    def validate_cache(expected_count, check_loadable) -> Dict
    def exists() -> bool
    def is_empty() -> bool
```

**Directory Structure**:
```
cache_root/
  └── hybrid_generated/
      └── {experiment_name}/
          ├── metadata.json
          ├── sample_000000.pt
          ├── sample_000001.pt
          └── ...
```

**Complexity**: ⚠️ **MEDIUM-HIGH**
- Comprehensive feature set
- Many methods for different use cases
- Could be streamlined

---

## 6. Factory Integration

### 6.1 TrainerFactory Methods

**File**: `src/factories/trainer_factory.py`

**Key Methods**:

#### 1. `create_data_loader_for_synthetic()`

**Purpose**: Create DataLoader for synthetic training

**Parameters**:
- `config`: Full configuration
- `sim_indices`: Simulations to load
- `batch_size`: Batch size
- `shuffle`: Shuffle flag
- `use_sliding_window`: Sliding window flag

**Flow**:
```python
def create_data_loader_for_synthetic(config, ...):
    # 1. Setup paths
    raw_data_dir = ...
    cache_dir = ...
    
    # 2. Create DataManager
    data_manager = DataManager(raw_data_dir, cache_dir, config)
    
    # 3. Extract field specs
    field_names = config['data']['fields']
    dynamic_fields = list(config['model']['synthetic']['output_specs'].keys())
    static_fields = [f for f in input_specs if f not in output_specs]
    
    # 4. Create base dataset
    dataset = HybridDataset(
        data_manager=data_manager,
        sim_indices=sim_indices,
        field_names=field_names,
        return_fields=False  # ← Tensors for synthetic
    )
    
    # 5. Check augmentation
    if augmentation_enabled:
        # Create AdaptiveAugmentedDataLoader
        loader = AdaptiveAugmentedDataLoader(...)
        return loader.get_loader(batch_size, shuffle)
    else:
        # Standard DataLoader
        return DataLoader(dataset, batch_size, shuffle)
```

**Complexity**: ⚠️ **HIGH**
- Many configuration extraction steps
- Complex branching logic
- Tight coupling with config structure

---

#### 2. `create_dataset_for_physical()`

**Purpose**: Create HybridDataset for physical training

**Parameters**:
- `config`: Full configuration
- `sim_indices`: Simulations to load
- `use_sliding_window`: Sliding window flag

**Flow**:
```python
def create_dataset_for_physical(config, ...):
    # 1. Setup paths (same as synthetic)
    raw_data_dir = ...
    cache_dir = ...
    
    # 2. Create DataManager
    data_manager = DataManager(raw_data_dir, cache_dir, config)
    
    # 3. Create dataset
    dataset = HybridDataset(
        data_manager=data_manager,
        sim_indices=sim_indices,
        field_names=field_names,
        return_fields=True  # ← Fields for physical
    )
    
    return dataset  # No DataLoader wrapper
```

**Complexity**: ⚠️ **MEDIUM**
- Simpler than synthetic (no augmentation)
- Still has config extraction overhead

---

## 7. Usage Patterns

### 7.1 Synthetic Training (No Augmentation)

```python
# In run.py or trainer
config = load_config()

# Create data loader via factory
data_loader = TrainerFactory.create_data_loader_for_synthetic(
    config=config,
    shuffle=True,
    use_sliding_window=True
)

# Train
for epoch in range(num_epochs):
    for initial_state, targets in data_loader:
        # initial_state: [B, C_all, H, W]
        # targets: [B, T, C_dynamic, H, W]
        predictions = model(initial_state)
        loss = criterion(predictions, targets)
        loss.backward()
```

### 7.2 Synthetic Training (With Augmentation)

```python
# In run.py or trainer
config = load_config()
config['trainer_params']['augmentation']['enabled'] = True
config['trainer_params']['augmentation']['strategy'] = 'cache'
config['trainer_params']['augmentation']['alpha'] = 0.1

# Create augmented data loader
data_loader = TrainerFactory.create_data_loader_for_synthetic(
    config=config,
    shuffle=True
)
# → Returns AdaptiveAugmentedDataLoader.get_loader()
#   → Wraps CachedAugmentedDataset
#      → Wraps HybridDataset

# Train (same as before)
for epoch in range(num_epochs):
    for initial_state, targets in data_loader:
        predictions = model(initial_state)
        loss = criterion(predictions, targets)
        loss.backward()
```

### 7.3 Physical Training

```python
# In run.py or trainer
config = load_config()

# Create dataset (no DataLoader)
dataset = TrainerFactory.create_dataset_for_physical(
    config=config,
    use_sliding_window=True
)

# Train (one sample at a time)
for initial_fields, target_fields in dataset:
    # initial_fields: Dict[str, Field]
    # target_fields: Dict[str, List[Field]]
    
    # Run optimization for this sample
    loss = optimize_parameters(model, initial_fields, target_fields)
```

### 7.4 Hybrid Training

```python
# In HybridTrainer
def _train_synthetic_with_augmentation(self, generated_data):
    # 1. Create real dataset
    real_dataset = self._create_hybrid_dataset(
        self.trainer_config["train_sim"],
        return_fields=False  # Tensors
    )
    
    # 2. Create augmented dataset
    augmented_dataset = AugmentedTensorDataset(
        real_dataset=real_dataset,
        generated_data=generated_data,  # From physical model
        alpha=self.alpha
    )
    
    # 3. Create DataLoader
    train_loader = DataLoader(augmented_dataset, batch_size=32)
    
    # 4. Train
    self.synthetic_trainer.train(train_loader, num_epochs)

def _train_physical_with_augmentation(self, generated_data):
    # 1. Create real dataset
    real_dataset = self._create_hybrid_dataset(
        self.trainer_config["train_sim"],
        return_fields=True  # Fields
    )
    
    # 2. Create augmented dataset
    augmented_dataset = AugmentedFieldDataset(
        real_dataset=real_dataset,
        generated_data=generated_data,  # From synthetic model
        alpha=self.alpha
    )
    
    # 3. Iterate directly (no DataLoader)
    for initial_fields, target_fields in augmented_dataset:
        self.physical_trainer._train_sample(initial_fields, target_fields)
```

---

## 8. Complexity Analysis

### 8.1 Quantitative Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| **Total Classes** | 10 | ⚠️ High |
| **Abstraction Layers** | 5 | ⚠️ Deep |
| **Public Methods** | 60+ | ⚠️ High |
| **Configuration Parameters** | 15+ | ⚠️ Complex |
| **Code Duplication** | Medium | ⚠️ Some redundancy |
| **Cyclomatic Complexity** | Medium | ✅ Acceptable |

### 8.2 Complexity by Component

| Component | LOC | Methods | Complexity | Maintainability |
|-----------|-----|---------|------------|-----------------|
| `DataManager` | ~400 | 6 | HIGH | Medium |
| `HybridDataset` | ~500 | 10 | HIGH | Medium |
| `AugmentedTensorDataset` | ~150 | 4 | LOW | High |
| `AugmentedFieldDataset` | ~150 | 4 | LOW | High |
| `CachedAugmentedDataset` | ~200 | 6 | MEDIUM | Medium |
| `AdaptiveAugmentedDataLoader` | ~250 | 8 | HIGH | Low |
| `CacheManager` | ~400 | 15 | MEDIUM-HIGH | Medium |
| `generation_utils` | ~300 | 5 | MEDIUM | Medium |
| `TrainerFactory` (data methods) | ~350 | 2 | HIGH | Low |

### 8.3 Coupling Analysis

**High Coupling**:
- ✅ `DataManager` ↔ Config structure (tight)
- ✅ `HybridDataset` ↔ `DataManager` (necessary)
- ✅ `TrainerFactory` ↔ Config structure (tight)
- ⚠️ `AdaptiveAugmentedDataLoader` ↔ Multiple dataset types (complex)

**Medium Coupling**:
- ✅ `AugmentedTensorDataset` ↔ `HybridDataset`
- ✅ `CachedAugmentedDataset` ↔ `CacheManager`

**Low Coupling**:
- ✅ Generation utilities (well encapsulated)

---

## 9. Pain Points & Redundancies

### 9.1 Major Pain Points

#### 1. **Deep Abstraction Layers** ⚠️⚠️⚠️

**Issue**: 5 layers deep makes debugging difficult

**Example Call Stack**:
```
TrainerFactory.create_data_loader_for_synthetic()
  → AdaptiveAugmentedDataLoader()
    → CachedAugmentedDataset()
      → HybridDataset()
        → DataManager()
          → torch.load()
```

**Impact**: 
- Hard to trace data flow
- Difficult to debug issues
- Steep learning curve for new developers

---

#### 2. **Dual-Mode HybridDataset** ⚠️⚠️

**Issue**: `return_fields` parameter creates two different behaviors

**Complexity**:
```python
if return_fields:
    # Convert tensors → Fields
    # Complex reconstruction logic
    return (initial_fields, target_fields)
else:
    # Slice tensors directly
    return (initial_state, rollout_targets)
```

**Impact**:
- Difficult to test both modes
- Error-prone conversions
- Cognitive overhead

---

#### 3. **Config Structure Coupling** ⚠️⚠️⚠️

**Issue**: Many components tightly coupled to config structure

**Example**:
```python
# In TrainerFactory
field_names = config['data']['fields']
input_specs = config['model']['synthetic']['input_specs']
output_specs = config['model']['synthetic']['output_specs']
dynamic_fields = list(output_specs.keys())
static_fields = [f for f in input_specs if f not in output_specs]
```

**Impact**:
- Brittle to config changes
- Hard to refactor
- Difficult to use outside of config system

---

#### 4. **Strategy Selection Complexity** ⚠️⚠️

**Issue**: `AdaptiveAugmentedDataLoader` has complex auto-selection logic

**Complexity**:
```python
if generated_data is not None:
    strategy = 'memory'
elif cache_dir exists and has_samples:
    strategy = 'cache'
else:
    strategy = 'on_the_fly'  # Not implemented
```

**Impact**:
- Hard to predict behavior
- Error-prone configuration
- Hidden failures (e.g., empty cache → fallback)

---

#### 5. **Redundant Dataset Creation** ⚠️

**Issue**: Similar code in multiple places

**Examples**:
1. `TrainerFactory.create_data_loader_for_synthetic()`
2. `TrainerFactory.create_dataset_for_physical()`
3. `HybridTrainer._create_hybrid_dataset()`

**Duplication**:
```python
# Repeated in 3 places:
raw_data_dir = project_root / data_config["data_dir"] / data_config["dset_name"]
cache_dir = project_root / data_config["data_dir"] / "cache"
data_manager = DataManager(raw_data_dir, cache_dir, config)
field_names = data_config["fields"]
# ... extract specs ...
dataset = HybridDataset(data_manager, sim_indices, field_names, ...)
```

---

#### 6. **Cache Management Complexity** ⚠️

**Issue**: Two separate caching systems

**Systems**:
1. **DataManager cache**: PhiFlow Scene → Tensors
   - Location: `data/cache/{dset_name}/`
   - Purpose: Avoid redundant Field conversions

2. **Augmentation cache**: Model predictions
   - Location: `data/cache/hybrid_generated/{experiment}/`
   - Purpose: Store generated predictions

**Impact**:
- Confusing for users
- Separate management APIs
- Disk space concerns

---

### 9.2 Code Duplication

#### 1. **Path Construction** (3 locations)

```python
# Duplicated in:
# - TrainerFactory.create_data_loader_for_synthetic()
# - TrainerFactory.create_dataset_for_physical()
# - HybridTrainer._create_hybrid_dataset()

project_root = Path(config.get("project_root", "."))
raw_data_dir = project_root / data_config["data_dir"] / data_config["dset_name"]
cache_dir = project_root / data_config["data_dir"] / "cache"
```

#### 2. **Field Spec Extraction** (2 locations)

```python
# Duplicated in:
# - TrainerFactory.create_data_loader_for_synthetic()
# - HybridTrainer._create_hybrid_dataset()

input_specs = model_config["synthetic"]["input_specs"]
output_specs = model_config["synthetic"]["output_specs"]
dynamic_fields = list(output_specs.keys())
static_fields = [f for f in input_specs.keys() if f not in output_specs]
```

#### 3. **DataManager Creation** (3 locations)

```python
# Same code in multiple places
data_manager = DataManager(
    raw_data_dir=str(raw_data_dir),
    cache_dir=str(cache_dir),
    config=config,
    validate_cache=data_config.get("validate_cache", True),
    auto_clear_invalid=data_config.get("auto_clear_invalid", False),
)
```

---

### 9.3 Unnecessary Complexity

#### 1. **AugmentedFieldDataset Not a PyTorch Dataset**

**Issue**: Different interface from `AugmentedTensorDataset`

```python
# AugmentedTensorDataset
class AugmentedTensorDataset(Dataset):  # ← Is a Dataset
    def __getitem__(self, idx): ...

# AugmentedFieldDataset
class AugmentedFieldDataset:  # ← NOT a Dataset
    def __iter__(self): ...
```

**Impact**: 
- Inconsistent interface
- Cannot use with DataLoader
- Harder to test

---

#### 2. **Separate Augmentation Classes**

**Issue**: `AugmentedTensorDataset` vs `AugmentedFieldDataset`

**Similarity**: ~90% identical code

**Difference**: Data types (tensors vs Fields)

**Could be**: Single class with generic typing

---

#### 3. **AdaptiveAugmentedDataLoader Wrapper**

**Issue**: Adds layer without clear benefit

**Current**:
```python
adaptive = AdaptiveAugmentedDataLoader(dataset, alpha, strategy)
loader = adaptive.get_loader(batch_size)
```

**Could be**:
```python
dataset = create_augmented_dataset(dataset, alpha, strategy)
loader = DataLoader(dataset, batch_size)
```

---

### 9.4 Configuration Complexity

**Config Parameters for Data Loading**:

```yaml
data:
  data_dir: "data"
  dset_name: "burgers_128"
  fields: ["velocity"]
  validate_cache: true
  auto_clear_invalid: false

model:
  synthetic:
    input_specs: {velocity: 2}
    output_specs: {velocity: 2}
  
trainer_params:
  train_sim: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  batch_size: 16
  num_predict_steps: 10
  
  augmentation:
    enabled: true
    alpha: 0.1
    strategy: "cache"  # or "memory" or "on_the_fly"
    
    cache:
      enabled: true
      experiment_name: "burgers_128_alpha0.1"
      max_memory_samples: 1000
      format: "dict"
    
    on_the_fly:
      batch_size: 32
      rollout_steps: 10

cache:
  root: "data/cache"
  auto_create: true
  validation:
    check_on_load: true
```

**Total Parameters**: 15+

**Issue**: Many parameters, complex interdependencies

---

## 10. Summary

### Current State

**Strengths** ✅:
- Comprehensive feature set
- Good separation of caching and loading
- Flexible augmentation strategies
- Well-documented individual components

**Weaknesses** ⚠️:
- Too many abstraction layers (5 layers)
- Complex factory methods
- Tight coupling to config structure
- Code duplication
- Dual-mode dataset adds complexity
- Strategy selection is error-prone

### Complexity Scores

| Aspect | Score | Status |
|--------|-------|--------|
| **Abstraction Depth** | 5 layers | ⚠️ Too Deep |
| **Number of Classes** | 10+ | ⚠️ High |
| **Configuration Complexity** | 15+ params | ⚠️ Complex |
| **Code Duplication** | Medium | ⚠️ Some |
| **Coupling** | High | ⚠️ Tight |
| **Overall Complexity** | 8/10 | ⚠️ Very High |

### Opportunities for Simplification

See separate document: **DATA_LOADING_SIMPLIFICATION_PROPOSAL.md**

Key areas:
1. 🎯 Reduce abstraction layers (5 → 3)
2. 🎯 Eliminate dual-mode dataset
3. 🎯 Simplify factory methods
4. 🎯 Reduce config coupling
5. 🎯 Consolidate augmentation classes
6. 🎯 Remove unnecessary wrappers

---

**Document Version**: 1.0  
**Last Updated**: November 3, 2025  
**Status**: Ready for simplification review
