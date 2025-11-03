# Phase 1 API - Usage Examples

## Overview

After Phase 1 migration, trainers now use **explicit data passing**. This document shows how to use the new API.

---

## Running Training (run.py)

### Command Line Usage (Unchanged)

```bash
# Synthetic training
python run.py --config-name=burgers_experiment

# Physical training
python run.py --config-name=burgers_physical_experiment

# Override parameters
python run.py --config-name=burgers_experiment trainer_params.epochs=200
```

### What Changed Internally

**Before (Old API)**:
```python
# run.py
trainer = TrainerFactory.create_trainer(config)
trainer.train()  # Trainer creates data internally
```

**After (Phase 1 API)**:
```python
# run.py
trainer = TrainerFactory.create_trainer(config)  # Creates model + trainer

if model_type == "synthetic":
    data_loader = TrainerFactory.create_data_loader_for_synthetic(config)
    trainer.train(data_source=data_loader, num_epochs=100)

elif model_type == "physical":
    dataset = TrainerFactory.create_dataset_for_physical(config)
    trainer.train(data_source=dataset, num_epochs=1)
```

---

## TrainerFactory API

### Creating Trainers

#### Synthetic Trainer

```python
from src.factories import TrainerFactory

# Create trainer (internally creates model and passes to SyntheticTrainer)
trainer = TrainerFactory.create_trainer(config)

# Or create components manually:
from src.factories import ModelFactory
from src.training.synthetic import SyntheticTrainer

model = ModelFactory.create_synthetic_model(config)
trainer = SyntheticTrainer(config, model)
```

#### Physical Trainer

```python
from src.factories import TrainerFactory

# Create trainer (internally creates model, extracts learnable params, passes both)
trainer = TrainerFactory.create_trainer(config)

# Or create components manually:
from src.factories import ModelFactory
from src.training.physical import PhysicalTrainer
from phi.math import math

model = ModelFactory.create_physical_model(config)
learnable_params = [
    math.tensor(0.01)  # viscosity initial guess
]
trainer = PhysicalTrainer(config, model, learnable_params)
```

---

## Creating Data Loaders/Datasets

### For Synthetic Training

```python
from src.factories import TrainerFactory

# Create DataLoader with defaults from config
data_loader = TrainerFactory.create_data_loader_for_synthetic(config)

# Custom data loader
data_loader = TrainerFactory.create_data_loader_for_synthetic(
    config,
    sim_indices=[0, 1, 2],  # Specific simulations
    batch_size=64,          # Custom batch size
    shuffle=True,           # Shuffle data
    use_sliding_window=True # Always True in Phase 1
)

# Train with data
trainer.train(data_source=data_loader, num_epochs=100)
```

### For Physical Training

```python
from src.factories import TrainerFactory

# Create HybridDataset with defaults from config
dataset = TrainerFactory.create_dataset_for_physical(config)

# Custom dataset
dataset = TrainerFactory.create_dataset_for_physical(
    config,
    sim_indices=[0],        # Single simulation
    use_sliding_window=True # Always True in Phase 1
)

# Train with data (physical typically single epoch per sample)
trainer.train(data_source=dataset, num_epochs=1)
```

---

## Manual Workflow (Advanced)

### Synthetic Training - Full Control

```python
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.factories import ModelFactory
from src.training.synthetic import SyntheticTrainer
from src.data import DataManager, HybridDataset

# 1. Create model
model = ModelFactory.create_synthetic_model(config)

# 2. Create trainer with model
trainer = SyntheticTrainer(config, model)

# 3. Setup data pipeline
project_root = Path(config["project_root"])
raw_data_dir = project_root / config["data"]["data_dir"] / config["data"]["dset_name"]
cache_dir = project_root / config["data"]["data_dir"] / "cache"

data_manager = DataManager(
    raw_data_dir=str(raw_data_dir),
    cache_dir=str(cache_dir),
    config=config,
)

# 4. Create dataset
dataset = HybridDataset(
    data_manager=data_manager,
    sim_indices=[0, 1, 2, 3, 4],
    field_names=config["data"]["fields"],
    num_frames=None,  # Load all frames
    num_predict_steps=config["trainer_params"]["num_predict_steps"],
    dynamic_fields=list(config["model"]["synthetic"]["output_specs"].keys()),
    static_fields=[
        f for f in config["model"]["synthetic"]["input_specs"].keys()
        if f not in config["model"]["synthetic"]["output_specs"]
    ],
    use_sliding_window=True,  # ALWAYS True in Phase 1
)

# 5. Create DataLoader
data_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

# 6. Train with explicit data
trainer.train(data_source=data_loader, num_epochs=100)

# 7. Optionally train with different data (reuses optimizer state!)
augmented_data_loader = DataLoader(...)
trainer.train(data_source=augmented_data_loader, num_epochs=50)
```

### Physical Training - Full Control

```python
from pathlib import Path
from phi.math import math

from src.factories import ModelFactory
from src.training.physical import PhysicalTrainer
from src.data import DataManager, HybridDataset

# 1. Create model
model = ModelFactory.create_physical_model(config)

# 2. Create learnable parameters
learnable_params = [
    math.tensor(0.01)  # viscosity initial guess
]

# 3. Create trainer with model and params
trainer = PhysicalTrainer(config, model, learnable_params)

# 4. Setup data pipeline
project_root = Path(config["project_root"])
raw_data_dir = project_root / config["data"]["data_dir"] / config["data"]["dset_name"]
cache_dir = project_root / config["data"]["data_dir"] / "cache"

data_manager = DataManager(
    raw_data_dir=str(raw_data_dir),
    cache_dir=str(cache_dir),
    config=config,
)

# 5. Create dataset (returns PhiFlow fields, not tensors)
dataset = HybridDataset(
    data_manager=data_manager,
    sim_indices=[0],  # Single simulation for physical
    field_names=config["data"]["fields"],
    num_frames=config["trainer_params"]["num_predict_steps"] + 1,
    num_predict_steps=config["trainer_params"]["num_predict_steps"],
    return_fields=True,  # IMPORTANT: PhiFlow fields, not tensors
    use_sliding_window=True,  # ALWAYS True in Phase 1
)

# 6. Train with explicit data
trainer.train(data_source=dataset, num_epochs=1)

# 7. Optionally continue with different data
augmented_dataset = HybridDataset(...)
trainer.train(data_source=augmented_dataset, num_epochs=1)
```

---

## Key Differences from Old API

### 1. Model Creation

**Old**: Trainers created models internally
```python
trainer = SyntheticTrainer(config)  # Creates model inside __init__
```

**New**: Models created externally and passed in
```python
model = ModelFactory.create_synthetic_model(config)
trainer = SyntheticTrainer(config, model)
```

### 2. Data Access

**Old**: Trainers created data loaders internally
```python
trainer = SyntheticTrainer(config)  # Creates data_loader inside __init__
trainer.train()  # Uses internal data_loader
```

**New**: Data loaders passed to train()
```python
trainer = SyntheticTrainer(config, model)  # No data loader in __init__
data_loader = TrainerFactory.create_data_loader_for_synthetic(config)
trainer.train(data_source=data_loader, num_epochs=100)
```

### 3. Training Invocation

**Old**: No arguments to train()
```python
trainer.train()  # Uses epochs from config, internal data
```

**New**: Explicit data and epochs
```python
trainer.train(data_source=data_loader, num_epochs=100)
```

### 4. Reusability

**Old**: One trainer per training run
```python
trainer = SyntheticTrainer(config)
trainer.train()
# Can't reuse with different data
```

**New**: Persistent trainers, multiple training runs
```python
trainer = SyntheticTrainer(config, model)

# First training run
trainer.train(data_source=real_data_loader, num_epochs=100)

# Second run with augmented data (preserves optimizer state!)
trainer.train(data_source=augmented_data_loader, num_epochs=50)
```

---

## Configuration Requirements

### Synthetic Training Config

```yaml
run_params:
  model_type: synthetic  # Specifies SyntheticTrainer

model:
  synthetic:
    name: UNet
    input_specs:
      velocity: 2
      pressure: 1
    output_specs:
      velocity: 2
      pressure: 1

data:
  dset_name: burgers_128
  data_dir: data
  fields: [velocity, pressure]

trainer_params:
  train_sim: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  batch_size: 32
  epochs: 100
  num_predict_steps: 32
  learning_rate: 0.001
```

### Physical Training Config

```yaml
run_params:
  model_type: physical  # Specifies PhysicalTrainer

model:
  physical:
    name: BurgersPDE
    pde_params:
      viscosity: 0.01  # True value (for testing)

data:
  dset_name: burgers_128
  data_dir: data
  fields: [velocity]

trainer_params:
  train_sim: [0]  # Single simulation
  epochs: 1       # Physical typically single epoch
  num_predict_steps: 32
  learnable_parameters:
    - name: viscosity
      initial_guess: 0.02  # Starting point for optimization
  method: L-BFGS-B
  abs_tol: 1e-6
  max_iterations: 100
  suppress_convergence_errors: false  # Set true for hybrid
```

---

## Troubleshooting

### ImportError: cannot import ModelFactory

**Solution**: Make sure you're importing from `src.factories`:
```python
from src.factories import ModelFactory, TrainerFactory
```

### TypeError: __init__() missing required positional argument 'model'

**Cause**: Trying to use old API `SyntheticTrainer(config)`

**Solution**: Use factory or pass model:
```python
# Option 1: Use factory
trainer = TrainerFactory.create_trainer(config)

# Option 2: Manual creation
model = ModelFactory.create_synthetic_model(config)
trainer = SyntheticTrainer(config, model)
```

### TypeError: train() missing required positional argument 'data_source'

**Cause**: Trying to call `trainer.train()` without arguments

**Solution**: Pass data source explicitly:
```python
# For synthetic
data_loader = TrainerFactory.create_data_loader_for_synthetic(config)
trainer.train(data_source=data_loader, num_epochs=100)

# For physical
dataset = TrainerFactory.create_dataset_for_physical(config)
trainer.train(data_source=dataset, num_epochs=1)
```

### ValueError: No 'learnable_parameters' defined

**Cause**: Physical trainer config missing learnable parameters

**Solution**: Add to config:
```yaml
trainer_params:
  learnable_parameters:
    - name: viscosity
      initial_guess: 0.02
```

---

## What's Next?

- ✅ **Phase 1 Complete**: Base trainers and factories updated
- ⏩ **Phase 2 Next**: Test standalone training with new API
- 🔜 **Phase 3**: Implement data augmentation (AugmentedDataset, CacheManager)
- 🔜 **Phase 4**: Implement HybridTrainer
- 🔜 **Phase 5**: Full integration testing

---

## Quick Reference

### Imports

```python
# Factories
from src.factories import ModelFactory, TrainerFactory

# Trainers
from src.training.synthetic import SyntheticTrainer
from src.training.physical import PhysicalTrainer

# Data
from src.data import DataManager, HybridDataset

# PyTorch
import torch
from torch.utils.data import DataLoader

# PhiFlow
from phi.math import math
```

### Factory Methods

```python
# Create trainers
trainer = TrainerFactory.create_trainer(config)

# Create models
synthetic_model = ModelFactory.create_synthetic_model(config)
physical_model = ModelFactory.create_physical_model(config)

# Create data
data_loader = TrainerFactory.create_data_loader_for_synthetic(config)
dataset = TrainerFactory.create_dataset_for_physical(config)
```

### Training

```python
# Synthetic
trainer.train(data_source=data_loader, num_epochs=100)

# Physical
trainer.train(data_source=dataset, num_epochs=1)
```
