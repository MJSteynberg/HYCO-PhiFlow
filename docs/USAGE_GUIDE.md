# HYCO-PhiFlow Usage Guide

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running Experiments](#running-experiments)
- [Configuration System](#configuration-system)
- [Workflow Modes](#workflow-modes)
- [Command Reference](#command-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## Introduction

**HYCO-PhiFlow** (Hybrid Computational Physics with PhiFlow) is a framework for training neural networks to emulate physics simulations. It combines:

- **PhiFlow**: High-performance PDE solvers for physics simulation
- **PyTorch**: Deep learning framework for neural network training
- **Hydra**: Flexible configuration management

### What Can You Do?

1. **Generate** physics simulation data
2. **Train** neural networks to emulate physics
3. **Evaluate** trained models against ground truth
4. **Experiment** with different PDEs, architectures, and hyperparameters

### Supported PDEs

- **Burgers' Equation**: 2D fluid flow with advection and diffusion
- **Heat Equation**: Thermal diffusion
- **Smoke Simulation**: Full fluid simulation with buoyancy

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- Git

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd HYCO-PhiFlow
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python run.py --help
   ```

---

## Quick Start

### 1. Generate Training Data

Generate 25 Burgers equation simulations:

```bash
python run.py --config-name=burgers_experiment run_params.mode=generate
```

This creates data in `data/burgers_128/sim_000000/` through `sim_000024/`.

### 2. Train a Model

Train a U-Net to emulate the Burgers equation:

```bash
python run.py --config-name=burgers_experiment run_params.mode=train
```

The trained model is saved to `results/models/burgers_unet_128.pt`.

### 3. Evaluate the Model

Evaluate the trained model on test simulations:

```bash
python run.py --config-name=burgers_experiment run_params.mode=evaluate
```

Results are saved to `results/evaluation/`.

### 4. Full Pipeline

Run all three steps sequentially:

```bash
python run.py --config-name=burgers_experiment run_params.mode=[generate,train,evaluate]
```

---

## Running Experiments

The main entry point is `run.py`, which uses Hydra for configuration management.

### Basic Syntax

```bash
python run.py --config-name=<experiment> [overrides...]
```

### Available Experiments

Located in `conf/`:

| Experiment | Description |
|------------|-------------|
| `burgers_experiment` | Full Burgers equation experiment (synthetic model) |
| `burgers_physical_experiment` | Burgers equation with physical model training |
| `burgers_quick_test` | Fast test with minimal data |
| `smoke_experiment` | Smoke simulation experiment |
| `smoke_physical_experiment` | Smoke with physical model training |
| `heat_physical_experiment` | Heat equation experiment |

### Experiment Structure

Each experiment configuration includes:

```yaml
defaults:
  - data: burgers_128           # Dataset configuration
  - model/physical: burgers      # Physical model
  - model/synthetic: unet        # Neural network model
  - trainer: synthetic           # Training configuration
  - generation: default          # Data generation settings
  - evaluation: default          # Evaluation settings
  - _self_

run_params:
  experiment_name: 'burgers_unet_128_v1'
  notes: 'Training UNet on Burgers 128x128'
  mode: ['train', 'evaluate']
  model_type: 'synthetic'
```

---

## Configuration System

HYCO-PhiFlow uses **Hydra's compositional configuration**. Configurations are organized hierarchically:

```
conf/
├── config.yaml                  # Base configuration
├── burgers_experiment.yaml      # Experiment-specific overrides
├── data/
│   ├── burgers_128.yaml        # Dataset: Burgers 128x128
│   ├── heat_64.yaml            # Dataset: Heat 64x64
│   └── smoke_128.yaml          # Dataset: Smoke 128x128
├── model/
│   ├── physical/
│   │   ├── burgers.yaml        # Burgers physical model config
│   │   ├── heat.yaml           # Heat physical model config
│   │   └── smoke.yaml          # Smoke physical model config
│   └── synthetic/
│       └── unet.yaml           # U-Net architecture config
├── trainer/
│   ├── synthetic.yaml          # Training hyperparameters
│   ├── synthetic_quick.yaml    # Fast training for testing
│   ├── physical.yaml           # Physical model training
│   └── physical_quick.yaml     # Fast physical training
├── generation/
│   └── default.yaml            # Data generation parameters
└── evaluation/
    └── default.yaml            # Evaluation parameters
```

### Configuration Hierarchy

Configurations are composed via the `defaults` list:

```yaml
defaults:
  - data: burgers_128           # Load conf/data/burgers_128.yaml
  - model/physical: burgers      # Load conf/model/physical/burgers.yaml
  - model/synthetic: unet        # Load conf/model/synthetic/unet.yaml
  - trainer: synthetic           # Load conf/trainer/synthetic.yaml
  - _self_                       # Apply current file's overrides last
```

### Configuration Parameters

#### **Run Parameters** (`run_params`)

```yaml
run_params:
  experiment_name: 'my_experiment'   # Experiment identifier
  notes: 'Description'               # Optional notes
  mode: ['generate', 'train', 'evaluate']  # Tasks to execute
  model_type: 'synthetic'            # 'synthetic' or 'physical'
```

#### **Data Parameters** (`data`)

```yaml
data_dir: 'data/'                    # Root data directory
dset_name: 'burgers_128'             # Dataset name (subdirectory)
fields: ['velocity']                 # Fields to load
fields_scheme: 'VV'                  # Field type scheme (V=vector, S=scalar)
cache_dir: 'data/cache'              # Cache directory
validate_cache: true                 # Validate cached data
auto_clear_invalid: true             # Auto-clear invalid cache
```

#### **Physical Model Parameters** (`model.physical`)

```yaml
name: 'BurgersModel'                 # Model class name (from registry)
domain:
  size_x: 100                        # Domain width
  size_y: 100                        # Domain height
resolution:
  x: 128                             # Grid resolution in x
  y: 128                             # Grid resolution in y
dt: 0.8                              # Time step
pde_params:
  batch_size: 1                      # Number of parallel simulations
  nu: 0.1                            # Model-specific parameters
```

#### **Synthetic Model Parameters** (`model.synthetic`)

```yaml
name: 'UNet'                         # Model class name (from registry)
model_path: 'results/models'         # Directory to save models
model_save_name: 'burgers_unet_128'  # Model filename

input_specs:                         # Input fields and channel counts
  velocity: 2

output_specs:                        # Output fields and channel counts
  velocity: 2

architecture:                        # Architecture parameters
  levels: 4
  filters: 64
  batch_norm: true
```

#### **Training Parameters** (`trainer_params`)

```yaml
learning_rate: 0.0001                # Learning rate
batch_size: 16                       # Training batch size
epochs: 100                          # Number of epochs
num_predict_steps: 4                 # Rollout length during training

train_sim: [0, 1, 2, ...]           # Simulation indices for training
val_sim: null                        # Validation simulations (null = auto)

use_sliding_window: true             # Use sliding window for temporal data

optimizer: 'adam'                    # Optimizer: 'adam' or 'sgd'
scheduler: 'cosine'                  # LR scheduler: 'cosine' or 'step'
weight_decay: 0.0                    # Weight decay

save_interval: 10                    # Save checkpoint every N epochs
save_best_only: true                 # Only save best model
```

#### **Generation Parameters** (`generation_params`)

```yaml
num_simulations: 25                  # Number of simulations to generate
total_steps: 100                     # Time steps per simulation
save_interval: 1                     # Save every N frames
seed: null                           # Random seed (null = random)
```

#### **Evaluation Parameters** (`evaluation_params`)

```yaml
test_sim: [20, 21, 22, 23, 24]      # Test simulation indices
num_frames: 51                       # Number of frames to evaluate
metrics: ['mse', 'mae', 'rmse']     # Metrics to compute

keyframe_count: 5                    # Number of keyframes to visualize
animation_fps: 10                    # Animation frame rate
save_animations: true                # Generate animations
save_plots: true                     # Generate plots

output_dir: 'results/evaluation'     # Output directory
```

---

## Workflow Modes

The `run_params.mode` parameter controls which tasks to execute.

### Mode: `generate`

Generates physics simulation data using PhiFlow.

**What it does:**
1. Instantiates the physical model from config
2. Runs simulations with random initial conditions
3. Saves trajectories to disk in PhiFlow format

**Output:**
```
data/burgers_128/
├── sim_000000/
│   ├── velocity_000000.npz
│   ├── velocity_000001.npz
│   └── ...
├── sim_000001/
└── ...
```

**Example:**
```bash
python run.py --config-name=burgers_experiment run_params.mode=generate
```

**Override examples:**
```bash
# Generate 50 simulations instead of default
python run.py --config-name=burgers_experiment run_params.mode=generate generation_params.num_simulations=50

# Use different viscosity
python run.py --config-name=burgers_experiment run_params.mode=generate model.physical.pde_params.nu=0.05

# Change resolution
python run.py --config-name=burgers_experiment run_params.mode=generate model.physical.resolution.x=256 model.physical.resolution.y=256
```

---

### Mode: `train`

Trains a neural network to emulate the physical model.

**What it does:**
1. Loads training data from disk
2. Instantiates the synthetic model
3. Trains using specified optimizer and loss function
4. Saves checkpoints and best model

**Output:**
```
results/models/
└── burgers_unet_128.pt          # Trained model weights

outputs/YYYY-MM-DD/HH-MM-SS/     # Hydra output directory
├── .hydra/
│   └── config.yaml              # Full resolved config
├── train.log                     # Training logs
└── checkpoints/
    ├── epoch_010.pt
    ├── epoch_020.pt
    └── ...
```

**Example:**
```bash
python run.py --config-name=burgers_experiment run_params.mode=train
```

**Override examples:**
```bash
# Train for 200 epochs with larger batch size
python run.py --config-name=burgers_experiment run_params.mode=train trainer_params.epochs=200 trainer_params.batch_size=32

# Use specific training simulations
python run.py --config-name=burgers_experiment run_params.mode=train 'trainer_params.train_sim=[0,1,2,3,4,5,6,7,8,9]'

# Change learning rate
python run.py --config-name=burgers_experiment run_params.mode=train trainer_params.learning_rate=0.0005

# Disable sliding window
python run.py --config-name=burgers_experiment run_params.mode=train trainer_params.use_sliding_window=false
```

---

### Mode: `evaluate`

Evaluates a trained model against ground truth physics.

**What it does:**
1. Loads the trained model
2. Loads test data
3. Generates predictions using the model
4. Computes metrics (MSE, MAE, RMSE)
5. Generates visualizations and animations

**Output:**
```
results/evaluation/
├── metrics.json                  # Quantitative metrics
├── plots/
│   ├── frame_000.png
│   ├── frame_025.png
│   └── frame_050.png
└── animations/
    ├── ground_truth.gif
    ├── prediction.gif
    └── comparison.gif
```

**Example:**
```bash
python run.py --config-name=burgers_experiment run_params.mode=evaluate
```

**Override examples:**
```bash
# Evaluate on different test simulations
python run.py --config-name=burgers_experiment run_params.mode=evaluate 'evaluation_params.test_sim=[25,26,27,28,29]'

# Evaluate more frames
python run.py --config-name=burgers_experiment run_params.mode=evaluate evaluation_params.num_frames=100

# Disable animations
python run.py --config-name=burgers_experiment run_params.mode=evaluate evaluation_params.save_animations=false
```

---

### Combined Modes

Execute multiple tasks sequentially:

```bash
# Full pipeline
python run.py --config-name=burgers_experiment run_params.mode=[generate,train,evaluate]

# Generate and train only
python run.py --config-name=burgers_experiment run_params.mode=[generate,train]

# Train and evaluate
python run.py --config-name=burgers_experiment run_params.mode=[train,evaluate]
```

---

## Command Reference

### Hydra Command-Line Syntax

#### Override a Parameter

```bash
python run.py --config-name=experiment key=value
```

#### Override Nested Parameters

```bash
python run.py --config-name=experiment parent.child.key=value
```

#### Override Lists

```bash
# Full list replacement
python run.py --config-name=experiment 'list_param=[1,2,3]'

# Note: Use quotes for lists with spaces
python run.py --config-name=experiment 'list_param=[1, 2, 3]'
```

#### Switch Configuration Groups

```bash
python run.py --config-name=experiment trainer=synthetic_quick
```

#### Multi-Run (Sweep)

```bash
python run.py --config-name=experiment --multirun trainer_params.learning_rate=0.0001,0.0005,0.001
```

### Common Override Patterns

#### Change Resolution

```bash
python run.py --config-name=burgers_experiment model.physical.resolution.x=256 model.physical.resolution.y=256
```

#### Change Model Save Name

```bash
python run.py --config-name=burgers_experiment model.synthetic.model_save_name=my_model_v2
```

#### Set Experiment Name

```bash
python run.py --config-name=burgers_experiment run_params.experiment_name=burgers_high_res
```

#### Control Training

```bash
python run.py --config-name=burgers_experiment \
  trainer_params.epochs=200 \
  trainer_params.batch_size=32 \
  trainer_params.learning_rate=0.0005
```

---

## Examples

### Example 1: Quick Test Run

Fast test with minimal data and training:

```bash
python run.py --config-name=burgers_quick_test
```

This configuration:
- Generates only 5 simulations
- Trains for 10 epochs
- Evaluates on 2 test simulations

### Example 2: High-Resolution Experiment

Train on high-resolution data:

```bash
# Generate data
python run.py --config-name=burgers_experiment \
  run_params.mode=generate \
  model.physical.resolution.x=256 \
  model.physical.resolution.y=256 \
  data.dset_name=burgers_256

# Train model
python run.py --config-name=burgers_experiment \
  run_params.mode=train \
  model.physical.resolution.x=256 \
  model.physical.resolution.y=256 \
  data.dset_name=burgers_256 \
  model.synthetic.model_save_name=burgers_unet_256
```

### Example 3: Smoke Simulation

Full smoke simulation pipeline:

```bash
python run.py --config-name=smoke_experiment \
  run_params.mode=[generate,train,evaluate]
```

### Example 4: Custom Training Split

Use specific simulations for training:

```bash
python run.py --config-name=burgers_experiment \
  run_params.mode=[train,evaluate] \
  'trainer_params.train_sim=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]' \
  'evaluation_params.test_sim=[15,16,17,18,19]'
```

### Example 5: Parameter Sweep

Sweep over different viscosities:

```bash
python run.py --config-name=burgers_experiment \
  --multirun \
  model.physical.pde_params.nu=0.01,0.05,0.1,0.5
```

This runs 4 separate experiments with different viscosity values.

### Example 6: Heat Equation

Run heat equation experiment:

```bash
python run.py --config-name=heat_physical_experiment \
  run_params.mode=[generate,train,evaluate]
```

### Example 7: Continuing Training

Continue training from a checkpoint:

```bash
python run.py --config-name=burgers_experiment \
  run_params.mode=train \
  trainer_params.epochs=200 \
  # Model will auto-load if exists and continue training
```

---

## Directory Structure

After running experiments, your directory structure will look like:

```
HYCO-PhiFlow/
├── conf/                        # Configuration files
├── src/                         # Source code
│   ├── models/                  # Physical and synthetic models
│   ├── data/                    # Data loading and generation
│   ├── training/                # Training loops
│   ├── evaluation/              # Evaluation code
│   └── utils/                   # Utilities
├── data/                        # Generated simulation data
│   ├── burgers_128/
│   │   ├── sim_000000/
│   │   └── ...
│   └── cache/                   # Cached tensors for fast loading
├── results/                     # Training and evaluation results
│   ├── models/                  # Saved model weights
│   │   └── burgers_unet_128.pt
│   └── evaluation/              # Evaluation outputs
│       ├── metrics.json
│       ├── plots/
│       └── animations/
├── outputs/                     # Hydra output directories
│   └── YYYY-MM-DD/
│       └── HH-MM-SS/
│           ├── .hydra/
│           │   └── config.yaml
│           └── train.log
├── run.py                       # Main entry point
└── README.md
```

---

## Troubleshooting

### Common Issues

#### 1. **Model not found**

**Error:**
```
ValueError: Physical model 'BurgersModel' not found in registry
```

**Solution:**
- Ensure the model is imported in `src/models/physical/__init__.py`
- Check that the model class is decorated with `@ModelRegistry.register_physical()`
- Verify the name in your config matches the registry name exactly

#### 2. **No data found**

**Error:**
```
FileNotFoundError: Data directory not found: data/burgers_128
```

**Solution:**
```bash
# Generate data first
python run.py --config-name=burgers_experiment run_params.mode=generate
```

#### 3. **CUDA out of memory**

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# Reduce batch size
python run.py --config-name=burgers_experiment trainer_params.batch_size=8

# Reduce rollout length
python run.py --config-name=burgers_experiment trainer_params.num_predict_steps=2

# Use smaller resolution
python run.py --config-name=burgers_experiment model.physical.resolution.x=64 model.physical.resolution.y=64
```

#### 4. **Training loss is NaN**

**Possible causes:**
- Learning rate too high
- Unstable physics parameters
- Data normalization issues

**Solutions:**
```bash
# Reduce learning rate
python run.py --config-name=burgers_experiment trainer_params.learning_rate=0.00001

# Check data validity
# Inspect a simulation manually to ensure it's not exploding
```

#### 5. **Cache validation errors**

**Error:**
```
Cache validation failed: resolution mismatch
```

**Solution:**
```bash
# Clear cache
rm -rf data/cache/burgers_128
# Or let auto-clear handle it
python run.py --config-name=burgers_experiment data.auto_clear_invalid=true
```

#### 6. **Hydra configuration override issues**

**Error:**
```
omegaconf.errors.ConfigAttributeError: Key 'xyz' not found
```

**Solution:**
- Check for typos in parameter names
- Use `+` prefix for new keys: `+new_key=value`
- Use correct nesting: `parent.child.key=value`

---

## Advanced Usage

### Custom Experiments

Create your own experiment configuration:

1. **Create config file**: `conf/my_experiment.yaml`

```yaml
defaults:
  - data: burgers_128
  - model/physical: burgers
  - model/synthetic: unet
  - trainer: synthetic
  - generation: default
  - evaluation: default
  - _self_

run_params:
  experiment_name: 'my_custom_experiment'
  notes: 'Testing custom parameters'
  mode: ['train', 'evaluate']
  model_type: 'synthetic'

# Custom overrides
trainer_params:
  epochs: 150
  batch_size: 24
  learning_rate: 0.0003
```

2. **Run it:**
```bash
python run.py --config-name=my_experiment
```

### Debugging

Enable debug mode for verbose logging:

```bash
# Hydra debug mode
python run.py --config-name=burgers_experiment --cfg job

# Print full resolved config
python run.py --config-name=burgers_experiment --cfg all

# Dry run (no execution)
python run.py --config-name=burgers_experiment hydra.run.dir=null
```

### Working with Outputs

Access Hydra outputs:

```bash
# Latest run
cd outputs/$(ls -t outputs | head -1)

# View resolved config
cat .hydra/config.yaml

# View logs
tail -f train.log
```

---

## Performance Tips

### Data Generation

- Use JIT compilation for physics steps (already implemented)
- Generate data in parallel if you have multiple GPUs:
  ```bash
  # Run multiple generation jobs
  python run.py --config-name=burgers_experiment run_params.mode=generate generation_params.num_simulations=10 data.dset_name=burgers_128_part1 &
  python run.py --config-name=burgers_experiment run_params.mode=generate generation_params.num_simulations=10 data.dset_name=burgers_128_part2 &
  ```

### Training

- Enable data caching (already enabled by default)
- Use `use_sliding_window=true` for efficient temporal sampling
- Increase `num_predict_steps` gradually during training
- Use mixed precision training (requires code modification)

### Evaluation

- Disable animations if you only need metrics:
  ```bash
  python run.py --config-name=burgers_experiment run_params.mode=evaluate evaluation_params.save_animations=false
  ```

---

## Next Steps

1. **Read the Models Documentation**: See `docs/MODELS_DOCUMENTATION.md` for detailed model information
2. **Explore Configurations**: Browse `conf/` directory to understand available options
3. **Customize**: Create your own experiment configurations
4. **Extend**: Add new physical models or neural architectures

---

## Support

For issues, questions, or contributions:
- Check the documentation in `docs/`
- Review existing configurations in `conf/`
- Examine example experiments

---

**Version**: 1.0  
**Last Updated**: October 31, 2025  
**Maintainer**: HYCO-PhiFlow Team
