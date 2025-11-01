# Evaluation System Documentation

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [The Evaluator Class](#the-evaluator-class)
- [Metrics System](#metrics-system)
- [Visualization System](#visualization-system)
- [Running Evaluations](#running-evaluations)
- [Understanding Results](#understanding-results)
- [Configuration Reference](#configuration-reference)
- [Advanced Usage](#advanced-usage)
- [Examples](#examples)

---

## Overview

The evaluation system in HYCO-PhiFlow provides comprehensive tools for assessing the performance of trained neural network models against ground truth physics simulations. It combines quantitative metrics with rich visualizations to give you deep insights into model behavior.

### Key Features

✅ **Comprehensive Metrics**
- MSE, MAE, RMSE, relative error, normalized error
- Per-timestep tracking
- Per-field and per-channel analysis
- Aggregate statistics across simulations

✅ **Rich Visualizations**
- Side-by-side animated comparisons (GIFs)
- Error evolution plots over time
- Keyframe comparisons at critical moments
- Error heatmaps for multi-channel fields
- Difference maps highlighting prediction errors

✅ **Autoregressive Rollout**
- Realistic evaluation using model output as next input
- Captures error accumulation over time
- Configurable rollout length

✅ **Multi-Field Support**
- Handles complex simulations (e.g., velocity + density + inflow)
- Automatic separation of static vs dynamic fields
- Field-specific visualizations and metrics

✅ **Organized Output**
- Structured directory organization
- JSON summaries for programmatic access
- Aggregate reports across multiple simulations

---

## Architecture

```
src/evaluation/
├── __init__.py
├── evaluator.py          # Main Evaluator class orchestrating evaluation
├── metrics.py            # Metric computation functions
└── visualizations.py     # Visualization generation functions
```

### Design Philosophy

1. **Separation of Concerns**: Metrics, visualizations, and orchestration are separate modules
2. **Composability**: Functions can be used independently or via the Evaluator
3. **Extensibility**: Easy to add new metrics or visualization types
4. **Automation**: One-command evaluation with comprehensive output

---

## The Evaluator Class

The `Evaluator` class is the main entry point for running evaluations. It orchestrates the complete workflow:

### Workflow

```
1. Initialize Evaluator from config
   ↓
2. Load trained model
   ↓
3. Setup data manager
   ↓
4. For each test simulation:
   a. Load ground truth data
   b. Run autoregressive inference
   c. Compute error metrics
   d. Generate visualizations
   e. Save results and JSON summary
   ↓
5. Create aggregate summary (if multiple simulations)
```

### Initialization

```python
from src.evaluation import Evaluator

# Initialize from Hydra config
evaluator = Evaluator(config)

# Configuration is extracted from:
# - config['data']: Dataset configuration
# - config['model']['synthetic']: Model configuration  
# - config['evaluation_params']: Evaluation parameters
```

### Main API Methods

#### `evaluate(sim_indices=None, base_save_dir=None)`

Run complete evaluation on multiple simulations.

**Arguments:**
- `sim_indices` (List[int], optional): Simulations to evaluate. Uses `test_sim` from config if None.
- `base_save_dir` (Path, optional): Base directory for results. Auto-generated if None.

**Returns:**
- `Dict[int, Dict]`: Results for each simulation

**Example:**
```python
# Evaluate test simulations from config
results = evaluator.evaluate()

# Evaluate specific simulations
results = evaluator.evaluate(sim_indices=[20, 21, 22])

# Custom save directory
results = evaluator.evaluate(base_save_dir='custom_results/my_eval')
```

#### `evaluate_simulation(sim_index, save_dir=None)`

Run evaluation on a single simulation.

**Arguments:**
- `sim_index` (int): Index of simulation to evaluate
- `save_dir` (Path, optional): Directory for this simulation's results

**Returns:**
- `Dict`: Results containing:
  - `metrics`: Computed error metrics
  - `visualizations`: Paths to generated visualizations
  - `inference_results`: Prediction and ground truth tensors
  - `save_dir`: Where results were saved

**Example:**
```python
result = evaluator.evaluate_simulation(sim_index=20)

# Access metrics
mse_stats = result['metrics']['aggregates']['velocity']['mse']
print(f"Mean MSE: {mse_stats['mean']:.6f}")

# Access predictions
prediction = result['inference_results']['prediction']  # [T, C, H, W]
```

#### `run_inference(sim_index, num_rollout_steps=None)`

Run autoregressive model inference on a simulation.

**Arguments:**
- `sim_index` (int): Simulation to run inference on
- `num_rollout_steps` (int, optional): Number of autoregressive steps. Defaults to `num_frames - 1`.

**Returns:**
- `Dict`: Inference results containing:
  - `prediction`: Model predictions [T, C, H, W]
  - `ground_truth`: True data [T, C, H, W]
  - `initial_state`: Initial condition [C, H, W]

**How it works:**
```python
# 1. Load initial state (t=0) from ground truth
# 2. Use as input to model
# 3. Model predicts next state (t=1)
# 4. Use prediction as next input (autoregressive)
# 5. Repeat for num_rollout_steps
# 6. Return full trajectory
```

#### `compute_metrics(prediction, ground_truth)`

Compute all error metrics.

**Arguments:**
- `prediction` (Tensor): Model predictions [T, C, H, W]
- `ground_truth` (Tensor): True data [T, C, H, W]

**Returns:**
- `Dict`: Metrics dictionary with:
  - `field_metrics`: Per-field, per-timestep errors
  - `aggregates`: Statistics (mean, std, min, max, etc.)
  - `field_specs`: Field specifications used

#### `generate_visualizations(prediction, ground_truth, sim_index, save_dir)`

Generate all visualizations for a simulation.

**Arguments:**
- `prediction` (Tensor): Model predictions [T, C, H, W]
- `ground_truth` (Tensor): True data [T, C, H, W]
- `sim_index` (int): Simulation index
- `save_dir` (Path): Directory to save visualizations

**Returns:**
- `Dict[str, Dict[str, Path]]`: Paths to generated files organized by type

---

## Metrics System

The metrics system (`metrics.py`) provides functions for computing various error metrics.

### Available Metrics

#### **Mean Squared Error (MSE)**
```python
mse = compute_mse_per_timestep(prediction, ground_truth, reduction='mean')
# Returns: [T, C] tensor
```

**Formula**: $\text{MSE} = \frac{1}{HW}\sum_{i,j}(y_{pred} - y_{true})^2$

**Use case**: Standard metric, sensitive to large errors (squared term amplifies outliers).

---

#### **Root Mean Squared Error (RMSE)**
```python
rmse = compute_rmse_per_timestep(prediction, ground_truth, reduction='mean')
# Returns: [T, C] tensor
```

**Formula**: $\text{RMSE} = \sqrt{\text{MSE}}$

**Use case**: Same units as original data, more interpretable than MSE.

---

#### **Mean Absolute Error (MAE)**
```python
mae = compute_mae_per_timestep(prediction, ground_truth, reduction='mean')
# Returns: [T, C] tensor
```

**Formula**: $\text{MAE} = \frac{1}{HW}\sum_{i,j}|y_{pred} - y_{true}|$

**Use case**: Less sensitive to outliers than MSE, represents average magnitude of errors.

---

#### **Relative Error**
```python
rel_err = compute_relative_error_per_timestep(
    prediction, ground_truth, epsilon=1e-8, reduction='mean'
)
# Returns: [T, C] tensor
```

**Formula**: $\text{RelErr} = \frac{|y_{pred} - y_{true}|}{|y_{true}| + \epsilon}$

**Use case**: Normalizes by ground truth magnitude, good for comparing across different scales.

---

#### **Normalized Error**
```python
norm_err = compute_normalized_error_per_timestep(prediction, ground_truth, reduction='mean')
# Returns: [T, C] tensor
```

**Formula**: $\text{NormErr} = \frac{|y_{pred} - y_{true}|}{\max(|y_{true}|) - \min(|y_{true}|)}$

**Use case**: Normalizes by data range, makes errors comparable across different fields.

---

### Reduction Options

All metric functions support three reduction modes:

- **`'mean'`**: Average over spatial dimensions → returns [T, C]
- **`'sum'`**: Sum over spatial dimensions → returns [T, C]
- **`'none'`**: Keep spatial dimensions → returns [T, C, H, W]

**Example:**
```python
# Per-timestep, per-channel averages
mse_tc = compute_mse_per_timestep(pred, gt, reduction='mean')  # [T, C]

# Full spatial error maps
mse_map = compute_mse_per_timestep(pred, gt, reduction='none')  # [T, C, H, W]

# Visualize spatial error at specific timestep
import matplotlib.pyplot as plt
plt.imshow(mse_map[10, 0, :, :])  # Frame 10, channel 0
```

---

### Aggregate Statistics

The `aggregate_metrics()` function computes summary statistics:

```python
stats = aggregate_metrics(errors, dim=0)
# Returns dictionary with:
# {
#   'mean': float,      # Mean error
#   'std': float,       # Standard deviation
#   'min': float,       # Minimum error
#   'max': float,       # Maximum error
#   'median': float,    # Median error
#   'q25': float,       # 25th percentile
#   'q75': float        # 75th percentile
# }
```

**Example:**
```python
mse = compute_mse_per_timestep(pred, gt)  # [T, C]
stats = aggregate_metrics(mse, dim=0)      # Aggregate over time

print(f"Mean MSE: {stats['mean']:.6f} ± {stats['std']:.6f}")
print(f"Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
```

---

### Multi-Field Metrics

For simulations with multiple fields (e.g., smoke = velocity + density + inflow):

```python
field_specs = {
    'velocity': 2,  # 2 channels (x, y)
    'density': 1,   # 1 channel
    'inflow': 1     # 1 channel (static)
}

# Compute metrics for each field separately
field_metrics = compute_metrics_per_field(
    prediction, ground_truth, field_specs, metrics=['mse', 'mae']
)

# Access field-specific metrics
velocity_mse = field_metrics['velocity']['mse']  # [T, 2]
density_mae = field_metrics['density']['mae']    # [T, 1]
```

---

## Visualization System

The visualization system (`visualizations.py`) generates various plots and animations.

### 1. Comparison Animations (GIFs)

**Function**: `create_comparison_gif()`

Creates animated GIF showing prediction vs ground truth over time.

**Features:**
- Side-by-side comparison
- Optional difference panel
- Consistent color scaling
- Automatic magnitude computation for vector fields
- Frame counter overlay

**Panels:**
1. **Ground Truth**: True physics simulation
2. **Prediction**: Model output
3. **Absolute Difference**: |prediction - ground truth| (optional)

**Example:**
```python
from src.evaluation.visualizations import create_comparison_gif

create_comparison_gif(
    prediction=pred,           # [T, C, H, W]
    ground_truth=gt,           # [T, C, H, W]
    field_name='velocity',
    save_path='velocity_comparison.gif',
    fps=10,
    show_difference=True
)
```

**Output:**
- GIF file with smooth animation
- Colorbar for value scale
- Title with field name and frame counter

---

### 2. Error vs Time Plots

**Function**: `plot_error_vs_time()`

Line plots showing error evolution over time.

**Features:**
- Multiple metrics on separate subplots
- Per-channel error tracking
- Mean error across channels
- Grid lines for readability
- Automatic channel naming

**Example:**
```python
from src.evaluation.visualizations import plot_error_vs_time

plot_error_vs_time(
    prediction=pred,
    ground_truth=gt,
    field_name='velocity',
    save_path='velocity_error.png',
    metrics=['mse', 'mae', 'rmse'],
    channel_names=['x-component', 'y-component']
)
```

**Output:**
- PNG file with subplots for each metric
- Legend distinguishing channels
- Markers for small datasets, lines for large

**Interpretation:**
- **Flat line**: Model maintains consistent accuracy
- **Increasing trend**: Error accumulation (drift)
- **Sudden jumps**: Instability or regime changes
- **Channel differences**: Some components harder to predict

---

### 3. Keyframe Comparisons

**Function**: `plot_keyframe_comparison()`

Grid layout showing key moments in the simulation.

**Features:**
- Evenly-spaced temporal sampling (t=0, T/4, T/2, 3T/4, T)
- Three columns: Ground Truth | Prediction | Difference
- Optional error metrics overlay
- Consistent color scaling
- Time labels (e.g., "t = T/2")

**Example:**
```python
from src.evaluation.visualizations import plot_keyframe_comparison

plot_keyframe_comparison(
    prediction=pred,
    ground_truth=gt,
    field_name='density',
    save_path='density_keyframes.png',
    num_keyframes=5,
    show_difference=True,
    show_metrics=True
)
```

**Output:**
- PNG file with grid layout (num_keyframes × 3)
- MSE and MAE values overlaid on prediction panels
- Time labels on left side

**Use case:**
- Quick visual assessment of model performance
- Identify at which time scales errors become visible
- Compare early vs late predictions
- Publication-ready figures

---

### 4. Error Heatmaps

**Function**: `plot_error_heatmap()`

Heatmap showing error across time and channels.

**Features:**
- Time on x-axis, channels on y-axis
- Color intensity represents error magnitude
- Useful for multi-channel fields
- Subsampling for long simulations

**Example:**
```python
from src.evaluation.visualizations import plot_error_heatmap

plot_error_heatmap(
    prediction=pred,
    ground_truth=gt,
    field_name='velocity',
    save_path='velocity_heatmap.png',
    metric='mse',
    max_frames=50
)
```

**Output:**
- PNG file with heatmap
- Colorbar showing error scale
- Channel labels (x, y for velocity)

**Interpretation:**
- **Horizontal bands**: One channel consistently more accurate
- **Vertical bands**: Certain timesteps particularly difficult
- **Gradients**: Error accumulation patterns
- **Hot spots**: Critical failure moments

---

### 5. Multi-Field Wrapper Functions

For convenience, wrapper functions handle multi-field data automatically:

```python
# Create animations for all fields
paths = create_comparison_gif_from_specs(
    prediction, ground_truth, field_specs, save_dir='animations/',
    fps=10, show_difference=True
)
# Returns: {'velocity': Path('velocity_comparison.gif'), 
#           'density': Path('density_comparison.gif')}

# Create error plots for all fields
paths = plot_error_vs_time_multi_field(
    prediction, ground_truth, field_specs, save_dir='plots/',
    metrics=['mse', 'mae']
)

# Create keyframe comparisons for all fields
paths = plot_keyframe_comparison_multi_field(
    prediction, ground_truth, field_specs, save_dir='plots/',
    num_keyframes=5
)
```

---

## Running Evaluations

### Via Command Line

**Basic usage:**
```bash
python run.py --config-name=burgers_experiment run_params.mode=evaluate
```

**Custom test simulations:**
```bash
python run.py --config-name=burgers_experiment \
  run_params.mode=evaluate \
  'evaluation_params.test_sim=[20,21,22,23,24]'
```

**More frames:**
```bash
python run.py --config-name=burgers_experiment \
  run_params.mode=evaluate \
  evaluation_params.num_frames=100
```

**Disable animations:**
```bash
python run.py --config-name=burgers_experiment \
  run_params.mode=evaluate \
  evaluation_params.save_animations=false
```

---

### Via Python Script

```python
from src.evaluation import Evaluator

# Load config (e.g., via Hydra or manual dict)
config = {...}

# Create evaluator
evaluator = Evaluator(config)

# Run evaluation
results = evaluator.evaluate()

# Access results
for sim_idx, sim_results in results.items():
    metrics = sim_results['metrics']['aggregates']
    print(f"Simulation {sim_idx}:")
    for field, field_metrics in metrics.items():
        print(f"  {field} MSE: {field_metrics['mse']['mean']:.6f}")
```

---

### Programmatic Usage

For custom evaluation workflows:

```python
import torch
from src.evaluation import Evaluator
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.visualizations import create_comparison_gif

# Load your data
prediction = torch.load('prediction.pt')      # [T, C, H, W]
ground_truth = torch.load('ground_truth.pt')  # [T, C, H, W]

# Compute metrics
metrics = compute_all_metrics(prediction, ground_truth, metrics=['mse', 'mae'])

# Get statistics
from src.evaluation.metrics import aggregate_metrics
mse_stats = aggregate_metrics(metrics['mse'], dim=0)
print(f"MSE: {mse_stats['mean']:.6f} ± {mse_stats['std']:.6f}")

# Create visualization
create_comparison_gif(
    prediction, ground_truth, 'my_field', 'comparison.gif', fps=10
)
```

---

## Understanding Results

### Directory Structure

After running evaluation, results are organized as follows:

```
results/evaluation/
└── burgers_128_burgers_unet_128/    # {dataset}_{model}
    ├── sim_000020/                   # Per-simulation results
    │   ├── animations/
    │   │   ├── velocity_comparison.gif
    │   │   └── ...
    │   ├── plots/
    │   │   ├── velocity_error_vs_time.png
    │   │   ├── velocity_keyframes.png
    │   │   ├── velocity_error_heatmap.png
    │   │   └── ...
    │   └── metrics/
    │       └── metrics_summary.json
    ├── sim_000021/
    │   └── ...
    └── summary/                       # Aggregate across simulations
        └── aggregate_metrics.json
```

---

### Metrics JSON Format

**Per-simulation metrics** (`metrics_summary.json`):

```json
{
  "aggregates": {
    "velocity": {
      "mse": {
        "mean": 0.000123,
        "std": 0.000045,
        "min": 0.000001,
        "max": 0.000567,
        "median": 0.000110,
        "q25": 0.000078,
        "q75": 0.000145
      },
      "mae": {...}
    },
    "density": {...}
  },
  "per_timestep": {
    "velocity": {
      "mse": [0.0001, 0.0002, 0.0003, ...],  // Per-timestep values
      "mae": [...]
    }
  }
}
```

**Aggregate metrics** (`aggregate_metrics.json`):

```json
{
  "velocity": {
    "mse": {
      "mean": 0.000150,      // Mean across all test simulations
      "std": 0.000020,       // Std across simulations
      "min": 0.000123,       // Best simulation
      "max": 0.000180        // Worst simulation
    },
    "mae": {...}
  }
}
```

---

### Interpreting Metrics

#### **MSE (Mean Squared Error)**

- **Scale**: Units squared (e.g., m²/s² for velocity)
- **Sensitivity**: Heavily penalizes large errors
- **Good value**: Problem-dependent; compare against baseline

**Example interpretation:**
- MSE = 0.001 → Typical error magnitude ≈ √0.001 = 0.032
- Compare to data range: if velocity is [-5, 5], error is ~0.3%

#### **MAE (Mean Absolute Error)**

- **Scale**: Same units as data
- **Sensitivity**: Linear, treats all errors equally
- **Good value**: Should be small fraction of data range

**Example interpretation:**
- MAE = 0.5 m/s for velocity field with range [0, 10] m/s
- Average error is 5% of maximum velocity

#### **RMSE (Root Mean Squared Error)**

- **Scale**: Same units as data
- **Sensitivity**: Between MSE and MAE
- **Good value**: Similar to MAE, but slightly higher due to squaring

#### **Relative Error**

- **Scale**: Dimensionless (percentage)
- **Interpretation**: Direct percentage error
- **Good value**: < 5% is excellent, < 10% is good

**Example interpretation:**
- Relative error = 0.05 → On average, predictions are within 5% of true values

---

### Error Evolution Patterns

#### **1. Stable Model**
```
Error ──────────────  (Flat line)
```
- Error remains constant over time
- Good long-term stability
- No error accumulation

#### **2. Linear Drift**
```
Error        ╱ (Linear increase)
           ╱
         ╱
```
- Error grows linearly with time
- Typical for autoregressive models
- Manageable if slope is small

#### **3. Error Explosion**
```
Error            ╱│ (Exponential growth)
                ╱ │
              ╱   │
            ╱     │
```
- Error grows rapidly
- Model becomes unstable
- Requires architecture improvements

#### **4. Periodic Oscillation**
```
Error  ╱╲╱╲╱╲  (Oscillation)
```
- Error varies periodically
- May indicate under-damping
- Or physical phenomena the model struggles with

---

## Configuration Reference

### Evaluation Parameters

Located in `conf/evaluation/default.yaml`:

```yaml
# Simulations to evaluate
test_sim: [20, 21, 22, 23, 24]

# Number of frames to evaluate (including initial state)
num_frames: 51

# Metrics to compute
metrics: ['mse', 'mae', 'rmse']

# Visualization settings
keyframe_count: 5        # Number of keyframes to show
animation_fps: 10        # Frames per second for GIFs
save_animations: true    # Generate animated GIFs
save_plots: true         # Generate static plots

# Output directory
output_dir: 'results/evaluation'
```

---

### Parameter Descriptions

| Parameter | Type | Description |
|-----------|------|-------------|
| `test_sim` | List[int] | Indices of simulations to evaluate |
| `num_frames` | int | Number of frames to rollout (including t=0) |
| `metrics` | List[str] | Metrics to compute: 'mse', 'mae', 'rmse', 'relative', 'normalized' |
| `keyframe_count` | int | Number of evenly-spaced keyframes for comparison |
| `animation_fps` | int | Animation frame rate (frames per second) |
| `save_animations` | bool | Whether to generate animated GIFs |
| `save_plots` | bool | Whether to generate static plots |
| `output_dir` | str | Base directory for evaluation results |

---

### Command-Line Overrides

```bash
# Evaluate different simulations
python run.py --config-name=experiment \
  'evaluation_params.test_sim=[30,31,32]'

# Longer rollout
python run.py --config-name=experiment \
  evaluation_params.num_frames=100

# Additional metrics
python run.py --config-name=experiment \
  'evaluation_params.metrics=[mse,mae,rmse,relative,normalized]'

# More keyframes
python run.py --config-name=experiment \
  evaluation_params.keyframe_count=10

# Fast evaluation (no visualizations)
python run.py --config-name=experiment \
  evaluation_params.save_animations=false \
  evaluation_params.save_plots=false
```

---

## Advanced Usage

### Custom Evaluation Pipeline

```python
from src.evaluation import Evaluator
from src.evaluation.metrics import compute_metrics_per_field
from src.evaluation.visualizations import create_comparison_gif
import torch

# Initialize evaluator
evaluator = Evaluator(config)
evaluator.load_model()
evaluator.setup_data_manager()

# Run inference
inference_results = evaluator.run_inference(sim_index=20)
prediction = inference_results['prediction']
ground_truth = inference_results['ground_truth']

# Compute custom metrics
field_specs = {'velocity': 2}
metrics = compute_metrics_per_field(
    prediction, ground_truth, field_specs, 
    metrics=['mse', 'mae', 'relative']
)

# Custom visualization
create_comparison_gif(
    prediction[:, :2, :, :],  # Only velocity channels
    ground_truth[:, :2, :, :],
    'velocity',
    'custom_animation.gif',
    fps=15,
    vmin=0, vmax=10
)

# Save custom metrics
import json
with open('custom_metrics.json', 'w') as f:
    json.dump({
        'velocity_mse': metrics['velocity']['mse'].mean().item(),
        'velocity_mae': metrics['velocity']['mae'].mean().item()
    }, f, indent=2)
```

---

### Batch Evaluation Script

```python
import json
from pathlib import Path
from src.evaluation import Evaluator

def evaluate_multiple_models(model_configs, test_sims):
    """Evaluate multiple models on same test set."""
    results_summary = {}
    
    for model_name, config in model_configs.items():
        print(f"\nEvaluating {model_name}...")
        
        # Update test simulations
        config['evaluation_params']['test_sim'] = test_sims
        
        # Run evaluation
        evaluator = Evaluator(config)
        results = evaluator.evaluate()
        
        # Collect aggregate metrics
        all_mse = []
        for sim_results in results.values():
            mse = sim_results['metrics']['aggregates']['velocity']['mse']['mean']
            all_mse.append(mse)
        
        results_summary[model_name] = {
            'mean_mse': sum(all_mse) / len(all_mse),
            'std_mse': (sum((x - sum(all_mse)/len(all_mse))**2 for x in all_mse) / len(all_mse))**0.5
        }
    
    # Save comparison
    with open('model_comparison.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    return results_summary

# Usage
model_configs = {
    'unet_small': config_small,
    'unet_large': config_large,
    'resnet': config_resnet
}

results = evaluate_multiple_models(model_configs, test_sims=[20, 21, 22, 23, 24])
```

---

### Error Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from src.evaluation.metrics import compute_mse_per_timestep

# Get spatial error maps
mse_spatial = compute_mse_per_timestep(pred, gt, reduction='none')  # [T, C, H, W]

# Analyze spatial error distribution
final_frame_error = mse_spatial[-1, 0, :, :]  # Last frame, first channel

# Find regions with high error
threshold = mse_spatial.mean() + 2 * mse_spatial.std()
high_error_mask = final_frame_error > threshold

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].imshow(gt[-1, 0, :, :].T, cmap='viridis')
axes[0].set_title('Ground Truth')

axes[1].imshow(pred[-1, 0, :, :].T, cmap='viridis')
axes[1].set_title('Prediction')

axes[2].imshow(high_error_mask.T, cmap='Reds')
axes[2].set_title('High Error Regions')

plt.tight_layout()
plt.savefig('error_analysis.png')
```

---

## Examples

### Example 1: Basic Evaluation

```bash
# Train a model
python run.py --config-name=burgers_experiment run_params.mode=train

# Evaluate on default test set
python run.py --config-name=burgers_experiment run_params.mode=evaluate
```

**Output:**
- Animations for each test simulation
- Error plots showing MSE, MAE, RMSE over time
- Keyframe comparisons
- JSON metrics summary

---

### Example 2: Long Rollout Evaluation

Test model stability over extended time:

```bash
python run.py --config-name=burgers_experiment \
  run_params.mode=evaluate \
  evaluation_params.num_frames=200 \
  'evaluation_params.test_sim=[20]'
```

**Use case:** Check if error grows linearly or exponentially over long trajectories.

---

### Example 3: Comprehensive Metrics

Compute all available metrics:

```bash
python run.py --config-name=burgers_experiment \
  run_params.mode=evaluate \
  'evaluation_params.metrics=[mse,mae,rmse,relative,normalized]'
```

---

### Example 4: Quick Metrics Only

Skip visualizations for fast metric computation:

```bash
python run.py --config-name=burgers_experiment \
  run_params.mode=evaluate \
  evaluation_params.save_animations=false \
  evaluation_params.save_plots=false
```

---

### Example 5: High-Quality Visualizations

More keyframes and higher FPS:

```bash
python run.py --config-name=burgers_experiment \
  run_params.mode=evaluate \
  evaluation_params.keyframe_count=10 \
  evaluation_params.animation_fps=20
```

---

### Example 6: Smoke Simulation Evaluation

Multi-field evaluation with static fields:

```bash
python run.py --config-name=smoke_experiment run_params.mode=evaluate
```

**Output:**
- Separate visualizations for velocity, density, and inflow
- Per-field metrics
- Aggregate summary across all fields

---

## Troubleshooting

### Issue: Model checkpoint not found

**Error:**
```
FileNotFoundError: Model checkpoint not found at results/models/model_name.pth
```

**Solution:**
- Ensure model is trained first: `python run.py --config-name=experiment run_params.mode=train`
- Check `model.synthetic.model_save_name` matches trained model
- Verify `model.synthetic.model_path` is correct

---

### Issue: Shape mismatch in metrics

**Error:**
```
Shape mismatch: prediction (50, 2, 128, 128) != ground_truth (50, 3, 128, 128)
```

**Solution:**
- Model output channels don't match expected fields
- Check `model.synthetic.output_specs` in config
- Verify model architecture produces correct number of channels

---

### Issue: Out of memory during evaluation

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce `num_frames`: `evaluation_params.num_frames=25`
- Evaluate one simulation at a time
- Use CPU: `device = torch.device('cpu')` in evaluator

---

### Issue: Animations not generating

**Symptom:** No GIF files created

**Solution:**
- Ensure `pillow` is installed: `pip install pillow`
- Check `evaluation_params.save_animations=true`
- Verify write permissions in output directory

---

### Issue: Metrics seem wrong

**Symptoms:** Very high or very low error values

**Checks:**
1. **Unit mismatch**: Are prediction and ground truth in same units/scale?
2. **Normalization**: Was data normalized during training but not during evaluation?
3. **Field ordering**: Are fields in the same order in prediction and ground truth?
4. **Time alignment**: Is initial state (t=0) included correctly?

**Debug:**
```python
print(f"Prediction range: [{pred.min():.3f}, {pred.max():.3f}]")
print(f"Ground truth range: [{gt.min():.3f}, {gt.max():.3f}]")
print(f"Prediction shape: {pred.shape}")
print(f"Ground truth shape: {gt.shape}")
```

---

## Best Practices

### 1. Evaluation Strategy

✅ **DO:**
- Evaluate on held-out test simulations not seen during training
- Use consistent random seeds for reproducibility
- Evaluate at multiple rollout lengths (10, 25, 50, 100 steps)
- Compare against physics baseline (how accurate is ground truth?)

❌ **DON'T:**
- Evaluate on training data (causes overfitting illusion)
- Change model during evaluation
- Compare models evaluated on different test sets

---

### 2. Choosing Metrics

- **MSE**: Standard choice, good for optimization
- **MAE**: More intuitive, robust to outliers
- **Relative Error**: When data has varying magnitudes
- **Multiple Metrics**: Use several for comprehensive assessment

---

### 3. Visualization Guidelines

- **Animations**: Essential for understanding temporal behavior
- **Keyframes**: Quick assessment, publication figures
- **Error Plots**: Diagnose stability issues
- **Heatmaps**: Multi-channel fields only

---

### 4. Reporting Results

Include in reports:
1. **Test set description**: Which simulations, how many, initial conditions
2. **Rollout length**: How many steps were predicted
3. **Multiple metrics**: MSE, MAE, and one normalized metric
4. **Error evolution**: Plot over time to show accumulation
5. **Qualitative assessment**: Visual comparison via keyframes
6. **Baseline comparison**: Compare to simple baselines (persistence, linear extrapolation)

---

## Further Reading

- **Models Documentation**: `docs/MODELS_DOCUMENTATION.md` - Model architecture details
- **Usage Guide**: `docs/USAGE_GUIDE.md` - How to run experiments
- **PhiFlow Docs**: https://tum-pbs.github.io/PhiFlow/ - Physics simulation library
- **PyTorch Docs**: https://pytorch.org/docs/ - Deep learning framework

---

**Version**: 1.0  
**Last Updated**: October 31, 2025  
**Maintainer**: HYCO-PhiFlow Team
