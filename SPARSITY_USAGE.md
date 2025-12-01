# Sparse Observations Feature - Usage Guide

This document explains how to use the sparse observations feature in HYCO-PhiFlow.

## Overview

The sparse observations feature allows you to train models with:
- **Temporal sparsity**: Only certain timesteps are accessible during training
- **Spatial sparsity**: Loss is computed only on specific spatial regions

## Configuration

### Using Preset Configurations

Add one of the preset sparsity configurations to your config file:

```yaml
# In your config file (e.g., burgers_2d.yaml)
sparsity:
  temporal:
    enabled: true
    mode: endpoints
    start_fraction: 0.1
    end_fraction: 0.1
    uniform_stride: 1
    custom_fractions: []

  spatial:
    enabled: true
    mode: range
    x_range: [0.0, 0.5]
    y_range: null
    center_fraction: 0.5
    num_random_points: 100
    random_seed: 42
```

### Available Preset Configurations

Located in `conf/sparsity/`:

1. **full.yaml** - No sparsity (default)
   - All timesteps visible
   - Full spatial domain observed

2. **time_endpoints.yaml** - Temporal sparsity only
   - First 10% and last 10% of trajectory visible
   - Full spatial domain

3. **time_uniform.yaml** - Uniform temporal sampling
   - Every 5th timestep visible
   - Full spatial domain

4. **space_partial.yaml** - Spatial sparsity only
   - All timesteps visible
   - Only left half (x: 0-50%) of domain observed

5. **space_center.yaml** - Center region observation
   - All timesteps visible
   - Only center 50% of domain observed

6. **combined.yaml** - Both temporal and spatial sparsity
   - Endpoints temporal + partial spatial

## Temporal Sparsity Modes

### 1. Endpoints Mode
Observe first and last portions of trajectory:
```yaml
temporal:
  enabled: true
  mode: endpoints
  start_fraction: 0.1  # First 10%
  end_fraction: 0.1    # Last 10%
```

### 2. Uniform Mode
Observe every Nth timestep:
```yaml
temporal:
  enabled: true
  mode: uniform
  uniform_stride: 5  # Every 5th timestep
```

### 3. Custom Mode
Specify exact timesteps (as fractions 0-1):
```yaml
temporal:
  enabled: true
  mode: custom
  custom_fractions: [0.0, 0.25, 0.5, 0.75, 1.0]
```

## Spatial Sparsity Modes

### 1. Range Mode
Observe rectangular region:
```yaml
spatial:
  enabled: true
  mode: range
  x_range: [0.0, 0.5]  # First half in x
  y_range: [0.25, 0.75]  # Middle half in y
```

### 2. Center Mode
Observe centered region:
```yaml
spatial:
  enabled: true
  mode: center
  center_fraction: 0.5  # Center 50% of domain
```

### 3. Random Points Mode
Observe random sparse points:
```yaml
spatial:
  enabled: true
  mode: random_points
  num_random_points: 100
  random_seed: 42
```

## Visualization

Generate visualization of your sparsity configuration:

```python
from src.data.sparsity import SparsityConfig, TemporalSparsityConfig, SpatialSparsityConfig
from src.visualization.sparsity_viz import create_sparsity_report
from phi.flow import spatial

# Create sparsity config
temporal = TemporalSparsityConfig(enabled=True, mode='endpoints',
                                  start_fraction=0.1, end_fraction=0.1)
spatial_cfg = SpatialSparsityConfig(enabled=True, mode='range',
                                     x_range=[0.0, 0.5])
config = SparsityConfig(temporal=temporal, spatial=spatial_cfg)

# Generate visualization report
create_sparsity_report(
    config=config,
    trajectory_length=100,
    spatial_shape=spatial(x=64, y=64),
    output_dir='results/sparsity_viz',
    sample_trajectory=None  # Optionally provide sample data
)
```

This will create:
- `temporal_mask.png` - Visualization of temporal sparsity pattern
- `spatial_mask.png` - Visualization of spatial mask
- `observation_summary.png` - Combined visualization
- `sparsity_summary.txt` - Text summary of configuration

## Running Training with Sparsity

Simply run your training as usual - the sparsity configuration will be automatically loaded:

```bash
python run.py --config-name=burgers_2d.yaml
```

The training logs will show when sparsity is enabled:
```
Sparsity configuration:
  Temporal: endpoints mode
  Spatial: range mode
Spatial mask initialized: 50.0% visible
```

## How It Works

### Temporal Sparsity (Dataset Level)
- Filters which timesteps are accessible
- Only visible timesteps are used for training samples
- Automatically adjusts `samples_per_sim` based on visible timesteps
- Finds nearest visible timesteps when needed

### Spatial Sparsity (Loss Level)
- Model sees full domain (no masking in forward pass)
- Loss is computed only on visible spatial region
- Mask is automatically normalized by visible area
- Separate masks can be used for real vs generated data

## Implementation Details

### Files Modified/Created

**Core Module:**
- `src/data/sparsity.py` - Sparsity configuration and masking classes

**Dataset Integration:**
- `src/data/dataset.py` - Temporal sparsity support

**Trainer Integration:**
- `src/training/synthetic/trainer.py` - Spatial masking in loss
- `src/training/physical/trainer.py` - Spatial masking in loss
- `src/training/hybrid/trainer.py` - Sparsity config parsing

**Factory:**
- `src/factories/dataloader_factory.py` - Temporal config passing

**Visualization:**
- `src/visualization/sparsity_viz.py` - Visualization utilities

**Configs:**
- `conf/sparsity/*.yaml` - Preset configurations

## Tips

1. **Start with no sparsity** - Use `sparsity: full` initially to establish baseline
2. **Temporal first** - Try temporal sparsity before spatial for easier debugging
3. **Visualize your masks** - Always generate visualization to verify your configuration
4. **Monitor loss scaling** - Spatial sparsity reduces effective training data, may need to adjust learning rate
5. **Check visible fraction** - Log output shows percentage of visible data

## Troubleshooting

**Issue**: Loss is very high with spatial sparsity
- **Solution**: Check that your visible region contains meaningful dynamics

**Issue**: Training is slow with temporal sparsity
- **Solution**: Ensure `samples_per_sim` is reasonable (check dataset size in logs)

**Issue**: Spatial mask not being applied
- **Solution**: Verify `sparsity.spatial.enabled: true` in config

**Issue**: Import error with `field`
- **Solution**: Fixed - ensure you're using latest version with explicit imports

## Example Configurations

### Sparse Endpoint Observations (HYCO-style)
```yaml
sparsity:
  temporal:
    enabled: true
    mode: endpoints
    start_fraction: 0.1
    end_fraction: 0.1
  spatial:
    enabled: false
```

### Partial Domain Observation
```yaml
sparsity:
  temporal:
    enabled: false
  spatial:
    enabled: true
    mode: range
    x_range: [0.0, 0.5]
```

### Sparse Sensor Network Simulation
```yaml
sparsity:
  temporal:
    enabled: false
  spatial:
    enabled: true
    mode: random_points
    num_random_points: 50
    random_seed: 42
```
