# PhiFlow & PhiML Best Practices Guide

**Purpose**: Avoid common pitfalls and leverage the full power of PhiFlow/PhiML
**Audience**: Developers working on HYCO-PhiFlow migration
**Based on**: Official phi/phiml documentation and reference examples

---

## Table of Contents

1. [Named Dimensions: The Core Concept](#1-named-dimensions-the-core-concept)
2. [Dimension Types and Their Behaviors](#2-dimension-types-and-their-behaviors)
3. [Common Pitfalls and How to Avoid Them](#3-common-pitfalls-and-how-to-avoid-them)
4. [Field Operations Best Practices](#4-field-operations-best-practices)
5. [Training Neural Networks in PhiML](#5-training-neural-networks-in-phiml)
6. [Performance Optimization](#6-performance-optimization)
7. [Caching and Memory Management](#7-caching-and-memory-management)
8. [PDE Simulations](#8-pde-simulations)
9. [Debugging and Troubleshooting](#9-debugging-and-troubleshooting)
10. [Migration from PyTorch Patterns](#10-migration-from-pytorch-patterns)

---

## 1. Named Dimensions: The Core Concept

### 1.1 The Fundamental Shift

**PyTorch/NumPy thinking**: Dimensions are positions (0, 1, 2, ...)
**PhiML thinking**: Dimensions are names with semantic meaning

```python
# ❌ OLD WAY (PyTorch)
data = torch.randn(32, 3, 64, 64)  # What does each dimension mean?
# Is it (batch, channels, height, width)?
# Or (batch, height, width, channels)?
# You have to remember or check documentation

# ✅ NEW WAY (PhiML)
from phiml import math
from phiml.math import batch, channel, spatial

data = math.random_normal(batch(examples=32),
                          channel(vector='x,y,z'),
                          spatial(x=64, y=64))
# Crystal clear: 32 examples, 3-component vectors, 64×64 spatial grid
```

### 1.2 Creating Named Tensors

```python
from phiml import math
from phiml.math import batch, spatial, channel, instance

# Spatial grids (for fields, images, volumes)
grid = math.zeros(spatial(x=100, y=100))

# Batch dimensions (for parallel processing)
batched = math.random_uniform(batch(examples=128), spatial(x=64, y=64))

# Channel dimensions (for features, vector components)
features = math.ones(channel(features=256))
velocity = math.zeros(channel(vector='x,y,z'))

# Instance dimensions (for particles, points)
particles = math.random_normal(instance(particles=1000),
                               channel(vector='x,y'))
```

### 1.3 Why This Matters

**Automatic reshaping**:
```python
# PhiML automatically transposes/reshapes based on names
a = math.random_normal(channel(vector='x,y,z'))  # Shape: (vector=3)
b = math.random_normal(channel(vector='z,y,x'))  # Shape: (vector=3)

# These are automatically aligned by name!
c = a + b  # Works! PhiML reorders b to match a
```

**No more dimension errors**:
```python
# ❌ PyTorch
a = torch.randn(3, 64, 64)  # (C, H, W)
b = torch.randn(64, 64, 3)  # (H, W, C)
c = a + b  # RuntimeError: shapes don't match!

# ✅ PhiML
a = math.random_normal(channel(c=3), spatial(x=64, y=64))
b = math.random_normal(spatial(x=64, y=64), channel(c=3))
c = a + b  # Just works! Dimensions aligned by name
```

---

## 2. Dimension Types and Their Behaviors

### 2.1 The Five Dimension Types

| Type | Purpose | Behavior in Operations | Example Use |
|------|---------|------------------------|-------------|
| `batch` | Parallel execution | Computed independently | Multiple simulations |
| `spatial` | Physical space | FFT, gradients, stencils | Grids, coordinates |
| `channel` | Features/components | Treated as data | RGB channels, vectors |
| `instance` | Individual objects | Pairwise operations | Particles, points |
| `dual` | Conjugate space | Advanced linear algebra | Implicit matrices |

### 2.2 How Operations Treat Dimension Types

```python
from phiml import math
from phiml.math import batch, spatial, channel

data = math.random_normal(batch(sims=10),
                          spatial(x=64, y=64),
                          channel(features=3))

# FFT operates on SPATIAL dimensions only
fft_data = math.fft(data)  # FFT over x, y (not batch or channel)

# Sum can target specific types
total = math.sum(data, 'spatial')  # Sum over x, y
# Result shape: (sims=10, features=3)

mean_feature = math.mean(data, 'channel')  # Average over features
# Result shape: (sims=10, x=64, y=64)

# Gradients work on SPATIAL dimensions
grad = math.spatial_gradient(data)
# Adds 'vector' channel dim for gradient components
```

### 2.3 Critical Rule: Match Dimension Types for Physics

```python
# ❌ WRONG: Velocity as batch dimension
velocity = math.zeros(batch(x=64, y=64), channel(vector='x,y'))
# This means "64 simulations in x, 64 in y" - not what you want!

# ✅ CORRECT: Velocity as spatial + channel
velocity = math.zeros(spatial(x=64, y=64), channel(vector='x,y'))
# This means "2D grid with 2-component vectors at each point"
```

### 2.4 Combining Multiple Batches

```python
# You can have multiple batch dimensions for parameter sweeps
results = math.zeros(batch(reynolds_number=10,
                           timestep=5,
                           initial_condition=20),
                     spatial(x=128, y=128))
# This creates 10×5×20 = 1000 parallel simulations!
```

---

## 3. Common Pitfalls and How to Avoid Them

### 3.1 Pitfall: Using `.native()` Too Early

```python
# ❌ BAD: Lose dimension information
tensor = math.random_normal(spatial(x=64, y=64))
native = tensor.native()  # Now just a PyTorch/JAX tensor
result = native + 1  # Lost all PhiML features!

# ✅ GOOD: Stay in PhiML as long as possible
tensor = math.random_normal(spatial(x=64, y=64))
result = tensor + 1  # Still a PhiML tensor with named dims
# Only convert at the very end if absolutely necessary
```

**Rule of thumb**: Avoid `.native()` unless interfacing with external libraries. PhiML operations are as fast or faster.

### 3.2 Pitfall: Manual Transpose/Reshape

```python
# ❌ BAD: Manual dimension manipulation (PyTorch habits)
data = math.random_normal(spatial(x=64, y=64), channel(c=3))
# Don't do: data.native().permute(2, 0, 1)

# ✅ GOOD: Use dimension names
data.channel[-1]  # Last channel
data.x[10:20]  # Slice in x
data.c['r']  # Access named channel (if using channel(c='r,g,b'))
```

### 3.3 Pitfall: Forgetting Dimension Types in Functions

```python
# ❌ BAD: Ambiguous dimensions
def my_function(x):
    return math.sum(x)  # Sum over ALL dimensions - probably not intended!

# ✅ GOOD: Explicit about dimension semantics
def my_function(x):
    """x should have shape (batch, spatial, channel)"""
    return math.sum(x, 'spatial')  # Clear intent
```

### 3.4 Pitfall: Not Using `jit_compile` for Iterative Code

```python
# ❌ SLOW: Python loop overhead
def simulate(state, num_steps):
    for _ in range(num_steps):
        state = step(state)
    return state

# ✅ FAST: JIT compilation (10-100x speedup!)
from phiml import math

@math.jit_compile
def simulate(state, num_steps):
    # Same code, but compiled!
    for _ in range(num_steps):
        state = step(state)
    return state

# ✅ EVEN BETTER: Use iterate()
from phi.flow import iterate

trajectory = iterate(step, batch(time=100), initial_state)
# Automatically stacks results along time dimension
```

### 3.5 Pitfall: Mixing Backends Accidentally

```python
# ❌ BAD: Inconsistent backends
from phi import torch  # Sets backend to PyTorch
from phi.flow import *

# Later in code...
from phi import jax  # Changes backend to JAX!
# Now your tensors are incompatible!

# ✅ GOOD: Set backend once at the top of your main script
# main.py
from phi import torch  # or jax, or tensorflow
from phi.torch.flow import *  # Everything uses this backend

# Other files
from phi.flow import *  # Inherits backend from main
```

### 3.6 Pitfall: Incorrect Vector Field Creation

```python
# ❌ WRONG: Scalar field, not vector field
velocity = math.zeros(spatial(x=64, y=64, vector=2))
# This creates a 3D grid (64×64×2), not a 2D vector field!

# ✅ CORRECT: Vector field
velocity = math.zeros(spatial(x=64, y=64), channel(vector='x,y'))
# This creates a 2D grid (64×64) where each point has a 2D vector
```

### 3.7 Pitfall: Not Checking Field Boundaries

```python
# ❌ BAD: Using wrong boundaries for advection
velocity = StaggeredGrid(0, 0, Box(x=100, y=100), x=64, y=64)
# Default boundary is 0 (zero vector) - but you might want ZERO_GRADIENT!

# ✅ GOOD: Explicit boundaries
from phi.flow import ZERO_GRADIENT, PERIODIC

velocity = StaggeredGrid(0, ZERO_GRADIENT, Box(x=100, y=100), x=64, y=64)
# or for periodic domains:
velocity = StaggeredGrid(0, PERIODIC, Box(x=100, y=100), x=64, y=64)
```

### 3.8 Pitfall: Inefficient Field Sampling

```python
# ❌ SLOW: Repeated field creation
for t in range(1000):
    smoke = CenteredGrid(some_value, ...)  # Creates new grid each time!

# ✅ FAST: Reuse field structure, update values
from phi.field import Field

initial_grid = CenteredGrid(0, ZERO_GRADIENT, Box(x=100, y=100), x=64, y=64)
for t in range(1000):
    smoke = Field.update(initial_grid, new_values)  # Reuses structure
```

---

## 4. Field Operations Best Practices

### 4.1 Creating Fields Efficiently

```python
from phi.flow import CenteredGrid, StaggeredGrid, Box, ZERO_GRADIENT

# For scalar fields (smoke, temperature, pressure)
smoke = CenteredGrid(
    initial_value,  # Can be: scalar, array, function, or Geometry
    boundary=ZERO_GRADIENT,
    bounds=Box(x=100, y=100),
    resolution=dict(x=64, y=64)
)

# For vector fields (velocity)
# Use StaggeredGrid for better incompressibility
velocity = StaggeredGrid(
    (0, 0),  # Initial velocity
    boundary=ZERO_GRADIENT,
    bounds=Box(x=100, y=100),
    resolution=dict(x=64, y=64)
)
```

### 4.2 Function-Based Initialization

```python
import numpy as np

# ✅ GOOD: Use lambda for mathematical expressions
smoke = CenteredGrid(
    lambda x, y: math.sin(2 * np.pi * x / 100) * math.cos(2 * np.pi * y / 100),
    ZERO_GRADIENT,
    Box(x=100, y=100),
    x=64, y=64
)

# ✅ GREAT: Use geometry for shapes
from phi.geom import Sphere

inflow = CenteredGrid(
    Sphere(x=50, y=20, radius=5),  # Circle of smoke
    ZERO_GRADIENT,
    Box(x=100, y=100),
    x=64, y=64
)
```

### 4.3 Resampling Between Grids

```python
# Common need: Transfer data between different grid types
smoke_centered = CenteredGrid(...)  # Scalar field
velocity_staggered = StaggeredGrid(...)  # Vector field

# ✅ Correct resampling
from phi.field import resample

# Resample smoke to velocity grid (for advection)
smoke_on_velocity_grid = resample(smoke_centered, to=velocity_staggered)

# Soft resampling for smooth geometries
from phi.geom import Sphere
sphere = Sphere(x=50, y=50, radius=10)
smooth_obstacle = resample(sphere, to=smoke_centered, soft=True)
# soft=True uses smooth approximation instead of hard 0/1
```

### 4.4 Field Arithmetic

```python
# Fields support natural arithmetic
smoke = CenteredGrid(1.0, ...)
smoke_doubled = smoke * 2  # Element-wise multiplication
smoke_decayed = smoke * 0.99  # Decay

# Combining fields
total_density = smoke + dust + fog  # Element-wise addition

# With geometries
from phi.geom import Box as GeomBox
heat_source = resample(GeomBox(x=(40, 60), y=(40, 60)), to=temperature)
temperature = temperature + heat_source * 10  # Add heat
```

---

## 5. Training Neural Networks in PhiML

### 5.1 Pure PhiML Training Loop (No PyTorch!)

```python
from phiml import math, nn
from phiml.math import batch, channel

# Create network - NO torch.nn imports needed
net = nn.mlp(
    in_channels=1,
    out_channels=1,
    layers=[128, 128, 128],
    activation='ReLU',
    batch_norm=False
)

# Create optimizer - NO torch.optim needed
optimizer = nn.adam(net, learning_rate=1e-3)

# Define loss function
def loss_function(x_batch, y_batch):
    prediction = math.native_call(net, x_batch)
    return math.l2_loss(prediction - y_batch)

# Training loop - NO .backward(), .step(), .zero_grad()!
for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        # ONE LINE does everything: forward, backward, step
        loss = nn.update_weights(net, optimizer, loss_function, x_batch, y_batch)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
```

### 5.2 Available Architectures

```python
from phiml import nn

# Multi-Layer Perceptron
mlp = nn.mlp(in_channels, out_channels, layers=[256, 256], activation='SiLU')

# Convolutional Network
convnet = nn.conv_net(
    in_channels=3,
    out_channels=3,
    layers=[32, 64, 128],
    batch_norm=True,
    activation='ReLU'
)

# ResNet
resnet = nn.res_net(
    in_channels=3,
    out_channels=3,
    layers=[16, 32, 64, 128],
    activation='ReLU'
)

# U-Net (best for field-to-field mappings)
unet = nn.u_net(
    in_channels=3,
    out_channels=3,
    levels=4,  # Depth of U-Net
    filters=32,  # Base number of filters
    activation='ReLU'
)
```

### 5.3 Optimizers

```python
from phiml import nn

# Adam (recommended for most cases)
opt = nn.adam(network, learning_rate=1e-3)

# SGD with momentum
opt = nn.sgd(network, learning_rate=1e-2, momentum=0.9)

# AdaGrad
opt = nn.adagrad(network, learning_rate=1e-2)
```

### 5.4 Best Practices for Network Training

```python
# ✅ GOOD: Use learning rate scheduling
base_lr = 1e-3
for epoch in range(num_epochs):
    # Cosine annealing
    lr = base_lr * 0.5 * (1 + math.cos(math.pi * epoch / num_epochs))
    optimizer = nn.adam(net, learning_rate=lr)

    loss = nn.update_weights(net, optimizer, loss_fn, x, y)

# ✅ GOOD: Use gradient clipping for stability
def loss_fn_with_clip(x, y):
    pred = math.native_call(net, x)
    loss = math.l2_loss(pred - y)

    # Compute gradients
    gradients = math.gradient(loss, get_output=False)

    # Clip gradients
    clipped = [math.clip(g, -1.0, 1.0) for g in gradients]

    return loss

# ✅ GOOD: Normalize inputs
def normalize(x):
    mean = math.mean(x, batch)
    std = math.std(x, batch)
    return (x - mean) / (std + 1e-8)
```

---

## 6. Performance Optimization

### 6.1 Always Use JIT Compilation

```python
from phiml import math

# ❌ SLOW: Python loop overhead (10-100x slower)
def step_function(state):
    # ... physics here ...
    return new_state

for i in range(1000):
    state = step_function(state)

# ✅ FAST: JIT compiled
@math.jit_compile
def step_function(state):
    # ... same physics code ...
    return new_state

for i in range(1000):
    state = step_function(state)

# ✅ FASTEST: JIT + iterate
from phi.flow import iterate

@math.jit_compile
def step_function(state):
    # ... same physics code ...
    return new_state

trajectory = iterate(step_function, batch(time=1000), initial_state)
```

### 6.2 JIT Compilation Gotchas

```python
# ❌ JIT can't compile if you:
# - Print inside the function
# - Use Python data structures that change (lists, dicts)
# - Call non-JIT-compatible libraries
# - Have dynamic control flow based on tensor values

@math.jit_compile
def bad_jit_function(x):
    print(x)  # ❌ Can't print
    if x.sum() > 0:  # ❌ Control flow on tensor value
        return x
    else:
        return -x

# ✅ JIT-compatible version:
@math.jit_compile
def good_jit_function(x):
    # Use math.where for conditional logic
    return math.where(math.sum(x) > 0, x, -x)
```

### 6.3 Batching for Parallelism

```python
# ❌ SLOW: Sequential processing
results = []
for param in parameter_values:
    result = simulate(param)
    results.append(result)

# ✅ FAST: Parallel batch processing
from phiml.math import batch

params = math.stack(parameter_values, batch('params'))
results = simulate(params)  # All computed in parallel!

# ✅ EXAMPLE: Parameter sweep
viscosities = math.linspace(0.01, 0.1, batch(viscosity=10))
initial_conditions = generate_ics(batch(ic=20))

# This runs 10×20 = 200 simulations in parallel!
results = simulate(viscosities, initial_conditions)
```

### 6.4 Memory Management

```python
# For large simulations, watch memory usage

# ❌ BAD: Stores entire trajectory in memory
trajectory = iterate(step, batch(time=10000), state)  # 10000 timesteps!

# ✅ BETTER: Store only specific times
from phiml.math import range_tensor

# Store every 100th timestep
save_indices = math.range(0, 10000, 100)  # 100 timesteps
trajectory = iterate(step, batch(time=save_indices), state)

# ✅ BEST: Use caching to disk (see section 7)
```

---

## 7. Caching and Memory Management

### 7.1 Using PhiML's Parallel Caching

```python
from phiml.dataclasses import parallel_compute, cached_property
from dataclasses import dataclass

@dataclass
class Simulation:
    initial_state: 'Tensor'
    viscosity: float

    @cached_property
    def trajectory(self):
        """This gets computed once and cached to disk"""
        return simulate(self.initial_state, self.viscosity)

    @cached_property
    def statistics(self):
        """This reuses cached trajectory"""
        return compute_stats(self.trajectory)

# Create simulation tasks
sims = [Simulation(state, visc) for state, visc in params]

# Compute in parallel with disk caching
parallel_compute(
    sims,
    [Simulation.trajectory, Simulation.statistics],
    batch_dim='simulations',
    cache_dir='./cache',
    memory_limit=4096  # MB
)
```

### 7.2 Cache Configuration

```python
from phiml.dataclasses import set_cache_ttl, load_cache_as

# Set cache time-to-live (how long to keep in memory)
set_cache_ttl(10.0)  # seconds

# Load cached data using specific backend
load_cache_as('torch')  # or 'jax', 'numpy', 'tensorflow'

# Use different backend for workers
load_cache_as('torch', worker_backend='numpy')
# Main process uses PyTorch, workers use NumPy (faster for CPU)
```

### 7.3 Best Practices for Large-Scale Computation

```python
# ✅ Strategy 1: Batch dimensions for parallelism
# Process 100 simulations at once
states = math.random_normal(batch(sims=100), spatial(x=128, y=128))
results = simulate_batch(states)

# ✅ Strategy 2: Split large batches
# If 100 is too memory-intensive, split into chunks
for i in range(10):
    states_chunk = states[{'sims': range(i*10, (i+1)*10)}]
    results_chunk = simulate_batch(states_chunk)
    save_to_disk(results_chunk, f'results_{i}.npz')

# ✅ Strategy 3: Use cached_property for expensive operations
@dataclass
class Experiment:
    @cached_property  # Computed once, reused thereafter
    def expensive_preprocessing(self):
        return very_slow_operation()

    @cached_property  # Can depend on other cached properties
    def analysis(self):
        return analyze(self.expensive_preprocessing)
```

---

## 8. PDE Simulations

### 8.1 Time Stepping Patterns

```python
from phi.flow import iterate
from phiml import math

# ✅ Pattern 1: Simple iterate
@math.jit_compile
def step(state, dt=0.1):
    # ... update state ...
    return new_state

trajectory = iterate(step, batch(time=100), initial_state, dt=0.1)

# ✅ Pattern 2: Multiple fields
@math.jit_compile
def step(velocity, smoke, pressure, dt=0.1):
    # ... update all fields ...
    return new_velocity, new_smoke, new_pressure

v_traj, s_traj, p_traj = iterate(
    step,
    batch(time=100),
    initial_v, initial_s, initial_p,
    dt=0.1
)

# ✅ Pattern 3: With forces/sources
@math.jit_compile
def step(v, s, p, dt, buoyancy_strength):
    # buoyancy_strength is NOT iterated (constant parameter)
    # ...
    return v, s, p

trajectory = iterate(
    step,
    batch(time=100),
    v0, s0, p0,
    dt=0.1,
    buoyancy_strength=0.5
)
```

### 8.2 Advection Best Practices

```python
from phi.flow import advect

# ✅ For smooth fields: use Mac-Cormack (2nd order accurate)
smoke_new = advect.mac_cormack(smoke, velocity, dt=0.1)

# ✅ For stability: use semi-Lagrangian (1st order, very stable)
smoke_new = advect.semi_lagrangian(smoke, velocity, dt=0.1)

# ✅ For sharp features: use WENO
smoke_new = advect.weno(smoke, velocity, dt=0.1)

# ❌ DON'T use simple forward Euler advection (unstable)
```

### 8.3 Diffusion Best Practices

```python
from phi.flow import diffuse

# ✅ For implicit diffusion (unconditionally stable)
smoke_new = diffuse.implicit(smoke, diffusivity=0.1, dt=0.1)

# ✅ For variable diffusivity (spatially-varying)
viscosity_field = CenteredGrid(...)  # Different viscosity at each point
velocity_new = diffuse.implicit(velocity, viscosity_field, dt=0.1)

# ❌ DON'T use explicit diffusion for large dt (unstable)
# Only use explicit if dt << dx^2 / diffusivity
```

### 8.4 Incompressibility (Pressure Projection)

```python
from phi.flow import fluid
from phiml.math import Solve

# ✅ For basic incompressible flow
velocity_new, pressure = fluid.make_incompressible(
    velocity,
    obstacles=(),  # Or list of Geometry objects
    solve=Solve(x0=pressure_guess)  # Warm start with previous pressure
)

# ✅ For higher accuracy: specify solver tolerance
velocity_new, pressure = fluid.make_incompressible(
    velocity,
    obstacles=(),
    solve=Solve(x0=pressure_guess,
                rel_tol=1e-5,  # Relative tolerance
                max_iterations=1000)
)

# ✅ For 4th-order time stepping (Runge-Kutta)
from phi.flow import fluid

velocity_new, pressure = fluid.incompressible_rk4(
    velocity,
    dt=0.1,
    diffusion=0.01,
    obstacles=()
)
```

### 8.5 Boundary Conditions

```python
from phi.flow import ZERO_GRADIENT, PERIODIC, BOUNDARY, extrapolation

# ✅ Zero gradient (Neumann) - common for velocity
velocity = StaggeredGrid(0, ZERO_GRADIENT, ...)

# ✅ Periodic boundaries - for turbulence
smoke = CenteredGrid(0, PERIODIC, ...)

# ✅ Mixed boundaries - different on each side
from phi.field import extrapolation as ext

boundaries = {
    'x': (ZERO_GRADIENT, ZERO_GRADIENT),  # Left and right
    'y': (ext.BOUNDARY, ZERO_GRADIENT)     # Bottom: no-slip, Top: free
}
velocity = StaggeredGrid(0, boundaries, ...)

# ✅ Spatially-varying boundaries
boundary_field = CenteredGrid(lambda x, y: ..., ...)
smoke = CenteredGrid(0, boundary_field.as_boundary(), ...)
```

---

## 9. Debugging and Troubleshooting

### 9.1 Inspecting Tensor Shapes

```python
from phiml import math

tensor = math.random_normal(batch(b=10), spatial(x=64, y=64), channel(c=3))

# ✅ Print shape information
print(tensor.shape)  # Full shape with names and types
# Output: (batchᵇ=10, spatialˣ=64, spatialʸ=64, channelᶜ=3)

# ✅ Check specific dimensions
print(tensor.shape.batch)  # Just batch dimensions
print(tensor.shape.spatial)  # Just spatial dimensions
print(tensor.shape.get_size('x'))  # Size of specific dimension

# ✅ Check if dimension exists
if 'x' in tensor.shape:
    print("Has x dimension")
```

### 9.2 Debugging JIT Compilation Issues

```python
# ❌ If JIT compilation fails, remove @jit_compile temporarily
# @math.jit_compile  # Comment out
def problematic_function(x):
    print(f"Debug: x = {x}")  # Now prints work
    return x * 2

# Run without JIT to see errors
result = problematic_function(data)

# ✅ Once debugged, add JIT back and remove prints
@math.jit_compile
def problematic_function(x):
    return x * 2
```

### 9.3 Checking for NaN/Inf

```python
from phiml import math

# ✅ Check for invalid values
if math.any(math.isnan(tensor)):
    print("Warning: NaN detected!")

if math.any(math.isinf(tensor)):
    print("Warning: Inf detected!")

# ✅ Replace NaN with default value
tensor = math.where(math.isnan(tensor), 0.0, tensor)

# ✅ Clip to prevent overflow
tensor = math.clip(tensor, -1e6, 1e6)
```

### 9.4 Visualizing Fields

```python
from phi import vis

# ✅ Plot a 2D field
vis.plot(smoke_field, title="Smoke Density")
vis.show()  # Display

# ✅ Animated trajectory
vis.plot(smoke_trajectory, animate='time', title="Smoke Evolution")

# ✅ Save to file
vis.plot(smoke_field, title="Smoke")
vis.savefig("smoke_frame.png")

# ✅ Multiple fields side-by-side
vis.plot({
    'Smoke': smoke,
    'Velocity': velocity,
    'Pressure': pressure
})
```

### 9.5 Profiling Performance

```python
import time
from phiml import math

# ✅ Time non-JIT vs JIT
def time_function(func, *args):
    start = time.time()
    result = func(*args)
    end = time.time()
    print(f"Time: {end - start:.4f} seconds")
    return result

# Compare
print("Without JIT:")
time_function(slow_function, data)

print("With JIT:")
@math.jit_compile
def fast_function(x):
    return slow_function(x)

time_function(fast_function, data)
```

---

## 10. Migration from PyTorch Patterns

### 10.1 Common PyTorch → PhiML Translations

| PyTorch | PhiML | Notes |
|---------|-------|-------|
| `torch.randn(32, 3, 64, 64)` | `math.random_normal(batch(b=32), channel(c=3), spatial(x=64, y=64))` | Named dimensions |
| `x.permute(0, 2, 3, 1)` | Just use `x` | Auto-aligns |
| `x.view(...)` | `math.pack_dims(x, ...)` | Rarely needed |
| `torch.cat([a, b], dim=1)` | `math.concat([a, b], 'dimension_name')` | Named concat |
| `x.mean(dim=1)` | `math.mean(x, 'dimension_name')` | Named reduction |
| `torch.nn.Linear(10, 20)` | `nn.mlp(10, 20, layers=[])` | PhiML network |
| `optimizer.zero_grad(); loss.backward(); optimizer.step()` | `nn.update_weights(net, opt, loss_fn, x, y)` | One line! |

### 10.2 DataLoader Replacement

```python
# ❌ OLD (PyTorch DataLoader)
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __getitem__(self, idx):
        return self.data[idx]

loader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch in loader:
    # ...

# ✅ NEW (PhiML batching)
from phiml import math
from phiml.math import batch

# Data is already batched with named dimension
data = math.stack(data_list, batch('examples'))

# Shuffle
indices = math.random_permutation(batch(examples=len(data_list)))
shuffled_data = data[indices]

# Chunk into batches
num_batches = len(data_list) // 32
for i in range(num_batches):
    batch_data = data.examples[i*32:(i+1)*32]
    # ...
```

### 10.3 Model Saving/Loading

```python
# ❌ OLD (PyTorch)
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))

# ✅ NEW (PhiML)
from phiml import math

# Save
math.save(network, 'model.phiml')

# Load
network = math.load('model.phiml')

# Or use native backend format
math.save(network, 'model.pth', format='torch')  # Saves as PyTorch file
```

### 10.4 Device Management

```python
# ❌ OLD (PyTorch)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)

# ✅ NEW (PhiML)
# Backend is set globally, no need to move tensors
from phi import torch  # or jax
from phi.torch.flow import *

# All operations automatically use GPU if available
# No .to(device) calls needed!
```

### 10.5 Gradient Computation

```python
# ❌ OLD (PyTorch)
x = torch.randn(10, requires_grad=True)
y = x ** 2
y.backward()
grad = x.grad

# ✅ NEW (PhiML)
from phiml import math

def f(x):
    return x ** 2

x = math.random_normal(batch(b=10))
grad = math.gradient(f, wrt='x', get_output=False)(x)
# Or get both output and gradient:
y, grad = math.gradient(f, wrt='x', get_output=True)(x)
```

---

## Quick Reference Card

### Essential Imports
```python
from phiml import math, nn
from phiml.math import batch, spatial, channel, instance
from phi.flow import *  # CenteredGrid, StaggeredGrid, etc.
```

### Creating Tensors
```python
math.zeros(spatial(x=64, y=64))
math.ones(batch(b=32), channel(c=3))
math.random_normal(spatial(x=64, y=64))
math.linspace(0, 1, spatial(x=100))
```

### Creating Fields
```python
CenteredGrid(0, ZERO_GRADIENT, Box(x=100, y=100), x=64, y=64)
StaggeredGrid((0, 0), PERIODIC, Box(x=100, y=100), x=64, y=64)
```

### Time Stepping
```python
@math.jit_compile
def step(state):
    return new_state

trajectory = iterate(step, batch(time=100), initial_state)
```

### Training
```python
net = nn.mlp(in_dim, out_dim, layers=[128, 128])
opt = nn.adam(net, learning_rate=1e-3)

def loss_fn(x, y):
    return math.l2_loss(math.native_call(net, x) - y)

loss = nn.update_weights(net, opt, loss_fn, x_batch, y_batch)
```

### Physics Operations
```python
advect.mac_cormack(field, velocity, dt)
diffuse.implicit(field, diffusivity, dt)
fluid.make_incompressible(velocity, obstacles)
math.spatial_gradient(field)
math.divergence(vector_field)
math.curl(vector_field)
```

---

## Summary: Top 10 Rules

1. **Always use named dimensions** - Never rely on dimension order
2. **Set backend once** - At top of main script, stick with it
3. **Use JIT compilation** - For any iterative/repeated code
4. **Stay in PhiML** - Avoid `.native()` until absolutely necessary
5. **Match dimension types** - spatial for grids, channel for vectors, batch for parallel
6. **Use `iterate()` for time stepping** - Automatic stacking, better performance
7. **One-line training** - `nn.update_weights()` does everything
8. **Explicit boundaries** - Always specify boundary conditions
9. **Use proper advection** - mac_cormack or semi_lagrangian, not manual
10. **Batch everything** - Leverage parallel batch dimensions for parameter sweeps

---

**Remember**: The PhiML philosophy is "write once, run anywhere, batch automatically." Trust the named dimensions, and the library handles the rest!
