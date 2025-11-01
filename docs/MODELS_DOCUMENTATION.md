# Models Documentation

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Physical Models](#physical-models)
- [Synthetic Models](#synthetic-models)
- [Model Registry](#model-registry)
- [Adding New Models](#adding-new-models)
- [Configuration Reference](#configuration-reference)

---

## Overview

The `src/models` folder contains the core modeling components of the HYCO-PhiFlow framework. It implements a **dual-model architecture** consisting of:

1. **Physical Models**: Physics-based PDE simulators built on PhiFlow
2. **Synthetic Models**: Neural networks (PyTorch) that learn to emulate physical models

The framework uses a **registry pattern** for automatic model discovery and instantiation, making it easy to add new models without modifying existing code.

---

## Architecture

```
src/models/
├── __init__.py
├── registry.py                 # Model registration and discovery system
├── physical/                   # Physics-based PDE models
│   ├── __init__.py
│   ├── base.py                # Abstract base class for physical models
│   ├── burgers.py             # Burgers' equation implementation
│   ├── heat.py                # Heat equation implementation
│   └── smoke.py               # Smoke simulation implementation
└── synthetic/                  # Neural network models
    ├── __init__.py
    ├── base.py                # Abstract base class for synthetic models
    └── unet.py                # U-Net implementation
```

### Design Principles

1. **Separation of Concerns**: Physical and synthetic models are clearly separated
2. **Extensibility**: New models can be added by implementing abstract base classes
3. **Registry Pattern**: Automatic model discovery via decorators
4. **Configuration-Driven**: All models are instantiated from YAML configurations
5. **Field-Based I/O**: Models work with PhiFlow Field objects (CenteredGrid, StaggeredGrid)

---

## Physical Models

Physical models solve partial differential equations (PDEs) using PhiFlow's physics simulation capabilities. They are used for:
- Generating training data
- Providing ground truth for evaluation
- Testing physical accuracy of synthetic models

### Base Class: `PhysicalModel`

All physical models inherit from `PhysicalModel` (defined in `physical/base.py`), which provides:

#### **Key Features**

1. **Automatic Configuration Parsing**
   - Domain setup (Box)
   - Resolution (Shape)
   - Time step (dt)
   - PDE-specific parameters

2. **PDE Parameter Declaration**
   ```python
   PDE_PARAMETERS = {
       'nu': {
           'type': float,
           'default': 0.01,
       },
       'buoyancy': {
           'type': float,
           'default': 1.0,
       }
   }
   ```

3. **Abstract Methods** (must be implemented by subclasses)
   ```python
   @abstractmethod
   def get_initial_state(self) -> Dict[str, Field]:
       """Generate initial state with batch dimension"""
       pass

   @abstractmethod
   def step(self, current_state: Dict[str, Field]) -> Dict[str, Field]:
       """Advance simulation by one time step"""
       pass
   ```

#### **Configuration Structure**

```yaml
name: 'ModelName'
domain:
  size_x: 100
  size_y: 100
resolution:
  x: 128
  y: 128
dt: 0.8
pde_params:
  batch_size: 1
  # Model-specific parameters
```

---

### Available Physical Models

#### 1. **BurgersModel** - Burgers' Equation

**Location**: `physical/burgers.py`

**Description**: Implements the 2D Burgers' equation, a fundamental PDE in fluid dynamics that exhibits both advection and diffusion.

**Equation**:
```
∂u/∂t + u·∇u = ν∇²u
```

**Parameters**:
- `nu` (float, default=0.01): Viscosity coefficient

**Fields**:
- **Input/Output**: `velocity` (StaggeredGrid, 2 components)

**Boundary Conditions**: Periodic

**Example Configuration**:
```yaml
name: 'BurgersModel'
domain:
  size_x: 100
  size_y: 100
resolution:
  x: 128
  y: 128
dt: 0.8
pde_params:
  batch_size: 1
  nu: 0.1
```

**Physics Implementation**:
```python
def _burgers_physics_step(velocity, dt, nu):
    velocity = diffuse.explicit(u=velocity, diffusivity=nu, dt=dt)
    velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)
    return velocity
```

---

#### 2. **HeatModel** - Heat Equation

**Location**: `physical/heat.py`

**Description**: Implements the 2D heat equation (diffusion equation), modeling heat transfer in a medium.

**Equation**:
```
∂T/∂t = α∇²T
```

**Parameters**:
- `diffusivity` (float, default=0.1): Thermal diffusivity coefficient

**Fields**:
- **Input/Output**: `temp` (CenteredGrid, 1 component)

**Boundary Conditions**: Periodic

**Initial Condition**: Cosine-based pattern
```python
temp_0 = cos(2πx/100) + cos(2πy/100)
```

**Example Configuration**:
```yaml
name: 'HeatModel'
domain:
  size_x: 100
  size_y: 100
resolution:
  x: 64
  y: 64
dt: 0.5
pde_params:
  batch_size: 1
  diffusivity: 0.1
```

---

#### 3. **SmokeModel** - Smoke Simulation

**Location**: `physical/smoke.py`

**Description**: Implements a full 2D smoke simulation with buoyancy, advection, diffusion, and pressure projection.

**Equations**:
```
∂ρ/∂t + u·∇ρ = 0                    (density advection)
∂u/∂t + u·∇u = -∇p + ν∇²u + ρ·b·ĝ  (velocity evolution)
∇·u = 0                              (incompressibility)
```

**Parameters**:
- `nu` (float, default=0.0): Viscosity
- `buoyancy` (float, default=1.0): Buoyancy force strength
- `inflow_radius` (float, default=10.0): Radius of smoke source
- `inflow_rate` (float, default=0.1): Rate of smoke injection
- `inflow_center` (tuple, optional): Position of smoke source (auto-generated if not set)
- `inflow_rand_x_range` (list, default=[0.2, 0.8]): Random x-position range
- `inflow_rand_y_range` (list, default=[0.15, 0.25]): Random y-position range

**Fields**:
- **Input/Output**: 
  - `velocity` (StaggeredGrid, 2 components)
  - `density` (CenteredGrid, 1 component)
  - `inflow` (CenteredGrid, 1 component) - static field

**Boundary Conditions**: Zero for velocity, boundary for density

**Example Configuration**:
```yaml
name: 'SmokeModel'
domain:
  size_x: 100
  size_y: 100
resolution:
  x: 128
  y: 128
dt: 1.0
pde_params:
  batch_size: 1
  nu: 0.0
  buoyancy: 1.0
  inflow_radius: 10.0
  inflow_rate: 0.1
```

**Physics Implementation**:
```python
def _smoke_physics_step(velocity, density, inflow, domain, dt, buoyancy_factor, nu):
    # Advect density and add inflow
    density = advect.mac_cormack(density, velocity, dt=dt) + dt * inflow
    
    # Apply buoyancy force
    buoyancy_force = (density * (0, buoyancy_factor)).at(velocity)
    velocity = velocity + dt * buoyancy_force
    
    # Advect velocity
    velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)
    
    # Diffuse velocity
    velocity = diffuse.explicit(velocity, nu, dt=dt)
    
    # Project to incompressible (∇·u = 0)
    velocity, pressure = fluid.make_incompressible(velocity)
    
    return velocity, density
```

---

## Synthetic Models

Synthetic models are neural networks that learn to emulate physical models. They enable:
- Fast inference (orders of magnitude faster than physics simulation)
- Hybrid physics-ML approaches
- Learning from data

### Base Class: `SyntheticModel`

Defined in `synthetic/base.py`, provides:

#### **Key Features**

1. **Field Specifications**
   ```python
   INPUT_SPECS = {'velocity': 2, 'density': 1}   # field: num_channels
   OUTPUT_SPECS = {'velocity': 2, 'density': 1}
   ```

2. **Abstract Method**
   ```python
   @abstractmethod
   def forward(self, state: Dict[str, Field], dt: float = 0.0) -> Dict[str, Field]:
       """Forward pass through the model"""
       pass
   ```

---

### Available Synthetic Models

#### **UNet** - U-Net Architecture

**Location**: `synthetic/unet.py`

**Description**: Implements a U-Net architecture for spatial field prediction. Works directly with PyTorch tensors in `[batch, channels, height, width]` format.

**Key Features**:

1. **Static vs Dynamic Field Handling**
   - Automatically preserves static fields (e.g., inflow in smoke simulation)
   - Only predicts dynamic fields (e.g., velocity, density)
   - Re-attaches static fields to output in original order

2. **Architecture Parameters**:
   - `levels` (int, default=4): Number of down/up-sampling levels
   - `filters` (int, default=64): Base number of filters
   - `batch_norm` (bool, default=true): Use batch normalization

3. **Flexible Input/Output**:
   - Input: All fields (static + dynamic)
   - Output: Full state with predictions + preserved static fields

**Example Configuration**:
```yaml
name: 'UNet'
model_path: 'results/models'
model_save_name: 'burgers_unet_128'

input_specs:
  velocity: 2

output_specs:
  velocity: 2

architecture:
  levels: 4
  filters: 64
  batch_norm: true
```

**For Smoke Simulation**:
```yaml
input_specs:
  velocity: 2
  density: 1
  inflow: 1      # Static field

output_specs:
  velocity: 2
  density: 1     # No inflow - it's static
```

**Architecture**:
```
Input: [B, in_channels, H, W]
    ↓
Encoder (down-sampling with conv + pooling)
    Level 1: filters
    Level 2: filters × 2
    Level 3: filters × 4
    Level 4: filters × 8
    ↓
Decoder (up-sampling with conv + upconv)
    Level 4 → 3 (+ skip connection)
    Level 3 → 2 (+ skip connection)
    Level 2 → 1 (+ skip connection)
    ↓
Output: [B, out_channels, H, W]
```

---

## Model Registry

The registry pattern enables automatic model discovery and instantiation without hard-coded dependencies.

**Location**: `registry.py`

### **Features**

1. **Decorator-Based Registration**
   ```python
   @ModelRegistry.register_physical('BurgersModel')
   class BurgersModel(PhysicalModel):
       pass
   ```

2. **Separate Registries**
   - Physical models: `_physical_models`
   - Synthetic models: `_synthetic_models`

3. **Dynamic Instantiation**
   ```python
   model = ModelRegistry.get_physical_model('BurgersModel', config)
   ```

### **API Reference**

#### Registration

```python
@ModelRegistry.register_physical(name: str)
@ModelRegistry.register_synthetic(name: str)
```
Decorators for registering model classes.

#### Instantiation

```python
ModelRegistry.get_physical_model(name: str, config: Dict) -> PhysicalModel
ModelRegistry.get_synthetic_model(name: str, config: Dict) -> SyntheticModel
```
Create model instances from configuration.

#### Inspection

```python
ModelRegistry.list_physical_models() -> List[str]
ModelRegistry.list_synthetic_models() -> List[str]
ModelRegistry.is_physical_model_registered(name: str) -> bool
ModelRegistry.is_synthetic_model_registered(name: str) -> bool
```
Query available models.

---

## Adding New Models

### Adding a Physical Model

1. **Create the model file**: `src/models/physical/mymodel.py`

2. **Implement the model**:
   ```python
   from .base import PhysicalModel
   from src.models.registry import ModelRegistry
   from phi.torch.flow import *
   from typing import Dict
   
   @jit_compile
   def _mymodel_physics_step(field, dt, param1):
       # Implement physics step
       return new_field
   
   @ModelRegistry.register_physical('MyModel')
   class MyModel(PhysicalModel):
       PDE_PARAMETERS = {
           'param1': {
               'type': float,
               'default': 1.0,
           }
       }
       
       def get_initial_state(self) -> Dict[str, Field]:
           b = batch(batch=self.batch_size)
           field_0 = CenteredGrid(
               0, extrapolation.PERIODIC,
               x=self.resolution.get_size('x'),
               y=self.resolution.get_size('y'),
               bounds=self.domain
           )
           field_0 = math.expand(field_0, b)
           return {"field": field_0}
       
       def step(self, current_state: Dict[str, Field]) -> Dict[str, Field]:
           new_field = _mymodel_physics_step(
               current_state["field"],
               self.dt,
               self.param1
           )
           return {"field": new_field}
   ```

3. **Create configuration**: `conf/model/physical/mymodel.yaml`
   ```yaml
   name: 'MyModel'
   domain:
     size_x: 100
     size_y: 100
   resolution:
     x: 64
     y: 64
   dt: 0.5
   pde_params:
     batch_size: 1
     param1: 1.0
   ```

4. **Import in `__init__.py`**: 
   ```python
   # src/models/physical/__init__.py
   from .mymodel import MyModel
   ```

### Adding a Synthetic Model

1. **Create the model file**: `src/models/synthetic/mynet.py`

2. **Implement the model**:
   ```python
   import torch
   import torch.nn as nn
   from src.models.registry import ModelRegistry
   from typing import Dict, Any
   
   @ModelRegistry.register_synthetic('MyNet')
   class MyNet(nn.Module):
       def __init__(self, config: Dict[str, Any]):
           super().__init__()
           self.config = config
           self.input_specs = config['input_specs']
           self.output_specs = config['output_specs']
           
           in_channels = sum(self.input_specs.values())
           out_channels = sum(self.output_specs.values())
           
           # Build your network
           self.net = nn.Sequential(
               nn.Conv2d(in_channels, 64, 3, padding=1),
               nn.ReLU(),
               nn.Conv2d(64, out_channels, 3, padding=1)
           )
       
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           return self.net(x)
   ```

3. **Create configuration**: `conf/model/synthetic/mynet.yaml`
   ```yaml
   name: 'MyNet'
   model_path: 'results/models'
   model_save_name: ???
   
   input_specs:
     field: 1
   
   output_specs:
     field: 1
   
   architecture:
     layers: 3
     hidden_dim: 64
   ```

4. **Import in `__init__.py`**:
   ```python
   # src/models/synthetic/__init__.py
   from .mynet import MyNet
   ```

---

## Configuration Reference

### Physical Model Configuration Template

```yaml
name: 'ModelName'              # Must match registry name
domain:
  size_x: 100                  # Domain width
  size_y: 100                  # Domain height
resolution:
  x: 128                       # Number of grid points in x
  y: 128                       # Number of grid points in y
dt: 0.8                        # Time step size
pde_params:
  batch_size: 1                # Number of parallel simulations
  # Model-specific parameters declared in PDE_PARAMETERS
```

### Synthetic Model Configuration Template

```yaml
name: 'ModelName'              # Must match registry name
model_path: 'results/models'   # Directory to save/load models
model_save_name: ???           # Specific model filename (required)

input_specs:                   # Input fields and their channel counts
  field1: 2
  field2: 1

output_specs:                  # Output fields and their channel counts
  field1: 2
  field2: 1

architecture:                  # Model-specific architecture params
  # Architecture parameters
```

---

## Best Practices

### For Physical Models

1. **Use JIT Compilation**: Wrap physics steps with `@jit_compile` for performance
2. **Batch Dimensions**: Always expand fields to batch size using `math.expand(field, b)`
3. **Field Types**: Choose appropriate field types:
   - `StaggeredGrid`: For vector fields (velocity)
   - `CenteredGrid`: For scalar fields (density, temperature, pressure)
4. **Boundary Conditions**: Match physics (PERIODIC, ZERO, BOUNDARY)
5. **Parameter Validation**: Use PDE_PARAMETERS for automatic parsing and validation

### For Synthetic Models

1. **Static Fields**: Properly handle static fields that don't change over time
2. **Channel Ordering**: Maintain consistent field ordering in specs
3. **Tensor Format**: Work with `[batch, channels, height, width]` format
4. **Skip Connections**: Use skip connections for better gradient flow (as in U-Net)
5. **Normalization**: Consider batch normalization for training stability

### For Both

1. **Registry Names**: Use descriptive, unique names for registration
2. **Documentation**: Add docstrings explaining model purpose and parameters
3. **Configuration**: Provide sensible defaults and clear parameter descriptions
4. **Testing**: Test with various resolutions and batch sizes
5. **Error Handling**: Provide clear error messages for configuration issues

---

## Troubleshooting

### Common Issues

**1. Model not found in registry**
```
ValueError: Physical model 'ModelName' not found in registry
```
**Solution**: Ensure the model is imported in `__init__.py` and decorated with `@ModelRegistry.register_physical()`

**2. Field dimension mismatch**
```
Shape mismatch: expected (batch=4, x=128, y=128), got (x=128, y=128)
```
**Solution**: Expand fields to batch dimension using `math.expand(field, batch(batch=N))`

**3. Channel count mismatch**
```
Expected input with 3 channels, got 2
```
**Solution**: Verify `input_specs` matches the actual fields in your state dictionary

**4. Configuration parameter missing**
```
Required parameter 'nu' not found in pde_params
```
**Solution**: Add the parameter to your YAML configuration or provide a default in PDE_PARAMETERS

---

## Additional Resources

- **PhiFlow Documentation**: https://tum-pbs.github.io/PhiFlow/
- **PyTorch Documentation**: https://pytorch.org/docs/
- **Hydra Configuration**: https://hydra.cc/
- **Repository README**: See `docs/USAGE_GUIDE.md` for running experiments

---

**Version**: 1.0  
**Last Updated**: October 31, 2025  
**Maintainer**: HYCO-PhiFlow Team
