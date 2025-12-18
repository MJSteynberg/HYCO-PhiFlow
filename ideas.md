# Inverse Problems for Navier-Stokes: Brainstorm for Pretty Solutions

## Context
Using **PhiFlow** (differentiable PDE solver) combined with ideas from **HYCO** (Hybrid-Cooperative Learning), we explore inverse problems that produce visually stunning fluid dynamics.

---

## 🌀 Category 1: Vortex & Wake Identification

### 1.1 Von Kármán Street Genesis
**Problem**: Given observations of a developed vortex street (alternating vortices behind a bluff body), identify the *obstacle shape* and *inflow velocity profile* that generated it.

**Why it's pretty**: The characteristic alternating vortex pattern creates mesmerizing periodic structures.

**Mathematical formulation**:
- **Forward**: `∂u/∂t + (u·∇)u = -∇p + ν∇²u + obstacle_forcing(θ)`
- **Inverse**: Find `θ` (obstacle geometry parameters) and `u_inlet` given sparse velocity observations

**PhiFlow approach**:
```python
# Parameterize obstacle as implicit function or mesh
obstacle = Obstacle(geometry=ParameterizedShape(theta))
# Backprop through simulation to optimize theta
```

### 1.2 Double Vortex Merger Reconstruction
**Problem**: Given the final merged vortex state, reconstruct the two initial vortices (positions, strengths, radii).

**Why it's pretty**: The merging process creates spiral arms and intricate filament structures.

---

## 🎨 Category 2: Mixing & Scalar Transport

### 2.1 Dye Injection Pattern Inverse
**Problem**: Given a beautiful final mixing pattern (e.g., marble-like swirls), identify the injection locations, timing, and rates of different dye sources.

**Why it's pretty**: Creates organic, art-like patterns reminiscent of marbling or latte art.

**Inverse variables**: 
- Source locations `(x_i, y_i)` 
- Injection rates `q_i(t)` as functions of time
- Possibly: background flow forcing

### 2.2 Optimal Stirring Discovery
**Problem**: Find the external forcing field `f(x,y,t)` that creates the most "aesthetically pleasing" mixing pattern (defined via image-based loss).

**Why it's pretty**: Can generate symmetric, fractal-like mixing boundaries.

**Loss function ideas**:
- Maximize mixing entropy
- Match target artistic pattern
- Minimize mixing time while maximizing visual complexity

---

## 🌊 Category 3: Force Field Identification

### 3.1 Hidden Forcing Reconstruction
**Problem**: A fluid is being driven by an unknown spatially-varying body force `f(x,y)`. Given velocity observations, recover the forcing field.

**Why it's pretty**: The force field itself can be visualized as a heatmap or vector field overlaid on the flow.

**Mathematical setup**:
```
∂u/∂t + (u·∇)u = -∇p + ν∇²u + f(x,y)
```

**PhiFlow implementation**:
```python
# Parameterize f as neural network or basis expansion
force_field = NeuralField(mlp) 
# or
force_field = sum(c_i * basis_function_i(x,y))
```

### 3.2 Buoyancy Field from Plume Observations
**Problem**: Given smoke/temperature observations of a rising plume, identify the heat source distribution at the bottom.

**Why it's pretty**: Plumes create mushroom clouds, entrainment patterns, and turbulent mixing layers.

---

## 🔄 Category 4: Boundary Condition Identification

### 4.1 Inlet Profile Reconstruction
**Problem**: Complex vortical structures develop downstream. Infer the inlet velocity profile `u(y, t)` that created them.

**Why it's pretty**: Time-varying inlet conditions create complex wave-like patterns.

### 4.2 Multi-Inlet Orchestra
**Problem**: Multiple inlets (like organ pipes) create an interference pattern. Identify the phase and amplitude of each inlet.

**Why it's pretty**: Creates standing wave patterns and nodal structures in the flow.

---

## 💫 Category 5: Viscosity & Parameter Identification

### 5.1 Spatially-Varying Viscosity Reconstruction
**Problem**: The fluid has heterogeneous viscosity `ν(x,y)` (like oil and water regions). Given flow observations, recover the viscosity field.

**Why it's pretty**: Different viscosity regions create sharp boundaries and interesting rheological patterns.

**Connection to HYCO**: This is exactly the type of coefficient identification problem HYCO handles well!

### 5.2 Reynolds Number Bifurcation Detection
**Problem**: Given a flow that is transitioning from laminar to turbulent, identify the local effective Reynolds number distribution.

**Why it's pretty**: Transition regions show beautiful intermittency and spot-like turbulent patches.

---

## 🎯 Category 6: Control & Optimization

### 6.1 Target Shape Morphing
**Problem**: Given initial smoke/dye configuration and desired final artistic pattern, find the velocity field control sequence.

**Why it's pretty**: The transformation itself is mesmerizing - watching smoke morph into shapes.

**This is well-supported by PhiFlow's differentiable physics!**

### 6.2 Vortex Choreography
**Problem**: Control multiple vortices to follow prescribed artistic trajectories (like synchronized swimming).

**Why it's pretty**: Coordinated vortex motion with trail visualizations.

---

## 🏆 Top 3 Recommendations for Implementation

### Recommended #1: **Dye Injection Source Identification** (2.1)
**Why**: 
- Creates stunning marble-like visuals
- Well-posed inverse problem
- Natural fit for HYCO methodology
- Easy to generate ground truth data

```python
# Sketch implementation
sources = [PointSource(x=tensor([...]), y=tensor([...]), rate=tensor([...]))]
loss = ||observed_dye - simulated_dye||² + regularization
```

### Recommended #2: **Hidden Force Field Reconstruction** (3.1)
**Why**:
- Fundamental inverse problem with clear physics
- Force field visualization adds extra visual layer
- Can parameterize with neural network (physics-informed)
- Connects to HYCO's strength in parameter identification

### Recommended #3: **Obstacle Shape from Vortex Street** (1.1)
**Why**:
- Iconic fluid mechanics problem
- Beautiful periodic structures
- Shape optimization via differentiable simulation
- Can use level-set or parametric representation

---

## Technical Considerations for PhiFlow

### Differentiable Components Available:
- ✅ Advection (semi-Lagrangian, MacCormack)
- ✅ Diffusion (explicit, implicit)
- ✅ Pressure projection (make_incompressible)
- ✅ Obstacle handling
- ✅ Gradient computation via autograd

### Implementation Pattern:
```python
from phi.torch.flow import *  # For PyTorch backend with gradients

# Forward model
@jit_compile  
def forward(params, n_steps):
    velocity, smoke = initial_state(params)
    for _ in range(n_steps):
        velocity, smoke = ns_step(velocity, smoke, params)
    return smoke

# Inverse solve
params = tensor([...], requires_grad=True)
for epoch in range(n_epochs):
    predicted = forward(params)
    loss = mse_loss(predicted, observations)
    grads = math.gradient(loss, params)
    params -= lr * grads
```

---

## Visual Enhancement Ideas

1. **Color mapping**: Use perceptually uniform colormaps (viridis, plasma)
2. **Streamlines**: Overlay streamlines on scalar fields
3. **Vorticity**: Compute and visualize `ω = ∇ × u`
4. **LIC (Line Integral Convolution)**: Beautiful texture-based flow visualization
5. **Particle tracing**: Release particles and trace pathlines

---

## Next Steps

1. Choose one problem from recommendations
2. Generate synthetic "ground truth" data
3. Implement forward model in PhiFlow
4. Add observation operator (sparse measurements)
5. Implement inverse solver (gradient-based or HYCO-style)
6. Visualize results with animations