"""
Playground for experimenting with PhiML named channel dimensions for multi-field support.

This demonstrates how to use named dimensions instead of dictionaries for multi-field data.
"""

from phi.torch.flow import *
from phiml.math import channel, spatial, batch, instance
import matplotlib.pyplot as plt

print("=" * 60)
print("HIERARCHICAL FIELD STRUCTURE: VECTORS + SCALARS")
print("=" * 60)

# =============================================================================
# 1. MULTIPLE CHANNEL DIMENSIONS (FIELD + VECTOR)
# =============================================================================
print("\n1. Multiple Channel Dimensions (field + vector)")
print("-" * 40)

# Create a tensor with TWO channel dimensions:
# - 'field' for different physical quantities (velocity, density)
# - 'vector' for vector components (x, y) - only applies to velocity

# First, let's see what happens with two channel dimensions
velocity = math.random_normal(spatial(x=32, y=32), channel(vector='x,y'))
density = math.random_normal(spatial(x=32, y=32))

print(f"Velocity shape: {velocity.shape}")
print(f"Density shape: {density.shape}")

# Stack them with a 'field' dimension
# Note: density needs to be expanded to have a vector dimension for stacking
state = math.stack({
    'velocity': velocity,
    'density': density,
}, channel('field'), expand_dims={'density': channel(vector='scalar')})

print(f"\nCombined state shape: {state.shape}")
print(f"State has dimensions: {state.shape.names}")

# =============================================================================
# 2. ACCESSING HIERARCHICAL FIELDS
# =============================================================================
print("\n2. Accessing Hierarchical Fields")
print("-" * 40)

# Access velocity field (includes all vector components)
vel = state.field['velocity']
print(f"Velocity field shape: {vel.shape}")

# Access specific velocity component
vel_x = state.field['velocity'].vector['x']
print(f"Velocity x-component shape: {vel_x.shape}")

# Access density field
dens = state.field['density']
print(f"Density field shape: {dens.shape}")

# Access density value (remove the 'scalar' vector dim)
dens_value = state.field['density'].vector['scalar']
print(f"Density value shape: {dens_value.shape}")

# =============================================================================
# 3. ALTERNATIVE: NON-UNIFORM TENSORS (different sizes per field)
# =============================================================================
print("\n3. Non-Uniform Tensors (different vector sizes per field)")
print("-" * 40)

# PhiML supports non-uniform tensors where different slices have different shapes
# This is perfect for mixing vectors (2 components) and scalars (1 component)

velocity_2d = math.random_normal(spatial(x=32, y=32), channel(vector='x,y'))
density_scalar = math.random_normal(spatial(x=32, y=32), channel(vector='c'))  # 'c' for component
pressure_scalar = math.random_normal(spatial(x=32, y=32), channel(vector='c'))

# Stack creates a non-uniform tensor if shapes don't match along stacked dim
try:
    non_uniform_state = math.stack({
        'velocity': velocity_2d,
        'density': density_scalar,
        'pressure': pressure_scalar,
    }, channel('field'))
    print(f"Non-uniform state shape: {non_uniform_state.shape}")
    print(f"Is uniform: {non_uniform_state.shape.is_uniform}")
except Exception as e:
    print(f"Non-uniform stacking result: {type(e).__name__}: {e}")

# =============================================================================
# 4. PRACTICAL APPROACH: FLATTEN FOR NETWORK, RESTRUCTURE FOR PHYSICS
# =============================================================================
print("\n4. Practical Approach: Flatten for Network")
print("-" * 40)

# For neural networks, we typically want a single flat channel dimension
# Create helper functions to flatten/unflatten

def flatten_state(velocity, density, pressure=None):
    """Flatten multi-field state into single channel dimension for network."""
    fields = [velocity.vector['x'], velocity.vector['y'], density]
    if pressure is not None:
        fields.append(pressure)
    names = ['vel_x', 'vel_y', 'density'] + (['pressure'] if pressure is not None else [])
    return math.stack(fields, channel(field=','.join(names)))

def unflatten_state(flat_state):
    """Unflatten network output back to structured fields."""
    vel_x = flat_state.field['vel_x']
    vel_y = flat_state.field['vel_y']
    velocity = math.stack([vel_x, vel_y], channel(vector='x,y'))
    density = flat_state.field['density']
    return velocity, density

# Test it
velocity_in = math.random_normal(spatial(x=32, y=32), channel(vector='x,y'))
density_in = math.random_normal(spatial(x=32, y=32))

flat = flatten_state(velocity_in, density_in)
print(f"Flattened state shape: {flat.shape}")
print(f"Flattened field names: {flat.shape['field'].item_names}")

# Unflatten
velocity_out, density_out = unflatten_state(flat)
print(f"Unflattened velocity shape: {velocity_out.shape}")
print(f"Unflattened density shape: {density_out.shape}")

# =============================================================================
# 5. USING PHIFLOW'S NATIVE VECTOR HANDLING
# =============================================================================
print("\n5. PhiFlow's Native Vector Handling with CenteredGrid")
print("-" * 40)

# PhiFlow CenteredGrid natively supports vector fields with channel(vector='x,y')
# The trick is that physics operations use 'vector' dimension

velocity_grid = CenteredGrid(
    math.random_normal(spatial(x=64, y=64), channel(vector='x,y')),
    PERIODIC,
    bounds=Box(x=64, y=64)
)
print(f"Velocity grid shape: {velocity_grid.shape}")
print(f"Velocity grid values shape: {velocity_grid.values.shape}")

# Scalar field (no vector dimension)
density_grid = CenteredGrid(
    math.random_normal(spatial(x=64, y=64)),
    PERIODIC,
    bounds=Box(x=64, y=64)
)
print(f"Density grid shape: {density_grid.shape}")

# Physics operations work naturally
advected_vel = advect.semi_lagrangian(velocity_grid, velocity_grid, dt=0.1)
advected_dens = advect.semi_lagrangian(density_grid, velocity_grid, dt=0.1)
print(f"Advected velocity shape: {advected_vel.values.shape}")
print(f"Advected density shape: {advected_dens.values.shape}")

# =============================================================================
# 6. COMBINING VECTOR AND SCALAR FOR NETWORK INPUT
# =============================================================================
print("\n6. Combining Vector and Scalar Fields for Network")
print("-" * 40)

# The cleanest approach: rename 'vector' to 'field' and concat
def prepare_network_input(velocity, density):
    """Prepare multi-field input for neural network."""
    # Rename velocity's vector dim to field with component names
    vel_flat = math.rename_dims(velocity, 'vector', channel(field='vel_x,vel_y'))
    # Expand density to have field dim
    dens_flat = math.expand(density, channel(field='density'))
    # Concat along field dimension
    return math.concat([vel_flat, dens_flat], 'field')

def parse_network_output(network_output):
    """Parse network output back to velocity and density."""
    # Extract and restructure
    vel_x = network_output.field['vel_x']
    vel_y = network_output.field['vel_y']
    velocity = math.stack([vel_x, vel_y], channel(vector='x,y'))
    density = network_output.field['density']
    return velocity, density

# Test
vel_in = math.random_normal(spatial(x=32, y=32), channel(vector='x,y'))
dens_in = math.random_normal(spatial(x=32, y=32))

net_input = prepare_network_input(vel_in, dens_in)
print(f"Network input shape: {net_input.shape}")
print(f"Network input field names: {net_input.shape['field'].item_names}")

# Simulate network output (same shape)
net_output = net_input * 0.9  # fake prediction

vel_out, dens_out = parse_network_output(net_output)
print(f"Parsed velocity shape: {vel_out.shape}")
print(f"Parsed density shape: {dens_out.shape}")

# =============================================================================
# 7. LOSS COMPUTATION
# =============================================================================
print("\n7. Loss Computation")
print("-" * 40)

# With unified tensor, loss is simple
prediction = math.random_normal(batch(batch=8), spatial(x=32, y=32), channel(field='vel_x,vel_y,density'))
target = math.random_normal(batch(batch=8), spatial(x=32, y=32), channel(field='vel_x,vel_y,density'))

loss = math.l2_loss(prediction - target)
mean_loss = math.mean(loss, 'batch')
print(f"Unified loss: {float(mean_loss):.4f}")

# Per-field weighting is also easy
vel_weight = 1.0
dens_weight = 0.5
weighted_loss = (
    vel_weight * math.l2_loss(prediction.field['vel_x,vel_y'] - target.field['vel_x,vel_y']) +
    dens_weight * math.l2_loss(prediction.field['density'] - target.field['density'])
)
print(f"Weighted loss: {float(math.mean(weighted_loss, 'batch')):.4f}")

# =============================================================================
# 8. COMPLETE SIMULATION EXAMPLE
# =============================================================================
print("\n8. Complete Simulation Example")
print("-" * 40)

# Define a step function that handles velocity (vector) and density (scalar)
def physics_step(velocity_grid, density_grid, dt=0.1):
    """One physics step for velocity and density."""
    # Self-advection of velocity
    velocity_new = advect.semi_lagrangian(velocity_grid, velocity_grid, dt)
    # Advect density by velocity
    density_new = advect.semi_lagrangian(density_grid, velocity_grid, dt)
    return velocity_new, density_new

# Initialize
v0 = CenteredGrid(Noise(vector='x,y', scale=1.0), PERIODIC, x=64, y=64, bounds=Box(x=64, y=64))
d0 = CenteredGrid(Noise(scale=0.5), PERIODIC, x=64, y=64, bounds=Box(x=64, y=64))

print(f"Initial velocity shape: {v0.values.shape}")
print(f"Initial density shape: {d0.values.shape}")

# Run a few steps
v, d = v0, d0
for step in range(5):
    v, d = physics_step(v, d, dt=0.1)

print(f"After 5 steps - velocity: {v.values.shape}, density: {d.values.shape}")

# For network training, flatten state
flat_state = prepare_network_input(v.values, d.values)
print(f"Flattened for network: {flat_state.shape}")

# =============================================================================
# 9. SUMMARY: TWO VIABLE APPROACHES
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY: TWO VIABLE APPROACHES")
print("=" * 60)

print("""
APPROACH 1: Hierarchical (field + vector dimensions)
----------------------------------------------------
state = math.stack({
    'velocity': velocity,  # has vector='x,y'
    'density': math.expand(density, channel(vector='scalar'))
}, channel('field'))

Pros: Preserves semantic structure
Cons: Non-uniform tensor, complex access patterns
Access: state.field['velocity'].vector['x']

APPROACH 2: Flattened (single field dimension)  [RECOMMENDED]
-------------------------------------------------------------
state = math.stack({
    'vel_x': velocity.vector['x'],
    'vel_y': velocity.vector['y'],
    'density': density
}, channel('field'))

Pros: Uniform tensor, works with networks, simple access
Cons: Loses vector semantics (but easy to reconstruct)
Access: state.field['vel_x'], state.field['vel_x,vel_y']

RECOMMENDATION:
- Use flattened approach for network input/output
- Use helper functions to convert to/from physics representation
- Keep velocity as proper vector field for physics operations
""")
