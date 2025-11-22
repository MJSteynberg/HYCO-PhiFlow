# Proposal: Multi-Field Support Using Named PhiML Tensor Dimensions

## Overview

This proposal outlines a better approach for handling multi-field data in HYCO-PhiFlow by leveraging PhiML's named dimension system instead of the current dictionary-based approach.

## Current Implementation

### Dictionary Approach

Currently, multi-field data is handled using Python dictionaries:

```python
# Current data structure
state = {
    'velocity': Tensor(batch?, x, y, vector=2),
    'density': Tensor(batch?, x, y),
    'pressure': Tensor(batch?, x, y)
}
```

**Issues with the current approach:**

1. **Repetitive concatenation/splitting**: Each forward pass requires:
   - Concatenating fields along the 'vector' dimension
   - Renaming 'vector' → 'channel' for the network
   - Running inference
   - Renaming 'channel' → 'vector'
   - Splitting back into separate fields

2. **Complex bookkeeping**: Must track:
   - Field order in the dictionary
   - Number of channels per field
   - Which fields are dynamic vs static

3. **Loss function complexity**: Must iterate over fields:
   ```python
   for field_name in targets.keys():
       target_t = targets[field_name].time[t]
       predicted_t = predictions[field_name]
       loss += l2_loss(predicted_t - target_t)
   ```

4. **Configuration overhead**: Requires `fields_scheme` ('dvv') and `fields_type` ('DSS') strings

## Proposed Solution: Named Field Dimension

### Core Concept

Use PhiML's `channel` dimension type with named components to represent multiple fields in a single tensor:

```python
from phiml.math import channel, spatial, batch

# Define field dimension with named components
field_dim = channel(field='velocity_x,velocity_y,density,pressure')

# Single unified state tensor
state = Tensor(
    shape=(batch(batch=64), spatial(x=128, y=128), channel(field='velocity_x,velocity_y,density,pressure'))
)
```

### Alternative: Hierarchical Channel Dimensions

For fields with varying dimensionality (e.g., velocity is 2D vector, density is scalar), use nested/hierarchical naming:

```python
# Option A: Flat naming with prefixes
field_dim = channel(field='vel_x,vel_y,density,pressure')

# Option B: Use separate dimensions for field and component
# This requires PhiML support for nested channels
# state.field['velocity'].vector['x']  # Access velocity_x component
```

### Recommended Approach: Field-Aware Channel Dimension

```python
from phiml.math import channel, concat, stack

# Define fields with their component counts
FIELD_SPEC = {
    'velocity': 2,  # 2D vector
    'density': 1,   # scalar
    'pressure': 1   # scalar
}

# Create named channel dimension
def create_field_channel(field_spec: dict) -> Shape:
    """Create a channel dimension with named field components."""
    names = []
    for field_name, num_components in field_spec.items():
        if num_components == 1:
            names.append(field_name)
        else:
            # For vectors, append component index or spatial dim name
            for i in range(num_components):
                names.append(f"{field_name}_{i}")
    return channel(field=','.join(names))

# Usage
field_shape = create_field_channel(FIELD_SPEC)
# channel(field='velocity_0,velocity_1,density,pressure')
```

## Simplified Forward Step

### Current Implementation (Dictionary-based)

```python
def __call__(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
    # 1. Concatenate fields
    field_order = list(self.input_specs.keys())
    field_tensors = [x[field_name] for field_name in field_order]
    concatenated = math.concat(field_tensors, 'vector')

    # 2. Rename vector -> channel
    network_input = math.rename_dims(concatenated, 'vector', 'channel')

    # 3. Handle static vs dynamic
    if self.num_static_channels == 0:
        predicted = math.native_call(self.network, network_input)
    else:
        dynamic = network_input.channel[0:self.num_dynamic_channels]
        static = network_input.channel[self.num_dynamic_channels:self.total_channels]
        predicted_dynamic = math.native_call(self.network, dynamic)
        predicted = math.concat([predicted_dynamic, static], 'channel')

    # 4. Rename channel -> vector
    predicted = math.rename_dims(predicted, 'channel', 'vector')

    # 5. Split back into fields
    output_dict = {}
    channel_idx = 0
    for field_name in field_order:
        num_channels = self.input_specs[field_name]
        output_dict[field_name] = predicted.vector[channel_idx:channel_idx + num_channels]
        channel_idx += num_channels

    return output_dict
```

### Proposed Implementation (Named Dimensions)

```python
def __call__(self, state: Tensor) -> Tensor:
    """
    Forward pass with unified state tensor.

    Args:
        state: Tensor with shape (batch?, x?, y?, field='velocity_0,velocity_1,density,...')

    Returns:
        Tensor with same shape, predicted next state
    """
    # No concatenation needed - state is already unified

    # Handle static fields by slicing using named dimension
    if self.static_fields:
        # Use named slicing - PhiML handles this elegantly
        dynamic_state = state.field[self.dynamic_field_names]
        static_state = state.field[self.static_field_names]

        # Network only predicts dynamic fields
        predicted_dynamic = math.native_call(self.network, dynamic_state)

        # Combine with static fields
        # PhiML's concat automatically handles dimension matching
        predicted = math.concat([predicted_dynamic, static_state], 'field')
    else:
        predicted = math.native_call(self.network, state)

    return predicted
```

**Benefits:**
- No dictionary iteration
- No manual concatenation/splitting
- No dimension renaming
- Named slicing for static/dynamic separation
- Single tensor throughout

## Simplified Training Loss

### Current Implementation

```python
def loss_function(init_state, rollout_targets):
    current_state = init_state  # Dict of tensors
    total_loss = 0.0

    for t in range(self.rollout_steps):
        next_state = self.model(current_state)
        step_loss = 0.0

        # Must iterate over fields
        for field_name in rollout_targets.keys():
            target_t = rollout_targets[field_name].time[t]
            predicted_t = next_state[field_name]
            field_loss = math.l2_loss(predicted_t - target_t)
            step_loss += math.mean(field_loss, 'batch')

        total_loss += step_loss
        current_state = next_state

    return total_loss / float(self.rollout_steps)
```

### Proposed Implementation

```python
def loss_function(init_state: Tensor, rollout_targets: Tensor) -> Tensor:
    """
    Compute loss with unified tensors.

    Args:
        init_state: Tensor(batch, x, y, field)
        rollout_targets: Tensor(batch, time, x, y, field)
    """
    current_state = init_state
    total_loss = math.tensor(0.0)

    for t in range(self.rollout_steps):
        next_state = self.model(current_state)

        # Single loss computation for all fields
        target_t = rollout_targets.time[t]
        # l2_loss automatically sums over non-batch dimensions (field, x, y)
        step_loss = math.l2_loss(next_state - target_t)
        step_loss = math.mean(step_loss, 'batch')

        total_loss += step_loss
        current_state = next_state

    return total_loss / float(self.rollout_steps)
```

**Benefits:**
- Single loss computation instead of loop over fields
- PhiML's `l2_loss` naturally handles the field dimension
- Cleaner, more readable code
- Better performance (single kernel call)

## Data Loading Changes

### Current Dataset Output

```python
class Batch:
    initial_state: Dict[str, Any]  # {field_name: Tensor(batch, x, y, vector)}
    targets: Dict[str, Any]        # {field_name: Tensor(batch, time, x, y, vector)}
```

### Proposed Dataset Output

```python
class Batch:
    initial_state: Tensor  # Tensor(batch, x, y, field='velocity_0,velocity_1,density,...')
    targets: Tensor        # Tensor(batch, time, x, y, field='velocity_0,velocity_1,density,...')
```

### Dataset Implementation

```python
def iterate_batches(self, batch_size: int, shuffle: bool = True):
    for batch_indices in self._get_batch_indices(batch_size, shuffle):
        samples = [self._get_sample(idx) for idx in batch_indices]

        # Stack all fields together with named dimension
        initial_states = math.stack(
            [s.state for s in samples],  # Each sample has unified state tensor
            math.batch('batch')
        )
        targets = math.stack(
            [s.targets for s in samples],
            math.batch('batch')
        )

        yield Batch(initial_state=initial_states, targets=targets)
```

## Configuration Simplification

### Current Configuration

```yaml
data:
  fields: ['velocity', 'density']
  fields_scheme: 'vvd'      # v=velocity(2), d=density(1) - cryptic!
  fields_type: 'DDS'        # D=dynamic, S=static - error prone!
```

### Proposed Configuration

```yaml
data:
  fields:
    velocity:
      components: 2          # Number of vector components
      type: dynamic          # Network predicts this
    density:
      components: 1
      type: dynamic
    pressure:
      components: 1
      type: static          # Network doesn't predict (passed through)
```

This is more explicit, self-documenting, and less error-prone.

## Field Access Patterns

With named dimensions, field access becomes intuitive:

```python
# Get velocity components from unified state
velocity = state.field['velocity_0,velocity_1']  # Or state.field[0:2]

# Get just density
density = state.field['density']

# Selective updates
state = state.field.replace('velocity_0,velocity_1', new_velocity)

# Compute field-specific metrics
velocity_magnitude = math.vec_length(state.field[0:2])
```

## Physical Model Integration

Physical models also benefit from unified state tensors:

```python
class BurgersModel(PhysicalModel):
    def forward(self, state: Tensor) -> Tensor:
        """
        Single physics step.

        Args:
            state: Tensor(batch?, x, y, field='velocity_x,velocity_y')
        """
        velocity = CenteredGrid(state, extrapolation.PERIODIC, bounds=self.domain)

        # Physics operations work directly on the tensor
        velocity = advect.semi_lagrangian(velocity, velocity, dt=self.dt)
        velocity = diffuse.explicit(velocity, self.diffusion_coeff, dt=self.dt)

        # Return tensor (Grid.values extracts the tensor)
        return velocity.values
```

## Migration Strategy

1. **Phase 1: Add unified tensor support alongside dictionaries**
   - Create helper functions to convert between formats
   - Update Dataset to optionally return unified tensors
   - Add `use_unified_tensors` config flag

2. **Phase 2: Update models to accept both formats**
   - SyntheticModel checks input type and handles accordingly
   - Maintain backward compatibility

3. **Phase 3: Deprecate dictionary format**
   - Update all configs to new format
   - Remove dictionary handling code
   - Simplify codebase

## Potential Challenges

1. **Variable number of components per field**: Some fields are scalars (1), others are vectors (2-3). Named dimensions handle this with explicit component naming.

2. **Field-specific operations**: Some operations may need to act on specific fields. Use named slicing: `state.field['velocity_0,velocity_1']`.

3. **Backward compatibility**: Existing checkpoints and configs use dictionary format. Provide conversion utilities.

4. **PhiML limitations**: Ensure all required operations work with the channel dimension approach. Test `l2_loss`, `native_call`, etc.

## Summary

Using PhiML's named dimensions for multi-field support provides:

| Aspect | Dictionary Approach | Named Dimension Approach |
|--------|--------------------|-----------------------|
| Data structure | `Dict[str, Tensor]` | Single `Tensor` |
| Forward pass | Concat → rename → infer → rename → split | Direct inference |
| Loss computation | Loop over fields | Single operation |
| Config format | Cryptic strings ('vvd', 'DDS') | Explicit YAML dict |
| Code complexity | High (manual bookkeeping) | Low (PhiML handles it) |
| Performance | Multiple operations | Fused operations |
| Type safety | Runtime errors | Dimension name checking |

The named dimension approach aligns better with PhiML's design philosophy and significantly simplifies the codebase while improving performance.
