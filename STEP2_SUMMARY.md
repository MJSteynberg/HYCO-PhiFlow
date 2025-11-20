# Step 2 Migration Summary - PhiML Training Integration

## Overview
Successfully integrated PhiML training into the HYCO-PhiFlow project. Running `python run.py --config-name=burgers.yaml` now uses **PhiML native training** instead of PyTorch training.

## Changes Made

### 1. Configuration ([conf/burgers.yaml](conf/burgers.yaml))
```yaml
# Changed from:
model:
  synthetic:
    name: 'UNet'  # PyTorch model

# To:
model:
  synthetic:
    name: 'PhiMLUNet'  # PhiML native model
```

### 2. Trainer Factory ([src/factories/trainer_factory.py](src/factories/trainer_factory.py))
- **Added import**: `from src.training.synthetic.phiml_trainer import PhiMLSyntheticTrainer`
- **Updated `_create_synthetic_trainer()`**: Now returns `PhiMLSyntheticTrainer` instead of `SyntheticTrainer`
- The PhiML trainer automatically detects model type (PyTorch vs PhiML) and handles appropriately

### 3. New Files Created

#### [src/training/synthetic/phiml_trainer.py](src/training/synthetic/phiml_trainer.py)
- **`PhiMLSyntheticTrainer`**: Supports both PyTorch and PhiML models
- **`torch_to_phiml()`**: Converts PyTorch tensors to PhiML format
- **`phiml_to_torch()`**: Converts PhiML tensors back to PyTorch format
- Uses `phiml.nn.update_weights()` for training (PhiML best practice)
- Uses `phiml.nn.adam()` optimizer for PhiML models

#### PhiML Models ([src/models/synthetic/](src/models/synthetic/))
- `phiml_base.py`: Base class for PhiML models
- `phiml_unet.py`: UNet in pure PhiML
- `phiml_resnet.py`: ResNet in pure PhiML
- `phiml_convnet.py`: ConvNet in pure PhiML

### 4. Test Files
- `test_step1.py`: Tests PhiML model creation and forward passes
- `test_step2.py`: Tests PhiML trainer with tensor conversions

## Training Pipeline Flow

```
┌──────────────────────────────────────────────────────────────┐
│ python run.py --config-name=burgers.yaml                     │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ↓
┌───────────────────────────────────────────────────────────────┐
│ TrainerFactory.create_trainer(config)                         │
│   - Reads config: mode='synthetic', name='PhiMLUNet'          │
│   - Calls _create_synthetic_trainer()                         │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ↓
┌───────────────────────────────────────────────────────────────┐
│ ModelFactory.create_synthetic_model(config)                   │
│   - ModelRegistry.get_synthetic_model()                       │
│   - Returns: PhiMLUNet instance                               │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ↓
┌───────────────────────────────────────────────────────────────┐
│ PhiMLSyntheticTrainer(config, model)                          │
│   - Detects model type: is_phiml_model = True                 │
│   - Creates PhiML optimizer: phiml.nn.adam()                  │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ↓
┌───────────────────────────────────────────────────────────────┐
│ trainer.train(data_source, num_epochs)                        │
│                                                                │
│ For each batch:                                                │
│   1. torch_to_phiml(): Convert PyTorch tensors → PhiML        │
│   2. Define loss_function(init_state, targets):               │
│       - Autoregressive rollout with PhiML model               │
│       - Compute phimath.l2_loss()                             │
│   3. phiml.nn.update_weights():                               │
│       - Handles backward pass                                 │
│       - Updates optimizer                                     │
│       - Returns loss                                          │
└───────────────────────────────────────────────────────────────┘
```

## Key Features

### Tensor Conversions
- **PyTorch format**: `(Batch, Channels, ..., Height, Width)`
- **PhiML format**: `(Batch, ..., Height, Width, Channels)`
- Proper transposition and dimension naming
- Handles time dimensions correctly

### PhiML Training Pattern
```python
def loss_function(init_state, targets):
    # Autoregressive rollout
    current_state = init_state
    for t in range(num_steps):
        next_state = model(current_state)
        loss += phimath.l2_loss(next_state - targets.time[t])
        current_state = next_state
    return loss / num_steps

# One-line training!
loss = phiml.nn.update_weights(network, optimizer, loss_function, inputs, targets)
```

### Backward Compatibility
The `PhiMLSyntheticTrainer` supports **both** PyTorch and PhiML models:
- **PhiML models**: Uses `phiml.nn.update_weights()` and PhiML optimizer
- **PyTorch models**: Falls back to `torch.optim.Adam()` and manual backward

## Verification

### Test Model and Trainer
```bash
source activate phi-env && python test_step2.py
```

Expected output:
- ✅ Tensor conversions work correctly
- ✅ Trainer created with PhiML model
- ✅ Forward passes successful
- ✅ DataLoader compatibility verified

### Run Actual Training (when data is ready)
```bash
source activate phi-env && python run.py --config-name=burgers.yaml
```

## What's Next

### NOT Yet Implemented:
- **Checkpoint Save/Load**: PhiML models don't save/load yet (will implement when needed)
- **Full Training Test**: Requires actual data generation
- **Step 3**: PhiML-native data pipeline (currently uses PyTorch DataLoader)

### To Complete Full PhiML Migration:
1. ✅ **Step 1**: PhiML models (DONE)
2. ✅ **Step 2**: PhiML trainer (DONE)
3. ⏳ **Step 3**: PhiML data pipeline (FUTURE)

## Benefits of Current Implementation

1. **Pure PhiML Training**: No PyTorch training loop, uses `nn.update_weights()`
2. **Automatic Type Detection**: Trainer works with both PyTorch and PhiML models
3. **Clean Tensor Conversions**: Proper handling of dimension ordering
4. **Best Practices**: Follows PhiML examples pattern
5. **Backward Compatible**: Old PyTorch models still work

## Technical Notes

### Dimension Handling
```python
# PyTorch: [B=2, V=3, T=4, H=64, W=64]
torch_tensor = torch.randn(2, 3, 4, 64, 64)

# Convert to PhiML: [B=2, T=4, H=64, W=64, V=3]
phiml_tensor = torch_to_phiml(torch_tensor, format="BVTHW")
# Result: (batchᵇ=2, timeᵇ=4, xˢ=64, yˢ=64, vectorᶜ=3)

# Convert back: [B=2, V=3, T=4, H=64, W=64]
torch_tensor = phiml_to_torch(phiml_tensor, target_format="BVTHW")
```

### Loss Computation
```python
# PhiML L2 loss (more efficient than PyTorch MSE)
loss = phimath.l2_loss(prediction - target)

# Equivalent to:
loss = torch.mean((prediction - target) ** 2)
```

---

**Status**: ✅ Step 2 Complete - PhiML training is now the default for synthetic models
