# Hybrid Training for Temporal Scarcity (5-10 Frames Only)
**Date:** November 4, 2025  
**Purpose:** Implementation guide for HYCO hybrid training with extremely limited temporal observations

---

## Executive Summary

This document addresses the **most challenging data scarcity scenario**: having only **5-10 temporal snapshots from a single simulation**. Traditional machine learning approaches completely fail in this scenario, but hybrid training can succeed by using physics to generate the missing temporal data.

**What This Document Covers:**
1. Why 5-10 frames is fundamentally different from "limited data"
2. How physics generates temporal training data
3. Complete configuration for extreme temporal scarcity
4. Expected outcomes and limitations
5. When this approach works vs. when it doesn't

### Key Insight

> **With only 5-6 frames, you don't have a "small" dataset—you have almost NO dataset.** The only way to train is to use physics to synthesize the missing temporal evolution.

---

## Part 1: The Challenge

### 1.1 What is Temporal Scarcity?

**Your Situation:**
- ✅ You have a **single simulation/experiment**
- ✅ You captured only **5-10 temporal snapshots** (frames)
- ❌ You cannot use sliding window (would give 0-2 samples)
- ❌ You cannot split train/validation (not enough data)
- ❌ Traditional neural network training is impossible

**Example:**
```
Captured Data:
t=0.0s: [velocity_field, density_field]
t=0.2s: [velocity_field, density_field]
t=0.4s: [velocity_field, density_field]
t=0.6s: [velocity_field, density_field]
t=0.8s: [velocity_field, density_field]
t=1.0s: [velocity_field, density_field]

Total: 6 snapshots
```

### 1.2 Why Traditional Approaches Fail Catastrophically

**Attempt 1: Standard Neural Network Training**
```python
num_frames = 6
num_predict_steps = 4  # Standard setting
num_samples = num_frames - num_predict_steps = 2  # Only 2 samples!

model_parameters = 1_000_000  # UNet parameters
samples = 2

# Result: 500,000 parameters per training sample
# Outcome: Complete overfitting, model learns nothing
```

**Attempt 2: Reduce Prediction Steps**
```python
num_frames = 6
num_predict_steps = 1  # Try single-step prediction
num_samples = num_frames - num_predict_steps = 5  # Only 5 samples!

# Result: Still 200,000 parameters per sample
# Outcome: Still catastrophic overfitting
```

**Attempt 3: Smaller Model**
```python
num_frames = 6
model_parameters = 50_000  # Tiny model
num_samples = 5

# Result: 10,000 parameters per sample
# Outcome: Model too simple to learn anything useful
```

**Conclusion**: You cannot train a neural network with 5-6 frames. Period.

### 1.3 Why Hybrid Training Can Work

The hybrid approach succeeds by **using physics to generate the missing temporal data**:

```
Step 1: Your 6 Real Frames
[t=0] → [t=0.2] → [t=0.4] → [t=0.6] → [t=0.8] → [t=1.0]
(6 observations)

Step 2: Physics Generates Temporal Rollouts
From frame t=0:    [t=0] → [t=0.1] → [t=0.2] → ... → [t=2.0] (20 physics steps)
From frame t=0.2:  [t=0.2] → [t=0.3] → [t=0.4] → ... → [t=2.2] (20 physics steps)
From frame t=0.4:  [t=0.4] → [t=0.5] → [t=0.6] → ... → [t=2.4] (20 physics steps)
... (repeat for each frame)

Result: 6 × 20 = 120 physics-generated training samples

Step 3: Combined Training Data
- Real transitions: 5 (between consecutive real frames)
- Physics-generated: 120
- Total training samples: 125

Step 4: Neural Network Training
Train on combined dataset → Can now learn temporal dynamics!
```

**Key Mechanism**: Physics acts as a **temporal data synthesizer** that respects physical laws while filling gaps in observations.

---

## Part 2: Implementation for 5-10 Frames

### 2.1 Core Configuration Strategy

**Critical Parameters:**

| Parameter | Setting | Reasoning |
|-----------|---------|-----------|
| `alpha` (augmentation ratio) | 10.0 - 20.0 | Need 10-20x more synthetic data than real frames |
| `num_predict_steps` | 1 | Single-step to maximize real samples |
| `num_cycles` | 40-50 | Many cycles for iterative refinement |
| `warmup_epochs` | 0 | Cannot afford warmup with so little data |
| `physical_epochs_per_cycle` | 15-20 | Heavy parameter learning from few observations |
| `synthetic_epochs_per_cycle` | 2-3 | Quick updates to avoid overfitting |
| `batch_size` | 2-4 | Tiny batches |
| `model` | unet_tiny | Smallest possible architecture |

### 2.2 Complete Configuration Example

```yaml
# conf/experiment/temporal_scarcity_6frames.yaml
# Configuration for training with ONLY 6 temporal snapshots

defaults:
  - /trainer: hybrid
  - /data: burgers_128
  - /model/synthetic: unet_tiny  # Custom tiny architecture
  - /model/physical: burgers
  - /logging: default

# Data configuration
data:
  dset_name: "burgers_128"
  # Create a special simulation with only 6 frames
  # You'll need to create: data/burgers_128/sim_sparse/
  # containing only 6 frame files

trainer_params:
  # Use the sparse simulation
  train_sim: [sparse]  # Your 6-frame simulation
  val_sim: []  # NO validation data (not enough frames)
  
  # Core settings
  epochs: 200  # Very long training
  batch_size: 2  # Minimal batch size
  num_predict_steps: 1  # CRITICAL: Single-step only
  
  # ============================================================================
  # AUGMENTATION: Extreme ratio for temporal scarcity
  # ============================================================================
  augmentation:
    alpha: 15.0  # Generate 15x more synthetic data
    # With 5 real frame-to-frame transitions × 15 = 75 synthetic samples
    # Total: 80 training samples (vs. 5 without augmentation)
  
  # ============================================================================
  # HYBRID TRAINING: Physics-heavy approach
  # ============================================================================
  hybrid:
    num_cycles: 50  # Many cycles for convergence
    
    # Emphasis on physics (it's your primary data source)
    synthetic_epochs_per_cycle: 2  # Minimal synthetic training
    physical_epochs_per_cycle: 20  # Heavy parameter learning
    
    # NO warmup (can't afford to waste data)
    warmup_synthetic_epochs: 0
  
  # ============================================================================
  # SYNTHETIC MODEL: Minimal architecture
  # ============================================================================
  synthetic:
    learning_rate: 1e-4
    weight_decay: 1e-3  # Strong regularization
    optimizer: adam
    scheduler: constant  # No schedule with so little data
  
  # ============================================================================
  # PHYSICAL MODEL: This is your data generator!
  # ============================================================================
  physical:
    method: 'L-BFGS-B'
    max_iterations: 200  # Allow many iterations
    abs_tol: 1e-8  # Tight tolerance for parameter learning
    
  # Checkpointing every cycle (don't lose progress!)
  checkpoint_freq: 1

# Model configuration - create tiny UNet
model:
  synthetic:
    unet:
      in_channels: 3
      out_channels: 2
      init_features: 16  # TINY (normally 64)
      num_levels: 2      # SHALLOW (normally 4)
      dropout: 0.3       # Heavy dropout
      batch_norm: true

run_params:
  model_type: "hybrid"
  task: "train"
```

### 2.3 Step-by-Step Implementation

**Step 1: Prepare Your Sparse Data**

```powershell
# Create directory for sparse simulation
mkdir data\burgers_128\sim_sparse

# Copy only your 6 frames to this directory
# Name them sequentially: frame_000000.npz, frame_000001.npz, ..., frame_000005.npz
```

**Step 2: Verify Physics Model Parameters**

```yaml
# conf/model/physical/burgers.yaml
model:
  physical:
    burgers:
      dt: 0.1  # Time step between your frames
      domain:
        size_x: 100
        size_y: 100
      resolution:
        x: 128
        y: 128
      
      # CRITICAL: These will be learned from your 6 frames
      learnable_parameters:
        - name: "nu"
          initial_guess: 0.01  # Your best estimate
          bounds: [0.001, 0.1]  # Reasonable physical range
```

**Step 3: Train**

```powershell
python run.py +experiment=temporal_scarcity_6frames
```

**Step 4: Monitor Carefully**

Since you have almost no data, watch these metrics:

```
Critical Metrics:
1. Physical parameter convergence
   - Is 'nu' stabilizing to a reasonable value?
   - Check: Are parameters within physical bounds?

2. Augmentation quality
   - Do physics predictions look realistic?
   - Visual inspection of generated samples

3. Synthetic loss trend
   - Should decrease slowly (don't expect dramatic drops)
   - Watch for: Sudden jumps (indicates instability)

4. Frame reconstruction
   - Can model reconstruct your 6 real frames?
   - This is your only measure of success
```

### 2.4 What to Expect

**Best Case Scenario:**
- ✅ Model learns basic temporal dynamics
- ✅ Can interpolate between your 6 observations
- ✅ Physics parameters converge to reasonable values
- ✅ Short-term predictions (1-2 steps) are decent
- ❌ Long-term rollouts (>5 steps) will drift/diverge
- ❌ Generalization to different initial conditions is limited

**Realistic Expectations:**
```python
# What you CAN achieve:
- Interpolation between observed frames: Good
- 1-step ahead prediction: Reasonable (~80% accuracy)
- 2-3 steps ahead: Moderate (~60% accuracy)
- Parameter identification: If physics is correct, can learn 1-2 parameters

# What you CANNOT achieve:
- Long rollouts (>10 steps): Will diverge
- Extrapolation beyond observed time range: Unreliable
- Generalization to different scenarios: Very limited
- Discovering unknown physics: Impossible with 6 frames
```

---

## Part 3: When This Approach Works

### 3.1 Requirements for Success

For hybrid training to work with 5-10 frames, you MUST have:

✅ **Known Physics Equations**
- You know the governing PDEs (Navier-Stokes, Burgers, etc.)
- Only unknown are parameter values (viscosity, diffusion coefficients)
- Physics model can run forward simulations

✅ **Good Initial Parameter Estimates**
- Your `initial_guess` for learnable parameters is reasonably close
- You know physical bounds (e.g., viscosity must be positive)

✅ **Consistent Frame Spacing**
- Frames are evenly spaced in time
- Time step dt is known

✅ **High-Quality Observations**
- Your 6 frames are noise-free or low-noise
- Spatial resolution is sufficient

✅ **Simple Dynamics**
- Single-field or few coupled fields
- Relatively smooth evolution (no shocks, discontinuities)

### 3.2 When This Approach Fails

❌ **Unknown Physics**
- You don't know the governing equations
- Need to discover physics from data (impossible with 6 frames)

❌ **Complex Multi-Scale Dynamics**
- Turbulence with many scales
- Complex boundary conditions
- Many interacting fields

❌ **Poor Physics Model**
- Your PDE approximation is very inaccurate
- Physics cannot reproduce observed behavior even roughly

❌ **Noisy Data**
- Your 6 frames have significant measurement noise
- Cannot reliably compute frame-to-frame differences

❌ **Irregular Sampling**
- Frames are not evenly spaced
- Unknown or variable time steps

---

## Part 4: Alternative Strategies

### 4.1 If You Have Any Additional Data...

**Even slight increases in data help dramatically:**

| Frames | Training Samples | Difficulty | Recommendation |
|--------|------------------|------------|----------------|
| 6 | 5 | Extreme | Hybrid with α=15-20 |
| 10 | 9 | Very Hard | Hybrid with α=10-15 |
| 15 | 14 | Hard | Hybrid with α=5-10 |
| 20 | 19 | Challenging | Hybrid with α=3-5 |
| 30+ | 29+ | Feasible | Standard hybrid (α=1-2) |

**Action**: If possible, capture even 5-10 more frames!

### 4.2 Multiple Short Simulations

If you can run multiple experiments:

**Option A: Single simulation with 6 frames (this document)**
```
Simulation 1: 6 frames
Total samples: 5
Difficulty: Extreme
```

**Option B: Three simulations with 6 frames each**
```
Simulation 1: 6 frames → 5 samples
Simulation 2: 6 frames → 5 samples
Simulation 3: 6 frames → 5 samples
Total samples: 15
Difficulty: Hard (but much better!)
```

**Recommendation**: If you can generate data, prefer multiple short simulations over one ultra-sparse simulation.

### 4.3 Transfer Learning

If you have related datasets:

```yaml
# Step 1: Pre-train on related dataset with more data
# Example: Burgers with nu=0.01 (100 frames)
python run.py +experiment=pretrain_burgers_nu001

# Step 2: Fine-tune on your 6-frame target dataset
model:
  synthetic:
    checkpoint: "results/models/pretrain_best.pt"  # Load pre-trained

trainer_params:
  train_sim: [sparse]  # Your 6 frames
  synthetic:
    learning_rate: 1e-5  # Very low for fine-tuning
    freeze_encoder: true  # Freeze some layers
```

**When this works:**
- Pre-training dataset has similar physics (same equations, different parameters)
- Can transfer learned representations

---

## Part 5: Practical Example

### 5.1 Complete Workflow: Burgers Equation with 6 Frames

**Your Data:**
```
data/burgers_128/sim_000000_sparse/
├── frame_000000.npz  # t=0.0s
├── frame_000001.npz  # t=0.1s
├── frame_000002.npz  # t=0.2s
├── frame_000003.npz  # t=0.3s
├── frame_000004.npz  # t=0.4s
└── frame_000005.npz  # t=0.5s
```

**Step 1: Configure Hybrid Training**

```yaml
# conf/experiment/burgers_6frames.yaml
defaults:
  - /trainer: hybrid
  - /data: burgers_128
  - /model/synthetic: unet_tiny
  - /model/physical: burgers

trainer_params:
  train_sim: [0]  # Single sparse simulation
  val_sim: []
  
  epochs: 200
  batch_size: 2
  num_predict_steps: 1
  
  augmentation:
    alpha: 15.0  # Generate 75 synthetic samples from 5 real
  
  hybrid:
    num_cycles: 50
    synthetic_epochs_per_cycle: 2
    physical_epochs_per_cycle: 20
    warmup_synthetic_epochs: 0
  
  synthetic:
    learning_rate: 1e-4
    weight_decay: 1e-3
  
  physical:
    method: 'L-BFGS-B'
    max_iterations: 200

model:
  physical:
    burgers:
      nu: 0.01  # Will be learned
      learnable_parameters:
        - name: "nu"
          initial_guess: 0.01
          bounds: [0.001, 0.1]
```

**Step 2: Train**

```powershell
python run.py +experiment=burgers_6frames
```

**Step 3: Monitor Training**

```
Expected Output:

Cycle 1:
  Physical parameter learning...
    Initial nu: 0.01000
    Optimizing on 5 real transitions...
    Learned nu: 0.01234
  
  Generating 75 synthetic samples from physics...
  
  Synthetic training (2 epochs)...
    Real samples: 5
    Augmented samples: 75
    Total: 80
    Train loss: 0.1234

Cycle 2:
  Physical parameter learning...
    Current nu: 0.01234
    Refining with synthetic predictions...
    Learned nu: 0.01189
  
  ... (continues for 50 cycles)

Cycle 50:
  Physical parameter learning...
    Current nu: 0.01198
    Parameter converged
  
  Final Results:
    Learned nu: 0.01198 (true value: ~0.012)
    Train loss: 0.0089
    Can interpolate between frames: Yes
    Can extrapolate beyond t=0.5s: Limited
```

**Step 4: Evaluate**

Since you have no validation data, evaluate on your training frames:

```python
# Reconstruct your 6 frames
predictions = model.predict(frame_0)
mse = compute_mse(predictions, [frame_1, frame_2, ..., frame_5])

# Expected MSE: 0.001 - 0.01 (good reconstruction)
# If MSE > 0.1: Model failed to learn
```

---

## Part 6: Troubleshooting

### 6.1 Physics Parameters Not Converging

**Symptom**: Parameter values oscillate or diverge

**Solutions:**
1. Better initial guess (closer to true value)
2. Tighter bounds on parameters
3. More physical epochs per cycle (25-30)
4. Reduce augmentation alpha temporarily (α=5 to check if physics is the issue)

### 6.2 Neural Network Not Learning

**Symptom**: Loss remains high or increases

**Solutions:**
1. Reduce model size further (init_features=8)
2. Increase weight_decay (1e-2)
3. Check augmented data quality (visualize physics predictions)
4. Reduce learning rate (5e-5)

### 6.3 Training Unstable

**Symptom**: Loss spikes, NaN values

**Solutions:**
1. Reduce learning rate dramatically (1e-5)
2. Clip gradients
3. Check physics model for numerical instabilities
4. Reduce time step dt in physics model

### 6.4 Poor Generalization

**Symptom**: Reconstructs training frames but poor rollouts

**Expected**: This is normal with 6 frames! You cannot expect good generalization.

**Mitigation:**
1. Collect more data (even 5 more frames help significantly)
2. Use ensemble of models
3. Accept limited applicability

---

## Part 7: Conclusion

### Summary

Training with only 5-10 frames is at the **extreme limit** of what's possible:

✅ **Hybrid training can work** by using physics to generate temporal data  
✅ **Best for**: Parameter identification, interpolation between observations  
✅ **Requires**: Known physics, good initial guesses, high-quality data  
❌ **Cannot achieve**: Long rollouts, extrapolation, generalization to new scenarios  
❌ **Needs**: Very careful tuning, realistic expectations

### Recommendations

**If you're in this situation:**

1. **First priority**: Try to get more data (even 5-10 more frames helps enormously)
2. **Second priority**: Multiple short simulations better than one ultra-sparse
3. **Third priority**: Use hybrid training as described in this document
4. **Be realistic**: You cannot expect miracles from 6 frames

**If you can't get more data:**

- Hybrid training is your only hope
- Focus on parameter learning, not prediction
- Use physics as much as possible
- Accept limited generalization

### Key Parameters Quick Reference

```yaml
# For 5-6 frames:
alpha: 15.0-20.0
num_cycles: 40-50
physical_epochs_per_cycle: 20
synthetic_epochs_per_cycle: 2
warmup_epochs: 0
batch_size: 2
model: unet_tiny (init_features=16, num_levels=2)
weight_decay: 1e-3

# For 10-15 frames:
alpha: 8.0-12.0
num_cycles: 30-40
physical_epochs_per_cycle: 15
synthetic_epochs_per_cycle: 3
warmup_epochs: 0
batch_size: 4
model: unet_small (init_features=32, num_levels=3)
weight_decay: 5e-4
```

---

*Last updated: November 4, 2025*  
*For questions about the HYCO architecture, see ARCHITECTURE_REVIEW.md*
