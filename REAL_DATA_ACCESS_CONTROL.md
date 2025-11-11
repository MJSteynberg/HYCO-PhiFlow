# Real Data Access Control in Hybrid Training

## Overview

The `real_data_access` parameter controls which models are allowed to train on real observational data versus generated/predicted data. This provides fine-grained control over the data flow in hybrid training.

## Parameter Options

```yaml
hybrid:
  real_data_access: 'option'  # Choose from options below
```

| Option | Synthetic Model Sees | Physical Model Sees | Use Case |
|--------|---------------------|---------------------|----------|
| `'both'` (default) | Real + Physics predictions | Real + Synthetic predictions | Standard hybrid training |
| `'synthetic_only'` | Real + Physics predictions | Synthetic predictions ONLY | Data-driven synthetic, adaptive physics |
| `'physical_only'` | Physics predictions ONLY | Real + Synthetic predictions | Physics-guided synthetic, data for calibration |
| `'neither'` | Physics predictions ONLY | Synthetic predictions ONLY | Self-supervised co-evolution (experimental) |

## Training Flow Diagrams

### Option 1: `real_data_access: 'both'` (Default)

```
Standard Hybrid Training - Both models see real data

Cycle N:
  ┌─────────────────────────────────────────────────┐
  │ Synthetic Model Training                        │
  │   Input: REAL data + Physics predictions        │
  │   Output: Learned dynamics                      │
  └─────────────────────────────────────────────────┘
                        ↓
  ┌─────────────────────────────────────────────────┐
  │ Physical Model Training                         │
  │   Input: REAL data + Synthetic predictions      │
  │   Output: Calibrated parameters                 │
  └─────────────────────────────────────────────────┘
```

### Option 2: `real_data_access: 'synthetic_only'`

```
Synthetic Sees Real Data - Physics adapts to synthetic

Cycle N:
  ┌─────────────────────────────────────────────────┐
  │ Synthetic Model Training                        │
  │   Input: REAL data ✓ + Physics predictions      │
  │   Output: Data-driven dynamics                  │
  └─────────────────────────────────────────────────┘
                        ↓
  ┌─────────────────────────────────────────────────┐
  │ Physical Model Training                         │
  │   Input: Synthetic predictions ONLY (no real ✗) │
  │   Output: Parameters adapted to synthetic       │
  └─────────────────────────────────────────────────┘
```

**Use when:**
- You have high-quality observational data
- You want synthetic model to be primary learner from data
- You want physics parameters to "reverse engineer" from synthetic predictions
- Testing if physics can discover parameters from learned dynamics

### Option 3: `real_data_access: 'physical_only'`

```
Physical Sees Real Data - Synthetic learns from physics

Cycle N:
  ┌─────────────────────────────────────────────────┐
  │ Physical Model Training                         │
  │   Input: REAL data ✓ + Synthetic predictions    │
  │   Output: Calibrated parameters from data       │
  └─────────────────────────────────────────────────┘
                        ↓
  ┌─────────────────────────────────────────────────┐
  │ Synthetic Model Training                        │
  │   Input: Physics predictions ONLY (no real ✗)   │
  │   Output: Physics-guided dynamics               │
  └─────────────────────────────────────────────────┘
```

**Use when:**
- You trust your physics model (correct equations)
- Real data is primarily for parameter identification/calibration
- You want synthetic model to be physics-guided
- Testing if synthetic can learn purely from calibrated physics

### Option 4: `real_data_access: 'neither'`

```
Self-Supervised Co-Evolution - Models learn from each other

Warmup Phase (Uses Real Data):
  ┌─────────────────────────────────────────────────┐
  │ Synthetic Model Warmup                          │
  │   Input: REAL data ✓ (establish baseline)       │
  │   Output: Initial learned dynamics              │
  └─────────────────────────────────────────────────┘

Then Cycle N (No Real Data):
  ┌─────────────────────────────────────────────────┐
  │ Synthetic Model Training                        │
  │   Input: Physics predictions ONLY (no real ✗)   │
  │   Output: Refined dynamics                      │
  └─────────────────────────────────────────────────┘
                        ↓
  ┌─────────────────────────────────────────────────┐
  │ Physical Model Training                         │
  │   Input: Synthetic predictions ONLY (no real ✗) │
  │   Output: Refined parameters                    │
  └─────────────────────────────────────────────────┘
```

**Use when:**
- Testing theoretical limits of self-supervised learning
- Investigating adversarial-like training dynamics
- Research into model co-evolution

**⚠️ WARNING:** Highly experimental! May diverge if models reinforce errors.

## Configuration Examples

### Example 1: Data-Driven Synthetic Model

```yaml
# conf/experiment/hybrid_synthetic_only_real_data.yaml
hybrid:
  real_data_access: 'synthetic_only'
  synthetic_epochs_per_cycle: 7  # More emphasis on synthetic
  physical_epochs_per_cycle: 3
  warmup_synthetic_epochs: 10    # Good warmup on real data

# Result: Synthetic learns from data, physics adapts to it
```

Run with:
```bash
python run.py +experiment=hybrid_synthetic_only_real_data
```

### Example 2: Physics-Guided Synthetic Model

```yaml
# conf/experiment/hybrid_physical_only_real_data.yaml
hybrid:
  real_data_access: 'physical_only'
  synthetic_epochs_per_cycle: 5
  physical_epochs_per_cycle: 5   # Balanced training
  warmup_synthetic_epochs: 0     # No warmup needed

# Result: Physics calibrated on data, synthetic learns from physics
```

Run with:
```bash
python run.py +experiment=hybrid_physical_only_real_data
```

### Example 3: Self-Supervised (Experimental)

```yaml
# conf/experiment/hybrid_no_real_data.yaml
hybrid:
  real_data_access: 'neither'
  num_cycles: 20                  # More cycles
  warmup_synthetic_epochs: 20     # CRITICAL: Good warmup!

# Result: Models co-evolve without real data (after warmup)
```

Run with:
```bash
python run.py +experiment=hybrid_no_real_data
```

## Implementation Details

### Code Location

The implementation is in `src/training/hybrid/trainer.py`:

```python
# Parameter initialization
self.real_data_access = self.hybrid_config.get("real_data_access", "both")

# In _train_synthetic_with_augmentation():
use_real_data = self.real_data_access in ['both', 'synthetic_only']
if not use_real_data:
    return self._train_synthetic_on_generated_only(generated_data)

# In _train_physical_with_augmentation():
use_real_data = self.real_data_access in ['both', 'physical_only']
if not use_real_data:
    return self._train_physical_on_generated_only(generated_data)
```

### Validation

The parameter is validated at initialization:

```python
valid_options = ['both', 'synthetic_only', 'physical_only', 'neither']
if self.real_data_access not in valid_options:
    raise ValueError(f"Invalid real_data_access: '{self.real_data_access}'")
```

### Logging

Training logs clearly indicate when real data is excluded:

```
Training synthetic model with augmented data...
  Synthetic model: using ONLY synthetic predictions (no real data)
  Training synthetic model on 120 generated samples only
  Synthetic training complete (loss: 0.0234)
```

## Use Cases and Recommendations

### Scientific Investigation

**Question:** Can physics parameters be discovered from learned dynamics?

**Setup:**
```yaml
real_data_access: 'synthetic_only'
```

**Expected Outcome:** Physics model should learn parameters that make its predictions match the synthetic model's learned dynamics.

---

**Question:** Can neural networks learn accurate dynamics from physics alone?

**Setup:**
```yaml
real_data_access: 'physical_only'
```

**Expected Outcome:** Synthetic model should learn dynamics that match the calibrated physics model.

---

**Question:** Can models co-evolve without ground truth?

**Setup:**
```yaml
real_data_access: 'neither'
warmup_synthetic_epochs: 20  # Critical!
```

**Expected Outcome:** After warmup, models should iteratively refine each other (or diverge).

### Practical Applications

#### Data Privacy

If real data is sensitive but physics model is shareable:

```yaml
# Train synthetic on protected data first
# Then share synthetic model to train physics remotely
real_data_access: 'synthetic_only'
```

#### Limited Data Quality

If real data is noisy but physics is reliable:

```yaml
# Use data only for parameter calibration
# Train synthetic on cleaner physics predictions
real_data_access: 'physical_only'
```

## Monitoring and Diagnostics

### Key Metrics to Watch

When using non-default `real_data_access` settings, monitor:

1. **Parameter Convergence**
   - Do physical parameters stabilize?
   - Are they physically reasonable?

2. **Loss Trends**
   - Both losses should generally decrease
   - Watch for divergence (losses increasing)

3. **Prediction Quality**
   - Validate on held-out real data
   - Check if predictions respect physics

4. **Cross-Model Agreement**
   - Do synthetic and physical predictions align?
   - Divergence indicates issues

### Warning Signs

🚨 **Divergence:**
- Losses increasing over cycles
- Physical parameters oscillating wildly
- Predictions becoming unrealistic

**Solution:** Reduce learning rates, increase regularization, or revert to `'both'`

🚨 **Mode Collapse:**
- One model outputs constant predictions
- No learning progress

**Solution:** Check warmup, adjust cycle balance, verify augmentation quality

## Comparison Table

| Aspect | `both` | `synthetic_only` | `physical_only` | `neither` |
|--------|--------|------------------|-----------------|-----------|
| **Synthetic sees real data** | ✓ | ✓ | ✗ | ✗ (only warmup) |
| **Physical sees real data** | ✓ | ✗ | ✓ | ✗ |
| **Stability** | High | Medium | Medium | Low |
| **Data efficiency** | Best | Good | Good | Poor |
| **Physics guidance** | Balanced | Low | High | Medium |
| **Data guidance** | Balanced | High | Low | None (after warmup) |
| **Use case** | Standard | Data-driven | Physics-driven | Experimental |
| **Recommended for production** | ✓ | ✓ | ✓ | ✗ |

## Backward Compatibility

**Default behavior is unchanged:**
```yaml
# If not specified, defaults to 'both' (standard hybrid training)
hybrid:
  # real_data_access: 'both'  # Implicit default
```

All existing configurations continue to work without modification.

## Future Extensions

Potential future enhancements:

1. **Dynamic Access Control:** Change `real_data_access` per cycle
2. **Partial Access:** Give models different subsets of real data
3. **Probabilistic Access:** Randomly sample which model sees real data each cycle
4. **Weighted Access:** Gradually transition from one mode to another

## References

- Main implementation: `src/training/hybrid/trainer.py`
- Base configuration: `conf/trainer/hybrid.yaml`
- Example experiments: `conf/experiment/hybrid_*_real_data.yaml`
- Architecture documentation: `ARCHITECTURE_REVIEW.md`

---

*Added: November 4, 2025*  
*Feature: Real Data Access Control*
