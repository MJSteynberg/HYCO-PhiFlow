# HYCO Implementation - Quick Reference

**Last Updated:** November 2, 2025

---

## 🔴 CRITICAL FIX REQUIRED FIRST

### Problem: Improper Non-Convergence Handling

**Current Code (WRONG):**
```python
except math.NotConverged as e:
    estimated_tensors = tuple(self.initial_guesses)  # ❌ Falls back to initial!
    if hasattr(e, 'result') and hasattr(e.result, 'x'):
        estimated_tensors = e.result.x
```

**Correct Solution: Use `suppress` in Solve**

```python
# In _setup_optimization():
suppress_convergence = self.trainer_config.get("suppress_convergence_errors", False)
suppress_list = []
if suppress_convergence:
    suppress_list.append(math.NotConverged)

return math.Solve(
    method=method,
    abs_tol=abs_tol,
    x0=self.initial_guesses,
    max_iterations=max_iterations,
    suppress=tuple(suppress_list)  # ← KEY FIX
)
```

**Why This Matters:**
- With `suppress=(math.NotConverged,)`, `math.minimize` returns best parameters found
- No exception raised, no fallback to initial guess
- Perfect for HYCO where we want a few optimization steps per epoch

**Config:**
```yaml
# conf/trainer/physical_hybrid.yaml
method: 'L-BFGS-B'
max_iterations: 5  # Intentionally low
suppress_convergence_errors: true  # Enable suppression
```

---

## 🎯 Implementation Phases

### Phase 0: Fix Critical Issues (MUST DO FIRST)
**Time:** 2-3 days  
**Files:**
- `src/training/physical/trainer.py`
- `src/config/trainer_config.py`
- `conf/trainer/physical_hybrid.yaml`

**Tasks:**
1. Add `suppress_convergence_errors` config option
2. Modify `_setup_optimization()` to use `suppress`
3. Simplify exception handling
4. Add tests

---

### Phase 1: Foundation
**Time:** 3-4 days  
**New Files:**
- `src/training/hybrid/__init__.py`
- `src/training/hybrid/trainer.py`
- `conf/trainer/hybrid.yaml`
- `conf/burgers_hybrid_experiment.yaml`

**Tasks:**
1. Create `HybridTrainer` skeleton
2. Register in `TrainerFactory`
3. Add configuration schemas

---

### Phase 2: Core Implementation
**Time:** 5-6 days

**Key Methods to Implement:**

```python
class HybridTrainer(AbstractTrainer):
    def _generate_synthetic_rollout(self, initial_state, num_steps) -> torch.Tensor
    def _generate_physical_rollout(self, initial_state, num_steps) -> Dict[str, Field]
    def _prepare_hybrid_training_data(self, real, pred, alpha) -> Tuple
    def _train_synthetic_epoch(self, real_targets, physical_preds) -> float
    def _optimize_physical_epoch(self, real_targets, synthetic_preds) -> float
    def train(self) -> Dict[str, Any]
```

---

## 📋 Configuration Templates

### Minimal Test Config

```yaml
# conf/burgers_hybrid_quick_test.yaml
defaults:
  - data: burgers_128
  - model/physical: burgers
  - model/synthetic: unet
  - trainer: hybrid
  - _self_

run_params:
  experiment_name: 'burgers_hybrid_test'
  mode: ['train']
  model_type: 'hybrid'

trainer_params:
  train_sim: [0, 1]
  epochs: 5
  num_predict_steps: 5
  alpha: 0.5
  
  synthetic:
    learning_rate: 1e-4
    batch_size: 16
  
  physical:
    method: 'L-BFGS-B'
    max_iterations: 3
    suppress_convergence_errors: true
```

### Production Config

```yaml
# conf/burgers_hybrid_experiment.yaml
defaults:
  - data: burgers_128
  - model/physical: burgers
  - model/synthetic: unet
  - trainer: hybrid
  - _self_

run_params:
  experiment_name: 'burgers_hybrid_full'
  mode: ['train', 'evaluate']
  model_type: 'hybrid'

trainer_params:
  train_sim: [0, 1, 2, 3, 4]
  val_sim: [5]
  epochs: 50
  num_predict_steps: 10
  
  # Hybrid parameters
  alpha: 0.5
  interleave_frequency: 1
  warmup_epochs: 5
  
  synthetic:
    learning_rate: 1e-4
    batch_size: 16
    optimizer: adam
    use_sliding_window: false
  
  physical:
    method: 'L-BFGS-B'
    abs_tol: 1e-6
    max_iterations: 5
    suppress_convergence_errors: true
```

---

## 🧪 Testing Checklist

### Unit Tests

- [ ] `test_suppress_convergence_disabled_by_default()`
- [ ] `test_suppress_convergence_enabled()`
- [ ] `test_physical_with_low_iterations()`
- [ ] `test_generate_synthetic_predictions()`
- [ ] `test_generate_physical_predictions()`
- [ ] `test_field_tensor_conversion_preserves_values()`

### Integration Tests

- [ ] `test_hybrid_training_one_epoch()`
- [ ] `test_hybrid_training_warmup()`
- [ ] `test_interleaved_training_loop()`

### System Tests

- [ ] `test_burgers_hybrid_training_end_to_end()`
- [ ] `test_heat_hybrid_training()`
- [ ] `test_smoke_hybrid_training()`

---

## 🚀 Quick Start Commands

### 1. Test Current Physical Trainer

```bash
conda activate torch-env
python run.py --config-name burgers_physical_quick_test
```

### 2. Apply Critical Fix

Edit `src/training/physical/trainer.py`:
- Add `suppress` parameter to `Solve()`
- Test with low iterations

### 3. Test Fixed Behavior

```bash
python run.py --config-name burgers_physical_quick_test \
  trainer_params.suppress_convergence_errors=true \
  trainer_params.max_iterations=3
```

### 4. Create Hybrid Skeleton

```bash
mkdir -p src/training/hybrid
touch src/training/hybrid/__init__.py
touch src/training/hybrid/trainer.py
touch conf/trainer/hybrid.yaml
```

### 5. Test Hybrid Trainer Registration

```python
from src.factories.trainer_factory import TrainerFactory
print(TrainerFactory.list_available_trainers())
# Should show: ['synthetic', 'physical', 'hybrid']
```

### 6. Run First Hybrid Training (after implementation)

```bash
python run.py --config-name burgers_hybrid_quick_test
```

---

## 📊 Key Metrics to Track

| Metric | Target | Notes |
|--------|--------|-------|
| Synthetic training time per epoch | < 30s | On GPU |
| Physical optimization time per epoch | < 60s | With max_iterations=5 |
| Memory usage | < 8GB | On GPU |
| Field-Tensor conversion overhead | < 10% | Of epoch time |
| Test coverage | > 80% | New code |

---

## ⚠️ Common Pitfalls

### 1. Forgetting to Suppress Convergence Errors

**Problem:** Physical optimization raises `NotConverged` every epoch.

**Solution:** Set `suppress_convergence_errors: true` in config.

### 2. Field-Tensor Shape Mismatches

**Problem:** Conversion fails with dimension errors.

**Solution:** Verify metadata matches, check batch dimensions.

### 3. Memory Overflow

**Problem:** Both models loaded, runs out of memory.

**Solution:** Use gradient checkpointing, reduce batch size, mixed precision.

### 4. Initial States Not Synchronized

**Problem:** Models start from different states.

**Solution:** Load data once, convert to both formats from same source.

### 5. Unstable Training

**Problem:** Losses oscillate or diverge.

**Solution:** Add warmup epochs, tune alpha, clip gradients.

---

## 🔗 Related Documents

- Full implementation strategy: `docs/HYCO_IMPLEMENTATION_STRATEGY.md`
- Original code review: (in this conversation)
- PhiML docs: `phiml_docs.json`
- Phi docs: `phi_docs.json`

---

## 📞 Key Decision Points

### Decision 1: Interleave Frequency

**Question:** How often to update physical model?

**Options:**
- Every epoch (frequency=1): More coupling, slower
- Every N epochs (frequency=N): Less coupling, faster

**Recommendation:** Start with 1, adjust based on convergence.

### Decision 2: Alpha (Real Data Weight)

**Question:** How to weight real vs synthetic data?

**Options:**
- 0.5: Equal weight
- 0.6-0.7: Prefer real data
- 0.3-0.4: Prefer synthetic data

**Recommendation:** Start with 0.5, tune based on validation loss.

### Decision 3: Physical Max Iterations

**Question:** How many optimization steps per epoch?

**Options:**
- 3-5: Fast, less accurate
- 10-20: Slower, more accurate
- Auto: Let it converge

**Recommendation:** Start with 5 for HYCO, no auto-convergence.

### Decision 4: Warmup Duration

**Question:** How many epochs to pre-train synthetic model?

**Options:**
- 0: Start hybrid immediately
- 5-10: Give synthetic head start
- 20+: Fully pre-train synthetic

**Recommendation:** Start with 5, adjust based on stability.

---

## 🎓 Learning Resources

### PhiML Solve Documentation

```python
# Key parameters:
Solve(
    method='L-BFGS-B',        # Optimization method
    abs_tol=1e-6,             # Absolute tolerance
    max_iterations=100,       # Maximum iterations
    suppress=(NotConverged,), # Suppress specific errors
    x0=initial_guess,         # Starting point
)
```

### Field-Tensor Conversion

```python
# Converter creation
from src.utils.field_conversion import make_batch_converter

converter = make_batch_converter(field_metadata_dict)

# Fields → Tensors
tensor = converter.fields_to_tensor_batch(fields_dict)

# Tensors → Fields
fields = converter.tensor_to_fields_batch(tensor, metadata)
```

---

## ✅ Success Checklist

### Before Implementation
- [ ] Read full implementation strategy
- [ ] Understand non-convergence handling
- [ ] Review existing trainer code
- [ ] Set up test environment

### Phase 0 Complete When
- [ ] `suppress` parameter working
- [ ] Tests pass for suppression
- [ ] Low max_iterations works without crash
- [ ] Documentation updated

### Phase 1 Complete When
- [ ] `HybridTrainer` skeleton created
- [ ] Registered in factory
- [ ] Configuration files created
- [ ] Can instantiate hybrid trainer

### Phase 2 Complete When
- [ ] Prediction generation works
- [ ] Data augmentation works
- [ ] Training loop completes
- [ ] Tests pass

### Production Ready When
- [ ] All tests pass
- [ ] Documentation complete
- [ ] Runs on all PDEs
- [ ] Performance acceptable
- [ ] No breaking changes

---

**Document Version:** 1.0  
**For detailed information, see:** `docs/HYCO_IMPLEMENTATION_STRATEGY.md`
