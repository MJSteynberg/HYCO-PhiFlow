# Memory Monitoring - Physical Trainer Integration

## Summary

Memory monitoring has been successfully added to `PhysicalTrainer` using the same clean, configuration-based approach as `SyntheticTrainer`.

## Integration Points

### 1. Initialization
Added memory monitor initialization in `__init__`:
```python
# --- Memory monitoring (optional, enabled by config) ---
enable_memory_monitoring = self.trainer_config.get('enable_memory_monitoring', False)
if enable_memory_monitoring:
    from src.utils.memory_monitor import EpochMemoryMonitor
    self.memory_monitor = EpochMemoryMonitor(...)
```

### 2. Training Method
Added monitoring at key points in `train()`:
- **Before optimization**: `memory_monitor.on_epoch_start()`
- **After loading data**: Print memory snapshot
- **During loss evaluation**: Monitor first N iterations
- **After optimization**: Print peak memory

### 3. Loss Function
Tracks loss function calls and prints memory for first N evaluations:
```python
loss_call_count = [0]

def loss_function(*learnable_tensors):
    if loss_call_count[0] < memory_monitor.verbose_batches:
        MemoryMonitor.print_memory_usage(f"Loss evaluation {loss_call_count[0]}")
        print(f"Iteration {loss_call_count[0]}: loss={final_loss}")
    loss_call_count[0] += 1
```

## Differences from Synthetic Trainer

| Aspect | Synthetic Trainer | Physical Trainer |
|--------|------------------|------------------|
| **Optimization** | PyTorch SGD/Adam | PhiFlow math.minimize (L-BFGS-B) |
| **Iterations** | Explicit batch loop | Black-box optimizer |
| **Monitoring Points** | Batch start/end | Loss function evaluations |
| **Memory Pattern** | Per-batch tracking | Per-evaluation tracking |
| **Peak Memory** | Per epoch | Per optimization run |

## Configuration

Use the same config structure for both trainers:
```yaml
trainer_params:
  enable_memory_monitoring: true
  memory_monitor_batches: 5  # First N batches/evaluations to print
```

## Usage

### Synthetic (PyTorch-based)
```bash
python run.py --config-name=burgers_experiment trainer=synthetic_with_memory
```

### Physical (PhiFlow-based)
```bash
python run.py --config-name=burgers_physical_experiment trainer=physical_with_memory
```

## Example Output Comparison

### Synthetic Trainer
```
Batch 0: GPU 23.1 MB, CPU 1634.7 MB, loss=0.005310
Batch 1: GPU 23.1 MB, CPU 1645.0 MB, loss=0.016940
...
Epoch end: GPU 14.4 MB allocated, 3242.0 MB reserved, peak 2880.3 MB
```

### Physical Trainer
```
After loading ground truth data: CPU: 674.1 MB | GPU: 0.8 MB
Loss evaluation 0: CPU: 853.2 MB | GPU: 0.8 MB
  Iteration 1: loss=(288.784) along batchᵇ
Loss evaluation 1: CPU: 1019.1 MB | GPU: 1.4 MB
  Iteration 2: loss=(288.784) along batchᵇ
...
After optimization: CPU: 1052.8 MB | GPU: 1.4 MB
Peak GPU memory during optimization: 5.9 MB
```

## Key Observations

### Memory Usage Patterns

**Physical Trainer (observed)**:
- Initial: ~674 MB CPU, ~0.8 MB GPU
- During optimization: ~1052 MB CPU, ~1.4 MB GPU  
- Peak GPU: ~6 MB
- Memory grows with each loss evaluation (PhiFlow computational graph)

**Synthetic Trainer (observed)**:
- Initial: ~1141 MB CPU, ~10 MB GPU
- During forward: ~1436 MB CPU, ~2745 MB GPU
- After backward: ~1572 MB CPU, ~18 MB GPU
- Peak GPU: ~2880 MB

**Key difference**: Physical trainer uses much less GPU memory because PhiFlow fields stay primarily in CPU memory, with only small computations on GPU.

## Files Modified

1. `src/training/physical/trainer.py`: Added memory monitoring
2. `conf/trainer/physical_with_memory.yaml`: New config file
3. `docs/MEMORY_MONITORING.md`: Updated with physical trainer info

## Unified API

Both trainers now support the same monitoring interface:
- Same config parameters
- Same `EpochMemoryMonitor` class
- Same optional dependency pattern
- Same clean, non-intrusive integration

## Benefits

✅ **Consistent API** across both trainer types  
✅ **Minimal code changes** - 3-4 monitoring points  
✅ **Configuration-driven** - no code changes to enable/disable  
✅ **Non-intrusive** - training logic stays clean  
✅ **Informative** - provides insights into memory patterns  
✅ **Flexible** - can adjust verbosity level  

## Testing

Tested with:
- ✅ `burgers_physical_quick_test` config
- ✅ Memory monitoring enabled
- ✅ First 5 loss evaluations tracked
- ✅ Peak memory reported correctly
- ✅ No performance impact when disabled

## Future Enhancements

Potential improvements:
- Add iteration progress tracking for L-BFGS-B
- Export memory metrics to logs
- Compare memory across different physical models
- Profile PhiFlow field operations
