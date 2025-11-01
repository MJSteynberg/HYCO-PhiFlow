# Memory Monitoring Guide

This guide explains the different approaches to memory monitoring in the HYCO-PhiFlow project, from simple to advanced.

**Supported trainers**: Both `SyntheticTrainer` (PyTorch-based) and `PhysicalTrainer` (PhiFlow-based) support memory monitoring.

## Quick Start

### Enable Memory Monitoring in Config

The simplest way to monitor memory during training is to enable it in your config:

```yaml
# In your trainer config (e.g., conf/trainer/synthetic.yaml)
trainer_params:
  enable_memory_monitoring: true
  memory_monitor_batches: 5  # Print detailed info for first 5 batches per epoch
```

Then run training normally:
```bash
# For synthetic (PyTorch) training
python run.py --config-name=burgers_experiment trainer=synthetic_with_memory

# For physical (PhiFlow) training
python run.py --config-name=heat_physical_experiment trainer=physical_with_memory
```

**Synthetic trainer** prints memory for the first 5 batches and peak memory at epoch end.

**Physical trainer** prints memory for the first 5 loss function evaluations during optimization.

---

## Monitoring Approaches

### 1. Configuration-Based Monitoring (Recommended)

**Best for**: Normal training runs where you want optional monitoring

The `EpochMemoryMonitor` is automatically initialized if `enable_memory_monitoring: true` in config.

**Features**:
- Zero code changes required in training logic
- Controlled via config
- Prints memory for first N batches
- Reports peak memory at epoch end
- Minimal performance overhead

**Example output (Synthetic trainer)**:
```
  Batch 0: GPU 23.1 MB, CPU 1645.1 MB, loss=0.014553
  Batch 1: GPU 23.1 MB, CPU 1645.2 MB, loss=0.013612
  ...
  Epoch end: GPU 23.1 MB allocated, 3242.0 MB reserved, peak 2880.3 MB
```

**Example output (Physical trainer)**:
```
After loading ground truth data: CPU: 850.2 MB | GPU Allocated: 15.3 MB | GPU Reserved: 20.0 MB
  Loss evaluation 0: CPU: 852.1 MB | GPU Allocated: 18.5 MB | GPU Reserved: 22.0 MB
    Iteration 1: loss=0.145320
  Loss evaluation 1: CPU: 852.3 MB | GPU Allocated: 18.5 MB | GPU Reserved: 22.0 MB
    Iteration 2: loss=0.098754
  ...
After optimization: CPU: 855.8 MB | GPU Allocated: 18.5 MB | GPU Reserved: 22.0 MB
  Peak GPU memory during optimization: 25.3 MB
```

---

### 2. Context Managers

**Best for**: Profiling specific code blocks

```python
from src.utils.memory_monitor import track_memory

# Track a specific operation
with track_memory("loading dataset"):
    dataset = HybridDataset(...)
```

**Example output**:
```
[Memory] loading dataset: CPU +450.2 MB, GPU +1250.5 MB
         Peak GPU: 1500.2 MB
```

---

### 3. Function Decorators

**Best for**: Profiling entire functions without modifying their code

```python
from src.utils.memory_monitor import monitor_memory

@monitor_memory("training epoch")
def _train_epoch(self):
    # ... training code ...
    pass
```

**Example output**:
```
[Memory] training epoch: CPU +200.5 MB, GPU +2500.8 MB
         Peak GPU: 2880.3 MB
```

---

### 4. Batch-Level Profiling

**Best for**: Detailed step-by-step analysis of training loops

```python
from src.utils.memory_monitor import BatchMemoryProfiler

profiler = BatchMemoryProfiler(enabled=True)

for batch_idx, batch in enumerate(dataloader):
    if batch_idx >= 5:
        profiler.disable()  # Only profile first 5 batches
    
    profiler.checkpoint("start")
    
    batch = batch.to(device)
    profiler.checkpoint("data_to_gpu")
    
    output = model(batch)
    profiler.checkpoint("forward")
    
    loss.backward()
    profiler.checkpoint("backward")
    
    optimizer.step()
    profiler.checkpoint("optimizer")
    
    profiler.print_summary()
    profiler.reset()
```

**Example output**:
```
    Memory profile:
      start → data_to_gpu: GPU 29.1 MB (+19.0 MB)
      data_to_gpu → forward: GPU 2749.9 MB (+2720.8 MB)
      forward → backward: GPU 23.1 MB (-2726.8 MB)
      backward → optimizer: GPU 23.1 MB (+0.0 MB)
```

---

### 5. Manual Tracking

**Best for**: Maximum control and custom metrics

```python
from src.utils.memory_monitor import MemoryMonitor, MemoryTracker

# Simple point-in-time measurement
MemoryMonitor.print_memory_usage("After model creation: ")

# Track multiple checkpoints
tracker = MemoryTracker()
tracker.record("initialization")
# ... do work ...
tracker.record("after_data_loading")
# ... more work ...
tracker.record("after_training")
tracker.print_summary()

# Get specific metrics
cpu_mb = MemoryMonitor.get_cpu_memory_mb()
gpu_mb = MemoryMonitor.get_gpu_memory_mb()
summary = MemoryMonitor.get_memory_summary()
```

---

## Understanding Memory Usage

### GPU Memory Breakdown

During training, GPU memory is used for:

1. **Model weights**: Size of the neural network parameters (~23 MB for typical UNet)
2. **Gradients**: Same size as weights (~23 MB)
3. **Optimizer state**: 2x weights for Adam (~46 MB)
4. **Activations**: Intermediate tensors during forward pass (**LARGEST**, ~2750 MB with batch_size=16)
5. **Input/output data**: Batched data on GPU (~30 MB)

**Total**: Model overhead (~92 MB) + Activations (scales with batch size)

### Why Activations Are Large

In autoregressive training:
- Forward pass for 4 timesteps creates many intermediate tensors
- All activations kept in memory for backward pass
- Scales linearly with batch size and number of rollout steps

**Example**:
- batch_size=16: ~2750 MB activations
- batch_size=32: ~5500 MB activations

### Memory Patterns

Normal pattern during batch processing:
```
Data to GPU:     ~30 MB    (input batch)
After forward:   ~2750 MB  (activations peak)
After backward:  ~23 MB    (activations freed)
After optimizer: ~23 MB    (stable)
```

---

## Optimizing GPU Memory

### 1. Reduce Batch Size
```yaml
trainer_params:
  batch_size: 16  # Instead of 32
```
**Impact**: Halves activation memory

### 2. Use Gradient Accumulation
```yaml
trainer_params:
  batch_size: 16
  gradient_accumulation_steps: 2  # Effective batch size = 32
```
**Impact**: Same effective batch size, lower memory

### 3. Mixed Precision Training (FP16)
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
**Impact**: ~40-50% memory reduction

### 4. Gradient Checkpointing
```python
from torch.utils.checkpoint import checkpoint

# In model forward pass
def forward(self, x):
    x = checkpoint(self.layer1, x)
    x = checkpoint(self.layer2, x)
    return x
```
**Impact**: Trade compute time for memory (useful for very deep networks)

---

## CPU Memory (Lazy Loading)

The lazy loading implementation (Phase 2.5) reduces CPU memory:

**Without lazy loading**: All simulations loaded at once (~1320 MB for 20 sims)
**With lazy loading** (cache=5): Only 5 most recent sims cached (~937 MB)

**Memory savings**: 29% with sliding window mode

Configure in data config:
```yaml
data:
  max_cached_sims: 5  # Number of simulations to keep in LRU cache
  pin_memory: true    # Pin memory for faster GPU transfer
```

---

## Best Practices

### For Development
- Enable memory monitoring: `enable_memory_monitoring: true`
- Use verbose for first 5 batches: `memory_monitor_batches: 5`
- Run with small dataset to quickly identify issues

### For Production
- Disable memory monitoring (small overhead)
- Set appropriate batch size based on GPU capacity
- Use gradient accumulation if needed

### For Debugging Memory Issues
1. Use `BatchMemoryProfiler` for detailed step-by-step analysis
2. Check peak memory with `torch.cuda.max_memory_allocated()`
3. Monitor with `nvidia-smi` in separate terminal
4. Profile with PyTorch memory snapshot:
   ```python
   torch.cuda.memory._record_memory_history(enabled=True)
   # ... run training ...
   torch.cuda.memory._dump_snapshot("memory.pickle")
   ```

### For Research/Experimentation
- Use context managers for specific operations
- Track memory across different configurations
- Log metrics for analysis

---

## Configuration Reference

### Enable Memory Monitoring
```yaml
trainer_params:
  enable_memory_monitoring: true     # Enable/disable monitoring
  memory_monitor_batches: 5          # Number of batches to print details for
```

### Lazy Loading Configuration
```yaml
data:
  max_cached_sims: 5                 # LRU cache size (number of simulations)
  pin_memory: true                   # Pin CPU memory for faster GPU transfer
```

### Memory-Efficient Training
```yaml
trainer_params:
  batch_size: 16                     # Smaller batch = less memory
  gradient_accumulation_steps: 2     # Maintain effective batch size
  use_amp: true                      # Mixed precision (if implemented)
```

---

## Troubleshooting

### "CUDA out of memory" Error

1. **Reduce batch size**:
   ```yaml
   trainer_params:
     batch_size: 8  # or lower
   ```

2. **Clear cache before training**:
   ```python
   torch.cuda.empty_cache()
   ```

3. **Check for memory leaks**:
   - Ensure you're not accumulating gradients unintentionally
   - Detach tensors that don't need gradients
   - Don't keep references to large tensors

### High CPU Memory Usage

1. **Reduce cached simulations**:
   ```yaml
   data:
     max_cached_sims: 3  # Lower cache size
   ```

2. **Check for data leaks**:
   - Review custom data loading code
   - Ensure proper garbage collection

### Memory Monitoring Not Working

1. **Check if enabled in config**:
   ```yaml
   trainer_params:
     enable_memory_monitoring: true
   ```

2. **Check imports**:
   ```python
   from src.utils.memory_monitor import EpochMemoryMonitor
   ```

3. **Verify psutil is installed**:
   ```bash
   pip install psutil
   ```

---

## API Reference

See `src/utils/memory_monitor.py` for complete API documentation.

**Key classes**:
- `MemoryMonitor`: Static methods for point-in-time measurements
- `MemoryTracker`: Track memory over multiple checkpoints
- `EpochMemoryMonitor`: Lightweight monitoring for training loops
- `BatchMemoryProfiler`: Detailed batch-level profiling

**Decorators and context managers**:
- `@monitor_memory()`: Function decorator
- `with track_memory()`: Context manager
