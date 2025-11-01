# Performance Monitoring Guide

## Overview

Performance monitoring combines **timing** and **memory tracking** to help you identify bottlenecks and optimize your code. This guide shows how to use the performance monitoring tools effectively.

## Quick Start

### Enable in Training Config

```yaml
# In conf/trainer/synthetic_with_memory.yaml or physical_with_memory.yaml
trainer_params:
  enable_memory_monitoring: true  # Enables both memory AND timing
  memory_monitor_batches: 5       # Verbose for first N batches/iterations
```

### Run Training

```bash
# Synthetic trainer
python run.py --config-name=burgers_experiment trainer=synthetic_with_memory

# Physical trainer  
python run.py --config-name=heat_physical_experiment trainer=physical_with_memory
```

### Example Output (Synthetic)

```
  Batch 0: 0.538s, GPU 23.1MB, CPU 1648.9MB, loss=0.009458
  Batch 1: 0.185s, GPU 23.1MB, CPU 1659.6MB, loss=0.016763
  ...
  Epoch end: 4.774s total, GPU 14.4MB, 3236MB reserved, peak 2880MB
  Batch timing: avg=0.201s, min=0.082s, max=0.538s
```

**Key insights**:
- First batch is slower (0.538s) - GPU warmup
- Subsequent batches are fast (0.185s) - consistent performance
- Batch 0 is a bottleneck - can be optimized with warmup

### Example Output (Physical)

```
==========================================================================================
PERFORMANCE SUMMARY
==========================================================================================
Operation                      Time (s)     CPU (MB)        GPU (MB)        GPU Peak
------------------------------------------------------------------------------------------
load_ground_truth                   0.396          +76.9           +0.8         1.7
optimization                        2.361          +40.9           +0.0         5.9
------------------------------------------------------------------------------------------
TOTAL                               2.758s
==========================================================================================
```

**Key insights**:
- Data loading: 0.396s (14% of total time)
- Optimization: 2.361s (86% of total time) - main computation
- Focus optimization efforts on the loss function

---

## Monitoring Tools

### 1. PerformanceMonitor (Detailed Analysis)

**Best for**: Profiling specific operations with detailed breakdowns

```python
from src.utils.memory_monitor import PerformanceMonitor

monitor = PerformanceMonitor()

with monitor.track("data loading"):
    data = load_large_dataset()

with monitor.track("preprocessing"):
    data = preprocess(data)

with monitor.track("model forward"):
    output = model(data)

monitor.print_summary()
```

**Output**:
```
==========================================================================================
PERFORMANCE SUMMARY
==========================================================================================
Operation                      Time (s)     CPU (MB)        GPU (MB)        GPU Peak
------------------------------------------------------------------------------------------
data loading                        0.450         +523.1           +0.0         0.0
preprocessing                       0.120         +128.5           +0.0         0.0
model forward                       0.850          +45.2        +1250.3      2880.5
------------------------------------------------------------------------------------------
TOTAL                               1.420s
==========================================================================================
```

### 2. Aggregated Operations

**Best for**: Analyzing repeated operations (batches, iterations)

```python
monitor = PerformanceMonitor()

for epoch in range(10):
    for batch in dataloader:
        with monitor.track("batch_forward", aggregate=True):
            output = model(batch)
        
        with monitor.track("batch_backward", aggregate=True):
            loss.backward()

monitor.print_summary(show_aggregate=True)
```

**Output**:
```
==========================================================================================
AGGREGATE STATISTICS (for repeated operations)
==========================================================================================
Operation                      Count    Total (s)    Mean (s)     Min (s)      Max (s)
------------------------------------------------------------------------------------------
batch_forward                  150      15.234       0.102        0.085        0.425
batch_backward                 150       8.451       0.056        0.048        0.112
==========================================================================================
```

**Analysis**:
- Forward pass: avg 0.102s, but max 0.425s (outlier!)
- First batch takes 4x longer - consider warmup
- Backward pass is consistent: 0.048-0.112s

### 3. Context Manager (Quick Profiling)

**Best for**: One-off profiling of code blocks

```python
from src.utils.memory_monitor import track_performance

with track_performance("critical_operation"):
    result = expensive_function()
```

**Output**:
```
[Performance] critical_operation: 1.234s, CPU +250.5MB, GPU +512.0MB
              GPU peak: 1024.3MB
```

### 4. Function Decorator

**Best for**: Profiling entire functions without modifying code

```python
from src.utils.memory_monitor import monitor_performance

@monitor_performance("data processing")
def process_batch(batch):
    # ... processing code ...
    return result

# Automatically prints timing when called
result = process_batch(data)
```

### 5. EpochPerformanceMonitor (Training Loops)

**Best for**: Monitoring training with minimal overhead

```python
from src.utils.memory_monitor import EpochPerformanceMonitor

monitor = EpochPerformanceMonitor(enabled=True, verbose_batches=5)

for epoch in range(epochs):
    monitor.on_epoch_start()
    
    for batch_idx, batch in enumerate(dataloader):
        monitor.on_batch_start(batch_idx)
        
        # ... training code ...
        
        monitor.on_batch_end(batch_idx, loss)
    
    monitor.on_epoch_end()
```

---

## Identifying Bottlenecks

### Strategy 1: Compare Operation Times

```python
monitor = PerformanceMonitor()

with monitor.track("data_loading"):
    data = load_data()

with monitor.track("preprocessing"):
    data = preprocess(data)

with monitor.track("model_forward"):
    output = model(data)

with monitor.track("post_processing"):
    result = post_process(output)

monitor.print_summary()

# Analyze
total_time = monitor.get_total_time()
for metric in monitor.get_metrics():
    percentage = (metric.duration_seconds / total_time) * 100
    if percentage > 50:
        print(f"⚠️  BOTTLENECK: {metric.operation} ({percentage:.1f}%)")
    elif percentage > 20:
        print(f"⚡ Consider optimizing: {metric.operation} ({percentage:.1f}%)")
```

### Strategy 2: Analyze Batch Timing Variance

```python
monitor = EpochPerformanceMonitor(enabled=True)

# After training
stats = monitor.get_statistics()

print(f"Average batch time: {stats['avg_batch_time']:.3f}s")
print(f"Min batch time: {stats['min_batch_time']:.3f}s")  
print(f"Max batch time: {stats['max_batch_time']:.3f}s")

# High variance = inconsistent performance
variance_ratio = stats['max_batch_time'] / stats['min_batch_time']
if variance_ratio > 2.0:
    print(f"⚠️  High variance ({variance_ratio:.1f}x) - investigate outliers!")
```

### Strategy 3: Profile Different Configurations

```python
results = {}

for batch_size in [8, 16, 32, 64]:
    monitor = PerformanceMonitor()
    
    with monitor.track(f"batch_size_{batch_size}"):
        train_with_batch_size(batch_size)
    
    results[batch_size] = monitor.get_total_time()

# Find optimal batch size
optimal = min(results, key=results.get)
print(f"Optimal batch size: {optimal} ({results[optimal]:.3f}s)")
```

---

## Common Bottlenecks and Solutions

### 1. Data Loading is Slow

**Symptoms**:
- `data_loading` takes >30% of total time
- High CPU usage, low GPU usage

**Solutions**:
```python
# Use more DataLoader workers
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

# Enable pin_memory for faster GPU transfer
dataloader = DataLoader(dataset, batch_size=32, pin_memory=True)

# Use prefetching
dataloader = DataLoader(dataset, batch_size=32, prefetch_factor=2)
```

### 2. First Batch is Slow

**Symptoms**:
- Batch 0: 0.538s
- Batch 1+: 0.185s (3x faster)

**Solutions**:
```python
# Add warmup batch before timing
with torch.no_grad():
    dummy_batch = next(iter(dataloader))
    model(dummy_batch.to(device))
    
# Now start timing
for batch in dataloader:
    # ... actual training ...
```

### 3. Forward Pass is Slow

**Symptoms**:
- `forward_pass` takes >50% of batch time
- GPU utilization high

**Solutions**:
```python
# Use mixed precision (FP16)
from torch.cuda.amp import autocast

with autocast():
    output = model(input)

# Reduce model size
# - Fewer filters
# - Fewer layers
# - Use depthwise separable convolutions

# Use gradient checkpointing (trades compute for memory)
from torch.utils.checkpoint import checkpoint

output = checkpoint(model, input)
```

### 4. Backward Pass is Slow

**Symptoms**:
- `backward_pass` takes >40% of batch time
- Memory usage spikes

**Solutions**:
```python
# Use gradient accumulation (smaller batches)
optimizer.zero_grad()
for i, batch in enumerate(mini_batches):
    loss = model(batch) / len(mini_batches)
    loss.backward()
optimizer.step()

# Clear unused gradients
torch.cuda.empty_cache()

# Use mixed precision backward
scaler = torch.cuda.amp.GradScaler()
with autocast():
    loss = model(input)
scaler.scale(loss).backward()
```

### 5. High Variance Between Batches

**Symptoms**:
- Some batches 2-3x slower than others
- Inconsistent training time

**Solutions**:
```python
# Check for variable-sized inputs
# - Use padding to uniform size
# - Sort batches by size

# Check for cache thrashing
# - Reduce cache size if memory constrained
# - Use pinned memory

# Check for data augmentation
# - Some augmentations are expensive (e.g., random transforms)
# - Pre-compute augmentations if possible
```

---

## Best Practices

### Development Phase

1. **Enable monitoring** for all experiments
2. **Profile different configurations** to find optimal settings
3. **Focus on the slowest operations** first
4. **Measure impact of changes** before/after optimization

```yaml
trainer_params:
  enable_memory_monitoring: true
  memory_monitor_batches: 10  # More verbose during development
```

### Production Phase

1. **Disable monitoring** (slight overhead)
2. **Use optimized settings** discovered during development
3. **Monitor at coarser granularity** (per-epoch, not per-batch)

```yaml
trainer_params:
  enable_memory_monitoring: false  # Disabled for production
```

### Research/Analysis Phase

1. **Use PerformanceMonitor** for detailed breakdowns
2. **Enable aggregation** for statistical analysis
3. **Export metrics** for visualization

```python
monitor = PerformanceMonitor()

# ... run experiments ...

# Export to pandas for analysis
import pandas as pd
metrics_df = pd.DataFrame([
    {
        'operation': m.operation,
        'time': m.duration_seconds,
        'cpu_mb': m.cpu_delta_mb,
        'gpu_mb': m.gpu_delta_mb
    }
    for m in monitor.get_metrics()
])

metrics_df.to_csv('performance_metrics.csv')
```

---

## Configuration Reference

### Synthetic Trainer
```yaml
trainer_params:
  enable_memory_monitoring: true   # Enable timing + memory
  memory_monitor_batches: 5        # Verbose for first N batches
  batch_size: 32                   # Affects timing significantly
  num_workers: 4                   # DataLoader parallelism
```

### Physical Trainer
```yaml
trainer_params:
  enable_memory_monitoring: true   # Enable timing + memory
  memory_monitor_batches: 5        # Verbose for first N iterations
  epochs: 50                       # Number of optimizer iterations
```

---

## API Reference

### PerformanceMonitor

```python
monitor = PerformanceMonitor(enabled=True, device=0)

with monitor.track("operation_name", aggregate=False):
    # ... code to profile ...
    pass

monitor.print_summary(show_aggregate=True)
monitor.get_total_time()  # Total time across all operations
monitor.get_operation_time("operation_name")  # Time for specific op
monitor.clear()  # Reset all metrics
```

### EpochPerformanceMonitor

```python
monitor = EpochPerformanceMonitor(
    enabled=True,
    verbose_batches=5,
    device=0
)

monitor.on_epoch_start()
monitor.on_batch_start(batch_idx)
monitor.on_batch_end(batch_idx, loss=None)
monitor.on_epoch_end()

stats = monitor.get_statistics()
# Returns: {
#     'total_epoch_time': float,
#     'avg_epoch_time': float,
#     'total_batch_time': float,
#     'avg_batch_time': float,
#     'min_batch_time': float,
#     'max_batch_time': float,
#     'num_batches': int,
#     'num_epochs': int
# }
```

### Context Managers & Decorators

```python
# Context manager
with track_performance("operation", print_result=True, device=0):
    # ... code ...
    pass

# Decorator
@monitor_performance("operation", device=0)
def my_function():
    # ... code ...
    pass
```

---

## Troubleshooting

### "My timing seems wrong"

- **Cause**: GPU operations are asynchronous
- **Solution**: Add `torch.cuda.synchronize()` before timing critical sections

```python
torch.cuda.synchronize()  # Wait for GPU operations to complete
start = time.perf_counter()
# ... operation ...
torch.cuda.synchronize()
end = time.perf_counter()
```

### "First batch is always slow"

- **Cause**: GPU initialization, JIT compilation, memory allocation
- **Solution**: This is normal - either exclude from timing or add warmup

### "Timing varies a lot between runs"

- **Causes**: 
  - System load
  - GPU state
  - Cache misses
  - Background processes
  
- **Solutions**:
  - Run multiple times and average
  - Close other applications
  - Use `torch.backends.cudnn.benchmark = False` for determinism

### "Memory doesn't match timing"

- **Cause**: Memory allocation is cached by PyTorch
- **Explanation**: High memory != slow performance (caching is good!)

---

## Examples

See `examples/performance_monitoring_examples.py` for runnable examples:

```bash
python examples\performance_monitoring_examples.py
```

Examples include:
1. Basic PerformanceMonitor usage
2. Aggregated operation tracking
3. Context manager usage
4. Function decorators
5. Training loop monitoring
6. Bottleneck identification

---

## Summary

**Performance monitoring helps you**:
- ✅ Identify bottlenecks (which operations are slow)
- ✅ Optimize effectively (focus on slowest parts)
- ✅ Validate improvements (measure before/after)
- ✅ Understand system behavior (memory + timing together)
- ✅ Make informed decisions (data-driven optimization)

**Key principle**: **Measure, don't guess!** Profile your code to find real bottlenecks, then optimize those specific areas.
