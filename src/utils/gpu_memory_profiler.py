"""
GPU Memory Profiler for Training

Monitors GPU memory during actual training runs.
This script provides instructions and tools to profile GPU memory usage.
"""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.memory_monitor import MemoryMonitor


def add_memory_monitoring_to_training():
    """
    Instructions for adding memory monitoring to actual training.

    This provides code snippets to add to the trainer for profiling.
    """
    print("=" * 80)
    print("GPU MEMORY PROFILING INSTRUCTIONS")
    print("=" * 80)

    print("\nTo profile GPU memory during training, add this code to your trainer:")
    print("\n1. In src/training/tensor_trainer.py, add at the top:")
    print("-" * 80)
    print(
        """
from src.utils.memory_monitor import MemoryMonitor
import torch
    """
    )

    print("\n2. In the _train_epoch method, add memory tracking:")
    print("-" * 80)
    print(
        """
def _train_epoch(self):
    '''Train for one epoch with memory monitoring.'''
    self.model.train()
    epoch_loss = 0.0
    
    # Print memory at epoch start
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        MemoryMonitor.print_memory_usage("Epoch start: ")
    
    for batch_idx, (initial_state, rollout_targets) in enumerate(self.train_loader):
        # Move to device
        initial_state = initial_state.to(self.device)
        rollout_targets = rollout_targets.to(self.device)
        
        # Print memory for first few batches
        if batch_idx < 5:
            MemoryMonitor.print_memory_usage(f"  Batch {batch_idx} after data to GPU: ")
        
        # Forward pass
        self.optimizer.zero_grad()
        predictions = self._rollout(initial_state, rollout_targets.shape[1])
        
        if batch_idx < 5:
            MemoryMonitor.print_memory_usage(f"  Batch {batch_idx} after forward: ")
        
        # Loss
        loss = self._compute_loss(predictions, rollout_targets)
        
        # Backward
        loss.backward()
        
        if batch_idx < 5:
            MemoryMonitor.print_memory_usage(f"  Batch {batch_idx} after backward: ")
        
        # Optimizer step
        self.optimizer.step()
        
        if batch_idx < 5:
            MemoryMonitor.print_memory_usage(f"  Batch {batch_idx} after optimizer: ")
            print(f"    Loss: {loss.item():.6f}")
        
        epoch_loss += loss.item()
    
    # Print peak memory
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        MemoryMonitor.print_memory_usage("Epoch end: ")
        print(f"  Peak GPU memory this epoch: {peak:.1f} MB")
    
    return epoch_loss / len(self.train_loader)
    """
    )

    print("\n3. Alternative: Quick profiler script")
    print("=" * 80)
    print("\nOr run training normally with CUDA memory profiling:")
    print(
        """
# Before training:
torch.cuda.memory._record_memory_history(enabled=True)

# After training:
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

# Then analyze with:
# python -m torch.cuda.memory viz trace memory_snapshot.pickle
    """
    )

    print("\n4. Quick memory check during training:")
    print("=" * 80)
    print(
        """
# Add this to your training loop to print memory every N batches:
if batch_idx % 10 == 0:
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"Batch {batch_idx}: GPU {allocated:.0f}MB allocated, {reserved:.0f}MB reserved")
    """
    )


def estimate_memory_requirements():
    """
    Estimate memory requirements for typical training scenario.
    """
    print("\n" + "=" * 80)
    print("MEMORY REQUIREMENT ESTIMATION")
    print("=" * 80)

    # Typical UNet configuration from burgers_experiment
    print("\nTypical UNet (4 levels, 64 filters):")
    print("-" * 80)

    # Rough parameter count for UNet
    # Level 0: 64 filters, ~100K params
    # Level 1: 128 filters, ~400K params
    # Level 2: 256 filters, ~1.6M params
    # Level 3: 512 filters, ~6.4M params
    # Total: ~8.5M parameters

    model_params = 8_500_000
    bytes_per_param = 4  # float32

    model_size_mb = model_params * bytes_per_param / (1024**2)
    gradients_mb = model_size_mb  # Same size as model
    optimizer_mb = model_size_mb * 2  # Adam keeps 2 states per parameter

    print(f"  Model parameters: {model_params:,}")
    print(f"  Model weights: {model_size_mb:.1f} MB")
    print(f"  Gradients: {gradients_mb:.1f} MB")
    print(f"  Optimizer state (Adam): {optimizer_mb:.1f} MB")
    print(
        f"  Total model+training overhead: {model_size_mb + gradients_mb + optimizer_mb:.1f} MB"
    )

    print("\nData/Activation Memory (depends on batch size):")
    print("-" * 80)

    # Burgers: 2 channels (velocity components), 128x128 resolution
    spatial_size = 128 * 128
    channels = 2
    batch_size = 4
    num_predict_steps = 4

    # Input batch
    input_mb = batch_size * channels * spatial_size * bytes_per_param / (1024**2)

    # Target batch (multiple timesteps)
    target_mb = (
        batch_size
        * num_predict_steps
        * channels
        * spatial_size
        * bytes_per_param
        / (1024**2)
    )

    # Activations (rough estimate: ~10x model size for deep networks)
    activations_mb = model_size_mb * 10

    print(f"  Input batch ({batch_size}x{channels}x{spatial_size}): {input_mb:.1f} MB")
    print(
        f"  Target batch ({batch_size}x{num_predict_steps}x{channels}x{spatial_size}): {target_mb:.1f} MB"
    )
    print(f"  Intermediate activations (estimate): {activations_mb:.1f} MB")
    print(f"  Total data/activation: {input_mb + target_mb + activations_mb:.1f} MB")

    print("\nTotal Estimated GPU Memory:")
    print("=" * 80)
    total_mb = (
        model_size_mb
        + gradients_mb
        + optimizer_mb
        + input_mb
        + target_mb
        + activations_mb
    )
    print(f"  {total_mb:.1f} MB ({total_mb/1024:.2f} GB)")

    print("\nPyTorch Memory Overhead:")
    print("-" * 80)
    print("  PyTorch caching allocator reserves extra memory for efficiency")
    print("  Typical overhead: +20-30% of allocated memory")
    total_with_overhead = total_mb * 1.25
    print(
        f"  Total with overhead: {total_with_overhead:.1f} MB ({total_with_overhead/1024:.2f} GB)"
    )

    print("\nRECOMMENDATIONS:")
    print("=" * 80)
    if total_with_overhead / 1024 > 8:
        print("  ⚠️  Estimated usage exceeds 8GB!")
        print("  Consider:")
        print("    - Reduce batch size")
        print("    - Use gradient accumulation")
        print("    - Use mixed precision training (FP16)")
        print("    - Reduce model size (fewer filters/levels)")
    elif total_with_overhead / 1024 > 6:
        print("  ⚠️  Estimated usage is high (6-8GB)")
        print("  Should work but close to limit. Monitor carefully.")
    else:
        print("  ✓ Estimated usage should be comfortable (<6GB)")


if __name__ == "__main__":
    add_memory_monitoring_to_training()
    estimate_memory_requirements()

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("\nTo actually profile GPU memory during training:")
    print("1. Modify src/training/tensor_trainer.py with the code above")
    print("2. Run: python run.py --config-name=burgers_experiment")
    print("3. Watch the memory output during training")
    print("\nOR")
    print("1. Run your training normally")
    print("2. Use nvidia-smi in another terminal to monitor GPU usage")
    print("3. Add simple memory prints to your training loop")
    print("=" * 80)
