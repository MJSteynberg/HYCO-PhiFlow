"""
Performance Monitoring Examples

Demonstrates how to use timing and memory monitoring together to identify
performance bottlenecks in your code.
"""

import torch
import sys
from pathlib import Path
import time as time_module

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.memory_monitor import (
    PerformanceMonitor,
    track_performance,
    monitor_performance,
    EpochPerformanceMonitor,
)


def example_1_performance_monitor():
    """Example 1: Using PerformanceMonitor for multiple operations"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Performance Monitor")
    print("=" * 80 + "\n")

    monitor = PerformanceMonitor()

    # Track data loading
    with monitor.track("data_loading"):
        data = torch.randn(1000, 1000)
        time_module.sleep(0.1)  # Simulate I/O delay

    # Track GPU transfer
    if torch.cuda.is_available():
        with monitor.track("gpu_transfer"):
            data_gpu = data.to("cuda")

    # Track computation
    with monitor.track("matrix_multiply"):
        if torch.cuda.is_available():
            result = data_gpu @ data_gpu.T
        else:
            result = data @ data.T

    # Track cleanup
    with monitor.track("cleanup"):
        if torch.cuda.is_available():
            del data_gpu, result
            torch.cuda.empty_cache()
        else:
            del result

    # Print detailed summary
    monitor.print_summary()


def example_2_aggregated_operations():
    """Example 2: Tracking repeated operations with aggregation"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Aggregated Operations")
    print("=" * 80 + "\n")

    monitor = PerformanceMonitor()

    # Simulate training loop with repeated operations
    for epoch in range(3):
        with monitor.track(f"epoch_{epoch}"):
            for batch in range(5):
                # Data loading (aggregated)
                with monitor.track("batch_data_load", aggregate=True):
                    data = torch.randn(32, 64, 128, 128)
                    time_module.sleep(0.01)

                # Forward pass (aggregated)
                with monitor.track("batch_forward", aggregate=True):
                    if torch.cuda.is_available():
                        data = data.to("cuda")
                        output = data * 2
                    else:
                        output = data * 2
                    time_module.sleep(0.02)

                # Backward pass (aggregated)
                with monitor.track("batch_backward", aggregate=True):
                    loss = output.mean()
                    time_module.sleep(0.015)

    # Print summary with aggregation
    monitor.print_summary(show_aggregate=True)


def example_3_context_manager():
    """Example 3: Quick performance tracking with context manager"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Context Manager")
    print("=" * 80 + "\n")

    # Track individual operations
    with track_performance("model initialization"):
        model_weights = torch.randn(1000, 1000)
        if torch.cuda.is_available():
            model_weights = model_weights.to("cuda")
        time_module.sleep(0.05)

    with track_performance("forward pass"):
        if torch.cuda.is_available():
            output = model_weights @ model_weights.T
        time_module.sleep(0.03)

    with track_performance("cleanup"):
        if torch.cuda.is_available():
            del model_weights, output
            torch.cuda.empty_cache()


def example_4_decorator():
    """Example 4: Function-level performance monitoring"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Function Decorator")
    print("=" * 80 + "\n")

    @monitor_performance("data processing")
    def process_data(size=1000):
        data = torch.randn(size, size)
        if torch.cuda.is_available():
            data = data.to("cuda")
        result = data @ data.T
        time_module.sleep(0.05)
        if torch.cuda.is_available():
            del data, result
            torch.cuda.empty_cache()
        return "Done"

    @monitor_performance("validation")
    def validate_model():
        time_module.sleep(0.1)
        return {"accuracy": 0.95}

    # Call decorated functions
    process_data(500)
    validate_model()


def example_5_epoch_performance_monitor():
    """Example 5: Training loop with EpochPerformanceMonitor"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Epoch Performance Monitor")
    print("=" * 80 + "\n")

    if not torch.cuda.is_available():
        print("GPU not available, using CPU (slower)")

    monitor = EpochPerformanceMonitor(enabled=True, verbose_batches=3)

    # Simulate 2 epochs
    for epoch in range(2):
        print(f"\nEpoch {epoch + 1}")
        print("-" * 40)

        monitor.on_epoch_start()

        # Simulate batches with varying complexity
        for batch_idx in range(5):
            monitor.on_batch_start(batch_idx)

            # Simulate batch processing
            data = torch.randn(16, 3, 128, 128)
            if torch.cuda.is_available():
                data = data.to("cuda")

            # Variable time to show timing differences
            time_module.sleep(0.02 + batch_idx * 0.01)

            output = data * 2
            loss = output.mean().item()

            monitor.on_batch_end(batch_idx, loss)

            del data, output

        monitor.on_epoch_end()

    # Get statistics
    stats = monitor.get_statistics()
    print(f"\nStatistics:")
    print(f"  Average batch time: {stats.get('avg_batch_time', 0):.3f}s")
    print(f"  Total epoch time: {stats.get('total_epoch_time', 0):.3f}s")


def example_6_bottleneck_identification():
    """Example 6: Identifying performance bottlenecks"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Bottleneck Identification")
    print("=" * 80 + "\n")

    monitor = PerformanceMonitor()

    print("Simulating a pipeline with different operation speeds...")

    # Fast operation
    with monitor.track("fast_operation"):
        x = torch.randn(100, 100)
        y = x * 2
        time_module.sleep(0.01)

    # Medium operation
    with monitor.track("medium_operation"):
        if torch.cuda.is_available():
            x = x.to("cuda")
        result = x @ x.T
        time_module.sleep(0.05)

    # Slow operation (bottleneck!)
    with monitor.track("slow_operation_BOTTLENECK"):
        time_module.sleep(0.2)  # Simulating I/O or complex computation

    # Fast operation
    with monitor.track("fast_operation2"):
        if torch.cuda.is_available():
            result2 = result * 2
        time_module.sleep(0.01)

    monitor.print_summary()

    # Analyze bottlenecks
    total_time = monitor.get_total_time()
    print(f"\nBottleneck Analysis:")
    for metric in monitor.get_metrics():
        percentage = (metric.duration_seconds / total_time) * 100
        if percentage > 50:
            print(
                f"  ⚠️  {metric.operation} takes {percentage:.1f}% of total time - OPTIMIZE THIS!"
            )
        elif percentage > 20:
            print(
                f"  ⚡ {metric.operation} takes {percentage:.1f}% of total time - consider optimizing"
            )
        else:
            print(
                f"  ✓  {metric.operation} takes {percentage:.1f}% of total time - acceptable"
            )


def run_all_examples():
    """Run all examples"""
    examples = [
        example_1_performance_monitor,
        example_2_aggregated_operations,
        example_3_context_manager,
        example_4_decorator,
        example_5_epoch_performance_monitor,
        example_6_bottleneck_identification,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            import traceback

            traceback.print_exc()

        # Cleanup between examples
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    print("=" * 80)
    print("PERFORMANCE MONITORING EXAMPLES")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("\nWARNING: CUDA not available. Some examples will use CPU (slower).")

    # Run all examples
    run_all_examples()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print(
        "  1. Use PerformanceMonitor.track() for detailed operation-by-operation analysis"
    )
    print("  2. Enable 'aggregate=True' to track repeated operations")
    print("  3. Use EpochPerformanceMonitor for training loops")
    print("  4. Identify bottlenecks by comparing operation times")
    print("  5. Focus optimization efforts on the slowest operations")
