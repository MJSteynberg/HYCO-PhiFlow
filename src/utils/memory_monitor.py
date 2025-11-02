"""
Memory and Performance Monitoring Utilities

Provides tools for monitoring CPU and GPU memory usage, as well as execution time
during training. Useful for diagnosing memory issues and performance bottlenecks.

Includes decorators and context managers for non-intrusive monitoring.
"""

import torch
import psutil
import os
import time
from typing import Optional, Dict, Callable, Any, List
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import defaultdict

from .logger import get_logger

logger = get_logger(__name__)


class MemoryMonitor:
    """Monitor CPU and GPU memory usage during training."""

    @staticmethod
    def get_cpu_memory_mb() -> float:
        """
        Get current process CPU memory usage in MB.

        Returns:
            Memory usage in megabytes
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    @staticmethod
    def get_gpu_memory_mb(device: int = 0) -> float:
        """
        Get current GPU memory usage in MB.

        Args:
            device: GPU device index (default: 0)

        Returns:
            Memory usage in megabytes, or 0.0 if CUDA not available
        """
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(device) / 1024 / 1024
        return 0.0

    @staticmethod
    def get_gpu_memory_reserved_mb(device: int = 0) -> float:
        """
        Get reserved GPU memory (allocated by PyTorch cache) in MB.

        Args:
            device: GPU device index (default: 0)

        Returns:
            Reserved memory in megabytes, or 0.0 if CUDA not available
        """
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved(device) / 1024 / 1024
        return 0.0

    @staticmethod
    def get_memory_summary(device: int = 0) -> Dict[str, float]:
        """
        Get comprehensive memory usage summary.

        Args:
            device: GPU device index (default: 0)

        Returns:
            Dictionary with memory usage statistics in MB
        """
        summary = {
            "cpu_memory_mb": MemoryMonitor.get_cpu_memory_mb(),
            "gpu_memory_allocated_mb": MemoryMonitor.get_gpu_memory_mb(device),
            "gpu_memory_reserved_mb": MemoryMonitor.get_gpu_memory_reserved_mb(device),
        }

        if torch.cuda.is_available():
            summary["gpu_memory_free_mb"] = (
                torch.cuda.get_device_properties(device).total_memory / 1024 / 1024
                - summary["gpu_memory_reserved_mb"]
            )

        return summary

    @staticmethod
    def print_memory_usage(prefix: str = "", device: int = 0):
        """
        Print current memory usage to console.

        Args:
            prefix: Optional prefix string for the output
            device: GPU device index (default: 0)
        """
        cpu_mem = MemoryMonitor.get_cpu_memory_mb()
        gpu_mem = MemoryMonitor.get_gpu_memory_mb(device)
        gpu_reserved = MemoryMonitor.get_gpu_memory_reserved_mb(device)

        if torch.cuda.is_available():
            logger.info(
                f"{prefix}CPU: {cpu_mem:.1f} MB | GPU Allocated: {gpu_mem:.1f} MB | GPU Reserved: {gpu_reserved:.1f} MB"
            )
        else:
            logger.info(f"{prefix}CPU: {cpu_mem:.1f} MB | GPU: Not available")

    @staticmethod
    def log_memory_usage(logger, prefix: str = "", device: int = 0):
        """
        Log memory usage to a logger.

        Args:
            logger: Logger instance (e.g., from logging module)
            prefix: Optional prefix string for the log message
            device: GPU device index (default: 0)
        """
        cpu_mem = MemoryMonitor.get_cpu_memory_mb()
        gpu_mem = MemoryMonitor.get_gpu_memory_mb(device)
        gpu_reserved = MemoryMonitor.get_gpu_memory_reserved_mb(device)

        if torch.cuda.is_available():
            logger.info(
                f"{prefix}CPU: {cpu_mem:.1f} MB | GPU Allocated: {gpu_mem:.1f} MB | GPU Reserved: {gpu_reserved:.1f} MB"
            )
        else:
            logger.info(f"{prefix}CPU: {cpu_mem:.1f} MB | GPU: Not available")


class MemoryTracker:
    """
    Track memory usage over time for analysis.

    Usage:
        tracker = MemoryTracker()
        tracker.record("start")
        # ... do some work ...
        tracker.record("after_loading")
        # ... more work ...
        tracker.record("after_training")
        tracker.print_summary()
    """

    def __init__(self, device: int = 0):
        """
        Initialize memory tracker.

        Args:
            device: GPU device index to monitor (default: 0)
        """
        self.device = device
        self.records = []

    def record(self, label: str):
        """
        Record current memory usage with a label.

        Args:
            label: Label for this memory snapshot
        """
        summary = MemoryMonitor.get_memory_summary(self.device)
        summary["label"] = label
        self.records.append(summary)

    def print_summary(self):
        """Print summary of all recorded memory snapshots."""
        if not self.records:
            logger.info("No memory records available")
            return

        logger.info("\n" + "=" * 80)
        logger.info("Memory Usage Summary")
        logger.info("=" * 80)
        logger.info(
            f"{'Label':<30} {'CPU (MB)':<12} {'GPU Alloc (MB)':<15} {'GPU Reserved (MB)':<15}"
        )
        logger.info("-" * 80)

        for record in self.records:
            label = record["label"]
            cpu = record["cpu_memory_mb"]
            gpu_alloc = record["gpu_memory_allocated_mb"]
            gpu_reserved = record["gpu_memory_reserved_mb"]
            logger.info(
                f"{label:<30} {cpu:>10.1f}   {gpu_alloc:>13.1f}   {gpu_reserved:>17.1f}"
            )

        logger.info("=" * 80 + "\n")

    def get_peak_memory(self) -> Dict[str, float]:
        """
        Get peak memory usage across all records.

        Returns:
            Dictionary with peak CPU and GPU memory usage
        """
        if not self.records:
            return {"cpu_peak_mb": 0.0, "gpu_peak_mb": 0.0}

        cpu_peak = max(r["cpu_memory_mb"] for r in self.records)
        gpu_peak = max(r["gpu_memory_allocated_mb"] for r in self.records)

        return {"cpu_peak_mb": cpu_peak, "gpu_peak_mb": gpu_peak}

    def clear(self):
        """Clear all recorded memory snapshots."""
        self.records = []


# ============================================================================
# DECORATORS AND CONTEXT MANAGERS
# ============================================================================


@contextmanager
def track_memory(label: str = "operation", device: int = 0, print_output: bool = True):
    """
    Context manager to track memory usage for a code block.

    Usage:
        with track_memory("data loading"):
            data = load_large_dataset()

    Args:
        label: Label for this operation
        device: GPU device index
        print_output: Whether to print memory changes
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    start_cpu = MemoryMonitor.get_cpu_memory_mb()
    start_gpu = MemoryMonitor.get_gpu_memory_mb(device)

    try:
        yield
    finally:
        end_cpu = MemoryMonitor.get_cpu_memory_mb()
        end_gpu = MemoryMonitor.get_gpu_memory_mb(device)

        cpu_delta = end_cpu - start_cpu
        gpu_delta = end_gpu - start_gpu

        if print_output:
            logger.info(f"[Memory] {label}: CPU {cpu_delta:+.1f} MB, GPU {gpu_delta:+.1f} MB")
            if torch.cuda.is_available():
                peak_gpu = torch.cuda.max_memory_allocated(device) / 1024 / 1024
                logger.info(f"         Peak GPU: {peak_gpu:.1f} MB")


def monitor_memory(label: Optional[str] = None, device: int = 0, verbose: bool = True):
    """
    Decorator to monitor memory usage of a function.

    Usage:
        @monitor_memory("training epoch")
        def train_epoch(self):
            # training code
            pass

    Args:
        label: Label for this operation (defaults to function name)
        device: GPU device index
        verbose: Whether to print memory changes
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            func_label = label or func.__name__

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            start_cpu = MemoryMonitor.get_cpu_memory_mb()
            start_gpu = MemoryMonitor.get_gpu_memory_mb(device)

            result = func(*args, **kwargs)

            end_cpu = MemoryMonitor.get_cpu_memory_mb()
            end_gpu = MemoryMonitor.get_gpu_memory_mb(device)

            if verbose:
                cpu_delta = end_cpu - start_cpu
                gpu_delta = end_gpu - start_gpu
                logger.info(
                    f"[Memory] {func_label}: CPU {cpu_delta:+.1f} MB, GPU {gpu_delta:+.1f} MB"
                )
                if torch.cuda.is_available():
                    peak_gpu = torch.cuda.max_memory_allocated(device) / 1024 / 1024
                    logger.info(f"         Peak GPU: {peak_gpu:.1f} MB")

            return result

        return wrapper

    return decorator


class EpochMemoryMonitor:
    """
    Lightweight memory monitor for training loops.

    Usage in trainer:
        self.memory_monitor = EpochMemoryMonitor(enabled=True, verbose_batches=5)

        # In training loop:
        for epoch in range(epochs):
            self.memory_monitor.on_epoch_start()

            for batch_idx, batch in enumerate(dataloader):
                self.memory_monitor.on_batch_start(batch_idx)
                # ... training code ...
                self.memory_monitor.on_batch_end(batch_idx)

            self.memory_monitor.on_epoch_end()
    """

    def __init__(self, enabled: bool = True, verbose_batches: int = 5, device: int = 0):
        """
        Initialize epoch memory monitor.

        Args:
            enabled: Whether monitoring is active
            verbose_batches: Number of batches to print detailed info for
            device: GPU device index
        """
        self.enabled = enabled
        self.verbose_batches = verbose_batches
        self.device = device
        self._batch_count = 0

    def on_epoch_start(self):
        """Call at the start of each epoch."""
        if not self.enabled:
            return

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        self._batch_count = 0

    def on_batch_start(self, batch_idx: int):
        """Call at the start of each batch."""
        if not self.enabled or batch_idx >= self.verbose_batches:
            return

        # Store starting memory for this batch
        self._batch_start_gpu = MemoryMonitor.get_gpu_memory_mb(self.device)

    def on_batch_end(self, batch_idx: int, loss: Optional[float] = None):
        """Call at the end of each batch."""
        if not self.enabled or batch_idx >= self.verbose_batches:
            return

        gpu_mem = MemoryMonitor.get_gpu_memory_mb(self.device)
        cpu_mem = MemoryMonitor.get_cpu_memory_mb()

        loss_str = f", loss={loss:.6f}" if loss is not None else ""
        logger.info(
            f"  Batch {batch_idx}: GPU {gpu_mem:.1f} MB, CPU {cpu_mem:.1f} MB{loss_str}"
        )

        self._batch_count = batch_idx + 1

    def on_epoch_end(self):
        """Call at the end of each epoch."""
        if not self.enabled:
            return

        if torch.cuda.is_available():
            peak_gpu = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            current_gpu = MemoryMonitor.get_gpu_memory_mb(self.device)
            reserved_gpu = MemoryMonitor.get_gpu_memory_reserved_mb(self.device)

            logger.info(
                f"  Epoch end: GPU {current_gpu:.1f} MB allocated, "
                f"{reserved_gpu:.1f} MB reserved, peak {peak_gpu:.1f} MB"
            )


class BatchMemoryProfiler:
    """
    Detailed memory profiler for individual training steps.

    Usage:
        profiler = BatchMemoryProfiler(enabled=True)

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 5:  # Only profile first 5 batches
                profiler.disable()

            profiler.checkpoint("data_loaded")

            batch = batch.to(device)
            profiler.checkpoint("data_to_gpu")

            output = model(batch)
            profiler.checkpoint("forward")

            loss = criterion(output, target)
            loss.backward()
            profiler.checkpoint("backward")

            optimizer.step()
            profiler.checkpoint("optimizer")

            profiler.print_summary()
            profiler.reset()
    """

    def __init__(self, enabled: bool = True, device: int = 0):
        """
        Initialize batch memory profiler.

        Args:
            enabled: Whether profiling is active
            device: GPU device index
        """
        self.enabled = enabled
        self.device = device
        self.checkpoints = []
        self._last_label = "start"
        self._last_gpu = (
            0.0
            if not torch.cuda.is_available()
            else MemoryMonitor.get_gpu_memory_mb(device)
        )

    def checkpoint(self, label: str):
        """Record a checkpoint with current memory usage."""
        if not self.enabled:
            return

        gpu_mem = MemoryMonitor.get_gpu_memory_mb(self.device)
        gpu_delta = gpu_mem - self._last_gpu

        self.checkpoints.append(
            {
                "label": label,
                "gpu_mb": gpu_mem,
                "gpu_delta": gpu_delta,
                "from": self._last_label,
            }
        )

        self._last_label = label
        self._last_gpu = gpu_mem

    def print_summary(self):
        """Print summary of all checkpoints."""
        if not self.enabled or not self.checkpoints:
            return

        logger.info(f"    Memory profile:")
        for cp in self.checkpoints:
            delta_str = f"{cp['gpu_delta']:+.1f}" if cp["gpu_delta"] != 0 else " 0.0"
            logger.info(
                f"      {cp['from']} → {cp['label']}: "
                f"GPU {cp['gpu_mb']:.1f} MB ({delta_str} MB)"
            )

    def reset(self):
        """Reset checkpoints for next batch."""
        self.checkpoints = []
        self._last_label = "start"
        self._last_gpu = MemoryMonitor.get_gpu_memory_mb(self.device)

    def disable(self):
        """Disable profiling."""
        self.enabled = False

    def enable(self):
        """Enable profiling."""
        self.enabled = True


# ============================================================================
# PERFORMANCE MONITORING (Time + Memory)
# ============================================================================


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    operation: str
    duration_seconds: float
    cpu_start_mb: float
    cpu_end_mb: float
    cpu_delta_mb: float
    gpu_start_mb: float
    gpu_end_mb: float
    gpu_delta_mb: float
    gpu_peak_mb: float = 0.0
    call_count: int = 1

    def __str__(self):
        return (
            f"{self.operation}: {self.duration_seconds:.3f}s, "
            f"CPU {self.cpu_delta_mb:+.1f}MB, GPU {self.gpu_delta_mb:+.1f}MB (peak {self.gpu_peak_mb:.1f}MB)"
        )


class PerformanceMonitor:
    """
    Monitor both execution time and memory usage for operations.

    Usage:
        monitor = PerformanceMonitor()

        with monitor.track("data loading"):
            data = load_data()

        with monitor.track("forward pass"):
            output = model(data)

        monitor.print_summary()
    """

    def __init__(self, enabled: bool = True, device: int = 0):
        """
        Initialize performance monitor.

        Args:
            enabled: Whether monitoring is active
            device: GPU device index (use 0 for default GPU, ignored if CUDA unavailable)
        """
        self.enabled = enabled
        # Validate device parameter
        if device < 0:
            device = 0  # Default to device 0 if invalid
        if torch.cuda.is_available() and device >= torch.cuda.device_count():
            device = 0  # Default to device 0 if out of range
        self.device = device
        self.metrics: List[PerformanceMetrics] = []
        self.aggregate_metrics: Dict[str, List[float]] = defaultdict(list)

    @contextmanager
    def track(self, operation: str, aggregate: bool = False):
        """
        Context manager to track performance of an operation.

        Args:
            operation: Name of the operation being tracked
            aggregate: If True, accumulate stats for this operation name

        Usage:
            with monitor.track("data loading"):
                data = load_data()
        """
        if not self.enabled:
            yield
            return

        # Reset peak stats (initialize CUDA if needed)
        if torch.cuda.is_available():
            try:
                # This will initialize CUDA if not already initialized
                torch.cuda.set_device(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
            except RuntimeError:
                # Device might be invalid, skip reset
                pass

        # Start measurements
        start_time = time.perf_counter()
        cpu_start = MemoryMonitor.get_cpu_memory_mb()
        gpu_start = MemoryMonitor.get_gpu_memory_mb(self.device)

        try:
            yield
        finally:
            # End measurements
            end_time = time.perf_counter()
            cpu_end = MemoryMonitor.get_cpu_memory_mb()
            gpu_end = MemoryMonitor.get_gpu_memory_mb(self.device)

            duration = end_time - start_time
            cpu_delta = cpu_end - cpu_start
            gpu_delta = gpu_end - gpu_start

            gpu_peak = 0.0
            if torch.cuda.is_available():
                gpu_peak = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024

            metrics = PerformanceMetrics(
                operation=operation,
                duration_seconds=duration,
                cpu_start_mb=cpu_start,
                cpu_end_mb=cpu_end,
                cpu_delta_mb=cpu_delta,
                gpu_start_mb=gpu_start,
                gpu_end_mb=gpu_end,
                gpu_delta_mb=gpu_delta,
                gpu_peak_mb=gpu_peak,
            )

            self.metrics.append(metrics)

            if aggregate:
                self.aggregate_metrics[operation].append(duration)

    def print_summary(self, show_aggregate: bool = True):
        """
        Print summary of all tracked operations.

        Args:
            show_aggregate: Whether to show aggregated statistics
        """
        if not self.metrics:
            logger.info("No performance metrics recorded")
            return

        logger.info("\n" + "=" * 90)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 90)
        logger.info(
            f"{'Operation':<30} {'Time (s)':<12} {'CPU (MB)':<15} {'GPU (MB)':<15} {'GPU Peak':<12}"
        )
        logger.info("-" * 90)

        for m in self.metrics:
            cpu_str = f"{m.cpu_delta_mb:+.1f}".rjust(7)
            gpu_str = f"{m.gpu_delta_mb:+.1f}".rjust(7)
            logger.info(
                f"{m.operation:<30} {m.duration_seconds:>10.3f}  {cpu_str:>13}  {gpu_str:>13}  {m.gpu_peak_mb:>10.1f}"
            )

        # Total time
        total_time = sum(m.duration_seconds for m in self.metrics)
        logger.info("-" * 90)
        logger.info(f"{'TOTAL':<30} {total_time:>10.3f}s")

        # Aggregate statistics
        if show_aggregate and self.aggregate_metrics:
            logger.info("\n" + "=" * 90)
            logger.info("AGGREGATE STATISTICS (for repeated operations)")
            logger.info("=" * 90)
            logger.info(
                f"{'Operation':<30} {'Count':<8} {'Total (s)':<12} {'Mean (s)':<12} {'Min (s)':<12} {'Max (s)':<12}"
            )
            logger.info("-" * 90)

            for op_name, durations in self.aggregate_metrics.items():
                count = len(durations)
                total = sum(durations)
                mean = total / count
                min_dur = min(durations)
                max_dur = max(durations)

                logger.info(
                    f"{op_name:<30} {count:<8} {total:<12.3f} {mean:<12.3f} {min_dur:<12.3f} {max_dur:<12.3f}"
                )

        logger.info("=" * 90 + "\n")

    def get_metrics(self) -> List[PerformanceMetrics]:
        """Get all recorded metrics."""
        return self.metrics

    def get_total_time(self) -> float:
        """Get total time across all operations."""
        return sum(m.duration_seconds for m in self.metrics)

    def get_operation_time(self, operation: str) -> float:
        """Get total time for a specific operation."""
        return sum(m.duration_seconds for m in self.metrics if m.operation == operation)

    def clear(self):
        """Clear all metrics."""
        self.metrics = []
        self.aggregate_metrics = defaultdict(list)


@contextmanager
def track_performance(operation: str, print_result: bool = True, device: int = 0):
    """
    Standalone context manager for quick performance tracking.

    Usage:
        with track_performance("data loading"):
            data = load_data()

    Args:
        operation: Name of the operation
        print_result: Whether to print the result immediately
        device: GPU device index (use 0 for default GPU, ignored if CUDA unavailable)
    """
    # Validate device parameter
    if device < 0:
        device = 0
    if torch.cuda.is_available() and device >= torch.cuda.device_count():
        device = 0

    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        except RuntimeError:
            pass

    start_time = time.perf_counter()
    cpu_start = MemoryMonitor.get_cpu_memory_mb()
    gpu_start = MemoryMonitor.get_gpu_memory_mb(device)

    try:
        yield
    finally:
        end_time = time.perf_counter()
        cpu_end = MemoryMonitor.get_cpu_memory_mb()
        gpu_end = MemoryMonitor.get_gpu_memory_mb(device)

        duration = end_time - start_time
        cpu_delta = cpu_end - cpu_start
        gpu_delta = gpu_end - gpu_start

        if print_result:
            logger.info(
                f"[Performance] {operation}: {duration:.3f}s, CPU {cpu_delta:+.1f}MB, GPU {gpu_delta:+.1f}MB"
            )
            if torch.cuda.is_available():
                gpu_peak = torch.cuda.max_memory_allocated(device) / 1024 / 1024
                logger.info(f"              GPU peak: {gpu_peak:.1f}MB")


def monitor_performance(operation: Optional[str] = None, device: int = 0):
    """
    Decorator for monitoring function performance.

    Usage:
        @monitor_performance("model training")
        def train_model():
            # training code
            pass

    Args:
        operation: Name of the operation (defaults to function name)
        device: GPU device index (use 0 for default GPU, ignored if CUDA unavailable)
    """
    # Validate device parameter
    validated_device = device
    if validated_device < 0:
        validated_device = 0
    if torch.cuda.is_available() and validated_device >= torch.cuda.device_count():
        validated_device = 0

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            op_name = operation or func.__name__

            if torch.cuda.is_available():
                try:
                    torch.cuda.set_device(validated_device)
                    torch.cuda.reset_peak_memory_stats(validated_device)
                except RuntimeError:
                    pass

            start_time = time.perf_counter()
            cpu_start = MemoryMonitor.get_cpu_memory_mb()
            gpu_start = MemoryMonitor.get_gpu_memory_mb(validated_device)

            result = func(*args, **kwargs)

            end_time = time.perf_counter()
            cpu_end = MemoryMonitor.get_cpu_memory_mb()
            gpu_end = MemoryMonitor.get_gpu_memory_mb(validated_device)

            duration = end_time - start_time
            cpu_delta = cpu_end - cpu_start
            gpu_delta = gpu_end - gpu_start

            logger.info(
                f"[Performance] {op_name}: {duration:.3f}s, CPU {cpu_delta:+.1f}MB, GPU {gpu_delta:+.1f}MB"
            )
            if torch.cuda.is_available():
                gpu_peak = torch.cuda.max_memory_allocated(device) / 1024 / 1024
                logger.info(f"              GPU peak: {gpu_peak:.1f}MB")

            return result

        return wrapper

    return decorator


class EpochPerformanceMonitor:
    """
    Combined time and memory monitor for training epochs.

    Extends EpochMemoryMonitor with timing capabilities.

    Usage:
        monitor = EpochPerformanceMonitor(enabled=True)

        monitor.on_epoch_start()
        for batch_idx, batch in enumerate(dataloader):
            monitor.on_batch_start(batch_idx)
            # ... training ...
            monitor.on_batch_end(batch_idx, loss)
        monitor.on_epoch_end()
    """

    def __init__(self, enabled: bool = True, verbose_batches: int = 5, device: int = 0):
        """
        Initialize epoch performance monitor.

        Args:
            enabled: Whether monitoring is active
            verbose_batches: Number of batches to print details for
            device: GPU device index (use 0 for default GPU, ignored if CUDA unavailable)
        """
        self.enabled = enabled
        self.verbose_batches = verbose_batches
        # Validate device parameter
        if device < 0:
            device = 0
        if torch.cuda.is_available() and device >= torch.cuda.device_count():
            device = 0
        self.device = device
        self._batch_count = 0
        self._epoch_start_time = 0.0
        self._batch_start_time = 0.0
        self._batch_times: List[float] = []
        self._epoch_times: List[float] = []

    def on_epoch_start(self):
        """Call at the start of each epoch."""
        if not self.enabled:
            return

        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
            except RuntimeError:
                pass

        self._epoch_start_time = time.perf_counter()
        self._batch_count = 0
        self._batch_times = []

    def on_batch_start(self, batch_idx: int):
        """Call at the start of each batch."""
        if not self.enabled:
            return

        self._batch_start_time = time.perf_counter()

    def on_batch_end(self, batch_idx: int, loss: Optional[float] = None):
        """Call at the end of each batch."""
        if not self.enabled:
            return

        batch_time = time.perf_counter() - self._batch_start_time
        self._batch_times.append(batch_time)

        if batch_idx < self.verbose_batches:
            gpu_mem = MemoryMonitor.get_gpu_memory_mb(self.device)
            cpu_mem = MemoryMonitor.get_cpu_memory_mb()

            loss_str = f", loss={loss:.6f}" if loss is not None else ""
            logger.info(
                f"  Batch {batch_idx}: {batch_time:.3f}s, GPU {gpu_mem:.1f}MB, CPU {cpu_mem:.1f}MB{loss_str}"
            )

        self._batch_count = batch_idx + 1

    def on_epoch_end(self):
        """Call at the end of each epoch."""
        if not self.enabled:
            return

        epoch_time = time.perf_counter() - self._epoch_start_time
        self._epoch_times.append(epoch_time)

        if torch.cuda.is_available():
            peak_gpu = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            current_gpu = MemoryMonitor.get_gpu_memory_mb(self.device)
            reserved_gpu = MemoryMonitor.get_gpu_memory_reserved_mb(self.device)

            logger.info(
                f"  Epoch end: {epoch_time:.3f}s total, GPU {current_gpu:.1f}MB allocated, "
                f"{reserved_gpu:.1f}MB reserved, peak {peak_gpu:.1f}MB"
            )
        else:
            logger.info(f"  Epoch end: {epoch_time:.3f}s total")

        # Batch statistics
        if self._batch_times:
            avg_batch = sum(self._batch_times) / len(self._batch_times)
            min_batch = min(self._batch_times)
            max_batch = max(self._batch_times)
            logger.info(
                f"  Batch timing: avg={avg_batch:.3f}s, min={min_batch:.3f}s, max={max_batch:.3f}s"
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get timing statistics."""
        if not self._batch_times:
            return {}

        return {
            "total_epoch_time": sum(self._epoch_times) if self._epoch_times else 0.0,
            "avg_epoch_time": (
                sum(self._epoch_times) / len(self._epoch_times)
                if self._epoch_times
                else 0.0
            ),
            "total_batch_time": sum(self._batch_times),
            "avg_batch_time": sum(self._batch_times) / len(self._batch_times),
            "min_batch_time": min(self._batch_times),
            "max_batch_time": max(self._batch_times),
            "num_batches": len(self._batch_times),
            "num_epochs": len(self._epoch_times),
        }
