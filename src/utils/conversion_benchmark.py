"""
Performance Benchmark for FieldTensorConverter

This module provides benchmarking utilities to measure the performance
of field-tensor conversions, which are critical for hybrid training efficiency.

Usage:
    python -m src.utils.conversion_benchmark
"""

import time
import torch
from typing import Dict, List, Tuple
import numpy as np
from phi.torch.flow import *
from phi import math
from phi.math import spatial, channel, batch as batch_dim
from phi.field import CenteredGrid, StaggeredGrid
from phi.geom import Box

from src.utils.field_conversion import (
    FieldTensorConverter,
    FieldMetadata
)


class ConversionBenchmark:
    """Benchmark utility for field-tensor conversions."""
    
    def __init__(self, resolutions: List[int], batch_sizes: List[int], num_runs: int = 100):
        """
        Initialize benchmark.
        
        Args:
            resolutions: List of spatial resolutions to test (e.g., [64, 128, 256])
            batch_sizes: List of batch sizes to test (e.g., [1, 4, 8])
            num_runs: Number of runs to average over
        """
        self.resolutions = resolutions
        self.batch_sizes = batch_sizes
        self.num_runs = num_runs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
    
    def _create_test_fields(
        self,
        resolution: int,
        batch_size: int,
        domain: Box
    ) -> Tuple[Dict[str, CenteredGrid], FieldTensorConverter]:
        """Create test fields and converter for benchmarking."""
        res = spatial(x=resolution, y=resolution)
        
        # Create metadata
        scalar_metadata = FieldMetadata(
            domain=domain,
            resolution=res,
            extrapolation=extrapolation.PERIODIC,
            field_type='centered',
            spatial_dims=('x', 'y'),
            channel_dims=()
        )
        
        vector_metadata = FieldMetadata(
            domain=domain,
            resolution=res,
            extrapolation=extrapolation.PERIODIC,
            field_type='centered',
            spatial_dims=('x', 'y'),
            channel_dims=('vector',)
        )
        
        # Create fields
        if batch_size > 1:
            density_data = torch.randn(batch_size, resolution, resolution).to(self.device)
            velocity_data = torch.randn(batch_size, resolution, resolution, 2).to(self.device)
            
            density_field = CenteredGrid(
                math.tensor(density_data, batch_dim('batch') & res),
                extrapolation.PERIODIC,
                bounds=domain
            )
            velocity_field = CenteredGrid(
                math.tensor(velocity_data, batch_dim('batch') & res & channel(vector='x,y')),
                extrapolation.PERIODIC,
                bounds=domain
            )
        else:
            density_data = torch.randn(resolution, resolution).to(self.device)
            velocity_data = torch.randn(resolution, resolution, 2).to(self.device)
            
            density_field = CenteredGrid(
                math.tensor(density_data, res),
                extrapolation.PERIODIC,
                bounds=domain
            )
            velocity_field = CenteredGrid(
                math.tensor(velocity_data, res & channel(vector='x,y')),
                extrapolation.PERIODIC,
                bounds=domain
            )
        
        fields = {
            'density': density_field,
            'velocity': velocity_field
        }
        
        converter = FieldTensorConverter({
            'density': scalar_metadata,
            'velocity': vector_metadata
        })
        
        return fields, converter
    
    def benchmark_fields_to_tensor(self, resolution: int, batch_size: int) -> float:
        """
        Benchmark fields -> tensor conversion.
        
        Returns:
            Average time in milliseconds
        """
        domain = Box(x=1, y=1)
        fields, converter = self._create_test_fields(resolution, batch_size, domain)
        
        # Warmup
        for _ in range(10):
            _ = converter.fields_to_tensors_batch(fields)
        
        # Synchronize if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.perf_counter()
        for _ in range(self.num_runs):
            _ = converter.fields_to_tensors_batch(fields)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        avg_time_ms = ((end_time - start_time) / self.num_runs) * 1000
        return avg_time_ms
    
    def benchmark_tensor_to_fields(self, resolution: int, batch_size: int) -> float:
        """
        Benchmark tensor -> fields conversion.
        
        Returns:
            Average time in milliseconds
        """
        domain = Box(x=1, y=1)
        fields, converter = self._create_test_fields(resolution, batch_size, domain)
        
        # Create tensor from fields
        tensor = converter.fields_to_tensors_batch(fields)
        
        # Warmup
        for _ in range(10):
            _ = converter.tensors_to_fields_batch(tensor)
        
        # Synchronize if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.perf_counter()
        for _ in range(self.num_runs):
            _ = converter.tensors_to_fields_batch(tensor)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        avg_time_ms = ((end_time - start_time) / self.num_runs) * 1000
        return avg_time_ms
    
    def benchmark_roundtrip(self, resolution: int, batch_size: int) -> float:
        """
        Benchmark full roundtrip conversion.
        
        Returns:
            Average time in milliseconds
        """
        domain = Box(x=1, y=1)
        fields, converter = self._create_test_fields(resolution, batch_size, domain)
        
        # Warmup
        for _ in range(10):
            tensor = converter.fields_to_tensors_batch(fields)
            _ = converter.tensors_to_fields_batch(tensor)
        
        # Synchronize if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.perf_counter()
        for _ in range(self.num_runs):
            tensor = converter.fields_to_tensors_batch(fields)
            _ = converter.tensors_to_fields_batch(tensor)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        avg_time_ms = ((end_time - start_time) / self.num_runs) * 1000
        return avg_time_ms
    
    def run_full_benchmark(self) -> Dict:
        """
        Run complete benchmark suite.
        
        Returns:
            Dictionary with benchmark results
        """
        print(f"\n{'='*80}")
        print(f"FieldTensorConverter Performance Benchmark")
        print(f"Device: {self.device}")
        print(f"Number of runs per test: {self.num_runs}")
        print(f"{'='*80}\n")
        
        results = {
            'device': str(self.device),
            'num_runs': self.num_runs,
            'tests': []
        }
        
        for resolution in self.resolutions:
            for batch_size in self.batch_sizes:
                print(f"Testing resolution={resolution}, batch_size={batch_size}")
                
                # Benchmark fields -> tensor
                fields_to_tensor_time = self.benchmark_fields_to_tensor(resolution, batch_size)
                print(f"  Fields → Tensor: {fields_to_tensor_time:.3f} ms")
                
                # Benchmark tensor -> fields
                tensor_to_fields_time = self.benchmark_tensor_to_fields(resolution, batch_size)
                print(f"  Tensor → Fields: {tensor_to_fields_time:.3f} ms")
                
                # Benchmark roundtrip
                roundtrip_time = self.benchmark_roundtrip(resolution, batch_size)
                print(f"  Roundtrip:       {roundtrip_time:.3f} ms")
                
                # Calculate throughput (elements per second)
                num_elements = resolution * resolution * 3 * batch_size  # 3 channels total
                throughput = (num_elements / roundtrip_time) * 1000 / 1e6  # Million elements/sec
                print(f"  Throughput:      {throughput:.2f} M elements/sec")
                print()
                
                results['tests'].append({
                    'resolution': resolution,
                    'batch_size': batch_size,
                    'fields_to_tensor_ms': fields_to_tensor_time,
                    'tensor_to_fields_ms': tensor_to_fields_time,
                    'roundtrip_ms': roundtrip_time,
                    'throughput_m_elems_per_sec': throughput
                })
        
        self.results = results
        return results
    
    def print_summary(self):
        """Print summary statistics."""
        if not self.results:
            print("No results available. Run benchmark first.")
            return
        
        print(f"\n{'='*80}")
        print("Summary Statistics")
        print(f"{'='*80}\n")
        
        # Find fastest and slowest
        tests = self.results['tests']
        
        fastest_roundtrip = min(tests, key=lambda x: x['roundtrip_ms'])
        slowest_roundtrip = max(tests, key=lambda x: x['roundtrip_ms'])
        
        highest_throughput = max(tests, key=lambda x: x['throughput_m_elems_per_sec'])
        
        print(f"Fastest roundtrip: {fastest_roundtrip['roundtrip_ms']:.3f} ms")
        print(f"  Resolution: {fastest_roundtrip['resolution']}, Batch: {fastest_roundtrip['batch_size']}")
        
        print(f"\nSlowest roundtrip: {slowest_roundtrip['roundtrip_ms']:.3f} ms")
        print(f"  Resolution: {slowest_roundtrip['resolution']}, Batch: {slowest_roundtrip['batch_size']}")
        
        print(f"\nHighest throughput: {highest_throughput['throughput_m_elems_per_sec']:.2f} M elements/sec")
        print(f"  Resolution: {highest_throughput['resolution']}, Batch: {highest_throughput['batch_size']}")
        
        # Average times
        avg_fields_to_tensor = np.mean([t['fields_to_tensor_ms'] for t in tests])
        avg_tensor_to_fields = np.mean([t['tensor_to_fields_ms'] for t in tests])
        avg_roundtrip = np.mean([t['roundtrip_ms'] for t in tests])
        
        print(f"\nAverage times across all tests:")
        print(f"  Fields → Tensor: {avg_fields_to_tensor:.3f} ms")
        print(f"  Tensor → Fields: {avg_tensor_to_fields:.3f} ms")
        print(f"  Roundtrip:       {avg_roundtrip:.3f} ms")
        print()
    
    def save_results(self, filepath: str):
        """Save results to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filepath}")


def main():
    """Run benchmark with default settings."""
    # Test configurations
    resolutions = [64, 128, 256]
    batch_sizes = [1, 4, 8, 16]
    num_runs = 100
    
    # Run benchmark
    benchmark = ConversionBenchmark(
        resolutions=resolutions,
        batch_sizes=batch_sizes,
        num_runs=num_runs
    )
    
    results = benchmark.run_full_benchmark()
    benchmark.print_summary()
    
    # Save results
    from pathlib import Path
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'benchmarks'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filepath = results_dir / f'conversion_benchmark_{timestamp}.json'
    benchmark.save_results(str(filepath))


if __name__ == '__main__':
    main()
