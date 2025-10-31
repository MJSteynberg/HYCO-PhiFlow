# HYCO-PhiFlow Refactoring Plan

**Date:** October 31, 2025  
**Author:** Development Team  
**Version:** 1.0

---

## Executive Summary

This document outlines a comprehensive refactoring plan for the HYCO-PhiFlow codebase with three primary objectives:

1. **Enhanced Dataset Validation** - Implement robust validation for dataset generation, caching, and reuse
2. **Hydra Configuration Management** - Migrate from manual YAML parsing to Hydra for structured configuration
3. **Improved Modularity** - Refactor components for better abstraction, testability, and maintainability

The plan is designed to be implemented incrementally without breaking existing functionality, allowing for continuous development and testing throughout the migration process.

---

## Table of Contents

1. [Current Architecture Analysis](#1-current-architecture-analysis)
2. [Proposed Changes Summary](#2-proposed-changes-summary)
3. [Detailed Implementation Plan](#3-detailed-implementation-plan)
4. [Migration Strategy](#4-migration-strategy)
5. [Testing Strategy](#5-testing-strategy)
6. [Rollback Plan](#6-rollback-plan)

---

## 1. Current Architecture Analysis

### 1.1 Current Strengths

- ✅ **Well-structured data pipeline** with `DataManager` and `HybridDataset`
- ✅ **Separation of concerns** between physical and synthetic models
- ✅ **Comprehensive testing** suite for data management
- ✅ **Efficient caching** system reducing redundant conversions
- ✅ **Unified configuration** schema supporting multiple workflows

### 1.2 Current Pain Points

#### 1.2.1 Dataset Validation Issues

**Problem:** Limited validation logic for determining when to regenerate vs. reuse data

**Current Behavior:**
```python
# DataManager.is_cached() - Basic validation
def is_cached(self, sim_index, field_names=None, num_frames=None):
    # Only checks: file exists, field names match, num_frames sufficient
    # Missing: PDE parameters, resolution, domain, generation params
```

**Issues:**
- No validation of PDE parameters (nu, buoyancy, etc.)
- No check for resolution/domain changes
- No verification of generation parameters (dt, save_interval)
- Cache can become stale without detection
- Users must manually delete cache when parameters change

#### 1.2.2 Configuration Management Issues

**Problem:** Manual YAML parsing with limited validation and no type safety

**Current Behavior:**
```python
# run.py
with open(args.config, 'r') as f:
    config: Dict[str, Any] = yaml.safe_load(f)

# Direct dictionary access throughout codebase
config['data']['dset_name']
config['model']['physical']['pde_params']['nu']
```

**Issues:**
- No validation until runtime (errors occur deep in execution)
- No type hints or IDE autocomplete for config values
- Difficult to track which config keys are required vs. optional
- No structured defaults or schema enforcement
- Hard to compose configs (e.g., base config + experiment overrides)
- No command-line override capability

#### 1.2.3 Modularity Issues

**Problem:** Tight coupling between components and limited abstraction

**Current Issues:**
1. **Hard-coded model creation:**
   ```python
   # In generator.py, trainer.py, evaluator.py
   if model_name == 'UNet':
       model = UNet(config)
   elif model_name == 'SomethingElse':
       model = SomethingElse(config)
   ```

2. **Config passed everywhere as raw dict:**
   ```python
   def __init__(self, config: Dict[str, Any]):
       self.data_config = config['data']
       self.model_config = config['model']['synthetic']
       # Manual extraction everywhere
   ```

3. **Limited interface contracts:**
   - No formal interface for trainers
   - Inconsistent method signatures
   - Hard to extend with new model types

4. **Duplication across trainers:**
   - Both `SyntheticTrainer` and `PhysicalTrainer` have similar data loading logic
   - Model creation logic duplicated
   - Checkpoint handling duplicated

---

## 2. Proposed Changes Summary

### 2.1 Enhanced Dataset Validation

**Goal:** Implement comprehensive validation to automatically detect when cache is invalid

**Key Features:**
1. **Cache Metadata Versioning**
   - Store complete generation context with cached data
   - Detect parameter mismatches automatically
   - Support backward compatibility

2. **Validation Chain**
   - Field names and counts
   - PDE parameters
   - Domain and resolution
   - Generation parameters (dt, timesteps)
   - PhiFlow version compatibility

3. **Smart Cache Management**
   - Automatic invalidation on parameter changes
   - Optional cache clearing utilities
   - Cache statistics and reporting

### 2.2 Hydra Configuration Management

**Goal:** Migrate to Hydra for structured, type-safe configuration

**Key Features:**
1. **Structured Configs (Dataclasses)**
   - Type-safe configuration objects
   - IDE autocomplete support
   - Validation at startup

2. **Composition and Inheritance**
   - Base configs + experiment configs
   - Mix-and-match model/dataset/trainer configs
   - Easy experimentation

3. **Command-line Overrides**
   - Override any parameter from CLI
   - No need to edit YAML files
   - Better for hyperparameter sweeps

4. **Configuration Groups**
   - `conf/model/physical/` - Physical model variants
   - `conf/model/synthetic/` - Synthetic model variants
   - `conf/data/` - Dataset configurations
   - `conf/trainer/` - Training configurations

### 2.3 Improved Modularity

**Goal:** Better abstraction, reduced coupling, increased extensibility

**Key Features:**
1. **Factory Pattern for Model Creation**
   - Centralized model registry
   - Automatic model discovery
   - Easy to add new models

2. **Base Trainer Abstract Class**
   - Shared functionality in base class
   - Consistent interface
   - Reduced code duplication

3. **Configuration Objects**
   - Structured config classes instead of dicts
   - Better type safety
   - Clearer dependencies

4. **Plugin Architecture**
   - Easily extensible for new model types
   - Decoupled components
   - Better testability

---

## 3. Detailed Implementation Plan

### Phase 1: Enhanced Dataset Validation (Week 1-2)

#### Step 1.1: Extend Cache Metadata Structure

**Location:** `src/data/data_manager.py`

**Changes:**
```python
# Add to DataManager.load_and_cache_simulation()

# Enhanced metadata structure
cache_data = {
    'tensor_data': tensor_data,
    'metadata': {
        'version': '2.0',  # Cache format version
        'created_at': datetime.now().isoformat(),
        'phiflow_version': phi.__version__,
        
        # Existing metadata
        'scene_metadata': scene_metadata,
        'field_metadata': field_metadata,
        'num_frames': len(frames_to_load),
        'frame_indices': frames_to_load,
        
        # NEW: Generation parameters
        'generation_params': {
            'pde_name': config['model']['physical']['name'],
            'pde_params': config['model']['physical']['pde_params'],
            'domain': config['model']['physical']['domain'],
            'resolution': config['model']['physical']['resolution'],
            'dt': config['model']['physical']['dt'],
        },
        
        # NEW: Data configuration
        'data_config': {
            'fields': field_names,
            'fields_scheme': config['data'].get('fields_scheme', 'unknown'),
        },
        
        # NEW: Checksums for validation
        'checksums': {
            'pde_params_hash': _compute_hash(config['model']['physical']['pde_params']),
            'resolution_hash': _compute_hash(config['model']['physical']['resolution']),
        }
    }
}
```

**Implementation:**
1. Create new file `src/data/validation.py`:
   ```python
   from typing import Dict, Any, List, Optional
   import hashlib
   import json
   
   class CacheValidator:
       """Validates cache against current configuration."""
       
       def __init__(self, config: Dict[str, Any]):
           self.config = config
       
       def validate_cache(
           self, 
           cached_metadata: Dict[str, Any],
           field_names: List[str],
           num_frames: int
       ) -> tuple[bool, List[str]]:
           """
           Validate cache against requirements.
           
           Returns:
               (is_valid, reasons_if_invalid)
           """
           reasons = []
           
           # Check version compatibility
           cache_version = cached_metadata.get('version', '1.0')
           if not self._is_version_compatible(cache_version):
               reasons.append(f"Cache version {cache_version} incompatible")
           
           # Check field names
           cached_fields = set(cached_metadata['data_config']['fields'])
           requested_fields = set(field_names)
           if cached_fields != requested_fields:
               reasons.append(
                   f"Field mismatch: cached={cached_fields}, "
                   f"requested={requested_fields}"
               )
           
           # Check num_frames
           cached_frames = cached_metadata['num_frames']
           if cached_frames < num_frames:
               reasons.append(
                   f"Insufficient frames: cached={cached_frames}, "
                   f"requested={num_frames}"
               )
           
           # Check PDE parameters
           if not self._validate_pde_params(cached_metadata):
               reasons.append("PDE parameter mismatch")
           
           # Check resolution
           if not self._validate_resolution(cached_metadata):
               reasons.append("Resolution mismatch")
           
           # Check domain
           if not self._validate_domain(cached_metadata):
               reasons.append("Domain mismatch")
           
           return (len(reasons) == 0, reasons)
       
       def _validate_pde_params(self, cached_metadata: Dict) -> bool:
           """Check if PDE parameters match."""
           cached_hash = cached_metadata.get('checksums', {}).get('pde_params_hash')
           current_hash = compute_hash(
               self.config['model']['physical']['pde_params']
           )
           return cached_hash == current_hash
       
       def _validate_resolution(self, cached_metadata: Dict) -> bool:
           """Check if resolution matches."""
           cached_res = cached_metadata['generation_params']['resolution']
           current_res = self.config['model']['physical']['resolution']
           return cached_res == current_res
       
       def _validate_domain(self, cached_metadata: Dict) -> bool:
           """Check if domain matches."""
           cached_domain = cached_metadata['generation_params']['domain']
           current_domain = self.config['model']['physical']['domain']
           return cached_domain == current_domain
       
       def _is_version_compatible(self, cache_version: str) -> bool:
           """Check version compatibility."""
           major_version = cache_version.split('.')[0]
           return major_version == '2'
   
   
   def compute_hash(obj: Any) -> str:
       """Compute stable hash of an object."""
       json_str = json.dumps(obj, sort_keys=True)
       return hashlib.sha256(json_str.encode()).hexdigest()
   ```

2. Update `DataManager.is_cached()`:
   ```python
   def is_cached(
       self, 
       sim_index: int,
       field_names: Optional[List[str]] = None,
       num_frames: Optional[int] = None,
       validate_params: bool = True  # NEW parameter
   ) -> bool:
       """
       Check if simulation is cached with optional parameter validation.
       
       Args:
           validate_params: If True, validates PDE params, resolution, domain
       """
       cache_path = self.get_cached_path(sim_index)
       
       if not cache_path.exists():
           return False
       
       # Basic existence check
       if field_names is None and num_frames is None and not validate_params:
           return True
       
       try:
           cached_data = torch.load(cache_path, weights_only=False)
           metadata = cached_data.get('metadata', {})
           
           # Use validator for comprehensive check
           if validate_params:
               validator = CacheValidator(self.config)
               is_valid, reasons = validator.validate_cache(
                   metadata, field_names or [], num_frames or 0
               )
               
               if not is_valid:
                   print(f"Cache invalid for sim {sim_index}:")
                   for reason in reasons:
                       print(f"  - {reason}")
                   return False
           else:
               # Legacy validation (backward compatible)
               if field_names is not None:
                   cached_fields = set(cached_data['tensor_data'].keys())
                   requested_fields = set(field_names)
                   if cached_fields != requested_fields:
                       return False
               
               if num_frames is not None:
                   cached_num_frames = metadata.get('num_frames', 0)
                   if cached_num_frames < num_frames:
                       return False
           
           return True
           
       except Exception as e:
           print(f"Error validating cache: {e}")
           return False
   ```

**Testing:**
1. Add tests in `tests/data/test_cache_validation.py`:
   ```python
   def test_cache_invalidated_on_pde_param_change():
       """Test that changing PDE parameters invalidates cache."""
       
   def test_cache_invalidated_on_resolution_change():
       """Test that changing resolution invalidates cache."""
   
   def test_cache_valid_with_same_params():
       """Test that cache is valid when params match."""
   
   def test_backward_compatibility_with_old_cache():
       """Test that old cache format still works."""
   ```

**Migration:**
- Old cache files without enhanced metadata will still work
- New cache files will have enhanced metadata
- Gradual migration as cache is regenerated

---

#### Step 1.2: Add Cache Management Utilities

**Location:** `src/data/cache_manager.py` (NEW FILE)

**Implementation:**
```python
"""
Utilities for cache management and inspection.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from datetime import datetime
import json
from rich.console import Console
from rich.table import Table


class CacheManager:
    """Manages cache inspection, statistics, and cleanup."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.console = Console()
    
    def list_cached_datasets(self) -> List[str]:
        """List all cached datasets."""
        if not self.cache_dir.exists():
            return []
        return [d.name for d in self.cache_dir.iterdir() if d.is_dir()]
    
    def get_cache_stats(self, dataset_name: str) -> Dict[str, Any]:
        """Get statistics for a cached dataset."""
        dataset_dir = self.cache_dir / dataset_name
        
        if not dataset_dir.exists():
            return {'exists': False}
        
        cache_files = list(dataset_dir.glob('sim_*.pt'))
        
        total_size = sum(f.stat().st_size for f in cache_files)
        
        # Sample first file for metadata
        metadata = None
        if cache_files:
            try:
                sample = torch.load(cache_files[0], weights_only=False)
                metadata = sample.get('metadata', {})
            except:
                pass
        
        return {
            'exists': True,
            'num_simulations': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_version': metadata.get('version', 'unknown') if metadata else 'unknown',
            'created_at': metadata.get('created_at', 'unknown') if metadata else 'unknown',
        }
    
    def print_cache_stats(self, dataset_name: Optional[str] = None):
        """Print cache statistics in a formatted table."""
        datasets = [dataset_name] if dataset_name else self.list_cached_datasets()
        
        table = Table(title="Cache Statistics")
        table.add_column("Dataset", style="cyan")
        table.add_column("Simulations", justify="right")
        table.add_column("Size (MB)", justify="right")
        table.add_column("Version", style="yellow")
        table.add_column("Created", style="green")
        
        for ds in datasets:
            stats = self.get_cache_stats(ds)
            if stats['exists']:
                table.add_row(
                    ds,
                    str(stats['num_simulations']),
                    f"{stats['total_size_mb']:.2f}",
                    stats['cache_version'],
                    stats['created_at']
                )
        
        self.console.print(table)
    
    def clear_cache(
        self, 
        dataset_name: str, 
        sim_indices: Optional[List[int]] = None,
        confirm: bool = True
    ):
        """
        Clear cache for dataset or specific simulations.
        
        Args:
            dataset_name: Name of dataset
            sim_indices: If None, clear all; otherwise clear specific sims
            confirm: If True, ask for confirmation
        """
        dataset_dir = self.cache_dir / dataset_name
        
        if not dataset_dir.exists():
            self.console.print(f"[yellow]No cache found for {dataset_name}[/yellow]")
            return
        
        if sim_indices is None:
            # Clear all
            files = list(dataset_dir.glob('sim_*.pt'))
            message = f"Delete all {len(files)} cached simulations for {dataset_name}?"
        else:
            # Clear specific
            files = [dataset_dir / f"sim_{i:06d}.pt" for i in sim_indices]
            files = [f for f in files if f.exists()]
            message = f"Delete {len(files)} cached simulations?"
        
        if confirm:
            response = input(f"{message} [y/N]: ")
            if response.lower() != 'y':
                self.console.print("[yellow]Cancelled[/yellow]")
                return
        
        for f in files:
            f.unlink()
        
        self.console.print(f"[green]Deleted {len(files)} cache files[/green]")
    
    def validate_cache(self, dataset_name: str, config: Dict[str, Any]) -> Dict[int, List[str]]:
        """
        Validate all cached simulations against current config.
        
        Returns:
            Dict mapping sim_index to list of validation errors
        """
        from src.data.validation import CacheValidator
        
        dataset_dir = self.cache_dir / dataset_name
        cache_files = list(dataset_dir.glob('sim_*.pt'))
        
        validator = CacheValidator(config)
        invalid_sims = {}
        
        for cache_file in cache_files:
            sim_idx = int(cache_file.stem.split('_')[1])
            
            try:
                cached_data = torch.load(cache_file, weights_only=False)
                metadata = cached_data.get('metadata', {})
                
                field_names = config['data']['fields']
                num_frames = config.get('generation_params', {}).get('total_steps', 50)
                
                is_valid, reasons = validator.validate_cache(
                    metadata, field_names, num_frames
                )
                
                if not is_valid:
                    invalid_sims[sim_idx] = reasons
                    
            except Exception as e:
                invalid_sims[sim_idx] = [f"Error loading cache: {e}"]
        
        return invalid_sims
    
    def print_validation_report(self, dataset_name: str, config: Dict[str, Any]):
        """Print validation report for a dataset."""
        invalid_sims = self.validate_cache(dataset_name, config)
        
        if not invalid_sims:
            self.console.print(f"[green]✓ All cache valid for {dataset_name}[/green]")
        else:
            self.console.print(f"[red]✗ Found {len(invalid_sims)} invalid cache files:[/red]")
            for sim_idx, reasons in invalid_sims.items():
                self.console.print(f"\n  Simulation {sim_idx}:")
                for reason in reasons:
                    self.console.print(f"    - {reason}")
            
            self.console.print(f"\n[yellow]Run with --clear-invalid-cache to remove invalid files[/yellow]")
```

**Add CLI utility:**

Create `scripts/manage_cache.py`:
```python
#!/usr/bin/env python
"""
Cache management utility.

Usage:
    python scripts/manage_cache.py list
    python scripts/manage_cache.py stats [dataset_name]
    python scripts/manage_cache.py validate <dataset_name> --config <config.yaml>
    python scripts/manage_cache.py clear <dataset_name> [--sims 0,1,2]
"""

import argparse
import yaml
from src.data.cache_manager import CacheManager


def main():
    parser = argparse.ArgumentParser(description="Cache management utility")
    parser.add_argument('command', choices=['list', 'stats', 'validate', 'clear'])
    parser.add_argument('dataset', nargs='?', help='Dataset name')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--sims', help='Comma-separated simulation indices')
    parser.add_argument('--cache-dir', default='data/cache', help='Cache directory')
    parser.add_argument('-y', '--yes', action='store_true', help='Skip confirmation')
    
    args = parser.parse_args()
    
    manager = CacheManager(args.cache_dir)
    
    if args.command == 'list':
        datasets = manager.list_cached_datasets()
        print("Cached datasets:")
        for ds in datasets:
            print(f"  - {ds}")
    
    elif args.command == 'stats':
        manager.print_cache_stats(args.dataset)
    
    elif args.command == 'validate':
        if not args.config:
            print("Error: --config required for validate")
            return
        
        with open(args.config) as f:
            config = yaml.safe_load(f)
        
        manager.print_validation_report(args.dataset, config)
    
    elif args.command == 'clear':
        sim_indices = None
        if args.sims:
            sim_indices = [int(s) for s in args.sims.split(',')]
        
        manager.clear_cache(args.dataset, sim_indices, confirm=not args.yes)


if __name__ == '__main__':
    main()
```

---

### Phase 2: Hydra Configuration Management (Week 3-4)

#### Step 2.1: Install and Setup Hydra

**Installation:**
```bash
pip install hydra-core omegaconf
```

**Add to requirements.txt:**
```
hydra-core>=1.3.0
omegaconf>=2.3.0
```

#### Step 2.2: Create Structured Config Dataclasses

**Location:** `src/config/` (NEW DIRECTORY)

Create `src/config/__init__.py`:
```python
"""Structured configuration using Hydra dataclasses."""

from .data_config import DataConfig
from .model_config import PhysicalModelConfig, SyntheticModelConfig
from .trainer_config import SyntheticTrainerConfig, PhysicalTrainerConfig
from .generation_config import GenerationConfig
from .evaluation_config import EvaluationConfig
from .experiment_config import ExperimentConfig

__all__ = [
    'DataConfig',
    'PhysicalModelConfig',
    'SyntheticModelConfig',
    'SyntheticTrainerConfig',
    'PhysicalTrainerConfig',
    'GenerationConfig',
    'EvaluationConfig',
    'ExperimentConfig',
]
```

Create `src/config/data_config.py`:
```python
from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import MISSING


@dataclass
class DataConfig:
    """Configuration for dataset."""
    
    data_dir: str = 'data/'
    dset_name: str = MISSING  # Required
    fields: List[str] = field(default_factory=list)  # Required
    fields_scheme: str = 'unknown'
    cache_dir: str = 'data/cache'
    
    # Validation options
    validate_cache: bool = True
    auto_clear_invalid: bool = False
```

Create `src/config/model_config.py`:
```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from omegaconf import MISSING


@dataclass
class DomainConfig:
    """Physical domain configuration."""
    size_x: float = 100.0
    size_y: float = 100.0
    size_z: Optional[float] = None


@dataclass
class ResolutionConfig:
    """Grid resolution configuration."""
    x: int = MISSING
    y: int = MISSING
    z: Optional[int] = None


@dataclass
class PhysicalModelConfig:
    """Configuration for physical PDE models."""
    
    name: str = MISSING  # e.g., 'BurgersModel', 'SmokeModel'
    domain: DomainConfig = field(default_factory=DomainConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    dt: float = 0.8
    pde_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchitectureConfig:
    """Neural network architecture parameters."""
    levels: int = 4
    filters: int = 64
    batch_norm: bool = True


@dataclass
class SyntheticModelConfig:
    """Configuration for synthetic (neural network) models."""
    
    name: str = MISSING  # e.g., 'UNet'
    model_path: str = 'results/models'
    model_save_name: str = MISSING
    
    input_specs: Dict[str, int] = field(default_factory=dict)
    output_specs: Dict[str, int] = field(default_factory=dict)
    
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
```

Create `src/config/trainer_config.py`:
```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SyntheticTrainerConfig:
    """Configuration for synthetic model training."""
    
    learning_rate: float = 1e-4
    batch_size: int = 16
    epochs: int = 100
    num_predict_steps: int = 4
    
    train_sim: List[int] = field(default_factory=list)
    val_sim: Optional[List[int]] = None
    
    use_sliding_window: bool = False
    
    # Optimizer settings
    optimizer: str = 'adam'
    scheduler: str = 'cosine'
    weight_decay: float = 0.0
    
    # Checkpoint settings
    save_interval: int = 10
    save_best_only: bool = True


@dataclass
class LearnableParameter:
    """Definition of a learnable parameter for inverse problems."""
    name: str
    initial_guess: float
    bounds: Optional[tuple] = None


@dataclass
class PhysicalTrainerConfig:
    """Configuration for physical model inverse problem training."""
    
    epochs: int = 100
    num_predict_steps: int = 10
    train_sim: List[int] = field(default_factory=list)
    
    learnable_parameters: List[LearnableParameter] = field(default_factory=list)
    
    # Optimizer settings
    method: str = 'L-BFGS-B'
    abs_tol: float = 1e-6
    max_iterations: Optional[int] = None
```

Create `src/config/generation_config.py`:
```python
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for data generation."""
    
    num_simulations: int = 10
    total_steps: int = 50
    save_interval: int = 1
    
    # Optional: random seed for reproducibility
    seed: Optional[int] = None
```

Create `src/config/evaluation_config.py`:
```python
from dataclasses import dataclass, field
from typing import List


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    test_sim: List[int] = field(default_factory=list)
    num_frames: int = 51
    metrics: List[str] = field(default_factory=lambda: ['mse', 'mae', 'rmse'])
    
    keyframe_count: int = 5
    animation_fps: int = 10
    save_animations: bool = True
    save_plots: bool = True
    
    output_dir: str = 'results/evaluation'
```

Create `src/config/experiment_config.py`:
```python
from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import MISSING

from .data_config import DataConfig
from .model_config import PhysicalModelConfig, SyntheticModelConfig
from .trainer_config import SyntheticTrainerConfig, PhysicalTrainerConfig
from .generation_config import GenerationConfig
from .evaluation_config import EvaluationConfig


@dataclass
class RunConfig:
    """Top-level run configuration."""
    experiment_name: str = MISSING
    notes: str = ""
    mode: List[str] = field(default_factory=list)  # ['generate', 'train', 'evaluate']
    model_type: str = 'synthetic'  # 'synthetic' or 'physical'


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    # Hydra settings
    defaults: List[Any] = field(default_factory=lambda: [
        '_self_',
        {'data': 'burgers_128'},
        {'model/physical': 'burgers'},
        {'model/synthetic': 'unet'},
        {'trainer': 'synthetic'},
    ])
    
    # Main config sections
    run_params: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Model configs (one will be used based on model_type)
    model: Dict[str, Any] = field(default_factory=dict)
    
    # Task-specific configs
    generation_params: Optional[GenerationConfig] = None
    trainer_params: Optional[Any] = None  # Can be Synthetic or Physical
    evaluation_params: Optional[EvaluationConfig] = None
    
    # Runtime
    project_root: str = '.'
```

#### Step 2.3: Create Hydra Config Structure

Create directory structure:
```
conf/
├── config.yaml                    # Base config
├── data/
│   ├── burgers_128.yaml
│   ├── smoke_128.yaml
│   └── heat_64.yaml
├── model/
│   ├── physical/
│   │   ├── burgers.yaml
│   │   ├── smoke.yaml
│   │   └── heat.yaml
│   └── synthetic/
│       ├── unet.yaml
│       └── unet_large.yaml
├── trainer/
│   ├── synthetic.yaml
│   ├── synthetic_long.yaml
│   └── physical.yaml
├── generation/
│   ├── default.yaml
│   └── long.yaml
├── evaluation/
│   └── default.yaml
└── experiment/
    ├── burgers_experiment.yaml
    ├── smoke_experiment.yaml
    └── burgers_physical_experiment.yaml
```

Example `conf/config.yaml`:
```yaml
defaults:
  - data: burgers_128
  - model/physical: burgers
  - model/synthetic: unet
  - trainer: synthetic
  - generation: default
  - evaluation: default
  - _self_

run_params:
  experiment_name: ???  # Must be specified
  notes: ""
  mode: [train]
  model_type: synthetic

project_root: ${hydra:runtime.cwd}
```

Example `conf/data/burgers_128.yaml`:
```yaml
# @package _global_.data
data_dir: 'data/'
dset_name: 'burgers_128'
fields: ['velocity']
fields_scheme: 'VV'
cache_dir: 'data/cache'
validate_cache: true
auto_clear_invalid: false
```

Example `conf/model/physical/burgers.yaml`:
```yaml
# @package _global_.model.physical
name: 'BurgersModel'
domain:
  size_x: 100
  size_y: 100
resolution:
  x: 128
  y: 128
dt: 0.8
pde_params:
  batch_size: 1
  nu: 0.1
```

Example `conf/model/synthetic/unet.yaml`:
```yaml
# @package _global_.model.synthetic
name: 'UNet'
model_path: 'results/models'
model_save_name: ???  # Must be specified

input_specs:
  velocity: 2

output_specs:
  velocity: 2

architecture:
  levels: 4
  filters: 64
  batch_norm: true
```

Example `conf/experiment/burgers_experiment.yaml`:
```yaml
# @package _global_
defaults:
  - /data: burgers_128
  - /model/physical: burgers
  - /model/synthetic: unet
  - /trainer: synthetic
  - override /generation: default
  - override /evaluation: default

run_params:
  experiment_name: 'burgers_unet_128_v1'
  notes: 'Training UNet on Burgers 128x128'
  mode: ['train', 'evaluate']
  model_type: 'synthetic'

model:
  synthetic:
    model_save_name: 'burgers_unet_128'

trainer_params:
  train_sim: [0, 1, 2, 3, 4, 5, 6, 7, 8]
  epochs: 100
  batch_size: 32

evaluation_params:
  test_sim: [0, 9]
  num_frames: 51
```

#### Step 2.4: Update run.py for Hydra

Create new `run_hydra.py`:
```python
"""
Hydra-based experiment runner.

Usage:
    python run_hydra.py experiment=burgers_experiment
    python run_hydra.py experiment=burgers_experiment trainer_params.epochs=200
    python run_hydra.py +experiment=smoke_experiment run_params.mode=[generate,train]
"""

import os
import sys
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generator import run_generation
from src.training.synthetic.trainer import SyntheticTrainer
from src.training.physical.trainer import PhysicalTrainer
from src.evaluation import Evaluator


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    
    # Print configuration (optional, for debugging)
    print("=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    # Convert to regular dict for compatibility with existing code
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Add project root
    config['project_root'] = str(PROJECT_ROOT)
    
    run_config = config['run_params']
    tasks = run_config['mode']
    
    if isinstance(tasks, str):
        tasks = [tasks]
    
    print(f"\n--- Experiment: {run_config['experiment_name']} ---")
    print(f"--- Tasks: {tasks} ---\n")
    
    # Execute tasks
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"RUNNING TASK: {task.upper()}")
        print(f"{'='*60}\n")
        
        if task == 'generate':
            run_generation(config)
        
        elif task == 'train':
            model_type = run_config.get('model_type', 'synthetic')
            
            if model_type == 'synthetic':
                trainer = SyntheticTrainer(config)
                trainer.train()
            elif model_type == 'physical':
                trainer = PhysicalTrainer(config)
                trainer.train()
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
        
        elif task == 'evaluate':
            model_type = run_config.get('model_type', 'synthetic')
            
            if model_type == 'synthetic':
                evaluator = Evaluator(config)
                evaluator.evaluate()
            else:
                print("Physical model evaluation not yet implemented.")
        
        else:
            print(f"Warning: Unknown task '{task}'")
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE: {run_config['experiment_name']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
```

**Benefits of Hydra approach:**
```bash
# Run experiment
python run_hydra.py experiment=burgers_experiment

# Override parameters from command line
python run_hydra.py experiment=burgers_experiment trainer_params.epochs=500

# Change mode
python run_hydra.py experiment=burgers_experiment run_params.mode=[generate,train,evaluate]

# Override nested config
python run_hydra.py experiment=burgers_experiment model.physical.pde_params.nu=0.2

# Multi-run (hyperparameter sweep)
python run_hydra.py -m experiment=burgers_experiment trainer_params.learning_rate=1e-3,1e-4,1e-5
```

---

### Phase 3: Improved Modularity (Week 5-6)

#### Step 3.1: Implement Model Registry Pattern

Create `src/models/registry.py`:
```python
"""
Model registry for automatic model discovery and instantiation.
"""

from typing import Dict, Type, Any, Callable
from abc import ABC


class ModelRegistry:
    """Registry for automatic model discovery."""
    
    _physical_models: Dict[str, Type] = {}
    _synthetic_models: Dict[str, Type] = {}
    
    @classmethod
    def register_physical(cls, name: str) -> Callable:
        """Decorator to register physical models."""
        def decorator(model_class: Type) -> Type:
            cls._physical_models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def register_synthetic(cls, name: str) -> Callable:
        """Decorator to register synthetic models."""
        def decorator(model_class: Type) -> Type:
            cls._synthetic_models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get_physical_model(cls, name: str, config: Dict[str, Any]):
        """Get physical model instance."""
        if name not in cls._physical_models:
            raise ValueError(
                f"Physical model '{name}' not found. "
                f"Available: {list(cls._physical_models.keys())}"
            )
        return cls._physical_models[name](config)
    
    @classmethod
    def get_synthetic_model(cls, name: str, config: Dict[str, Any]):
        """Get synthetic model instance."""
        if name not in cls._synthetic_models:
            raise ValueError(
                f"Synthetic model '{name}' not found. "
                f"Available: {list(cls._synthetic_models.keys())}"
            )
        return cls._synthetic_models[name](config)
    
    @classmethod
    def list_physical_models(cls) -> list:
        """List registered physical models."""
        return list(cls._physical_models.keys())
    
    @classmethod
    def list_synthetic_models(cls) -> list:
        """List registered synthetic models."""
        return list(cls._synthetic_models.keys())
```

Update physical models to use registry:

`src/models/physical/burgers.py`:
```python
from src.models.registry import ModelRegistry
from .base import PhysicalModel

@ModelRegistry.register_physical('BurgersModel')
class BurgersModel(PhysicalModel):
    # ... existing implementation ...
```

`src/models/physical/smoke.py`:
```python
from src.models.registry import ModelRegistry
from .base import PhysicalModel

@ModelRegistry.register_physical('SmokeModel')
class SmokeModel(PhysicalModel):
    # ... existing implementation ...
```

`src/models/synthetic/unet.py`:
```python
from src.models.registry import ModelRegistry
from .base import SyntheticModel

@ModelRegistry.register_synthetic('UNet')
class UNet(SyntheticModel):
    # ... existing implementation ...
```

Update `__init__.py` files to trigger registration:
```python
# src/models/physical/__init__.py
from .base import PhysicalModel
from .burgers import BurgersModel
from .smoke import SmokeModel
from .heat import HeatModel

__all__ = ['PhysicalModel', 'BurgersModel', 'SmokeModel', 'HeatModel']
```

#### Step 3.2: Create Base Trainer Class

Create `src/training/base_trainer.py`:
```python
"""
Base trainer class with shared functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import torch
import torch.nn as nn


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.
    
    Provides common functionality:
    - Model loading/saving
    - Device management
    - Config parsing
    - Checkpoint handling
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize base trainer."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # To be set by subclasses
        self.model: Optional[nn.Module] = None
        self.checkpoint_path: Optional[Path] = None
    
    @abstractmethod
    def _create_model(self):
        """Create model instance. Must be implemented by subclass."""
        pass
    
    @abstractmethod
    def _create_data_loader(self):
        """Create data loader. Must be implemented by subclass."""
        pass
    
    @abstractmethod
    def _train_epoch(self):
        """Train one epoch. Must be implemented by subclass."""
        pass
    
    @abstractmethod
    def train(self):
        """Main training loop. Must be implemented by subclass."""
        pass
    
    def save_checkpoint(
        self,
        epoch: int,
        loss: float,
        optimizer_state: Optional[Dict] = None,
        is_best: bool = False
    ):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            loss: Current loss
            optimizer_state: Optimizer state dict
            is_best: If True, also save as 'best.pth'
        """
        if self.checkpoint_path is None:
            raise ValueError("checkpoint_path not set")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'config': self.config,
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        # Save regular checkpoint
        torch.save(checkpoint, self.checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_path.parent / 'best.pth'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint. If None, uses self.checkpoint_path
            
        Returns:
            Checkpoint dictionary
        """
        if path is None:
            path = self.checkpoint_path
        
        if path is None or not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return checkpoint
    
    def get_parameter_count(self) -> int:
        """Get total number of model parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
    
    def get_trainable_parameter_count(self) -> int:
        """Get number of trainable parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def print_model_summary(self):
        """Print model summary information."""
        total_params = self.get_parameter_count()
        trainable_params = self.get_trainable_parameter_count()
        
        print("\n" + "="*60)
        print("MODEL SUMMARY")
        print("="*60)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Device: {self.device}")
        print("="*60 + "\n")
```

Update `SyntheticTrainer` to inherit from `BaseTrainer`:
```python
from src.training.base_trainer import BaseTrainer


class SyntheticTrainer(BaseTrainer):
    """Tensor-based trainer for synthetic models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Parse config sections
        self.data_config = config['data']
        self.model_config = config['model']['synthetic']
        self.trainer_config = config['trainer_params']
        
        # ... rest of initialization ...
        
        # Setup components
        self.train_loader = self._create_data_loader()
        self.model = self._create_model()
        
        # Print summary
        self.print_model_summary()
    
    # Implement abstract methods
    def _create_model(self):
        # ... existing implementation ...
    
    def _create_data_loader(self):
        # ... existing implementation ...
    
    def _train_epoch(self):
        # ... existing implementation ...
    
    def train(self):
        # ... existing implementation ...
```

#### Step 3.3: Implement Factory Pattern

Create `src/factories/` directory:

`src/factories/model_factory.py`:
```python
"""Factory for creating models."""

from typing import Dict, Any
import torch.nn as nn
from src.models.registry import ModelRegistry


class ModelFactory:
    """Factory for creating model instances."""
    
    @staticmethod
    def create_physical_model(config: Dict[str, Any]):
        """Create physical model from config."""
        model_config = config['model']['physical']
        model_name = model_config['name']
        return ModelRegistry.get_physical_model(model_name, model_config)
    
    @staticmethod
    def create_synthetic_model(config: Dict[str, Any]) -> nn.Module:
        """Create synthetic model from config."""
        model_config = config['model']['synthetic']
        model_name = model_config['name']
        return ModelRegistry.get_synthetic_model(model_name, model_config)
    
    @staticmethod
    def list_available_models():
        """List all available models."""
        return {
            'physical': ModelRegistry.list_physical_models(),
            'synthetic': ModelRegistry.list_synthetic_models(),
        }
```

`src/factories/trainer_factory.py`:
```python
"""Factory for creating trainers."""

from typing import Dict, Any
from src.training.base_trainer import BaseTrainer
from src.training.synthetic.trainer import SyntheticTrainer
from src.training.physical.trainer import PhysicalTrainer


class TrainerFactory:
    """Factory for creating trainer instances."""
    
    _trainers = {
        'synthetic': SyntheticTrainer,
        'physical': PhysicalTrainer,
    }
    
    @staticmethod
    def create_trainer(config: Dict[str, Any]) -> BaseTrainer:
        """
        Create trainer from config.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Trainer instance
        """
        model_type = config['run_params']['model_type']
        
        if model_type not in TrainerFactory._trainers:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Available: {list(TrainerFactory._trainers.keys())}"
            )
        
        TrainerClass = TrainerFactory._trainers[model_type]
        return TrainerClass(config)
    
    @staticmethod
    def register_trainer(name: str, trainer_class: type):
        """Register a new trainer type."""
        TrainerFactory._trainers[name] = trainer_class
```

Update `run_hydra.py` to use factories:
```python
from src.factories.model_factory import ModelFactory
from src.factories.trainer_factory import TrainerFactory

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    config['project_root'] = str(PROJECT_ROOT)
    
    tasks = config['run_params']['mode']
    
    for task in tasks:
        if task == 'generate':
            run_generation(config)
        
        elif task == 'train':
            # Use factory to create trainer
            trainer = TrainerFactory.create_trainer(config)
            trainer.train()
        
        elif task == 'evaluate':
            evaluator = Evaluator(config)
            evaluator.evaluate()
```

---

## 4. Migration Strategy

### 4.1 Backward Compatibility

**Key Principle:** Old code continues to work during migration

1. **Keep existing `run.py`**
   - Original YAML configs still work
   - Users can gradually migrate

2. **Dual config support in classes**
   ```python
   class SyntheticTrainer:
       def __init__(self, config):
           # Support both dict and OmegaConf
           if isinstance(config, DictConfig):
               config = OmegaConf.to_container(config, resolve=True)
           # ... rest of init ...
   ```

3. **Cache format versioning**
   - Old cache files still load
   - New cache has version='2.0'
   - Gradual regeneration

### 4.2 Migration Phases

**Phase 1 (Week 1-2): Dataset Validation**
- ✅ No breaking changes
- ✅ Backward compatible with old cache
- ✅ Add validation as opt-in feature
- ✅ Test thoroughly before making required

**Phase 2 (Week 3-4): Hydra Setup**
- ✅ Create `run_hydra.py` alongside `run.py`
- ✅ Keep both working simultaneously
- ✅ Documentation for both approaches
- ✅ Users choose when to migrate

**Phase 3 (Week 5-6): Modularity**
- ✅ Refactor without changing interfaces
- ✅ Factories optional (old way still works)
- ✅ BaseTrainer as mixin (not required parent)
- ✅ Gradual adoption

**Phase 4 (Week 7): Deprecation (Optional)**
- Mark old `run.py` as deprecated
- Encourage migration to Hydra
- Plan eventual removal

### 4.3 Rollback Strategy

**For each phase:**
1. **Feature flags** control new behavior
2. **Git branches** for each phase
3. **Comprehensive tests** before merging
4. **Documentation** of rollback procedure

**If issues arise:**
```bash
# Rollback validation
git revert <validation-commits>

# Rollback Hydra
git checkout main
# Users continue using run.py

# Rollback modularity
git revert <refactor-commits>
```

---

## 5. Testing Strategy

### 5.1 Unit Tests

**For Phase 1 (Validation):**
```python
# tests/data/test_enhanced_validation.py

def test_cache_invalidated_on_pde_param_change():
    """Test PDE parameter validation."""

def test_cache_invalidated_on_resolution_change():
    """Test resolution validation."""

def test_backward_compatibility_with_v1_cache():
    """Test old cache still works."""

def test_validation_chain_complete():
    """Test all validation criteria."""

def test_cache_manager_cli():
    """Test cache management utilities."""
```

**For Phase 2 (Hydra):**
```python
# tests/config/test_hydra_configs.py

def test_structured_config_validation():
    """Test dataclass validation."""

def test_config_composition():
    """Test Hydra composition."""

def test_command_line_overrides():
    """Test CLI overrides work."""

def test_backward_compatibility_with_yaml():
    """Test old YAML still works."""
```

**For Phase 3 (Modularity):**
```python
# tests/models/test_registry.py

def test_model_registration():
    """Test model registration."""

def test_model_factory_creation():
    """Test factory creates models."""

# tests/training/test_base_trainer.py

def test_base_trainer_interface():
    """Test base trainer contract."""

def test_checkpoint_saving_loading():
    """Test checkpoint functionality."""
```

### 5.2 Integration Tests

```python
# tests/integration/test_end_to_end.py

def test_generate_train_evaluate_pipeline():
    """Test complete workflow."""

def test_cache_reuse_across_runs():
    """Test cache reuse works."""

def test_hydra_experiment_run():
    """Test Hydra-based experiment."""

def test_factory_based_workflow():
    """Test factory-based creation."""
```

### 5.3 Validation Tests

Before each phase merge:
1. ✅ All existing tests pass
2. ✅ New tests added and passing
3. ✅ Integration tests pass
4. ✅ Manual smoke test of workflows
5. ✅ Performance benchmarks (no regression)

---

## 6. Rollback Plan

### 6.1 Per-Phase Rollback

**Phase 1: Dataset Validation**
```bash
# Disable validation
export DISABLE_CACHE_VALIDATION=1

# Or in code
DataManager(..., validate_cache=False)

# Or revert commits
git revert <validation-commits>
```

**Phase 2: Hydra**
```bash
# Continue using old run.py
python run.py --config configs/burgers_experiment.yaml

# Or revert Hydra commits
git revert <hydra-commits>
```

**Phase 3: Modularity**
```bash
# Old interfaces still work
# Just don't use factories

# Or revert refactor
git revert <modularity-commits>
```

### 6.2 Emergency Rollback

If critical issues:
```bash
# Return to last stable version
git checkout last-stable-tag

# Or feature flag everything off
export HYCO_LEGACY_MODE=1
```

---

## 7. Documentation Updates

### 7.1 User Documentation

Create/update:
1. `docs/USER_GUIDE.md` - Updated workflows
2. `docs/HYDRA_MIGRATION.md` - Migration guide
3. `docs/CONFIG_REFERENCE.md` - Config schema reference
4. `docs/CACHE_MANAGEMENT.md` - Cache utilities guide

### 7.2 Developer Documentation

Create/update:
1. `docs/ARCHITECTURE.md` - Updated architecture
2. `docs/ADDING_MODELS.md` - How to add new models
3. `docs/EXTENDING_TRAINERS.md` - Trainer extension guide
4. `docs/API_REFERENCE.md` - API documentation

### 7.3 Migration Examples

Create `examples/migration/`:
```
examples/migration/
├── 01_old_style_experiment.py
├── 02_new_style_experiment.py
├── 03_hydra_basic.py
├── 04_hydra_advanced.py
└── README.md
```

---

## 8. Success Criteria

### 8.1 Phase 1: Dataset Validation

- ✅ Cache validation detects parameter changes
- ✅ Users notified when cache is invalid
- ✅ Cache utilities work correctly
- ✅ No performance regression
- ✅ All existing tests pass

### 8.2 Phase 2: Hydra

- ✅ All experiments runnable with Hydra
- ✅ CLI overrides work
- ✅ Config composition works
- ✅ Backward compatibility maintained
- ✅ Documentation complete

### 8.3 Phase 3: Modularity

- ✅ New models added via registry
- ✅ Less code duplication
- ✅ Cleaner interfaces
- ✅ Better testability
- ✅ Easier to extend

---

## 9. Timeline

| Week | Phase | Tasks | Deliverables |
|------|-------|-------|--------------|
| 1-2 | Validation | Implement validation, cache manager, tests | Enhanced validation system |
| 3-4 | Hydra | Setup Hydra, create configs, update run.py | Hydra-based runner |
| 5-6 | Modularity | Factories, base classes, refactoring | Modular architecture |
| 7 | Polish | Documentation, examples, deprecations | Complete migration |

---

## 10. Next Steps

1. **Review this plan** with team
2. **Create feature branches** for each phase
3. **Start with Phase 1** (lowest risk)
4. **Iterate and refine** based on feedback
5. **Maintain momentum** with weekly reviews

---

## Appendix A: Example Usage

### Old Style (Current)
```bash
python run.py --config configs/burgers_experiment.yaml
```

### New Style (After Migration)
```bash
# Basic
python run_hydra.py experiment=burgers_experiment

# With overrides
python run_hydra.py experiment=burgers_experiment \
    trainer_params.epochs=500 \
    run_params.mode=[train,evaluate]

# Hyperparameter sweep
python run_hydra.py -m experiment=burgers_experiment \
    trainer_params.learning_rate=1e-3,1e-4,1e-5
```

### Cache Management
```bash
# List cached datasets
python scripts/manage_cache.py list

# View statistics
python scripts/manage_cache.py stats burgers_128

# Validate cache
python scripts/manage_cache.py validate burgers_128 \
    --config configs/burgers_experiment.yaml

# Clear invalid cache
python scripts/manage_cache.py clear burgers_128 --yes
```

---

## Appendix B: Configuration Examples

### Hydra Config Structure
```yaml
# conf/experiment/my_experiment.yaml
defaults:
  - /data: burgers_128
  - /model/physical: burgers
  - /model/synthetic: unet
  - /trainer: synthetic

run_params:
  experiment_name: my_experiment
  mode: [train, evaluate]

# Override specific parameters
model:
  synthetic:
    architecture:
      filters: 128  # Double the filters

trainer_params:
  learning_rate: 5e-4
  epochs: 200
```

---

## Appendix C: Performance Considerations

### Validation Overhead
- **First cache check:** ~1-5ms (hash computation)
- **Subsequent checks:** <1ms (cached results)
- **Overall impact:** Negligible (<0.1% of total time)

### Hydra Overhead
- **Config loading:** ~100-200ms (one-time)
- **Runtime overhead:** None (resolved at startup)
- **Benefit:** Better UX, fewer errors

### Modularity Overhead
- **Factory pattern:** <1ms per instantiation
- **Base class:** No overhead
- **Benefit:** Cleaner code, easier maintenance

---

**Document Version:** 1.0  
**Last Updated:** October 31, 2025  
**Status:** Ready for Implementation
