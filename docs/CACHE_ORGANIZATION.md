# Cache Organization and Data Hierarchy

**Document Version:** 1.0  
**Date:** November 3, 2025  
**Status:** Design Specification

---

## Overview

As the project grows, we need a clear hierarchy for different types of cached and generated data. This document defines the organization strategy for:
1. Raw simulation data (PhiFlow Scenes)
2. Preprocessed tensor cache (converted Fields)
3. Generated predictions (hybrid training)
4. Evaluation results and metrics

---

## Directory Structure

```
HYCO-PhiFlow/
├── data/                           # Raw simulation data
│   ├── burgers_128/               # Dataset: Burgers equation, 128x128
│   │   ├── sim_000000/           # Individual simulation (PhiFlow Scene)
│   │   ├── sim_000001/
│   │   └── ...
│   ├── smoke_128/                 # Dataset: Smoke simulation
│   ├── advection_64/              # Dataset: Advection
│   └── heat_64/                   # Dataset: Heat equation
│
├── cache/                          # Preprocessed and temporary data
│   ├── tensors/                   # Converted tensor cache (from DataManager)
│   │   ├── burgers_128/          # One subfolder per dataset
│   │   │   ├── sim_000000.pt     # Cached tensors + metadata
│   │   │   ├── sim_000001.pt
│   │   │   └── ...
│   │   ├── smoke_128/
│   │   └── advection_64/
│   │
│   ├── generated/                 # Generated predictions (hybrid training)
│   │   ├── synthetic_preds/      # Predictions FROM synthetic model
│   │   │   ├── cycle_000/        # Per hybrid cycle
│   │   │   │   ├── burgers_128/  # Per dataset
│   │   │   │   │   ├── gen_000000.pt
│   │   │   │   │   ├── gen_000001.pt
│   │   │   │   │   └── ...
│   │   │   ├── cycle_001/
│   │   │   └── ...
│   │   │
│   │   └── physical_preds/       # Predictions FROM physical model
│   │       ├── cycle_000/
│   │       │   ├── burgers_128/
│   │       │   │   ├── gen_000000.pt
│   │       │   │   └── ...
│   │       ├── cycle_001/
│   │       └── ...
│   │
│   └── evaluation/                # Evaluation-specific cache
│       ├── burgers_128/
│       │   ├── eval_tensors/     # Cached eval data tensors
│       │   └── predictions/      # Model predictions on eval set
│       └── smoke_128/
│
├── results/                        # Permanent results (not cache)
│   ├── models/                    # Saved model checkpoints
│   │   ├── synthetic/
│   │   │   ├── burgers_unet_epoch_100.pt
│   │   │   └── ...
│   │   ├── physical/
│   │   │   ├── burgers_diff_epoch_50.pt
│   │   │   └── ...
│   │   └── hybrid/
│   │       ├── burgers_hybrid_cycle_10.pt
│   │       └── ...
│   │
│   └── evaluation/                # Evaluation metrics and visualizations
│       ├── burgers_experiment_1/
│       │   ├── metrics.json
│       │   ├── rollout_plots/
│       │   └── error_analysis/
│       └── ...
│
└── logs/                          # Training logs
    ├── 2025-11-03_10-30-15/
    └── ...
```

---

## Configuration Mapping

### Current Config Structure

```yaml
# conf/config.yaml (defaults)
paths:
  # Raw data location
  raw_data_root: "data"
  
  # Cache locations (can be on different drives for performance)
  cache_root: "cache"
  tensor_cache_dir: "${paths.cache_root}/tensors"
  generated_cache_dir: "${paths.cache_root}/generated"
  eval_cache_dir: "${paths.cache_root}/evaluation"
  
  # Permanent results
  results_root: "results"
  model_checkpoint_dir: "${paths.results_root}/models"
  evaluation_output_dir: "${paths.results_root}/evaluation"
  
  # Logs
  log_dir: "logs"

data:
  dset_name: "burgers_128"  # Dataset name
  raw_data_dir: "${paths.raw_data_root}/${data.dset_name}"
  
  # Cache for THIS dataset
  cache_dir: "${paths.tensor_cache_dir}/${data.dset_name}"

trainer:
  # Hybrid training specific
  alpha: 0.1
  memory_budget_gb: 2.0  # Configurable memory budget
  
  # Generated data cache (cleaned up after each cycle)
  synthetic_pred_cache: "${paths.generated_cache_dir}/synthetic_preds"
  physical_pred_cache: "${paths.generated_cache_dir}/physical_preds"
```

### Example Experiment Config

```yaml
# conf/burgers_hybrid_experiment.yaml
defaults:
  - config
  - data: burgers_128
  - model: hybrid
  - trainer: hybrid

# Override paths for high-performance storage
paths:
  cache_root: "/mnt/ssd/cache"  # Fast SSD for cache
  raw_data_root: "/data/phiflow_sims"  # Bulk storage for raw data

trainer:
  alpha: 0.15
  memory_budget_gb: 4.0  # Larger budget for this experiment
  num_cycles: 20
  synthetic_epochs_per_cycle: 10
  physical_epochs_per_cycle: 5
```

---

## Cache Management Utilities

### Automatic Cleanup

```python
class CacheManager:
    """
    Centralized cache management for the hybrid training system.
    
    Handles:
    - Cache directory creation
    - Automatic cleanup of temporary caches
    - Disk space monitoring
    - Cache invalidation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Extract all cache paths
        self.tensor_cache_root = Path(config["paths"]["tensor_cache_dir"])
        self.generated_cache_root = Path(config["paths"]["generated_cache_dir"])
        self.eval_cache_root = Path(config["paths"]["eval_cache_dir"])
        
        # Dataset-specific paths
        self.dataset_name = config["data"]["dset_name"]
        self.dataset_tensor_cache = self.tensor_cache_root / self.dataset_name
        
        # Hybrid training paths
        self.synthetic_pred_cache = Path(config["trainer"]["synthetic_pred_cache"])
        self.physical_pred_cache = Path(config["trainer"]["physical_pred_cache"])
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create all necessary cache directories."""
        for path in [
            self.tensor_cache_root,
            self.generated_cache_root,
            self.eval_cache_root,
            self.dataset_tensor_cache,
            self.synthetic_pred_cache,
            self.physical_pred_cache,
        ]:
            path.mkdir(parents=True, exist_ok=True)
    
    def get_synthetic_pred_cache_dir(self, cycle: int) -> Path:
        """
        Get cache directory for synthetic predictions in a specific cycle.
        
        Args:
            cycle: Hybrid training cycle number
            
        Returns:
            Path to cache directory for this cycle
        """
        cache_dir = self.synthetic_pred_cache / f"cycle_{cycle:03d}" / self.dataset_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def get_physical_pred_cache_dir(self, cycle: int) -> Path:
        """
        Get cache directory for physical predictions in a specific cycle.
        
        Args:
            cycle: Hybrid training cycle number
            
        Returns:
            Path to cache directory for this cycle
        """
        cache_dir = self.physical_pred_cache / f"cycle_{cycle:03d}" / self.dataset_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def cleanup_cycle_cache(self, cycle: int):
        """
        Clean up generated prediction cache for a specific cycle.
        
        Args:
            cycle: Cycle number to clean up
        """
        import shutil
        
        # Clean synthetic predictions
        synthetic_cycle_dir = self.synthetic_pred_cache / f"cycle_{cycle:03d}"
        if synthetic_cycle_dir.exists():
            shutil.rmtree(synthetic_cycle_dir)
            logger.info(f"Cleaned up synthetic prediction cache for cycle {cycle}")
        
        # Clean physical predictions
        physical_cycle_dir = self.physical_pred_cache / f"cycle_{cycle:03d}"
        if physical_cycle_dir.exists():
            shutil.rmtree(physical_cycle_dir)
            logger.info(f"Cleaned up physical prediction cache for cycle {cycle}")
    
    def cleanup_all_generated_cache(self):
        """Clean up ALL generated prediction caches."""
        import shutil
        
        if self.synthetic_pred_cache.exists():
            shutil.rmtree(self.synthetic_pred_cache)
            self.synthetic_pred_cache.mkdir(parents=True, exist_ok=True)
            logger.info("Cleaned up all synthetic prediction cache")
        
        if self.physical_pred_cache.exists():
            shutil.rmtree(self.physical_pred_cache)
            self.physical_pred_cache.mkdir(parents=True, exist_ok=True)
            logger.info("Cleaned up all physical prediction cache")
    
    def get_cache_size(self, cache_type: str = "all") -> float:
        """
        Get total size of cache in GB.
        
        Args:
            cache_type: One of "tensors", "generated", "eval", or "all"
            
        Returns:
            Size in GB
        """
        def get_dir_size(path: Path) -> int:
            """Recursively get directory size in bytes."""
            total = 0
            if path.exists():
                for item in path.rglob("*"):
                    if item.is_file():
                        total += item.stat().st_size
            return total
        
        if cache_type == "tensors":
            size_bytes = get_dir_size(self.tensor_cache_root)
        elif cache_type == "generated":
            size_bytes = get_dir_size(self.generated_cache_root)
        elif cache_type == "eval":
            size_bytes = get_dir_size(self.eval_cache_root)
        elif cache_type == "all":
            size_bytes = (
                get_dir_size(self.tensor_cache_root) +
                get_dir_size(self.generated_cache_root) +
                get_dir_size(self.eval_cache_root)
            )
        else:
            raise ValueError(f"Unknown cache_type: {cache_type}")
        
        return size_bytes / 1e9  # Convert to GB
    
    def check_disk_space(self, required_gb: float) -> bool:
        """
        Check if sufficient disk space is available.
        
        Args:
            required_gb: Required space in GB
            
        Returns:
            True if sufficient space available
        """
        import shutil
        
        stat = shutil.disk_usage(self.tensor_cache_root)
        available_gb = stat.free / 1e9
        
        if available_gb < required_gb:
            logger.warning(
                f"Insufficient disk space: {available_gb:.2f} GB available, "
                f"{required_gb:.2f} GB required"
            )
            return False
        
        return True
    
    def print_cache_summary(self):
        """Print summary of cache usage."""
        logger.info("\n" + "="*60)
        logger.info("CACHE USAGE SUMMARY")
        logger.info("="*60)
        logger.info(f"Tensor cache:    {self.get_cache_size('tensors'):.2f} GB")
        logger.info(f"Generated cache: {self.get_cache_size('generated'):.2f} GB")
        logger.info(f"Eval cache:      {self.get_cache_size('eval'):.2f} GB")
        logger.info(f"Total cache:     {self.get_cache_size('all'):.2f} GB")
        logger.info("="*60 + "\n")
```

---

## Integration with HybridTrainer

```python
class HybridTrainer(AbstractTrainer):
    """
    Hybrid trainer with proper cache management.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model_registry: ModelRegistry,
        synthetic_trainer: SyntheticTrainer,
        physical_trainer: PhysicalTrainer,
        converter: BatchConcatenationConverter,
        data_manager: DataManager,
    ):
        super().__init__(config)
        
        self.model_registry = model_registry
        self.synthetic_trainer = synthetic_trainer
        self.physical_trainer = physical_trainer
        self.converter = converter
        self.data_manager = data_manager
        
        # Initialize cache manager
        self.cache_manager = CacheManager(config)
        
        # Training parameters
        self.alpha = config.get("trainer", {}).get("alpha", 0.1)
        self.memory_budget_gb = config.get("trainer", {}).get("memory_budget_gb", 2.0)
        self.num_cycles = config.get("trainer", {}).get("num_cycles", 10)
        self.synthetic_epochs_per_cycle = config.get("trainer", {}).get(
            "synthetic_epochs_per_cycle", 10
        )
        self.physical_epochs_per_cycle = config.get("trainer", {}).get(
            "physical_epochs_per_cycle", 5
        )
        
        # Determine augmentation strategy
        self._determine_augmentation_strategy()
        
        # Print initial cache summary
        self.cache_manager.print_cache_summary()
    
    def train(self, num_cycles: int):
        """Main hybrid training loop with proper cache management."""
        
        for cycle in range(num_cycles):
            logger.info(f"\n{'='*60}")
            logger.info(f"HYBRID CYCLE {cycle + 1}/{num_cycles}")
            logger.info(f"{'='*60}\n")
            
            try:
                # ===================================================================
                # Phase 1: Generate and cache physical predictions
                # ===================================================================
                logger.info("Phase 1: Generating physical predictions...")
                
                field_dataset = self._create_base_field_dataset()
                
                if self.augmentation_strategy == "on_the_fly":
                    # Don't pre-generate
                    physical_cache_dir = None
                else:
                    # Generate and cache
                    physical_cache_dir = self.cache_manager.get_physical_pred_cache_dir(cycle)
                    generated_tensors = self._generate_physical_predictions(field_dataset)
                    
                    if self.augmentation_strategy == "cache":
                        self._save_generated_tensors(generated_tensors, physical_cache_dir)
                        generated_tensors = None  # Free memory
                
                # ===================================================================
                # Phase 2: Train synthetic model
                # ===================================================================
                logger.info("\nPhase 2: Training synthetic model...")
                
                real_tensor_dataset = self._create_base_tensor_dataset()
                augmented_loader = self._create_augmented_tensor_loader(
                    real_tensor_dataset,
                    generated_data=generated_tensors if self.augmentation_strategy == "memory" else None,
                    cache_dir=physical_cache_dir if self.augmentation_strategy == "cache" else None
                )
                
                synthetic_metrics = self.synthetic_trainer.train(
                    augmented_loader,
                    num_epochs=self.synthetic_epochs_per_cycle
                )
                
                # ===================================================================
                # Phase 3: Generate and cache synthetic predictions
                # ===================================================================
                logger.info("\nPhase 3: Generating synthetic predictions...")
                
                if self.augmentation_strategy == "on_the_fly":
                    synthetic_cache_dir = None
                else:
                    synthetic_cache_dir = self.cache_manager.get_synthetic_pred_cache_dir(cycle)
                    generated_fields = self._generate_synthetic_predictions(real_tensor_dataset)
                    
                    if self.augmentation_strategy == "cache":
                        self._save_generated_fields(generated_fields, synthetic_cache_dir)
                        generated_fields = None
                
                # ===================================================================
                # Phase 4: Train physical model
                # ===================================================================
                logger.info("\nPhase 4: Training physical model...")
                
                augmented_source = self._create_augmented_field_source(
                    field_dataset,
                    generated_data=generated_fields if self.augmentation_strategy == "memory" else None,
                    cache_dir=synthetic_cache_dir if self.augmentation_strategy == "cache" else None
                )
                
                physical_metrics = self.physical_trainer.train(
                    augmented_source,
                    num_epochs=self.physical_epochs_per_cycle
                )
                
                # ===================================================================
                # Log cycle results
                # ===================================================================
                logger.info(f"\nCycle {cycle + 1} Complete:")
                logger.info(f"  Synthetic - Loss: {synthetic_metrics['final_loss']:.6f}")
                logger.info(f"  Physical  - Loss: {physical_metrics['final_loss']:.6f}")
                
            finally:
                # ===================================================================
                # Cleanup: Always clean up cycle cache (even on error)
                # ===================================================================
                if self.augmentation_strategy == "cache":
                    self.cache_manager.cleanup_cycle_cache(cycle)
                
                # Print cache summary after cycle
                self.cache_manager.print_cache_summary()
        
        logger.info("\n" + "="*60)
        logger.info("HYBRID TRAINING COMPLETE")
        logger.info("="*60)
        
        # Final cache summary
        self.cache_manager.print_cache_summary()
        
        return {
            "num_cycles": num_cycles,
            "final_synthetic_loss": synthetic_metrics["final_loss"],
            "final_physical_loss": physical_metrics["final_loss"],
        }
```

---

## CLI Tools for Cache Management

### Cache Inspection Tool

```python
# scripts/inspect_cache.py
"""
Utility script for inspecting cache contents and usage.

Usage:
    python scripts/inspect_cache.py --config conf/burgers_experiment.yaml
    python scripts/inspect_cache.py --cache-type tensors
    python scripts/inspect_cache.py --clean generated
"""

import argparse
from pathlib import Path
from omegaconf import OmegaConf

def main():
    parser = argparse.ArgumentParser(description="Inspect and manage cache")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--cache-type", choices=["tensors", "generated", "eval", "all"],
                       default="all", help="Type of cache to inspect")
    parser.add_argument("--clean", choices=["tensors", "generated", "eval", "all"],
                       help="Clean specified cache type")
    
    args = parser.parse_args()
    
    if args.config:
        config = OmegaConf.load(args.config)
        cache_manager = CacheManager(config)
    else:
        # Use default config
        config = OmegaConf.load("conf/config.yaml")
        cache_manager = CacheManager(config)
    
    if args.clean:
        # Clean specified cache
        if args.clean == "generated":
            cache_manager.cleanup_all_generated_cache()
        elif args.clean == "all":
            response = input("Are you sure you want to clean ALL caches? (yes/no): ")
            if response.lower() == "yes":
                cache_manager.cleanup_all_generated_cache()
                # Note: We don't clean tensor cache automatically as it's expensive to rebuild
                logger.info("Cleaned generated caches. Tensor cache preserved.")
        else:
            logger.warning(f"Cleaning {args.clean} cache not implemented")
    else:
        # Inspect cache
        cache_manager.print_cache_summary()

if __name__ == "__main__":
    main()
```

---

## Best Practices

### 1. Cache Separation
- **Never mix** raw data and cache in the same directory
- Keep generated predictions separate from preprocessed tensors
- Use different drives for cache vs results if possible (SSD for cache, HDD for results)

### 2. Cleanup Strategy
- **Generated cache**: Clean after each hybrid cycle (temporary)
- **Tensor cache**: Keep until explicitly deleted (expensive to rebuild)
- **Eval cache**: Clean when dataset or model changes

### 3. Configuration
- Always use configurable paths in YAML
- Use Hydra's interpolation for path composition
- Override paths per experiment as needed

### 4. Monitoring
- Log cache sizes before/after training
- Check disk space before starting long training runs
- Set up alerts for low disk space

---

## Migration Guide

### Updating Existing Configs

```yaml
# OLD (before)
data:
  cache_dir: "data/cache/burgers_128"

# NEW (after)
paths:
  cache_root: "cache"
  tensor_cache_dir: "${paths.cache_root}/tensors"
  generated_cache_dir: "${paths.cache_root}/generated"

data:
  dset_name: "burgers_128"
  cache_dir: "${paths.tensor_cache_dir}/${data.dset_name}"
```

### Updating Code

```python
# OLD
cache_dir = config["data"]["cache_dir"]

# NEW
cache_manager = CacheManager(config)
cache_dir = cache_manager.dataset_tensor_cache
```

---

## Future Enhancements

1. **Automatic cache pruning**: Remove least recently used caches when disk space low
2. **Cache compression**: Compress generated predictions to save space
3. **Distributed cache**: Support for shared cache across multiple machines
4. **Cache statistics**: Track hit/miss rates, access patterns
5. **Smart cleanup**: Keep recent cycles, clean old ones automatically

---

**Document Status:** Ready for implementation  
**Last Updated:** November 3, 2025
