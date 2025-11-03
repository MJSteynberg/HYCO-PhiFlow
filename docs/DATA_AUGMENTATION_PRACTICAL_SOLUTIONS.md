# Data Augmentation: Practical Solutions

**Document Version:** 1.0  
**Date:** November 3, 2025  
**Status:** Design Refinement

---

## Critical Challenges

### Challenge 1: Memory Constraints with Full Generation
**Problem:** Generating predictions for ALL samples (especially with sliding window) would require massive memory.

**Example:**
```
Real data: 30 simulations × 47 samples/sim (sliding window) = 1,410 samples
If we generate for all: 1,410 additional samples in memory
Each sample: ~64×64×C×T floats = potentially GBs of data
```

### Challenge 2: Generation Frequency vs Storage
**Problem:** Need to regenerate data for each hybrid epoch, but reuse across sub-trainer epochs.
- Generate once per hybrid epoch ✓
- Reuse for N sub-trainer epochs ✓
- But storing all generated data → memory issue ✗

### Challenge 3: Unified Data Access Pattern
**Problem:** Physical and Synthetic trainers should access data the same way.
- Both should support sliding window
- Both should work with same DataLoader interface
- But PhysicalTrainer currently uses different iteration pattern

---

## Solution 1: Batch-Level Generation with DataLoader

### Concept: Generate Augmented Data On-Demand in DataLoader

Instead of pre-generating all predictions, we generate them **batch-by-batch** during training.

#### Custom DataLoader with Generation

```python
class AugmentedBatchDataLoader:
    """
    DataLoader that generates augmented batches on-the-fly.
    
    Strategy:
    1. Each batch contains mix of real and generated samples
    2. Real samples: loaded from HybridDataset
    3. Generated samples: created on-demand per batch
    4. Generation happens in parallel with training (prefetching)
    
    Memory usage: Only one batch of generated data at a time!
    """
    
    def __init__(
        self,
        real_dataset: HybridDataset,
        generator_model: nn.Module,
        batch_size: int,
        alpha: float = 0.1,
        num_workers: int = 0,
        shuffle: bool = True,
        device: torch.device = None,
    ):
        """
        Args:
            real_dataset: HybridDataset with real data
            generator_model: Model to generate predictions (synthetic or physical)
            batch_size: Total batch size (real + generated)
            alpha: Weight for generated samples
            num_workers: Number of data loading workers
            shuffle: Whether to shuffle data
            device: Device for model inference (for generation)
        """
        self.real_dataset = real_dataset
        self.generator_model = generator_model
        self.batch_size = batch_size
        self.alpha = alpha
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Calculate split: how many real vs generated samples per batch
        # alpha represents the *weight* of generated data, but we need to decide
        # the *proportion* of generated samples
        # Let's use alpha as the proportion: e.g., alpha=0.1 → 10% generated samples
        self.num_generated_per_batch = max(1, int(batch_size * alpha))
        self.num_real_per_batch = batch_size - self.num_generated_per_batch
        
        # Create DataLoader for real data
        self.real_loader = DataLoader(
            real_dataset,
            batch_size=self.num_real_per_batch,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        self.generator_model.eval()
        self.generator_model.to(self.device)
        
    def __len__(self):
        """Number of batches per epoch."""
        return len(self.real_loader)
    
    def __iter__(self):
        """
        Yields batches with mixed real and generated data.
        
        Yields:
            Tuple of (inputs, targets, weights) where:
            - inputs: [batch_size, ...] 
            - targets: [batch_size, ...]
            - weights: [batch_size] tensor with 1.0 for real, alpha for generated
        """
        for real_batch in self.real_loader:
            # Unpack real batch
            real_inputs, real_targets = real_batch
            
            # Generate augmented samples for this batch
            generated_inputs, generated_targets = self._generate_batch_samples(
                num_samples=self.num_generated_per_batch,
                reference_batch=real_batch
            )
            
            # Combine real and generated
            combined_inputs = torch.cat([real_inputs, generated_inputs], dim=0)
            combined_targets = torch.cat([real_targets, generated_targets], dim=0)
            
            # Create weight tensor
            weights = torch.cat([
                torch.ones(len(real_inputs)),  # Real samples: weight = 1.0
                torch.full((len(generated_inputs),), self.alpha)  # Generated: weight = alpha
            ])
            
            # Shuffle within batch to mix real and generated
            indices = torch.randperm(len(combined_inputs))
            combined_inputs = combined_inputs[indices]
            combined_targets = combined_targets[indices]
            weights = weights[indices]
            
            yield combined_inputs, combined_targets, weights
    
    def _generate_batch_samples(
        self, 
        num_samples: int,
        reference_batch: Tuple
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic samples for augmentation.
        
        Strategy: Sample random indices from real dataset and generate predictions.
        
        Args:
            num_samples: Number of samples to generate
            reference_batch: Real batch for reference (shape, device, etc.)
            
        Returns:
            Tuple of (generated_inputs, generated_targets)
        """
        with torch.no_grad():
            generated_inputs = []
            generated_targets = []
            
            # Sample random indices from dataset
            indices = torch.randint(0, len(self.real_dataset), (num_samples,))
            
            for idx in indices:
                # Get real sample
                real_input, real_target = self.real_dataset[idx.item()]
                
                # Generate prediction using model
                input_tensor = real_input.unsqueeze(0).to(self.device)
                
                # Generate trajectory (autoregressive for synthetic, simulation for physical)
                generated_trajectory = self._generate_trajectory(input_tensor, real_target.shape[0])
                
                generated_inputs.append(real_input)  # Use same input
                generated_targets.append(generated_trajectory)
            
            return (
                torch.stack(generated_inputs),
                torch.stack(generated_targets)
            )
    
    def _generate_trajectory(self, initial_state: torch.Tensor, num_steps: int) -> torch.Tensor:
        """
        Generate trajectory from initial state.
        
        Args:
            initial_state: [1, C, H, W] tensor
            num_steps: Number of prediction steps
            
        Returns:
            [num_steps, C, H, W] tensor
        """
        # This method is model-specific
        # For SyntheticModel: autoregressive rollout
        # For PhysicalModel: simulation
        raise NotImplementedError("Subclass must implement trajectory generation")


class SyntheticAugmentedDataLoader(AugmentedBatchDataLoader):
    """
    Augmented DataLoader for synthetic training.
    Uses physical model to generate tensor predictions.
    """
    
    def __init__(
        self,
        real_dataset: HybridDataset,  # return_fields=False
        physical_model: Any,  # PhiFlow model
        converter: BatchConcatenationConverter,
        batch_size: int,
        alpha: float = 0.1,
        dt: float = 0.01,
        **kwargs
    ):
        self.physical_model = physical_model
        self.converter = converter
        self.dt = dt
        
        # Parent init (will call _generate_trajectory)
        super().__init__(
            real_dataset=real_dataset,
            generator_model=physical_model,  # Not actually used, but satisfies signature
            batch_size=batch_size,
            alpha=alpha,
            **kwargs
        )
    
    def _generate_trajectory(self, initial_state: torch.Tensor, num_steps: int) -> torch.Tensor:
        """
        Generate trajectory using physical simulation.
        
        Args:
            initial_state: [1, C, H, W] tensor with all fields
            num_steps: Number of prediction steps
            
        Returns:
            [num_steps, C_dynamic, H, W] tensor with dynamic fields only
        """
        # Convert initial tensor to fields
        initial_fields = self.converter.tensor_to_fields(initial_state[0])
        
        # Run physical simulation
        predicted_fields = self.physical_model.predict_trajectory(
            initial_fields,
            steps=num_steps,
            dt=self.dt
        )
        
        # Convert predicted fields back to tensors (dynamic fields only)
        trajectory_tensors = []
        for t in range(num_steps):
            fields_at_t = {field: predicted_fields[field][t] for field in predicted_fields.keys()}
            tensor_at_t = self.converter.fields_to_tensor(fields_at_t)
            trajectory_tensors.append(tensor_at_t)
        
        return torch.stack(trajectory_tensors, dim=0)


class PhysicalAugmentedDataLoader(AugmentedBatchDataLoader):
    """
    Augmented DataLoader for physical training.
    Uses synthetic model to generate field predictions.
    
    NOTE: This returns FIELDS, not tensors!
    """
    
    def __init__(
        self,
        real_dataset: HybridDataset,  # return_fields=True
        synthetic_model: nn.Module,
        converter: BatchConcatenationConverter,
        batch_size: int,
        alpha: float = 0.1,
        **kwargs
    ):
        self.synthetic_model = synthetic_model
        self.converter = converter
        
        super().__init__(
            real_dataset=real_dataset,
            generator_model=synthetic_model,
            batch_size=batch_size,
            alpha=alpha,
            **kwargs
        )
    
    def __iter__(self):
        """
        Yields batches with mixed real and generated FIELD data.
        
        NOTE: Physical trainer doesn't use batching in the traditional sense.
        Instead, it processes one sample at a time.
        So this actually yields individual samples, not batches.
        
        Yields:
            Tuple of (initial_fields, target_fields, weight)
        """
        # For physical training, we don't batch
        # Instead, we create an iterator that yields individual samples
        
        # Get all indices
        all_indices = list(range(len(self.real_dataset)))
        if self.shuffle:
            import random
            random.shuffle(all_indices)
        
        # Determine how many generated samples to create
        num_real = len(all_indices)
        num_generated = int(num_real * self.alpha)
        
        # Yield real samples
        for idx in all_indices:
            initial_fields, target_fields = self.real_dataset[idx]
            yield initial_fields, target_fields, 1.0
        
        # Yield generated samples
        generated_indices = torch.randint(0, len(self.real_dataset), (num_generated,))
        for idx in generated_indices:
            initial_fields, target_fields = self._generate_field_sample(idx.item())
            yield initial_fields, target_fields, self.alpha
    
    def _generate_field_sample(self, idx: int) -> Tuple[Dict, Dict]:
        """
        Generate a field sample using synthetic model.
        
        Args:
            idx: Index to use for initial condition
            
        Returns:
            Tuple of (initial_fields, predicted_target_fields)
        """
        with torch.no_grad():
            # Get real initial fields and convert to tensor
            initial_fields, _ = self.real_dataset[idx]
            initial_tensor = self.converter.fields_to_tensor(initial_fields).unsqueeze(0)
            
            # Generate trajectory using synthetic model
            initial_tensor = initial_tensor.to(self.device)
            predictions = []
            state = initial_tensor
            
            for step in range(self.real_dataset.num_predict_steps):
                pred = self.synthetic_model(state)
                predictions.append(pred)
                state = pred  # Autoregressive
            
            # Convert predictions back to fields
            predicted_fields = {field: [] for field in initial_fields.keys()}
            for pred_tensor in predictions:
                fields_at_t = self.converter.tensor_to_fields(pred_tensor[0])
                for field_name, field_obj in fields_at_t.items():
                    predicted_fields[field_name].append(field_obj)
            
            return initial_fields, predicted_fields
```

**Pros:**
- ✅ Memory efficient: only one batch of generated data at a time
- ✅ Fresh predictions: generated with current model state
- ✅ Parallel with training: can prefetch while training previous batch

**Cons:**
- ⚠️ Computational overhead: generation happens during training
- ⚠️ Slower epochs: each batch takes longer
- ⚠️ Complex implementation

---

## Solution 2: Cached Generation with Lazy Loading

### Concept: Generate Once, Cache to Disk, Load Lazily

Generate all predictions once per hybrid epoch, save to disk, load lazily during sub-trainer epochs.

```python
class CachedAugmentedDataset(Dataset):
    """
    Dataset that combines real data with cached generated predictions.
    
    Generated data is saved to disk and loaded lazily to manage memory.
    """
    
    def __init__(
        self,
        real_dataset: HybridDataset,
        generated_cache_dir: Path,
        alpha: float = 0.1,
        max_cached_samples: int = 100,  # LRU cache size
    ):
        self.real_dataset = real_dataset
        self.generated_cache_dir = Path(generated_cache_dir)
        self.alpha = alpha
        
        # Count generated samples
        self.generated_files = sorted(self.generated_cache_dir.glob("gen_*.pt"))
        self.num_real = len(real_dataset)
        self.num_generated = len(self.generated_files)
        
        # Create LRU cache for lazy loading
        self._load_generated_sample_cached = lru_cache(maxsize=max_cached_samples)(
            self._load_generated_sample_uncached
        )
    
    def __len__(self):
        return self.num_real + self.num_generated
    
    def __getitem__(self, idx: int) -> Tuple:
        if idx < self.num_real:
            # Real sample
            sample = self.real_dataset[idx]
            return (*sample, 1.0)
        else:
            # Generated sample (load lazily from cache)
            gen_idx = idx - self.num_real
            sample = self._load_generated_sample_cached(gen_idx)
            return (*sample, self.alpha)
    
    def _load_generated_sample_uncached(self, gen_idx: int) -> Tuple:
        """Load a generated sample from disk."""
        file_path = self.generated_files[gen_idx]
        return torch.load(file_path, weights_only=False)


class HybridTrainerWithCaching:
    """
    HybridTrainer that generates and caches predictions to disk.
    """
    
    def __init__(self, ...):
        # ... existing init ...
        
        # Cache directories for generated data
        self.cache_root = Path(config["paths"]["cache_dir"]) / "hybrid_generated"
        self.tensor_cache_dir = self.cache_root / "tensors"
        self.field_cache_dir = self.cache_root / "fields"
    
    def _generate_and_cache_physical_predictions(
        self, 
        field_dataset: HybridDataset,
        cycle: int
    ):
        """
        Generate physical predictions and save to disk.
        
        Args:
            field_dataset: Dataset with field data
            cycle: Current hybrid cycle number (for cache organization)
        """
        cache_dir = self.tensor_cache_dir / f"cycle_{cycle}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear old cache
        for old_file in cache_dir.glob("*.pt"):
            old_file.unlink()
        
        logger.info(f"Generating and caching physical predictions to {cache_dir}")
        
        physical_model = self.model_registry.get_physical_model()
        physical_model.eval()
        
        # Determine how many samples to generate (proportional to alpha)
        num_samples = int(len(field_dataset) * self.alpha)
        
        # Sample random indices
        indices = torch.randperm(len(field_dataset))[:num_samples]
        
        for i, idx in enumerate(tqdm(indices, desc="Generating")):
            with torch.no_grad():
                # Get sample and generate prediction
                initial_fields, target_fields = field_dataset[idx]
                
                # Generate prediction
                predicted_fields = physical_model.predict_trajectory(
                    initial_fields,
                    steps=field_dataset.num_predict_steps,
                    dt=self.config["model"]["physical"]["dt"]
                )
                
                # Convert to tensors
                input_tensor = self.converter.fields_to_tensor(initial_fields)
                target_tensor = self._convert_trajectory_to_tensor(predicted_fields)
                
                # Save to disk
                torch.save(
                    (input_tensor, target_tensor),
                    cache_dir / f"gen_{i:06d}.pt"
                )
        
        logger.info(f"Cached {len(indices)} generated samples")
        return cache_dir
    
    def train(self, base_data_manager, num_cycles: int):
        """Main training loop with disk caching."""
        
        for cycle in range(num_cycles):
            logger.info(f"\n{'='*60}")
            logger.info(f"HYBRID CYCLE {cycle + 1}/{num_cycles}")
            logger.info(f"{'='*60}\n")
            
            # ===================================================================
            # Phase 1: Generate and cache physical predictions
            # ===================================================================
            field_dataset = self._create_base_field_dataset()
            tensor_cache_dir = self._generate_and_cache_physical_predictions(
                field_dataset, cycle
            )
            
            # ===================================================================
            # Phase 2: Train synthetic model with cached augmented data
            # ===================================================================
            real_tensor_dataset = self._create_base_tensor_dataset()
            augmented_dataset = CachedAugmentedDataset(
                real_dataset=real_tensor_dataset,
                generated_cache_dir=tensor_cache_dir,
                alpha=self.alpha
            )
            
            augmented_loader = DataLoader(
                augmented_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
            
            # Train for multiple epochs (reuses cached data)
            synthetic_metrics = self.synthetic_trainer.train(
                augmented_loader,
                num_epochs=self.synthetic_epochs_per_cycle
            )
            
            # ===================================================================
            # Phase 3: Generate and cache synthetic predictions
            # ===================================================================
            field_cache_dir = self._generate_and_cache_synthetic_predictions(
                real_tensor_dataset, cycle
            )
            
            # ===================================================================
            # Phase 4: Train physical model with cached augmented data
            # ===================================================================
            augmented_field_dataset = CachedAugmentedDataset(
                real_dataset=field_dataset,
                generated_cache_dir=field_cache_dir,
                alpha=self.alpha
            )
            
            physical_metrics = self.physical_trainer.train(
                augmented_field_dataset,
                num_epochs=self.physical_epochs_per_cycle
            )
            
            # ===================================================================
            # Cleanup: Remove cached generated data
            # ===================================================================
            import shutil
            shutil.rmtree(tensor_cache_dir)
            shutil.rmtree(field_cache_dir)
```

**Pros:**
- ✅ Memory efficient: lazy loading with LRU cache
- ✅ Fast training: no generation overhead during sub-trainer epochs
- ✅ Reproducible: can inspect generated samples

**Cons:**
- ⚠️ Disk I/O overhead
- ⚠️ Requires disk space
- ⚠️ Stale predictions within hybrid epoch (but fresh each hybrid epoch)

---

## Solution 3: Hybrid Approach (RECOMMENDED)

### Concept: Smart Selection Based on Memory Budget

Combine both approaches based on available memory and dataset size:

```python
class AdaptiveAugmentedDataLoader:
    """
    Adaptively chooses between in-memory, cached, or on-the-fly generation.
    
    Strategy:
    1. Small datasets → generate all, keep in memory
    2. Medium datasets → generate all, cache to disk, lazy load
    3. Large datasets → generate on-the-fly per batch
    """
    
    def __init__(
        self,
        real_dataset: HybridDataset,
        generator_model: Any,
        batch_size: int,
        alpha: float = 0.1,
        memory_budget_gb: float = 2.0,  # Max memory for generated data
        cache_dir: Optional[Path] = None,
        **kwargs
    ):
        self.real_dataset = real_dataset
        self.generator_model = generator_model
        self.batch_size = batch_size
        self.alpha = alpha
        
        # Estimate memory requirements
        sample_input, sample_target = real_dataset[0]
        sample_size_bytes = (
            sample_input.element_size() * sample_input.numel() +
            sample_target.element_size() * sample_target.numel()
        )
        num_generated = int(len(real_dataset) * alpha)
        total_size_gb = (sample_size_bytes * num_generated) / 1e9
        
        # Choose strategy
        if total_size_gb < memory_budget_gb / 2:
            # Strategy 1: In-memory
            logger.info(f"Using IN-MEMORY generation ({total_size_gb:.2f} GB < {memory_budget_gb/2:.2f} GB)")
            self.strategy = "memory"
            self._generate_all_in_memory()
            
        elif cache_dir is not None and total_size_gb < memory_budget_gb:
            # Strategy 2: Disk cache
            logger.info(f"Using DISK CACHE generation ({total_size_gb:.2f} GB)")
            self.strategy = "cache"
            self.cache_dir = Path(cache_dir)
            self._generate_and_cache_to_disk()
            
        else:
            # Strategy 3: On-the-fly
            logger.info(f"Using ON-THE-FLY generation ({total_size_gb:.2f} GB > {memory_budget_gb:.2f} GB)")
            self.strategy = "on_the_fly"
        
    def __iter__(self):
        if self.strategy == "memory":
            return self._iter_from_memory()
        elif self.strategy == "cache":
            return self._iter_from_cache()
        else:
            return self._iter_on_the_fly()
```

---

## Solution 4: Unified Physical Trainer with DataLoader

### Making Physical Trainer Compatible with DataLoader

Currently physical trainer doesn't use DataLoader. Let's unify this:

```python
class PhysicalTrainer(FieldTrainer):
    """
    Physical trainer with unified data access.
    
    Now supports BOTH:
    1. DataLoader-style iteration (for hybrid training)
    2. Direct field iteration (for standalone training)
    """
    
    def train(self, data_source: Union[DataLoader, Iterable], num_epochs: int) -> Dict:
        """
        Train for specified epochs.
        
        Args:
            data_source: Either:
                - DataLoader yielding batches (for hybrid training with sliding window)
                - Iterable yielding samples (for backward compatibility)
            num_epochs: Number of epochs
        """
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_samples = 0
            
            for item in data_source:
                # Unpack based on length (3-tuple has weight, 2-tuple doesn't)
                if len(item) == 3:
                    initial_fields, target_fields, weight = item
                else:
                    initial_fields, target_fields = item
                    weight = 1.0
                
                # Train on sample
                self.optimizer.zero_grad()
                
                predicted_fields = self.model.predict_trajectory(
                    initial_fields,
                    steps=len(target_fields[next(iter(target_fields))]),
                    dt=self.config["model"]["physical"]["dt"]
                )
                
                loss = self._compute_field_loss(predicted_fields, target_fields)
                weighted_loss = loss * weight
                
                weighted_loss.backward()
                self.optimizer.step()
                
                epoch_loss += weighted_loss.item()
                num_samples += 1
            
            avg_loss = epoch_loss / num_samples
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")
        
        return {"final_loss": avg_loss, "num_epochs": num_epochs}


# Now both trainers can use the same data access pattern!
# Just need different Dataset return modes:

# For synthetic training:
tensor_dataset = HybridDataset(..., return_fields=False, use_sliding_window=True)
augmented_tensor_dataset = AugmentedTensorDataset(tensor_dataset, generated_data, alpha)
tensor_loader = DataLoader(augmented_tensor_dataset, batch_size=32, shuffle=True)
synthetic_trainer.train(tensor_loader, num_epochs=10)

# For physical training:
field_dataset = HybridDataset(..., return_fields=True, use_sliding_window=True)  # ← NOW WITH SLIDING WINDOW!
augmented_field_dataset = AugmentedFieldDataset(field_dataset, generated_data, alpha)
# Physical trainer can iterate directly (no batching needed for field operations)
physical_trainer.train(augmented_field_dataset, num_epochs=5)
```

---

## Recommended Final Design

### Memory-Aware Adaptive Strategy

```python
class HybridTrainer:
    """
    Hybrid trainer with adaptive data augmentation strategy.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model_registry: ModelRegistry,
        synthetic_trainer: SyntheticTrainer,
        physical_trainer: PhysicalTrainer,
        converter: BatchConcatenationConverter,
        data_manager: DataManager,  # ← Added
    ):
        super().__init__(config)
        
        self.model_registry = model_registry
        self.synthetic_trainer = synthetic_trainer
        self.physical_trainer = physical_trainer
        self.converter = converter
        self.data_manager = data_manager  # ← Store for metadata access
        
        # Augmentation parameters
        self.alpha = config.get("trainer", {}).get("alpha", 0.1)
        self.memory_budget_gb = config.get("trainer", {}).get("memory_budget_gb", 2.0)
        self.cache_root = Path(config["paths"]["cache_dir"]) / "hybrid_generated"
        
        # Determine augmentation strategy based on dataset size
        self._determine_augmentation_strategy()
    
    def _determine_augmentation_strategy(self):
        """
        Determine optimal augmentation strategy based on dataset size and memory budget.
        """
        # Create a sample dataset to estimate memory requirements
        sample_dataset = self._create_base_tensor_dataset()
        sample_input, sample_target = sample_dataset[0]
        
        sample_size_bytes = (
            sample_input.element_size() * sample_input.numel() +
            sample_target.element_size() * sample_target.numel()
        )
        
        num_generated = int(len(sample_dataset) * self.alpha)
        total_size_gb = (sample_size_bytes * num_generated) / 1e9
        
        if total_size_gb < self.memory_budget_gb / 2:
            self.augmentation_strategy = "memory"
            logger.info(f"Augmentation strategy: IN-MEMORY ({total_size_gb:.2f} GB)")
        elif total_size_gb < self.memory_budget_gb:
            self.augmentation_strategy = "cache"
            logger.info(f"Augmentation strategy: DISK CACHE ({total_size_gb:.2f} GB)")
        else:
            self.augmentation_strategy = "on_the_fly"
            logger.info(f"Augmentation strategy: ON-THE-FLY ({total_size_gb:.2f} GB)")
    
    def train(self, num_cycles: int):
        """
        Main hybrid training loop with adaptive augmentation.
        """
        for cycle in range(num_cycles):
            logger.info(f"\n{'='*60}")
            logger.info(f"HYBRID CYCLE {cycle + 1}/{num_cycles}")
            logger.info(f"{'='*60}\n")
            
            # ===================================================================
            # Phase 1: Generate physical predictions
            # ===================================================================
            logger.info("Phase 1: Generating physical predictions...")
            
            field_dataset = self._create_base_field_dataset()
            
            if self.augmentation_strategy == "on_the_fly":
                # Don't pre-generate, will generate during training
                generated_tensors = None
            else:
                # Pre-generate (either in-memory or cached)
                generated_tensors = self._generate_physical_predictions(field_dataset)
                
                if self.augmentation_strategy == "cache":
                    # Save to disk and clear from memory
                    cache_dir = self._cache_generated_tensors(generated_tensors, cycle)
                    generated_tensors = None  # Free memory
            
            # ===================================================================
            # Phase 2: Train synthetic model
            # ===================================================================
            logger.info("\nPhase 2: Training synthetic model...")
            
            real_tensor_dataset = self._create_base_tensor_dataset()
            
            if self.augmentation_strategy == "on_the_fly":
                # Use on-the-fly DataLoader
                augmented_loader = SyntheticAugmentedDataLoader(
                    real_dataset=real_tensor_dataset,
                    physical_model=self.model_registry.get_physical_model(),
                    converter=self.converter,
                    batch_size=self.config["trainer"]["batch_size"],
                    alpha=self.alpha,
                    dt=self.config["model"]["physical"]["dt"]
                )
            elif self.augmentation_strategy == "cache":
                # Use cached DataLoader
                augmented_dataset = CachedAugmentedDataset(
                    real_dataset=real_tensor_dataset,
                    generated_cache_dir=cache_dir,
                    alpha=self.alpha
                )
                augmented_loader = DataLoader(
                    augmented_dataset,
                    batch_size=self.config["trainer"]["batch_size"],
                    shuffle=True,
                    num_workers=self.config["trainer"]["num_workers"],
                    pin_memory=True
                )
            else:  # memory
                # Use in-memory DataLoader
                augmented_dataset = AugmentedTensorDataset(
                    real_dataset=real_tensor_dataset,
                    generated_data=generated_tensors,
                    alpha=self.alpha
                )
                augmented_loader = DataLoader(
                    augmented_dataset,
                    batch_size=self.config["trainer"]["batch_size"],
                    shuffle=True,
                    num_workers=self.config["trainer"]["num_workers"],
                    pin_memory=True
                )
            
            # Train synthetic model
            synthetic_metrics = self.synthetic_trainer.train(
                augmented_loader,
                num_epochs=self.synthetic_epochs_per_cycle
            )
            
            # ===================================================================
            # Phase 3: Generate synthetic predictions
            # ===================================================================
            logger.info("\nPhase 3: Generating synthetic predictions...")
            
            if self.augmentation_strategy == "on_the_fly":
                generated_fields = None
            else:
                generated_fields = self._generate_synthetic_predictions(real_tensor_dataset)
                
                if self.augmentation_strategy == "cache":
                    field_cache_dir = self._cache_generated_fields(generated_fields, cycle)
                    generated_fields = None
            
            # ===================================================================
            # Phase 4: Train physical model
            # ===================================================================
            logger.info("\nPhase 4: Training physical model...")
            
            if self.augmentation_strategy == "on_the_fly":
                # Use on-the-fly iterator
                augmented_source = PhysicalAugmentedDataLoader(
                    real_dataset=field_dataset,
                    synthetic_model=self.model_registry.get_synthetic_model(),
                    converter=self.converter,
                    batch_size=1,  # Physical trainer doesn't batch
                    alpha=self.alpha
                )
            elif self.augmentation_strategy == "cache":
                augmented_source = CachedAugmentedDataset(
                    real_dataset=field_dataset,
                    generated_cache_dir=field_cache_dir,
                    alpha=self.alpha
                )
            else:  # memory
                augmented_source = AugmentedFieldDataset(
                    real_dataset=field_dataset,
                    generated_data=generated_fields,
                    alpha=self.alpha
                )
            
            physical_metrics = self.physical_trainer.train(
                augmented_source,
                num_epochs=self.physical_epochs_per_cycle
            )
            
            # ===================================================================
            # Cleanup
            # ===================================================================
            if self.augmentation_strategy == "cache":
                import shutil
                shutil.rmtree(cache_dir, ignore_errors=True)
                shutil.rmtree(field_cache_dir, ignore_errors=True)
            
            logger.info(f"\nCycle {cycle + 1} Complete")
```

---

## Summary of Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Memory Management** | Adaptive strategy based on dataset size | Handles both small and large datasets efficiently |
| **Memory Budget** | Configurable via `memory_budget_gb` in config | User control over memory usage |
| **Cache Hierarchy** | Separate locations for real data, generated data, eval cache | Clean organization, easy cleanup |
| **Generation Frequency** | Once per hybrid epoch, reused for sub-trainer epochs | Balance between freshness and efficiency |
| **Generation Amount** | `num_generated = len(real_data) * alpha` (proportional) | Efficient: 10% samples for 10% weight |
| **Loss Weighting** | Weight applied ONCE in loss calculation, not in count | Avoid double-scaling |
| **Batch Mixing** | Random shuffling | Agreed upon |
| **DataManager** | Managed by HybridTrainer | Agreed upon |
| **Validation** | Real data only | Agreed upon |
| **Alpha Schedule** | Constant for now | Agreed upon |
| **Unified Access** | Both trainers use sliding window + same data interface | Consistency and flexibility |
| **Physical Sliding Window** | Always enabled (use_sliding_window=True) | Consistent with synthetic training |
| **Strategy Selection** | Small → memory, Medium → cache, Large → on-the-fly | Automatic adaptation |

---

## Next Steps

1. **Implement AdaptiveAugmentedDataLoader classes**
2. **Modify PhysicalTrainer to support sliding window**
3. **Update HybridTrainer with adaptive strategy**
4. **Test with small dataset first** (memory strategy)
5. **Test with larger dataset** (cache strategy)
6. **Profile performance** of each strategy
7. **Document memory budget configuration**

---

**Document Status:** Ready for implementation  
**Last Updated:** November 3, 2025
