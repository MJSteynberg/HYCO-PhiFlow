# Data Augmentation Strategy for Hybrid Training

**Document Version:** 1.0  
**Date:** November 3, 2025  
**Status:** Design Discussion

---

## Problem Statement

In hybrid training, we need to:
1. **Generate predictions** from both models (synthetic→physical, physical→synthetic)
2. **Combine real and generated data** with appropriate weighting
3. **Pass augmented data** to sub-trainers efficiently
4. **Track generated data** throughout training cycles

Key constraints:
- Trainers receive data via `train(data_source, num_epochs)` - NO internal data management
- Sub-trainers should remain agnostic to whether data is real or generated
- Need flexible weighting scheme (e.g., alpha for generated data)
- Must be memory efficient (don't duplicate large datasets)

---

## Current Infrastructure Analysis

### Existing DataLoader Pipeline

**SyntheticTrainer (Tensor-based):**
```python
# Current setup in _create_data_loaders():
train_dataset = HybridDataset(
    data_manager=data_manager,
    sim_indices=train_sim_indices,
    field_names=field_names,
    num_frames=num_frames,
    num_predict_steps=num_predict_steps,
    use_sliding_window=True,  # Multiple samples per simulation
    return_fields=False,      # Returns tensors
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
)

# Yields batches: (initial_state, rollout_targets)
# - initial_state: [B, C_all, H, W]
# - rollout_targets: [B, T, C_dynamic, H, W]
```

**PhysicalTrainer (Field-based):**
```python
# Current setup in _create_field_dataset():
field_dataset = HybridDataset(
    data_manager=data_manager,
    sim_indices=train_sim_indices,
    field_names=field_names,
    num_frames=num_frames,
    num_predict_steps=num_predict_steps,
    use_sliding_window=False,  # Single sample per simulation
    return_fields=True,         # Returns PhiFlow Fields
)

# Yields samples: (initial_fields, target_fields)
# - initial_fields: Dict[field_name, Field] at t=0
# - target_fields: Dict[field_name, List[Field]] for t=1..T
```

### Key Infrastructure Components

1. **DataManager**: Caches PhiFlow Scene data as tensors with metadata
2. **HybridDataset**: PyTorch Dataset wrapper with:
   - Lazy loading with LRU cache
   - Sliding window support for multiple samples per simulation
   - Dual mode: tensor or field output
3. **PyTorch DataLoader**: Handles batching, shuffling, multi-worker loading

---

## Design Options for Data Augmentation

### Option 1: Augmented Dataset Classes (RECOMMENDED)

**Concept:** Create specialized Dataset classes that wrap generated predictions alongside real data.

#### For Synthetic Training (Tensor-based):

```python
class AugmentedTensorDataset(Dataset):
    """
    Combines real data with generated predictions for synthetic training.
    
    Attributes:
        real_dataset: HybridDataset with return_fields=False
        generated_data: List of (input_tensor, target_tensor) tuples from physical model
        alpha: Weight for generated samples (real samples have weight 1.0)
    """
    
    def __init__(
        self,
        real_dataset: HybridDataset,
        generated_data: List[Tuple[torch.Tensor, torch.Tensor]],
        alpha: float = 0.1
    ):
        self.real_dataset = real_dataset
        self.generated_data = generated_data
        self.alpha = alpha
        
        # Total length is real + generated samples
        self.num_real = len(real_dataset)
        self.num_generated = len(generated_data)
        
    def __len__(self):
        return self.num_real + self.num_generated
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Returns: (input, target, weight)
        - weight = 1.0 for real data
        - weight = alpha for generated data
        """
        if idx < self.num_real:
            # Real data sample
            input_tensor, target_tensor = self.real_dataset[idx]
            weight = 1.0
        else:
            # Generated data sample
            gen_idx = idx - self.num_real
            input_tensor, target_tensor = self.generated_data[gen_idx]
            weight = self.alpha
            
        return input_tensor, target_tensor, weight
```

#### For Physical Training (Field-based):

```python
class AugmentedFieldDataset:
    """
    Combines real field data with generated predictions for physical training.
    
    Note: Not a PyTorch Dataset because PhysicalTrainer doesn't use DataLoader.
    Instead, provides an iterator interface.
    
    Attributes:
        real_dataset: HybridDataset with return_fields=True
        generated_data: List of (initial_fields, target_fields) tuples from synthetic model
        alpha: Weight for generated samples
    """
    
    def __init__(
        self,
        real_dataset: HybridDataset,
        generated_data: List[Tuple[Dict[str, Field], Dict[str, List[Field]]]],
        alpha: float = 0.1
    ):
        self.real_dataset = real_dataset
        self.generated_data = generated_data
        self.alpha = alpha
        
        self.num_real = len(real_dataset)
        self.num_generated = len(generated_data)
        
    def __len__(self):
        return self.num_real + self.num_generated
    
    def __iter__(self):
        """
        Yields: (initial_fields, target_fields, weight)
        """
        # Yield real samples
        for idx in range(self.num_real):
            initial_fields, target_fields = self.real_dataset[idx]
            yield initial_fields, target_fields, 1.0
            
        # Yield generated samples
        for initial_fields, target_fields in self.generated_data:
            yield initial_fields, target_fields, self.alpha
```

**Pros:**
- ✅ Clean separation of concerns
- ✅ Easy to understand and debug
- ✅ Flexible - can add more augmentation strategies
- ✅ Compatible with existing DataLoader/Dataset infrastructure
- ✅ Weights are part of the data contract (explicit)

**Cons:**
- ⚠️ Need to store generated data in memory (but only for one cycle)
- ⚠️ Requires two dataset classes (tensor vs field)

---

### Option 2: Dynamic Dataset with Generation Callbacks

**Concept:** Dataset that generates predictions on-the-fly during iteration.

```python
class DynamicAugmentedDataset(Dataset):
    """
    Dataset that generates augmented samples on-the-fly.
    """
    
    def __init__(
        self,
        real_dataset: HybridDataset,
        generator_model: nn.Module,  # Could be synthetic or physical model
        alpha: float = 0.1,
        num_generated_samples: int = 100
    ):
        self.real_dataset = real_dataset
        self.generator_model = generator_model
        self.alpha = alpha
        self.num_generated_samples = num_generated_samples
        
    def __getitem__(self, idx: int):
        if idx < len(self.real_dataset):
            # Real sample
            return (*self.real_dataset[idx], 1.0)
        else:
            # Generate on-the-fly
            gen_idx = idx - len(self.real_dataset)
            # ... generate from model ...
            return input_tensor, target_tensor, self.alpha
```

**Pros:**
- ✅ No need to store generated data in memory
- ✅ Can generate infinite augmented samples

**Cons:**
- ❌ Generation happens during training loop (slows down training)
- ❌ Models need to be passed to dataset (tight coupling)
- ❌ Hard to track what was generated (reproducibility issues)
- ❌ Complicates error handling

---

### Option 3: Data Iterator with Caching

**Concept:** Create iterators that yield mixed real/generated data with internal caching.

```python
def create_augmented_iterator(
    real_dataset: HybridDataset,
    generated_data: List[Tuple],
    alpha: float,
    batch_size: int
):
    """
    Creates an iterator that yields mixed batches.
    """
    # Combine all samples with weights
    all_samples = []
    
    # Add real samples
    for sample in real_dataset:
        all_samples.append((*sample, 1.0))
    
    # Add generated samples
    for sample in generated_data:
        all_samples.append((*sample, alpha))
    
    # Shuffle and batch
    random.shuffle(all_samples)
    
    for i in range(0, len(all_samples), batch_size):
        batch = all_samples[i:i+batch_size]
        yield collate_fn(batch)
```

**Pros:**
- ✅ Flexible control over batching and shuffling
- ✅ Simple implementation

**Cons:**
- ❌ Need to manually handle shuffling, batching, multi-worker loading
- ❌ Loses benefits of PyTorch DataLoader (pin_memory, prefetching, etc.)
- ❌ More code to maintain

---

## Recommended Approach: Option 1 (Augmented Dataset Classes)

**Rationale:**
1. **Clean separation**: Dataset handles data, trainer handles training
2. **Leverages existing infrastructure**: Works with PyTorch DataLoader
3. **Explicit weights**: Natural part of the data contract
4. **Memory efficient**: Generated data only stored for one cycle
5. **Easy to debug**: Can inspect generated data separately

---

## Implementation in HybridTrainer

### Storage of Generated Data

```python
class HybridTrainer(AbstractTrainer):
    """
    Manages interleaved training of synthetic and physical models.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model_registry: ModelRegistry,
        synthetic_trainer: SyntheticTrainer,
        physical_trainer: PhysicalTrainer,
        converter: BatchConcatenationConverter,
    ):
        super().__init__(config)
        
        self.model_registry = model_registry
        self.synthetic_trainer = synthetic_trainer
        self.physical_trainer = physical_trainer
        self.converter = converter
        
        # Training parameters
        self.alpha = config.get("trainer", {}).get("alpha", 0.1)
        self.num_cycles = config.get("trainer", {}).get("num_cycles", 10)
        self.synthetic_epochs_per_cycle = config.get("trainer", {}).get(
            "synthetic_epochs_per_cycle", 10
        )
        self.physical_epochs_per_cycle = config.get("trainer", {}).get(
            "physical_epochs_per_cycle", 5
        )
        
        # Storage for generated predictions (cleared each cycle)
        self.generated_tensors: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.generated_fields: List[Tuple[Dict[str, Field], Dict[str, List[Field]]]] = []
```

### Prediction Generation

```python
def _generate_physical_predictions(self, field_dataset: HybridDataset) -> List[Tuple]:
    """
    Use physical model to generate tensor predictions for synthetic training.
    
    Args:
        field_dataset: Dataset that yields (initial_fields, target_fields)
    
    Returns:
        List of (input_tensor, target_tensor) tuples
    """
    logger.info("Generating predictions from physical model...")
    
    physical_model = self.model_registry.get_physical_model()
    physical_model.eval()
    
    generated_tensors = []
    
    # Iterate through field dataset (no batching for physical model)
    for idx in tqdm(range(len(field_dataset)), desc="Physical predictions"):
        with torch.no_grad():
            # Get initial fields
            initial_fields, target_fields = field_dataset[idx]
            
            # Run physical simulation
            predicted_fields = physical_model.predict_trajectory(
                initial_fields,
                steps=len(target_fields[next(iter(target_fields))]),
                dt=self.config["model"]["physical"]["dt"]
            )
            
            # Convert fields to tensors for synthetic training
            # Initial state: all fields at t=0
            input_tensor = self.converter.fields_to_tensor(initial_fields)
            
            # Target: predicted trajectory (dynamic fields only)
            dynamic_fields = self.config["data"]["fields_scheme"]["dynamic"]
            target_tensors = []
            for t in range(len(predicted_fields[dynamic_fields[0]])):
                fields_at_t = {
                    field: predicted_fields[field][t] 
                    for field in dynamic_fields
                }
                tensor_at_t = self.converter.fields_to_tensor(fields_at_t)
                target_tensors.append(tensor_at_t)
            
            target_tensor = torch.stack(target_tensors, dim=0)  # [T, C, H, W]
            
            generated_tensors.append((input_tensor, target_tensor))
    
    logger.info(f"Generated {len(generated_tensors)} tensor samples from physical model")
    return generated_tensors


def _generate_synthetic_predictions(self, tensor_dataset: HybridDataset) -> List[Tuple]:
    """
    Use synthetic model to generate field predictions for physical training.
    
    Args:
        tensor_dataset: Dataset that yields (initial_state, rollout_targets) tensors
    
    Returns:
        List of (initial_fields, target_fields) tuples
    """
    logger.info("Generating predictions from synthetic model...")
    
    synthetic_model = self.model_registry.get_synthetic_model()
    synthetic_model.eval()
    
    # Need field metadata for conversion
    data_manager = self.config["data"]["data_manager"]  # Need to pass this in somehow
    sample_sim_idx = self.config["data"]["train_sim_indices"][0]
    cached_data = data_manager.load_from_cache(sample_sim_idx)
    field_metadata = cached_data["metadata"]["field_metadata"]
    
    generated_fields = []
    
    # Iterate through tensor dataset
    for idx in tqdm(range(len(tensor_dataset)), desc="Synthetic predictions"):
        with torch.no_grad():
            # Get initial state tensor
            initial_state, _ = tensor_dataset[idx]
            initial_state = initial_state.unsqueeze(0).to(synthetic_model.device)
            
            # Run autoregressive rollout
            predictions = []
            state = initial_state
            for step in range(self.config["data"]["num_predict_steps"]):
                pred = synthetic_model(state)
                predictions.append(pred)
                state = pred  # Use prediction as next input
            
            # Convert tensors to fields
            # Initial fields: convert initial_state to all fields
            initial_fields = self.converter.tensor_to_fields(
                initial_state[0],
                field_metadata,
                all_fields=self.config["data"]["fields_scheme"]["all"]
            )
            
            # Target fields: convert predictions to dynamic fields
            target_fields = {field: [] for field in self.config["data"]["fields_scheme"]["dynamic"]}
            for pred_tensor in predictions:
                fields_at_t = self.converter.tensor_to_fields(
                    pred_tensor[0],
                    field_metadata,
                    all_fields=self.config["data"]["fields_scheme"]["dynamic"]
                )
                for field_name, field_obj in fields_at_t.items():
                    target_fields[field_name].append(field_obj)
            
            generated_fields.append((initial_fields, target_fields))
    
    logger.info(f"Generated {len(generated_fields)} field samples from synthetic model")
    return generated_fields
```

### Training Loop Integration

```python
def train(self, base_data_manager, num_cycles: int):
    """
    Main hybrid training loop.
    
    Args:
        base_data_manager: DataManager for loading real data
        num_cycles: Number of hybrid training cycles
    """
    for cycle in range(num_cycles):
        logger.info(f"\n{'='*60}")
        logger.info(f"HYBRID TRAINING CYCLE {cycle + 1}/{num_cycles}")
        logger.info(f"{'='*60}\n")
        
        # ===================================================================
        # Phase 1: Generate predictions from physical model
        # ===================================================================
        logger.info("Phase 1: Generating physical predictions...")
        
        # Create base field dataset (for generation)
        field_dataset = HybridDataset(
            data_manager=base_data_manager,
            sim_indices=self.config["data"]["train_sim_indices"],
            field_names=self.config["data"]["fields_scheme"]["all"],
            num_frames=self.config["data"]["num_frames"],
            num_predict_steps=self.config["data"]["num_predict_steps"],
            use_sliding_window=False,  # One sample per sim for generation
            return_fields=True
        )
        
        self.generated_tensors = self._generate_physical_predictions(field_dataset)
        
        # ===================================================================
        # Phase 2: Train synthetic model with augmented data
        # ===================================================================
        logger.info("\nPhase 2: Training synthetic model...")
        
        # Create base tensor dataset (real data)
        real_tensor_dataset = HybridDataset(
            data_manager=base_data_manager,
            sim_indices=self.config["data"]["train_sim_indices"],
            field_names=self.config["data"]["fields_scheme"]["all"],
            num_frames=self.config["data"]["num_frames"],
            num_predict_steps=self.config["data"]["num_predict_steps"],
            use_sliding_window=True,  # Multiple samples for training
            return_fields=False
        )
        
        # Create augmented dataset
        augmented_tensor_dataset = AugmentedTensorDataset(
            real_dataset=real_tensor_dataset,
            generated_data=self.generated_tensors,
            alpha=self.alpha
        )
        
        # Create DataLoader
        augmented_loader = DataLoader(
            augmented_tensor_dataset,
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
        # Phase 3: Generate predictions from synthetic model
        # ===================================================================
        logger.info("\nPhase 3: Generating synthetic predictions...")
        
        # Use same real_tensor_dataset for generation
        self.generated_fields = self._generate_synthetic_predictions(real_tensor_dataset)
        
        # ===================================================================
        # Phase 4: Train physical model with augmented data
        # ===================================================================
        logger.info("\nPhase 4: Training physical model...")
        
        # Create augmented field dataset
        augmented_field_dataset = AugmentedFieldDataset(
            real_dataset=field_dataset,
            generated_data=self.generated_fields,
            alpha=self.alpha
        )
        
        # Train physical model (PhysicalTrainer expects an iterable, not DataLoader)
        physical_metrics = self.physical_trainer.train(
            augmented_field_dataset,
            num_epochs=self.physical_epochs_per_cycle
        )
        
        # ===================================================================
        # Clear generated data to free memory
        # ===================================================================
        self.generated_tensors.clear()
        self.generated_fields.clear()
        
        # Log cycle metrics
        logger.info(f"\nCycle {cycle + 1} Complete:")
        logger.info(f"  Synthetic - Loss: {synthetic_metrics['final_loss']:.6f}")
        logger.info(f"  Physical  - Loss: {physical_metrics['final_loss']:.6f}")
```

---

## Trainer Signature Modifications

### TensorTrainer Changes

```python
class TensorTrainer(AbstractTrainer):
    """Base class for tensor-based trainers."""
    
    def __init__(self, config: Dict[str, Any], model: nn.Module):
        """
        Args:
            config: Configuration dictionary
            model: Pre-created PyTorch model
        """
        # ... existing device setup ...
        self.model = model.to(self.device)
        self.optimizer = self._create_optimizer()
    
    def train(self, data_source: DataLoader, num_epochs: int) -> Dict[str, Any]:
        """
        Train for specified epochs using provided data.
        
        Args:
            data_source: PyTorch DataLoader yielding (input, target, weight) tuples
            num_epochs: Number of epochs to train
        
        Returns:
            Dictionary of training metrics
        """
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            self.model.train()
            for batch in data_source:
                # Unpack batch (now includes weight)
                input_batch, target_batch, weight_batch = batch
                
                # Standard training step with weighted loss
                self.optimizer.zero_grad()
                predictions = self.model(input_batch)
                
                # Compute per-sample loss
                per_sample_loss = self.loss_fn(predictions, target_batch)
                
                # Apply weights and aggregate
                weighted_loss = (per_sample_loss * weight_batch).mean()
                
                weighted_loss.backward()
                self.optimizer.step()
                
                epoch_loss += weighted_loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.6f}")
        
        return {"final_loss": avg_loss, "num_epochs": num_epochs}
```

### FieldTrainer Changes

```python
class FieldTrainer(AbstractTrainer):
    """Base class for field-based trainers."""
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        model: Any,  # PhiFlow model
        learnable_params: List[torch.nn.Parameter]
    ):
        """
        Args:
            config: Configuration dictionary
            model: Pre-created PhiFlow model
            learnable_params: List of learnable parameters to optimize
        """
        # ... existing setup ...
        self.model = model
        self.learnable_params = learnable_params
        self.optimizer = self._create_optimizer(learnable_params)
    
    def train(self, data_source: Iterable, num_epochs: int) -> Dict[str, Any]:
        """
        Train for specified epochs using provided data.
        
        Args:
            data_source: Iterable yielding (initial_fields, target_fields, weight) tuples
            num_epochs: Number of epochs to train
        
        Returns:
            Dictionary of training metrics
        """
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_samples = 0
            
            for initial_fields, target_fields, weight in data_source:
                # Standard training step with weighted loss
                self.optimizer.zero_grad()
                
                # Run simulation
                predicted_fields = self.model.predict_trajectory(
                    initial_fields,
                    steps=len(target_fields[next(iter(target_fields))]),
                    dt=self.config["model"]["physical"]["dt"]
                )
                
                # Compute loss
                loss = self._compute_field_loss(predicted_fields, target_fields)
                
                # Apply weight
                weighted_loss = loss * weight
                
                weighted_loss.backward()
                self.optimizer.step()
                
                epoch_loss += weighted_loss.item()
                num_samples += 1
            
            avg_loss = epoch_loss / num_samples
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.6f}")
        
        return {"final_loss": avg_loss, "num_epochs": num_epochs}
```

---

## Open Questions for Discussion

### 1. **Memory Management for Generated Data**

**Question:** Should we limit the number of generated samples per cycle?

**Options:**
- A) Generate from all real samples (could be 1000s if sliding window)
- B) Generate from subset (e.g., 100 samples per cycle)
- C) Generate proportional to alpha (e.g., if alpha=0.1, generate 10% as many samples as real)

**Recommendation:** Option C - keeps data balanced and memory usage predictable.

---

### 2. **Generation Frequency**

**Question:** Should we regenerate predictions every cycle, or reuse for multiple cycles?

**Options:**
- A) Regenerate every cycle (models evolving → better predictions)
- B) Reuse for N cycles (faster, but stale predictions)
- C) Hybrid: regenerate every K cycles

**Recommendation:** Option A - fresh predictions better reflect current model state.

---

### 3. **Batch Mixing Strategy**

**Question:** How should real and generated samples be mixed in batches?

**Options:**
- A) Random mixing (current approach - shuffle all together)
- B) Separate batches (some batches all-real, some all-generated)
- C) Fixed ratio per batch (e.g., 90% real, 10% generated)

**Recommendation:** Option A - simplest and most flexible via weighting.

---

### 4. **DataManager Passing**

**Question:** How should HybridTrainer access DataManager for field conversions?

**Options:**
- A) Pass DataManager to HybridTrainer.__init__()
- B) Pass metadata dict directly
- C) Access from config (if config stores DataManager reference)
- D) Create helper method in converter that takes config

**Recommendation:** Option A - cleanest dependency injection.

---

### 5. **Validation Data Augmentation**

**Question:** Should validation also use augmented data?

**Options:**
- A) No - validation on real data only (current plan)
- B) Yes - validation also augmented
- C) Separate validation metrics for real vs augmented

**Recommendation:** Option A - validation should measure real-world performance.

---

### 6. **Progressive Alpha Schedule**

**Question:** Should alpha change over training?

**Options:**
- A) Fixed alpha throughout training
- B) Increasing alpha (start low, increase as models improve)
- C) Decreasing alpha (start high for bootstrapping, decrease later)
- D) Curriculum: low → high → low

**Recommendation:** Start with Option A, add scheduling later if needed.

---

## Next Steps

1. **Resolve open questions** (discuss with team/advisor)
2. **Finalize AugmentedDataset implementations**
3. **Update TensorTrainer and FieldTrainer signatures**
4. **Implement HybridTrainer prediction generation methods**
5. **Test with simple case** (e.g., Burgers equation with small dataset)
6. **Profile memory usage** and optimize if needed
7. **Add logging/visualization** of generated vs real samples

---

## References

- [Hybrid Trainer Design](./HYBRID_TRAINER_DESIGN.md)
- [Field Conversion Documentation](./FIELD_CONVERSION.md)
- PyTorch Dataset/DataLoader documentation

---

**Document Status:** Awaiting feedback on open questions  
**Last Updated:** November 3, 2025
