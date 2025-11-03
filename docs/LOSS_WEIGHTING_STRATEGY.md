# Loss Weighting Strategy: Avoiding Double-Scaling

**Document Version:** 1.0  
**Date:** November 3, 2025  
**Status:** Critical Implementation Detail

---

## Problem Statement

When training with augmented data, we need to weight generated samples differently than real samples. However, we must be careful **not to double-scale** the contribution of generated data.

### Naive (Incorrect) Approach - Double Scaling ❌

```python
# WRONG: This double-scales the generated data contribution!

# Step 1: Generate 10% as many samples (alpha=0.1)
num_generated = int(len(real_data) * 0.1)  # e.g., 1000 real → 100 generated

# Step 2: Weight the loss
for sample in dataset:
    loss = compute_loss(model(input), target)
    weighted_loss = loss * weight  # weight=1.0 for real, 0.1 for generated
    
# Result: Generated data contributes (100 samples × 0.1 weight) = 10 effective samples
# This is 1% of real data, not 10%! We've scaled twice!
```

**Why this is wrong:**
- We already reduced the number of generated samples to 10%
- Then we further reduce their contribution by weighting with 0.1
- Total contribution: 10% × 10% = 1% (not intended!)

---

## Correct Approach: Single Scaling Point

We have **two valid strategies**. Choose ONE, not both:

### Strategy A: Count-based (Proportional Samples, Equal Weight) ✅

Generate proportional number of samples, give them equal weight.

```python
# Generate proportional samples
num_real = len(real_dataset)
num_generated = int(num_real * alpha)  # alpha=0.1 → 10% as many samples

# All samples have equal weight
for sample in combined_dataset:
    loss = compute_loss(model(input), target)
    # NO weighting - all samples contribute equally
    total_loss += loss

avg_loss = total_loss / (num_real + num_generated)

# Contribution:
# - Real: 1000 samples × 1.0 weight = 1000 effective
# - Generated: 100 samples × 1.0 weight = 100 effective
# - Ratio: 100/1000 = 10% ✓
```

**Pros:**
- Simple implementation
- No special loss weighting logic needed
- Easy to understand

**Cons:**
- For very small alpha (e.g., 0.01), might generate too few samples to matter
- Requires generating different number of samples per cycle

---

### Strategy B: Weight-based (Equal Samples, Proportional Weight) ✅

Generate equal number of samples, weight them proportionally.

```python
# Generate equal number of samples
num_real = len(real_dataset)
num_generated = num_real  # Same number

# Weight samples by alpha
for sample in combined_dataset:
    input, target, is_generated = sample
    loss = compute_loss(model(input), target)
    
    weight = alpha if is_generated else 1.0
    weighted_loss = loss * weight
    total_loss += weighted_loss

# Contribution:
# - Real: 1000 samples × 1.0 weight = 1000 effective
# - Generated: 1000 samples × 0.1 weight = 100 effective
# - Ratio: 100/1000 = 10% ✓
```

**Pros:**
- Always have meaningful number of generated samples
- Can use batch-level weighting naturally
- More fine-grained control

**Cons:**
- More complex implementation
- Need to track weights through data pipeline
- Memory usage higher (more samples generated)

---

## Chosen Strategy: Count-based (Strategy A)

**Rationale:**
- You specified: "I think A would be the most efficient"
- Less memory (fewer generated samples)
- Simpler implementation (no weight tracking in loss)
- Natural interpretation: 10% generated samples = 10% contribution

---

## Implementation Details

### Dataset Implementation (No Weights in Output)

```python
class AugmentedTensorDataset(Dataset):
    """
    Augmented dataset with proportional sampling (Strategy A).
    
    NO weights in output - all samples treated equally!
    """
    
    def __init__(
        self,
        real_dataset: HybridDataset,
        generated_data: List[Tuple[torch.Tensor, torch.Tensor]],
        alpha: float = 0.1
    ):
        self.real_dataset = real_dataset
        self.alpha = alpha
        
        # Calculate how many generated samples to use
        # This is where we apply alpha - in the COUNT
        self.num_real = len(real_dataset)
        self.num_generated_to_use = int(self.num_real * alpha)
        
        # Subsample generated data if we have more than needed
        if len(generated_data) > self.num_generated_to_use:
            import random
            indices = random.sample(range(len(generated_data)), self.num_generated_to_use)
            self.generated_data = [generated_data[i] for i in indices]
        else:
            self.generated_data = generated_data
            self.num_generated_to_use = len(generated_data)
        
        logger.info(f"Augmented dataset: {self.num_real} real + {self.num_generated_to_use} generated")
        logger.info(f"Generated contribution: {self.num_generated_to_use / self.num_real * 100:.1f}%")
        
    def __len__(self):
        return self.num_real + self.num_generated_to_use
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (input, target) - NO WEIGHT!
        
        All samples are treated equally in the loss.
        """
        if idx < self.num_real:
            # Real data sample
            return self.real_dataset[idx]
        else:
            # Generated data sample
            gen_idx = idx - self.num_real
            return self.generated_data[gen_idx]
```

### Trainer Loss Computation (No Weighting)

```python
class TensorTrainer(AbstractTrainer):
    """Tensor trainer with standard loss (no per-sample weighting)."""
    
    def train(self, data_source: DataLoader, num_epochs: int) -> Dict[str, Any]:
        """
        Train with standard loss calculation.
        
        Args:
            data_source: DataLoader yielding (input, target) tuples
                        NO weights - all samples treated equally!
        """
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            self.model.train()
            for batch in data_source:
                # Unpack batch (2-tuple, no weights)
                input_batch, target_batch = batch
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                
                # Standard training step
                self.optimizer.zero_grad()
                predictions = self.model(input_batch)
                
                # Compute loss (standard, no weighting)
                loss = self.loss_fn(predictions, target_batch)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.6f}")
        
        return {"final_loss": avg_loss, "num_epochs": num_epochs}


class FieldTrainer(AbstractTrainer):
    """Field trainer with standard loss (no per-sample weighting)."""
    
    def train(self, data_source: Iterable, num_epochs: int) -> Dict[str, Any]:
        """
        Train with standard loss calculation.
        
        Args:
            data_source: Iterable yielding (initial_fields, target_fields) tuples
                        NO weights - all samples treated equally!
        """
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_samples = 0
            
            for item in data_source:
                # Unpack (2-tuple, no weights)
                initial_fields, target_fields = item
                
                # Train on sample
                self.optimizer.zero_grad()
                
                predicted_fields = self.model.predict_trajectory(
                    initial_fields,
                    steps=len(target_fields[next(iter(target_fields))]),
                    dt=self.config["model"]["physical"]["dt"]
                )
                
                loss = self._compute_field_loss(predicted_fields, target_fields)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_samples += 1
            
            avg_loss = epoch_loss / num_samples
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")
        
        return {"final_loss": avg_loss, "num_epochs": num_epochs}
```

### Generation Logic (Proportional Count)

```python
def _generate_physical_predictions(
    self, 
    field_dataset: HybridDataset
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate tensor predictions from physical model.
    
    Generates proportional number based on alpha.
    
    Args:
        field_dataset: Dataset with field data
        
    Returns:
        List of (input_tensor, target_tensor) tuples
        Length = int(len(field_dataset) * self.alpha)
    """
    logger.info("Generating predictions from physical model...")
    
    physical_model = self.model_registry.get_physical_model()
    physical_model.eval()
    
    # Calculate how many to generate (THIS IS WHERE WE APPLY ALPHA)
    num_to_generate = int(len(field_dataset) * self.alpha)
    
    logger.info(f"  Real samples: {len(field_dataset)}")
    logger.info(f"  Generating: {num_to_generate} samples ({self.alpha*100:.1f}%)")
    
    # Sample random indices
    indices = torch.randperm(len(field_dataset))[:num_to_generate]
    
    generated_tensors = []
    
    for idx in tqdm(indices, desc="Physical predictions"):
        with torch.no_grad():
            # Get initial fields
            initial_fields, _ = field_dataset[idx]
            
            # Run physical simulation
            predicted_fields = physical_model.predict_trajectory(
                initial_fields,
                steps=field_dataset.num_predict_steps,
                dt=self.config["model"]["physical"]["dt"]
            )
            
            # Convert to tensors
            input_tensor = self.converter.fields_to_tensor(initial_fields)
            target_tensor = self._convert_trajectory_to_tensor(predicted_fields)
            
            generated_tensors.append((input_tensor, target_tensor))
    
    logger.info(f"Generated {len(generated_tensors)} tensor samples")
    return generated_tensors
```

---

## Validation: Checking Effective Contribution

```python
def validate_augmentation_balance(
    real_dataset: HybridDataset,
    generated_data: List,
    alpha: float
):
    """
    Validate that generated data contributes the expected proportion.
    
    Args:
        real_dataset: Real data dataset
        generated_data: List of generated samples
        alpha: Expected proportion
    """
    num_real = len(real_dataset)
    num_generated = len(generated_data)
    
    actual_ratio = num_generated / num_real
    expected_ratio = alpha
    
    logger.info("\n" + "="*60)
    logger.info("AUGMENTATION BALANCE VALIDATION")
    logger.info("="*60)
    logger.info(f"Real samples:      {num_real}")
    logger.info(f"Generated samples: {num_generated}")
    logger.info(f"Actual ratio:      {actual_ratio:.4f} ({actual_ratio*100:.2f}%)")
    logger.info(f"Expected ratio:    {expected_ratio:.4f} ({expected_ratio*100:.2f}%)")
    
    if abs(actual_ratio - expected_ratio) < 0.01:
        logger.info("✓ Balance is correct!")
    else:
        logger.warning(f"⚠ Balance mismatch: {abs(actual_ratio - expected_ratio)*100:.2f}% off")
    
    logger.info("="*60 + "\n")
    
    # Sanity check: NO weights should be present in dataset
    sample = AugmentedTensorDataset(real_dataset, generated_data, alpha)[0]
    if len(sample) == 2:
        logger.info("✓ Dataset returns 2-tuple (no weights) - correct!")
    elif len(sample) == 3:
        logger.error("✗ Dataset returns 3-tuple (with weights) - DOUBLE SCALING RISK!")
        raise ValueError("Dataset should not include weights with count-based strategy!")
```

---

## Example: Full Pipeline with Validation

```python
def train_with_augmentation(self, cycle: int):
    """
    Training cycle with augmentation balance validation.
    """
    # Generate predictions
    field_dataset = self._create_base_field_dataset()
    generated_tensors = self._generate_physical_predictions(field_dataset)
    
    # Validate balance BEFORE training
    validate_augmentation_balance(
        real_dataset=field_dataset,
        generated_data=generated_tensors,
        alpha=self.alpha
    )
    
    # Create augmented dataset (no weights)
    real_tensor_dataset = self._create_base_tensor_dataset()
    augmented_dataset = AugmentedTensorDataset(
        real_dataset=real_tensor_dataset,
        generated_data=generated_tensors,
        alpha=self.alpha  # Only used to subsample if needed
    )
    
    # Create DataLoader
    augmented_loader = DataLoader(
        augmented_dataset,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=self.num_workers,
        pin_memory=True
    )
    
    # Train (standard loss, no weighting)
    metrics = self.synthetic_trainer.train(
        augmented_loader,
        num_epochs=self.synthetic_epochs_per_cycle
    )
    
    return metrics
```

---

## Summary

### Key Points

1. **Single scaling point**: Apply alpha ONLY in sample count, NOT in loss weight
2. **No weights in dataset**: `__getitem__` returns 2-tuple `(input, target)`
3. **Proportional generation**: Generate `int(len(real_data) * alpha)` samples
4. **Equal treatment**: All samples contribute equally to loss
5. **Validation**: Always check that actual ratio matches expected alpha

### Mathematical Verification

For alpha = 0.1 (10% contribution from generated data):

```
Real samples:      1000
Generated samples: 100  (= 1000 × 0.1)

Loss contribution:
- Real:      1000 samples × 1.0 weight = 1000 effective
- Generated: 100 samples × 1.0 weight = 100 effective

Ratio: 100 / 1000 = 0.1 = 10% ✓

Total loss = (sum of real losses + sum of generated losses) / 1100
```

No double scaling! ✓

---

**Document Status:** Critical - Must follow for correct training  
**Last Updated:** November 3, 2025
