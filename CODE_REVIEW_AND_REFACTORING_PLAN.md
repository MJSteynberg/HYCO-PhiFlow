# Code Review and Refactoring Plan for HYCO-PhiFlow

**Date:** October 31, 2025  
**Reviewer:** GitHub Copilot  
**Branch:** feature/improved-modularity

---

## Executive Summary

This document provides a comprehensive code review of the HYCO-PhiFlow codebase with specific focus on:
1. **Trainer Architecture Issues** - Physical and synthetic trainers have fundamentally different requirements but share a common base class
2. **Interleaved Training Design** - Architecture for training physical and synthetic models together
3. **Backward Compatibility Cleanup** - Removing redundant code while maintaining current functionality

### Key Design Principle for Hybrid Training

**Important Clarification:** The HYCO (Hybrid Co-training) trainer uses **predictions from one model as training data for the other model**, NOT as additional terms in the loss function.

- Physical model trains to match synthetic model predictions (synthetic predictions are the "ground truth")
- Synthetic model trains to match physical model predictions (physical predictions are the "ground truth")
- Ground truth data is only used for initial conditions and generating predictions
- The HYCOTrainer **orchestrates** this process but **does not implement training logic itself**
- It delegates all actual training to existing FieldTrainer and TensorTrainer implementations
- HYCOTrainer inherits from a generic **HybridTrainer** base class for extensibility

---

## 1. Current Architecture Analysis

### 1.1 Trainer Hierarchy Issues

#### Current Structure
```
BaseTrainer (Abstract)
├── Device management (CPU/GPU)
├── Model checkpoint saving/loading  [PyTorch-specific]
├── Model parameter counting         [PyTorch-specific]
├── Abstract methods:
    ├── _create_model()
    ├── _create_data_loader()
    ├── _train_epoch()
    └── train()

PhysicalTrainer(BaseTrainer)
├── Uses PhiFlow Fields
├── Uses math.minimize optimizer
├── No epochs (single optimization run)
├── No DataLoader (loads entire sim at once)
├── Stub implementations for abstract methods

SyntheticTrainer(BaseTrainer)
├── Uses PyTorch tensors
├── Uses Adam optimizer
├── Epoch-based training
├── Uses PyTorch DataLoader
├── Full implementations of abstract methods
```

#### Problems Identified

**Problem 1: Forced Interface Incompatibility**
- `BaseTrainer` assumes PyTorch models with `.parameters()`, `.state_dict()`, etc.
- `PhysicalTrainer` doesn't have a PyTorch model, has stub methods for `_train_epoch()`
- The base class methods `get_parameter_count()`, `get_trainable_parameter_count()`, `move_model_to_device()` are meaningless for physical models

**Problem 2: Semantic Mismatch**
- `_train_epoch()` has no meaning for physical trainer (uses math.minimize, not epochs)
- `_create_data_loader()` returns None for physical trainer
- `checkpoint_path` uses PyTorch's `.pth` format, but physical models don't save PyTorch state dicts

**Problem 3: Different Optimization Paradigms**
```python
# Synthetic: Iterative gradient descent
for epoch in range(epochs):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()

# Physical: Black-box optimization
def loss_fn(*params):
    return simulate_and_compare()

optimized_params = math.minimize(loss_fn, initial_params)
```

These are fundamentally different patterns that don't fit into a common interface.

---

### 1.2 Code Quality Assessment

#### Strengths
✅ **Excellent modularity** - Registry pattern for models, factory pattern for trainers  
✅ **Clear separation** - Data management, models, training, and evaluation are well-separated  
✅ **Good documentation** - Comprehensive docstrings and comments  
✅ **Type hints** - Good use of type annotations  
✅ **Modern practices** - Uses Hydra for configuration, dataclasses, etc.

#### Areas for Improvement
⚠️ **Trainer hierarchy** - Forces incompatible interfaces together  
⚠️ **Backward compatibility code** - Some fallback logic in data_manager.py and hybrid_dataset.py for old cache formats  
⚠️ **Physical model validation** - No validation logic (commented out)  

---

## 2. Recommended Refactoring: Trainer Architecture

### 2.1 Proposed Architecture

Create **separate base classes** for fundamentally different trainer types:

```
AbstractTrainer (Minimal common interface)
├── config: Dict[str, Any]
├── train() -> Any
└── get_results() -> Dict[str, Any]

TensorTrainer(AbstractTrainer)
├── device: torch.device
├── model: nn.Module
├── optimizer: torch.optim.Optimizer
├── dataloader: DataLoader
├── _train_epoch() -> float
├── save_checkpoint()
├── load_checkpoint()
└── train() -> Dict[str, Any]

FieldTrainer(AbstractTrainer)
├── data_manager: DataManager
├── model: PhysicalModel
├── learnable_params: List[Tensor]
├── _setup_optimization() -> math.Solve
├── _load_ground_truth() -> Dict[str, Field]
└── train() -> Dict[str, Any]

HybridTrainer(AbstractTrainer)
├── tensor_trainer: TensorTrainer
├── field_trainer: FieldTrainer
├── converter: FieldTensorConverter
├── data_manager: DataManager
├── _generate_synthetic_predictions() -> Dict[str, Field]
├── _generate_physical_predictions() -> Dict[str, Field]
├── _load_ground_truth() -> Dict[str, Field]
└── train() -> Dict[str, Any] (abstract - subclasses define strategy)

HYCOTrainer(HybridTrainer)
├── Implements specific co-training strategy
├── _train_physical_with_synthetic_data()
├── _train_synthetic_with_physical_data()
└── train() -> Dict[str, Any] (interleaved co-training)
```

### 2.2 Benefits of This Approach

1. **No forced interfaces** - Each base class only includes what its subclasses actually need
2. **Clear semantics** - TensorTrainer has epochs, FieldTrainer has optimization runs
3. **Future extensibility** - Easy to add new trainer types (e.g., JAX-based, hybrid)
4. **Interleaved training support** - InterleavedTrainer can orchestrate both types

### 2.3 Migration Path

**Phase 1: Extract Common Interface**
```python
# src/training/abstract_trainer.py
class AbstractTrainer(ABC):
    """Minimal interface all trainers must implement."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project_root = config.get('project_root', '.')
    
    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Execute training and return results."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration used for this trainer."""
        return self.config
```

**Phase 2: Create TensorTrainer**
```python
# src/training/tensor_trainer.py
class TensorTrainer(AbstractTrainer):
    """Base class for PyTorch tensor-based trainers."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.dataloader: Optional[DataLoader] = None
    
    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Create and return the PyTorch model."""
        pass
    
    @abstractmethod
    def _create_dataloader(self) -> DataLoader:
        """Create and return the PyTorch DataLoader."""
        pass
    
    @abstractmethod
    def _train_epoch(self) -> float:
        """Train for one epoch, return average loss."""
        pass
    
    def train(self) -> Dict[str, Any]:
        """Execute epoch-based training loop."""
        # Common epoch-based training logic here
        pass
    
    # Include all current BaseTrainer methods that are PyTorch-specific
    def save_checkpoint(self, epoch, loss, ...):
        """Save PyTorch model checkpoint."""
        pass
    
    def load_checkpoint(self, path):
        """Load PyTorch model checkpoint."""
        pass
```

**Phase 3: Create FieldTrainer**
```python
# src/training/field_trainer.py
class FieldTrainer(AbstractTrainer):
    """Base class for PhiFlow field-based trainers."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_manager: Optional[DataManager] = None
        self.model: Optional[PhysicalModel] = None
        self.learnable_params: List[Tensor] = []
    
    @abstractmethod
    def _create_data_manager(self) -> DataManager:
        """Create and return DataManager."""
        pass
    
    @abstractmethod
    def _create_model(self) -> PhysicalModel:
        """Create and return physical model."""
        pass
    
    @abstractmethod
    def _setup_optimization(self) -> math.Solve:
        """Setup optimization parameters."""
        pass
    
    def train(self) -> Dict[str, Any]:
        """Execute optimization-based training."""
        # Common optimization logic here
        pass
    
    def save_results(self, path: Path, results: Dict[str, Any]):
        """Save optimization results (not a PyTorch checkpoint)."""
        pass
```

**Phase 4: Update Existing Trainers**
```python
# src/training/synthetic/trainer.py
class SyntheticTrainer(TensorTrainer):  # Changed from BaseTrainer
    """Tensor-based trainer for synthetic models."""
    # Current implementation mostly stays the same
    # But now inherits from TensorTrainer instead

# src/training/physical/trainer.py
class PhysicalTrainer(FieldTrainer):  # Changed from BaseTrainer
    """Field-based trainer for physical models."""
    # Remove stub methods
    # Cleaner implementation without forced interfaces
```

---

## 3. Interleaved Training Design

### 3.1 Architecture Overview

```python
class InterleavedTrainer(AbstractTrainer):
    """
    Trainer for interleaved physical-synthetic model training.
    
    Training Loop:
    Each epoch:
        1. Train physical model using: ground_truth + synthetic_predictions
        2. Train synthetic model using: ground_truth + physical_predictions
    
    This creates a co-training scenario where both models improve together.
    """
```

### 3.2 Detailed Design

#### Key Components

**A. Field-Tensor Converter**
```python
# src/utils/field_tensor_converter.py
class FieldTensorConverter:
    """
    Bidirectional converter between PhiFlow Fields and PyTorch tensors.
    
    This is the bridge between physical (Field-based) and synthetic 
    (tensor-based) models.
    """
    
    def __init__(self, field_metadata: Dict[str, FieldMetadata]):
        self.field_metadata = field_metadata
    
    def fields_to_tensors_batch(
        self, 
        fields: Dict[str, Field]
    ) -> torch.Tensor:
        """
        Convert dict of Fields to concatenated tensor for synthetic model.
        
        Returns:
            Tensor of shape [B, C, H, W] ready for UNet input
        """
        tensors = []
        for field_name in self.field_metadata.keys():
            tensor = field_to_tensor(fields[field_name], ensure_cpu=False)
            tensors.append(tensor)
        return torch.cat(tensors, dim=1)  # Concatenate on channel dim
    
    def tensors_to_fields_batch(
        self, 
        tensor: torch.Tensor
    ) -> Dict[str, Field]:
        """
        Convert concatenated tensor back to dict of Fields.
        
        Args:
            tensor: Tensor of shape [B, C, H, W] from UNet output
            
        Returns:
            Dict mapping field names to Field objects
        """
        fields = {}
        offset = 0
        for field_name, metadata in self.field_metadata.items():
            num_channels = metadata.channel_dims[0] if metadata.channel_dims else 1
            field_tensor = tensor[:, offset:offset+num_channels, :, :]
            fields[field_name] = tensor_to_field(field_tensor, metadata)
            offset += num_channels
        return fields
```

**B. Hybrid Trainer Base Class**
```python
# src/training/hybrid/base_trainer.py
class HybridTrainer(AbstractTrainer):
    """
    Abstract base class for hybrid physical-synthetic training strategies.
    
    This class provides common functionality for coordinating training
    between physical (Field-based) and synthetic (tensor-based) models.
    
    Subclasses implement specific hybrid training strategies by overriding
    the train() method.
    
    Common functionality:
    - Manages both TensorTrainer and FieldTrainer instances
    - Handles Field ↔ Tensor conversions
    - Generates predictions from both models
    - Loads and manages ground truth data
    
    Subclasses define:
    - Specific training orchestration strategy
    - How and when to exchange data between models
    - Custom data preparation or augmentation
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Create trainers for both model types
        self.physical_trainer = self._create_physical_trainer(config)
        self.synthetic_trainer = self._create_synthetic_trainer(config)
        
        # Converter for Field ↔ Tensor transformations
        self.converter = self._create_converter()
        
        # Data management (shared by both trainers)
        self.data_manager = DataManager(
            raw_data_dir=str(Path(config['project_root']) / config['data']['data_dir'] / config['data']['dset_name']),
            cache_dir=str(Path(config['project_root']) / config['data']['data_dir'] / 'cache'),
            config=config,
            validate_cache=config['data'].get('validate_cache', True),
            auto_clear_invalid=config['data'].get('auto_clear_invalid', False)
        )
        
        # Training parameters
        self.num_epochs = config['trainer_params']['epochs']
        self.num_predict_steps = config['trainer_params']['num_predict_steps']
        self.train_sim = config['trainer_params']['train_sim']
    
    def _create_physical_trainer(self, config: Dict[str, Any]) -> FieldTrainer:
        """Create physical trainer instance."""
        from src.training.physical.trainer import PhysicalTrainer
        return PhysicalTrainer(config)
    
    def _create_synthetic_trainer(self, config: Dict[str, Any]) -> TensorTrainer:
        """Create synthetic trainer instance."""
        from src.training.synthetic.trainer import SyntheticTrainer
        return SyntheticTrainer(config)
    
    def _create_converter(self) -> FieldTensorConverter:
        """Create field-tensor converter with proper metadata."""
        field_names = self.config['data']['fields']
        metadata = create_field_metadata_from_model(
            self.physical_trainer.model,
            field_names,
            field_types={'velocity': 'staggered'}
        )
        return FieldTensorConverter(metadata)
    
    def _generate_synthetic_predictions(
        self, 
        gt_data: Dict[str, Field]
    ) -> Dict[str, Field]:
        """
        Generate predictions from synthetic model.
        
        Args:
            gt_data: Ground truth data with initial conditions
            
        Returns:
            Dictionary of Fields containing synthetic model predictions
        """
        self.synthetic_trainer.model.eval()
        
        # Get initial state
        initial_state = {name: field.time[0] for name, field in gt_data.items()}
        
        # Convert to tensor
        current_tensor = self.converter.fields_to_tensors_batch(initial_state)
        current_tensor = current_tensor.to(self.synthetic_trainer.device)
        
        # Generate predictions autoregressively
        predictions = {name: [initial_state[name]] for name in initial_state.keys()}
        
        with torch.no_grad():
            for t in range(self.num_predict_steps):
                pred_tensor = self.synthetic_trainer.model(current_tensor)
                pred_fields = self.converter.tensors_to_fields_batch(pred_tensor)
                
                for name, field in pred_fields.items():
                    predictions[name].append(field)
                
                current_tensor = pred_tensor
        
        # Stack predictions along time dimension
        stacked_predictions = {}
        for name, fields in predictions.items():
            stacked_predictions[name] = stack(fields, batch('time'))
        
        return stacked_predictions
    
    def _generate_physical_predictions(
        self,
        gt_data: Dict[str, Field]
    ) -> Dict[str, Field]:
        """
        Generate predictions from physical model.
        
        Args:
            gt_data: Ground truth data with initial conditions
            
        Returns:
            Dictionary of Fields containing physical model predictions
        """
        # Get initial state
        initial_state = {name: field.time[0] for name, field in gt_data.items()}
        
        # Run physical simulation
        current_state = initial_state
        predictions = {name: [initial_state[name]] for name in initial_state.keys()}
        
        for t in range(self.num_predict_steps):
            current_state = self.physical_trainer.model.step(current_state)
            for name, field in current_state.items():
                predictions[name].append(field)
        
        # Stack predictions along time dimension
        stacked_predictions = {}
        for name, fields in predictions.items():
            stacked_predictions[name] = stack(fields, batch('time'))
        
        return stacked_predictions
    
    def _load_ground_truth(self) -> Dict[str, Field]:
        """
        Load ground truth data for training.
        
        Returns:
            Dictionary of Fields with time dimension
        """
        sim_idx = self.train_sim[0]  # For now, use first simulation
        
        # Load from cache
        cached_data = self.data_manager.get_or_load_simulation(
            sim_idx,
            field_names=self.config['data']['fields'],
            num_frames=self.num_predict_steps + 1
        )
        
        # Convert tensors to fields
        from src.utils.field_conversion import tensors_to_fields, FieldMetadata
        from phi.geom import Box
        
        field_metadata = {}
        for name, meta in cached_data['metadata']['field_metadata'].items():
            # Reconstruct metadata
            bounds_lower = meta['bounds_lower']
            bounds_upper = meta['bounds_upper']
            if len(bounds_lower) == 2:
                domain = Box(x=(bounds_lower[0], bounds_upper[0]), y=(bounds_lower[1], bounds_upper[1]))
            
            resolution_sizes = {dim: cached_data['tensor_data'][name].shape[i+2] 
                               for i, dim in enumerate(meta['spatial_dims'])}
            resolution = spatial(**resolution_sizes)
            
            field_metadata[name] = FieldMetadata.from_cache_metadata(meta, domain, resolution)
        
        # Convert all timesteps
        gt_fields = {}
        for name, tensor in cached_data['tensor_data'].items():
            # tensor shape: [time, channels, x, y]
            fields_list = []
            for t in range(tensor.shape[0]):
                tensor_t = tensor[t:t+1]
                if isinstance(tensor_t, torch.Tensor):
                    tensor_t = tensor_t.cuda() if torch.cuda.is_available() else tensor_t
                field_t = tensor_to_field(tensor_t, field_metadata[name], time_slice=0)
                fields_list.append(field_t)
            
            gt_fields[name] = stack(fields_list, batch('time'))
        
        return gt_fields
    
    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Execute hybrid training strategy.
        
        Subclasses must implement this to define their specific
        hybrid training orchestration strategy.
        """
        pass


# src/training/hybrid/hyco_trainer.py
class HYCOTrainer(HybridTrainer):
    """
    HYCO: Hybrid Co-training Trainer
    
    Implements interleaved co-training strategy where physical and synthetic
    models train on each other's predictions.
    
    Training strategy:
        1. Load ground truth data
        2. For each epoch:
            a. Generate synthetic predictions → Pass to physical trainer as data
            b. Physical trainer trains using synthetic predictions
            c. Generate physical predictions → Pass to synthetic trainer as data
            d. Synthetic trainer trains using physical predictions
    
    Key principle: This class does NOT implement training logic.
    It only orchestrates the sequence by:
        - Generating predictions from one model
        - Passing data to existing trainers
        - Managing the training cycle
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # All initialization handled by HybridTrainer base class
    
    def _create_converter(self) -> FieldTensorConverter:
        """Create field-tensor converter with proper metadata."""
        # Get field metadata from physical model
        field_names = self.config['data']['fields']
        metadata = create_field_metadata_from_model(
            self.physical_model,
            field_names,
            field_types={'velocity': 'staggered'}  # Configure as needed
        )
        return FieldTensorConverter(metadata)
    
    def train(self) -> Dict[str, Any]:
        """
        Execute HYCO interleaved co-training strategy.
        
        This method coordinates training but delegates actual training
        to the physical_trainer and synthetic_trainer.
        """
        
        # Load ground truth data (needed for generating predictions)
        gt_data = self._load_ground_truth()
        
        results = {
            'physical_losses': [],
            'synthetic_losses': [],
            'epochs': []
        }
        
        for epoch in range(self.num_epochs):
            print(f"\n=== HYCO Training Epoch {epoch+1}/{self.num_epochs} ===")
            
            # Step 1: Generate synthetic predictions to use as physical training data
            print("  → Generating synthetic predictions...")
            synthetic_predictions = self._generate_synthetic_predictions(gt_data)
            
            # Step 2: Train physical model using synthetic predictions as data
            print("  → Training physical model on synthetic predictions...")
            physical_result = self._train_physical_with_data(synthetic_predictions, gt_data)
            results['physical_losses'].append(physical_result['loss'])
            
            # Step 3: Generate physical predictions to use as synthetic training data
            print("  → Generating physical predictions...")
            physical_predictions = self._generate_physical_predictions(gt_data)
            
            # Step 4: Train synthetic model using physical predictions as data
            print("  → Training synthetic model on physical predictions...")
            synthetic_result = self._train_synthetic_with_data(physical_predictions, gt_data)
            results['synthetic_losses'].append(synthetic_result['loss'])
            
            results['epochs'].append(epoch)
            
            print(f"  Physical Loss: {physical_result['loss']:.6f}, "
                  f"Synthetic Loss: {synthetic_result['loss']:.6f}")
            
            # Save checkpoints periodically
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, results)
        
        return results
    
    # Note: _generate_synthetic_predictions, _generate_physical_predictions,
    # _train_physical_with_data, _train_synthetic_with_data, and other
    # helper methods are inherited from HybridTrainer or defined below
    
    def _train_physical_with_data(
        self, 
        gt_data: Dict[str, Field]
    ) -> Dict[str, Any]:
        """
        Train physical model using synthetic predictions as target data.
        self,
        synthetic_data: Dict[str, Field],
        gt_data: Dict[str, Field]
    ) -> Dict[str, Any]:
        """
        Train physical model using synthetic predictions as target data.
        
        This method delegates to the physical trainer's existing train() method
        but provides synthetic predictions as the training target.
        
        Args:
            synthetic_data: Predictions from synthetic model (used as training target)
            gt_data: Ground truth data (for initial conditions)
            
        Returns:
            Dictionary with training results from physical trainer
        """
        # Temporarily replace the physical trainer's ground truth target
        # with synthetic predictions
        original_gt = self.physical_trainer._load_ground_truth_data
        
        def load_synthetic_as_gt(sim_index):
            # Return synthetic predictions as if they were ground truth
            return synthetic_data
        
        # Monkey patch (or use a cleaner injection pattern)
        self.physical_trainer._load_ground_truth_data = load_synthetic_as_gt
        
        try:
            # Let the physical trainer do its thing with synthetic data
            self.physical_trainer.train()
            
            # Extract results
            result = {
                'loss': self.physical_trainer.final_loss if hasattr(self.physical_trainer, 'final_loss') else 0.0,
                'optimized_params': self.physical_trainer.model.__dict__
            }
        finally:
            # Restore original method
            self.physical_trainer._load_ground_truth_data = original_gt
        
        return result
    
    def _train_synthetic_with_data(
        self,
        gt_data: Dict[str, Field]
    ) -> Dict[str, Any]:
        """
        Train synthetic model using physical predictions as target data.
        self,
        physical_data: Dict[str, Field],
        gt_data: Dict[str, Field]
    ) -> Dict[str, Any]:
        """
        Train synthetic model using physical predictions as target data.
        
        This method delegates to the synthetic trainer's existing train() method
        but modifies the data loader to use physical predictions.
        
        Args:
            physical_data: Predictions from physical model (used as training target)
            gt_data: Ground truth data (for initial conditions)
            
        Returns:
            Dictionary with training results from synthetic trainer
        """
        # Create a custom dataset that uses physical predictions as targets
        # instead of ground truth
        
        # Convert physical predictions to tensors and cache them
        physical_cache = {}
        for name, field in physical_data.items():
            physical_cache[name] = field_to_tensor(field, ensure_cpu=False)
        
        # Temporarily modify the data manager to return physical predictions
        # as if they were cached ground truth
        original_load = self.synthetic_trainer.data_manager.load_from_cache
        
        def load_physical_as_cache(sim_index):
            return {
                'tensor_data': physical_cache,
                'metadata': self.data_manager.load_from_cache(sim_index)['metadata']
            }
        
        self.synthetic_trainer.data_manager.load_from_cache = load_physical_as_cache
        
        try:
            # Run one epoch of synthetic training with physical data
            synthetic_loss = self.synthetic_trainer._train_epoch()
            
            result = {
                'loss': synthetic_loss,
                'model_state': self.synthetic_trainer.model.state_dict()
            }
        finally:
            # Restore original method
            self.synthetic_trainer.data_manager.load_from_cache = original_load
        
        return result
    
    def _save_checkpoint(self, epoch: int, results: Dict[str, Any]):
        """
        Save checkpoints for both models.
        """
        checkpoint_dir = Path(self.config['project_root']) / 'results' / 'models' / 'hyco'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save physical model (just the optimized parameters)
        physical_checkpoint = {
            'epoch': epoch,
            'optimized_params': {
                param['name']: getattr(self.physical_trainer.model, param['name'])
                for param in self.physical_trainer.learnable_params_config
            },
            'losses': results['physical_losses']
        }
        
        physical_path = checkpoint_dir / f'physical_epoch_{epoch}.pt'
        torch.save(physical_checkpoint, physical_path)
        
        # Save synthetic model (full PyTorch checkpoint)
        synthetic_path = checkpoint_dir / f'synthetic_epoch_{epoch}.pth'
        self.synthetic_trainer.save_checkpoint(
            epoch=epoch,
            loss=results['synthetic_losses'][-1],
            optimizer_state=self.synthetic_trainer.optimizer.state_dict()
        )
        
        print(f"  Saved checkpoints at epoch {epoch}")
```

### 3.3 Hybrid Training Variants

The HybridTrainer architecture allows for different hybrid training strategies.
HYCOTrainer is one specific implementation, but others can be created:

**HYCO (Hybrid Co-training) - Implemented**
- Interleaved training where models use each other's predictions as data

**Potential Future Hybrid Strategies** (all inherit from HybridTrainer):

1. **HYCOTrainer** (current implementation)
   - Full interleaved: Each epoch, both models train on each other's predictions

2. **EnsembleTrainer** (future)
   - Train both models independently, combine predictions with learned weights
   
3. **CascadeTrainer** (future)
   - Physical model trains first, synthetic refines its predictions
   
4. **ConsensusTrainer** (future)
   - Both models trained to agree on predictions (consensus loss)

5. **AdversarialTrainer** (future)
   - Physical model as generator, synthetic as discriminator (or vice versa)

**HYCO Training Variants:**

**Variant 1: Warm-start Sequential**
```python
# Train synthetic model on ground truth first for N epochs
for epoch in range(warmstart_epochs):
    # Use ground truth data
    synthetic_result = self.synthetic_trainer._train_epoch()

# Then start interleaved training
for epoch in range(interleaved_epochs):
    # Now use predictions from each other
    synthetic_preds = self._generate_synthetic_predictions(gt_data)
    self._train_physical_with_data(synthetic_preds, gt_data)
    
    physical_preds = self._generate_physical_predictions(gt_data)
    self._train_synthetic_with_data(physical_preds, gt_data)
```

**Variant 2: Mixed Data Training**
- Blend ground truth with model predictions as training data
- Useful for stability during early training

```python
# Mix GT and predictions: use alpha fraction of GT, (1-alpha) of predictions
def create_mixed_target(gt_data, pred_data, alpha):
    mixed = {}
    for name in gt_data.keys():
        mixed[name] = alpha * gt_data[name] + (1 - alpha) * pred_data[name]
    return mixed

# Train with blended data
mixed_data = create_mixed_target(gt_data, synthetic_preds, alpha=0.7)
self._train_physical_with_data(mixed_data, gt_data)
```

### 3.4 Configuration Example

```yaml
# conf/hybrid/burgers_hyco.yaml
defaults:
  - data: burgers_128
  - model/physical: burgers
  - model/synthetic: unet
  - trainer: hyco
  - _self_

run_params:
  experiment_name: 'burgers_hyco_v1'
  mode: ['train', 'evaluate']
  model_type: 'hyco'

trainer_params:
  train_sim: [0, 1, 2, 3, 4]
  epochs: 100
  num_predict_steps: 4
  
  # Interleaved-specific params
  alpha: 0.5  # Weight between GT and predictions
  warmstart_epochs: 20  # Train synthetic alone first
  
  # Physical optimization
  physical_solver:
    method: 'L-BFGS-B'
    max_iterations: 10  # Per epoch
  
  # Synthetic optimization
  synthetic_optimizer: 'adam'
  learning_rate: 0.001
```

---

## 4. Backward Compatibility Cleanup

### 4.1 Code to Remove

#### A. Old Cache Format Support

**File:** `src/data/data_manager.py`

**Line 313:** Remove comment about backward compatibility
```python
# REMOVE:
# Original metadata (preserved for backward compatibility)

# KEEP (but simplify):
'scene_metadata': scene_metadata,
'field_metadata': field_metadata,
```

**File:** `src/data/hybrid_dataset.py`

**Line 227:** Remove fallback for old cache files
```python
# REMOVE this entire else block:
else:
    # Fallback for old cache files without bounds_lower/upper
    try:
        bounds_str = meta['bounds']
        domain = eval(bounds_str, {"Box": Box})
    except:
        domain = Box(x=1, y=1)

# REPLACE with error handling:
if 'bounds_lower' not in meta or 'bounds_upper' not in meta:
    raise ValueError(
        f"Invalid cache format for field '{name}'. "
        f"Please clear cache and regenerate data."
    )
```

#### B. Validation Strictness

**File:** `src/data/validation.py`

Currently allows version 1.x and 2.x caches. Since you want current version only:

```python
# CHANGE from:
def _is_version_compatible(self, cache_version: str) -> bool:
    cache_major = int(cache_version.split('.')[0])
    current_major = 2
    
    if self.strict:
        return cache_major == current_major
    else:
        # Allow version 1.x and 2.x caches (backward compatible)
        return cache_major in [1, 2]

# TO:
def _is_version_compatible(self, cache_version: str) -> bool:
    """Only accept current cache version (2.x)."""
    cache_major = int(cache_version.split('.')[0])
    current_major = 2
    return cache_major == current_major
```

Remove the `strict` parameter entirely since we always require current version.

#### C. Physical Model Validation

**File:** `src/models/physical/base.py`

The PDE parameter validation was designed but not enforced. If you don't need validation:

```python
# REMOVE validator concept:
PDE_PARAMETERS = {
    'nu': {'type': float, 'default': 0.01}  # Remove 'validator' key
}

# REMOVE this from _parse_pde_parameters:
# validator = param_spec.get('validator')
# if validator and not validator(value):
#     raise ValueError(f"Validation failed for {param_name}={value}")
```

Or, if you want validation, implement it:
```python
# ADD validation logic:
validator = param_spec.get('validator')
if validator and not validator(value):
    raise ValueError(
        f"Validation failed for parameter '{param_name}': "
        f"value={value} does not satisfy validation function"
    )
```

#### D. BaseTrainer Stubs in PhysicalTrainer

**File:** `src/training/physical/trainer.py`

Remove these stub methods (after refactoring to FieldTrainer):
```python
# REMOVE (Lines 206-227):
def _create_model(self):
    """Physical trainer uses model from config directly."""
    pass

def _create_data_loader(self):
    """Physical trainer doesn't use DataLoader."""
    pass

def _train_epoch(self, epoch: int) -> float:
    """Physical trainer doesn't use epoch-based training."""
    pass
```

---

## 5. Implementation Roadmap

### Phase 1: Refactor Trainer Hierarchy (Week 1)
**Goal:** Separate physical and synthetic trainer base classes

**Tasks:**
1. Create `AbstractTrainer` with minimal common interface
2. Create `TensorTrainer` from current `BaseTrainer` (keep all PyTorch-specific code)
3. Create `FieldTrainer` with PhiFlow-specific methods
4. Update `SyntheticTrainer` to inherit from `TensorTrainer`
5. Update `PhysicalTrainer` to inherit from `FieldTrainer`
6. Remove stub methods from `PhysicalTrainer`
7. Update `TrainerFactory` to handle new hierarchy
8. Run all existing tests to ensure nothing breaks

**Testing:**
- Run existing synthetic training: `python run.py --config-name=burgers_experiment`
- Run existing physical training: `python run.py --config-name=burgers_physical_experiment`
- Verify checkpoints load/save correctly
- Verify evaluation still works

---

### Phase 2: Cleanup Backward Compatibility (Week 1-2)
**Goal:** Remove all legacy code paths

**Tasks:**
1. Update cache version requirement to 2.x only
2. Remove fallback code in `hybrid_dataset.py`
3. Simplify validation logic (remove `strict` parameter)
4. Clear all existing cache: `rm -rf data/cache/*`
5. Regenerate cache with current version
6. Add clear error messages if old cache is encountered

**Testing:**
- Delete cache and regenerate data
- Verify validation catches mismatches (change config slightly, verify it detects)
- Run full training pipeline

---

### Phase 2.5: Fix Data Loading Memory Issues (Week 2)
**Goal:** Implement efficient batch-based data loading to handle large datasets

**Problem:**
Current implementation loads entire simulations into memory:
- HybridDataset pre-caches all simulations in `_cache_all_simulations()`
- Training with 20+ simulations causes memory issues
- No lazy loading or on-demand data fetching

**Proposed Solution:**
Implement lazy loading with LRU caching for simulation data:

```python
# src/data/hybrid_dataset.py (UPDATED)
from functools import lru_cache
from typing import Optional

class HybridDataset(Dataset):
    """
    PyTorch Dataset with lazy loading and LRU caching.
    
    Key improvements:
    - Simulations loaded on-demand, not all at once
    - LRU cache keeps N most recently used simulations in memory
    - Automatic memory management
    - Configurable cache size
    """
    
    def __init__(
        self,
        data_manager: DataManager,
        sim_indices: List[int],
        field_names: List[str],
        num_frames: int,
        num_predict_steps: int,
        dynamic_fields: List[str] = None,
        static_fields: List[str] = None,
        use_sliding_window: bool = False,
        return_fields: bool = False,
        max_cached_sims: int = 5  # NEW: Max simulations in memory
    ):
        self.data_manager = data_manager
        self.sim_indices = sim_indices
        self.field_names = field_names
        self.num_frames = num_frames
        self.num_predict_steps = num_predict_steps
        self.use_sliding_window = use_sliding_window
        self.return_fields = return_fields
        self.max_cached_sims = max_cached_sims
        
        # Handle static vs dynamic field distinction
        if dynamic_fields is None and static_fields is None:
            self.dynamic_fields = field_names
            self.static_fields = []
        elif dynamic_fields is not None:
            self.dynamic_fields = dynamic_fields
            self.static_fields = static_fields if static_fields is not None else []
        else:
            self.static_fields = static_fields
            self.dynamic_fields = [f for f in field_names if f not in static_fields]
        
        # Calculate channel indices for unpacking
        self._build_channel_map()
        
        # CHANGED: Don't pre-cache all simulations
        # Instead, verify cache exists and get metadata
        self._validate_cache_exists()
        
        # Build sample index mapping for sliding window
        if self.use_sliding_window:
            self._build_sliding_window_index()
        
        # CHANGED: Create LRU cache for simulation data
        self._cached_load_simulation = lru_cache(maxsize=self.max_cached_sims)(
            self._load_simulation_uncached
        )
    
    def _validate_cache_exists(self):
        """
        Validate that cache exists for all simulations without loading them.
        Also determine num_frames if it was None.
        """
        print(f"Validating cache for {len(self.sim_indices)} simulations...")
        
        for sim_idx in self.sim_indices:
            if not self.data_manager.is_cached(sim_idx, self.field_names, self.num_frames):
                print(f"  Cache missing for sim {sim_idx}, will generate on first access")
                # Could optionally trigger caching here, but we'll do it lazily
        
        # If num_frames was None, determine from first simulation
        if self.num_frames is None:
            first_sim = self.sim_indices[0]
            data = self.data_manager.get_or_load_simulation(
                first_sim,
                field_names=self.field_names,
                num_frames=None
            )
            first_field = self.field_names[0]
            self.num_frames = data['tensor_data'][first_field].shape[0]
            print(f"  Determined num_frames from cache: {self.num_frames}")
    
    def _load_simulation_uncached(self, sim_idx: int) -> Dict[str, Any]:
        """
        Load a single simulation from cache or disk.
        This is wrapped by LRU cache to keep recent simulations in memory.
        
        Args:
            sim_idx: Simulation index to load
            
        Returns:
            Cached data dictionary
        """
        return self.data_manager.get_or_load_simulation(
            sim_idx,
            field_names=self.field_names,
            num_frames=self.num_frames
        )
    
    def __getitem__(self, idx: int):
        """Get a training sample with lazy loading."""
        # Determine simulation and starting frame
        if self.use_sliding_window:
            sim_idx, start_frame = self.sample_index[idx]
        else:
            sim_idx = self.sim_indices[idx]
            start_frame = 0
        
        # Load simulation data (from LRU cache if available)
        data = self._cached_load_simulation(sim_idx)
        
        # Rest of the method stays the same...
        if self.return_fields:
            return self._convert_to_fields_with_start(data, start_frame)
        
        # Tensor-based mode
        all_field_tensors = [data['tensor_data'][name] for name in self.field_names]
        all_data = torch.cat(all_field_tensors, dim=1)
        initial_state = all_data[start_frame]
        
        dynamic_field_tensors = [data['tensor_data'][name] for name in self.dynamic_fields]
        dynamic_data = torch.cat(dynamic_field_tensors, dim=1)
        
        target_start = start_frame + 1
        target_end = start_frame + 1 + self.num_predict_steps
        rollout_targets = dynamic_data[target_start:target_end]
        
        return initial_state, rollout_targets
    
    def clear_cache(self):
        """Manually clear the simulation cache to free memory."""
        self._cached_load_simulation.cache_clear()
        print("Cleared simulation cache")
```

**Additional Optimization: Pin Memory Loading**

For faster GPU transfer, implement pinned memory loading:

```python
# src/data/hybrid_dataset.py (ENHANCEMENT)
class HybridDataset(Dataset):
    def __init__(self, ..., pin_memory: bool = True):
        ...
        self.pin_memory = pin_memory and torch.cuda.is_available()
    
    def _load_simulation_uncached(self, sim_idx: int) -> Dict[str, Any]:
        """Load simulation with optional pinned memory."""
        data = self.data_manager.get_or_load_simulation(...)
        
        if self.pin_memory:
            # Pin tensors for faster GPU transfer
            for name in data['tensor_data'].keys():
                tensor = data['tensor_data'][name]
                if not tensor.is_pinned():
                    data['tensor_data'][name] = tensor.pin_memory()
        
        return data
```

**Configuration Changes:**

```yaml
# conf/trainer/synthetic.yaml
# Add data loading parameters
data_loading:
  max_cached_sims: 5  # Number of simulations to keep in memory
  pin_memory: true    # Use pinned memory for faster GPU transfer
  num_workers: 2      # DataLoader workers (keep low to avoid memory duplication)
```

**DataLoader Configuration:**

Update trainer to use optimal DataLoader settings:

```python
# src/training/synthetic/trainer.py (MODIFIED)
def _create_data_loader(self):
    """Creates DataManager and HybridDataset with memory-efficient settings."""
    ...
    
    # Get data loading config
    data_loading_config = self.trainer_config.get('data_loading', {})
    max_cached_sims = data_loading_config.get('max_cached_sims', 5)
    pin_memory = data_loading_config.get('pin_memory', True)
    num_workers = data_loading_config.get('num_workers', 0)
    
    # Create HybridDataset with lazy loading
    dataset = HybridDataset(
        data_manager=data_manager,
        sim_indices=self.train_sim,
        field_names=self.field_names,
        num_frames=self.num_frames,
        num_predict_steps=self.num_predict_steps,
        dynamic_fields=self.dynamic_fields,
        static_fields=self.static_fields,
        use_sliding_window=self.use_sliding_window,
        max_cached_sims=max_cached_sims  # NEW
    )
    
    # Create PyTorch DataLoader with optimized settings
    loader = DataLoader(
        dataset,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=num_workers,  # Use config value
        pin_memory=pin_memory if torch.cuda.is_available() else False,
        persistent_workers=num_workers > 0  # Keep workers alive between epochs
    )
    
    return loader
```

**Memory Monitoring:**

Add memory monitoring utilities:

```python
# src/utils/memory_monitor.py (NEW)
import torch
import psutil
import os

class MemoryMonitor:
    """Monitor CPU and GPU memory usage during training."""
    
    @staticmethod
    def get_cpu_memory_mb() -> float:
        """Get current process CPU memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def get_gpu_memory_mb() -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    @staticmethod
    def print_memory_usage(prefix: str = ""):
        """Print current memory usage."""
        cpu_mem = MemoryMonitor.get_cpu_memory_mb()
        gpu_mem = MemoryMonitor.get_gpu_memory_mb()
        print(f"{prefix}CPU Memory: {cpu_mem:.1f} MB, GPU Memory: {gpu_mem:.1f} MB")
    
    @staticmethod
    def log_memory_usage(logger, prefix: str = ""):
        """Log memory usage to logger."""
        cpu_mem = MemoryMonitor.get_cpu_memory_mb()
        gpu_mem = MemoryMonitor.get_gpu_memory_mb()
        logger.info(f"{prefix}CPU: {cpu_mem:.1f} MB, GPU: {gpu_mem:.1f} MB")

# Usage in trainer
from src.utils.memory_monitor import MemoryMonitor

def _train_epoch(self):
    """Train one epoch with memory monitoring."""
    MemoryMonitor.print_memory_usage("Start of epoch: ")
    
    for batch_idx, (initial_state, rollout_targets) in enumerate(self.train_loader):
        # Training code...
        
        if batch_idx % 10 == 0:
            MemoryMonitor.print_memory_usage(f"Batch {batch_idx}: ")
    
    MemoryMonitor.print_memory_usage("End of epoch: ")
```

**Tasks:**
1. Implement lazy loading with LRU cache in HybridDataset
2. Add `max_cached_sims` parameter to control memory usage
3. Implement pinned memory loading for GPU training
4. Update DataLoader configuration in trainers
5. Create MemoryMonitor utility for tracking memory usage
6. Add data_loading config section to trainer configs
7. Test with increasing number of simulations (5, 10, 20, 50)
8. Document memory requirements and optimal settings

**Testing:**
- Train with 5 simulations: baseline memory usage
- Train with 20 simulations: verify no memory issues
- Train with 50+ simulations: stress test
- Monitor memory usage throughout training
- Verify LRU cache is working (check hit rate)
- Test with different `max_cached_sims` values (3, 5, 10)
- Verify training speed is acceptable
- Test with and without pinned memory

**Expected Benefits:**
- ✅ Handle 20+ simulations without memory issues
- ✅ Configurable memory usage (adjust `max_cached_sims`)
- ✅ Automatic cache management (no manual clearing needed)
- ✅ Faster GPU transfer with pinned memory
- ✅ Better monitoring and debugging with MemoryMonitor

**Memory Usage Estimates:**
- Single simulation (~128x128, 75 frames, 2 fields): ~100 MB
- With `max_cached_sims=5`: ~500 MB in memory at once
- With `max_cached_sims=10`: ~1 GB in memory at once
- Adjust based on available RAM and GPU memory

---

### Phase 3: Field-Tensor Converter (Week 2-3)
**Goal:** Create robust conversion utilities

**Tasks:**
1. Extend existing `field_conversion.py` to create `FieldTensorConverter` class
2. Add batch conversion methods (currently only single-field)
3. Add comprehensive tests for:
   - Scalar fields (density)
   - Vector fields (velocity)
   - Staggered ↔ Centered grid conversions
   - Batch dimension handling
4. Benchmark conversion performance

**Testing:**
- Unit tests for all field types
- Test on actual trained models (convert predictions back and forth, check accuracy)

---

### Phase 4: Implement HybridTrainer and HYCOTrainer (Week 3-5)
**Goal:** Create working hybrid training infrastructure

**Prerequisites:** Phase 2.5 must be complete (memory-efficient data loading)

**Tasks:**
1. Create `src/training/hybrid/` directory
2. Implement `HybridTrainer` abstract base class
3. Implement `HYCOTrainer` as first concrete implementation
4. Create configuration templates
4. Implement helper methods:
   - `_get_synthetic_rollout()`
   - `_run_physical_simulation()`
   - `_update_physical_params()`
5. Add logging and monitoring
6. Add checkpoint saving for both models

**Testing:**
- Start with small test: 1 simulation, 5 epochs, 2 predict steps
- Verify both models can load and predict
- Verify conversions work correctly
- Monitor losses to ensure both models are learning
- Compare to separate training (does HYCO help?)
- Test that HybridTrainer provides good foundation for future strategies

---

### Phase 5: Experiments and Optimization (Week 6+)
**Goal:** Validate HYCO training improves results

**Tasks:**
1. Run baseline experiments:
   - Synthetic model alone
   - Physical model alone
2. Run HYCO experiments with different variants:
   - Full HYCO (default)
   - Warm-start sequential
   - Mixed data training
3. Evaluate on held-out test simulations
4. Compare metrics (MSE, RMSE, physical accuracy)
5. Visualize improvements

**Metrics to Track:**
- Prediction accuracy (MSE vs ground truth)
- Physical parameter accuracy (for Burgers: estimated nu vs true nu)
- Generalization (performance on unseen simulations)
- Training time and convergence speed

---

## 6. Breaking Changes and Migration Guide

### 6.1 API Changes

#### Trainer Creation
**Before:**
```python
from src.training.base_trainer import BaseTrainer
trainer = TrainerFactory.create_trainer(config)
```

**After:**
```python
from src.training.abstract_trainer import AbstractTrainer
trainer = TrainerFactory.create_trainer(config)  # Same interface
```

#### Custom Trainers
**Before:**
```python
class MyTrainer(BaseTrainer):
    def _train_epoch(self):
        # Had to implement even if not needed
        pass
```

**After:**
```python
# For tensor-based trainers:
class MyTrainer(TensorTrainer):
    def _train_epoch(self):
        # Only implement if you're doing epoch-based training
        pass

# For field-based trainers:
class MyTrainer(FieldTrainer):
    # No need for _train_epoch stub
    pass
```

### 6.2 Configuration Changes

#### Interleaved Training Config
**New file:** `conf/trainer/interleaved.yaml`
```yaml
# @package _global_.trainer_params
learning_rate: 0.0001
epochs: 100
num_predict_steps: 4
train_sim: []

# Interleaved-specific
alpha: 0.5
warmstart_epochs: 0
physical_iterations_per_epoch: 10

# Physical solver
physical_solver:
  method: 'L-BFGS-B'
  abs_tol: 1e-6

# Synthetic optimizer
synthetic_optimizer: 'adam'
synthetic_scheduler: 'cosine'
```

### 6.3 Cache Invalidation

**IMPORTANT:** All existing cache must be cleared and regenerated.

```bash
# Clear cache
rm -rf data/cache/*

# Regenerate data
python run.py --config-name=burgers_experiment +run_params.mode=[generate]
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

Create comprehensive unit tests for new components:

```python
# tests/training/test_field_tensor_converter.py
def test_scalar_field_conversion():
    """Test converting scalar field to tensor and back."""
    pass

def test_vector_field_conversion():
    """Test converting vector field to tensor and back."""
    pass

def test_batch_conversion():
    """Test batch conversion with multiple fields."""
    pass

# tests/training/test_interleaved_trainer.py
def test_interleaved_initialization():
    """Test InterleavedTrainer initializes correctly."""
    pass

def test_physical_step():
    """Test physical training step."""
    pass

def test_synthetic_step():
    """Test synthetic training step."""
    pass

def test_full_epoch():
    """Test complete interleaved epoch."""
    pass
```

### 7.2 Integration Tests

```python
# tests/integration/test_end_to_end.py
def test_synthetic_training_pipeline():
    """Test complete synthetic training pipeline."""
    pass

def test_physical_training_pipeline():
    """Test complete physical training pipeline."""
    pass

def test_interleaved_training_pipeline():
    """Test complete interleaved training pipeline."""
    pass
```

### 7.3 Validation Experiments

Run small-scale validation before full experiments:
```bash
# Quick test configs
python run.py --config-name=burgers_quick_test
python run.py --config-name=burgers_physical_quick_test
python run.py --config-name=burgers_interleaved_quick_test
```

---

## 8. Risk Assessment and Mitigation

### 8.1 Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Memory issues with large datasets | **High** | **High** | **Phase 2.5: Lazy loading with LRU cache** |
| Breaking existing code | High | Medium | Comprehensive testing, gradual rollout |
| HYCO training doesn't converge | Medium | Medium | Implement multiple strategies, tune hyperparameters |
| Field-tensor conversion overhead | Medium | Low | Benchmark, optimize if needed, cache conversions |
| Cache invalidation disrupts workflow | Low | High | Clear communication, automated regeneration scripts |
| LRU cache misses slow training | Medium | Medium | Tune `max_cached_sims`, monitor hit rate |

### 8.2 Rollback Strategy

Keep old code in a separate branch:
```bash
git checkout -b backup/before-refactoring
git push origin backup/before-refactoring
```

Use feature flags for gradual rollout:
```python
# config.yaml
features:
  use_new_trainer_hierarchy: true
  use_interleaved_training: false
```

---

## 9. Documentation Updates Needed

1. **Architecture Diagram** - Update to show new trainer hierarchy
2. **Training Guide** - Add section on interleaved training
3. **API Reference** - Document new classes and methods
4. **Migration Guide** - Help users update custom trainers
5. **Configuration Reference** - Document new config options

---

## 10. Open Questions for Discussion

1. **Data injection method**: Should we use monkey patching (as shown) or modify trainers to accept external data sources?
2. **Training frequency**: Should both models train every epoch, or alternate epochs (e.g., 2 physical steps per 1 synthetic step)?
3. **Checkpoint strategy**: Save both models together or separately? Current approach saves separately.
4. **Evaluation**: How to evaluate hybrid models? 
   - Use both models together?
   - Evaluate each model independently?
   - Create ensemble predictions?
5. **Multi-simulation training**: Should HYCO trainer support multiple simulations at once?
6. **Convergence criteria**: When to stop HYCO training? 
   - When both models converge?
   - When combined performance plateaus?
   - Fixed number of epochs?
7. **Future hybrid strategies**: What other hybrid training strategies would be valuable?
   - Ensemble methods?
   - Cascade approaches?
   - Adversarial training?

---

## 11. Summary and Recommendations

### Immediate Actions (This Week)
1. ✅ Review this document
2. 🔨 Start Phase 1: Refactor trainer hierarchy
3. 🧹 Start Phase 2: Remove backward compatibility code
4. 🧪 Set up testing infrastructure

### Short Term (2-4 Weeks)
1. Complete Phase 3: Field-tensor converter
2. Begin Phase 4: Implement InterleavedTrainer
3. Run initial experiments

### Long Term (1-2 Months)
1. Complete Phase 5: Full experimental validation
2. Write paper/documentation
3. Optimize performance
4. Add advanced features (curriculum learning, multi-simulation, etc.)

### Key Takeaways

**Problem 1: Trainer Architecture** ✓
- **Solution**: Separate base classes for TensorTrainer and FieldTrainer
- **Benefit**: No more forced interfaces, cleaner code, easier to extend

**Problem 2: Memory Issues with Large Datasets** ✓ 🚨 **HIGH PRIORITY**
- **Solution**: Lazy loading with LRU cache in HybridDataset
- **Benefit**: Handle 20+ simulations, configurable memory usage, automatic cache management

**Problem 3: Hybrid Training** ✓
- **Solution**: HybridTrainer base class with HYCOTrainer as first implementation
- **Benefit**: Co-training improves both models, extensible architecture for future strategies

**Problem 4: Backward Compatibility** ✓
- **Solution**: Remove all fallback code, require cache regeneration
- **Benefit**: Simpler codebase, easier to maintain, faster execution

---

## Appendix A: File Structure After Refactoring

```
src/
├── training/
│   ├── abstract_trainer.py         [NEW] Minimal common interface
│   ├── tensor_trainer.py            [NEW] PyTorch-based training
│   ├── field_trainer.py             [NEW] PhiFlow-based training
│   ├── base_trainer.py              [DEPRECATED - keep temporarily]
│   ├── synthetic/
│   │   └── trainer.py               [MODIFIED] Inherits from TensorTrainer
│   ├── physical/
│   │   └── trainer.py               [MODIFIED] Inherits from FieldTrainer
│   └── hybrid/
│       ├── __init__.py              [NEW]
│       ├── base_trainer.py          [NEW] HybridTrainer abstract base
│       ├── hyco_trainer.py          [NEW] HYCOTrainer implementation
│       └── [future_trainer.py]      [FUTURE] Other hybrid strategies
├── utils/
│   ├── field_conversion.py          [MODIFIED] Add FieldTensorConverter
│   └── conversion_benchmark.py      [NEW] Performance tests
└── ...
```

---

## Appendix B: Configuration Template for HYCO Training

```yaml
# conf/burgers_hyco_experiment.yaml
defaults:
  - data: burgers_128
  - model/physical: burgers
  - model/synthetic: unet
  - trainer: interleaved
  - evaluation: default
  - _self_

run_params:
  experiment_name: 'burgers_interleaved_v1'
  notes: 'Interleaved training of physical and synthetic models'
  mode: ['train', 'evaluate']
  model_type: 'interleaved'

model:
  synthetic:
    model_save_name: 'burgers_unet_hyco'
  physical:
    model_save_name: 'burgers_physical_hyco'

trainer_params:
  train_sim: [0, 1, 2, 3, 4, 5]
  epochs: 100
  num_predict_steps: 4
  
  # HYCO-specific strategy
  hyco_strategy: 'full'  # 'full' | 'warmstart' | 'mixed_data'
  warmstart_epochs: 0  # Only for 'warmstart' strategy
  
  # Mixed data training (only for 'mixed_data' strategy)
  alpha: 0.7  # Blend: 0.7*GT + 0.3*predictions (gradually decrease to 0.0)
  
  # Physical model optimization
  physical_iterations_per_epoch: 10
  physical_solver:
    method: 'L-BFGS-B'
    abs_tol: 1e-6
    max_iterations: 10
  
  # Learnable physical parameters
  learnable_parameters:
    - name: 'nu'
      initial_guess: 0.005
  
  # Synthetic model optimization
  synthetic_optimizer: 'adam'
  learning_rate: 0.001
  weight_decay: 0.0
  scheduler: 'cosine'
  batch_size: 1  # Interleaved trains on one sim at a time

evaluation_params:
  test_sim: [20, 21, 22, 23]
  num_frames: 75
  evaluate_both: true  # Evaluate both physical and synthetic

project_root: ${hydra:runtime.cwd}
```

---

**End of Document**

This refactoring plan provides a clear path forward while maintaining code quality and minimizing risks. The phased approach allows for incremental testing and validation at each step.
