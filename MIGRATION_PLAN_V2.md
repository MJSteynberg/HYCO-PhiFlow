# PhiML Migration Plan V2: Backward Migration Strategy

**Objective**: Migrate HYCO-PhiFlow from hybrid PyTorch-PhiFlow to pure PhiML implementation
**Strategy**: Backward migration - start from models, move to trainers, then data pipeline
**Validation**: After each step, run `python run.py --config-name=burgers.yaml` to ensure nothing breaks

**Expected Benefits**:
- 🚀 15-20% performance improvement (eliminate conversion overhead)
- 🧹 ~30% code reduction (simpler data pipeline)
- 💪 10-100x speedup with JIT compilation
- 🎯 Better type safety with named dimensions
- 🔧 Easier maintenance and debugging

---

## Table of Contents

1. [Migration Overview](#1-migration-overview)
2. [Step 1: Synthetic Models to PhiML](#step-1-synthetic-models-to-phiml)
3. [Step 2: Synthetic Trainer to PhiML](#step-2-synthetic-trainer-to-phiml)
4. [Step 3: Data Pipeline Refactor](#step-3-data-pipeline-refactor)
5. [Future Work: Physical Models Integration](#future-work-physical-models-integration)
6. [Testing Strategy](#testing-strategy)
7. [Rollback Plan](#rollback-plan)

---

## 1. Migration Overview

### 1.1 Backward Migration Philosophy

```
Current State:
┌─────────────┐      ┌──────────────┐      ┌──────────────┐
│ Data        │─────→│  Trainer     │─────→│   Model      │
│ (PyTorch)   │      │  (PyTorch)   │      │  (PyTorch)   │
└─────────────┘      └──────────────┘      └──────────────┘

Step 1: Convert Model
┌─────────────┐      ┌──────────────┐      ┌──────────────┐
│ Data        │─────→│  Trainer     │─────→│   Model      │
│ (PyTorch)   │      │  (PyTorch)   │ conv │  (PhiML) ✓   │
└─────────────┘      └──────────────┘ ────→└──────────────┘

Step 2: Convert Trainer
┌─────────────┐      ┌──────────────┐      ┌──────────────┐
│ Data        │─────→│  Trainer     │ conv │   Model      │
│ (PyTorch)   │ conv │  (PhiML) ✓   │─────→│  (PhiML) ✓   │
└─────────────┘ ────→└──────────────┘      └──────────────┘

Step 3: Convert Data
┌─────────────┐      ┌──────────────┐      ┌──────────────┐
│ Data        │─────→│  Trainer     │─────→│   Model      │
│ (PhiML) ✓   │      │  (PhiML) ✓   │      │  (PhiML) ✓   │
└─────────────┘      └──────────────┘      └──────────────┘
```

### 1.2 Key Design Decisions

1. **PhiML Dataclasses for Caching**: Use `@dataclass` with `@cached_property` as shown in [references/cached_parallel_example.py](references/cached_parallel_example.py)
2. **Unified Dataset**: Single dataset class that works for both training modes
3. **Minimal Conversions**: Each step only converts at the boundary, then moves boundary inward
4. **Continuous Validation**: Run `python run.py --config-name=burgers.yaml` after each major change

### 1.3 Migration Timeline

```
Step 1: Synthetic Models (1-2 days)
   ↓
Step 2: Synthetic Trainer (2-3 days)
   ↓
Step 3: Data Pipeline (3-5 days)
   ↓
Total: ~1.5 weeks
```

---

## Step 1: Synthetic Models to PhiML

**Duration**: 1-2 days
**Risk**: Low
**Goal**: Convert synthetic models to pure PhiML, add conversion layer in trainer
**Validation**: `python run.py --config-name=burgers.yaml` must work

### 1.1 Understanding Current Model Structure

Current synthetic models (UNet, ResNet, ConvNet):
```python
class UNet(SyntheticModel):  # Inherits from nn.Module
    def __init__(self, config):
        super().__init__(config)
        # Uses phiml.nn.u_net internally
        self.model = u_net(...)

    def forward(self, x):  # PyTorch-style forward
        # x is torch.Tensor
        return self.model(x)
```

### 1.2 Create PhiML-Native Model Base

Create `src/models/synthetic/phiml_base.py`:

```python
"""
PhiML-native synthetic model base class.
Models work entirely with PhiML tensors (no PyTorch dependency).
"""

from phiml import math
from phiml.math import channel
import logging
from typing import Optional


class PhiMLSyntheticModel:
    """
    Base class for synthetic models using pure PhiML.
    No torch.nn.Module inheritance.
    """

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Get channel info from config
        data_config = config["data"]
        self.num_dynamic_fields = len(data_config["fields_scheme"]["dynamic"])
        self.num_static_fields = len(data_config["fields_scheme"]["static"])
        self.total_fields = self.num_dynamic_fields + self.num_static_fields

        self.logger.info(
            f"Initializing {self.__class__.__name__}: "
            f"{self.num_dynamic_fields} dynamic, {self.num_static_fields} static fields"
        )

        # Network will be set by subclasses
        self.network = None

    def __call__(self, x: 'Tensor') -> 'Tensor':
        """
        Forward pass with residual learning.

        Args:
            x: PhiML Tensor with shape (..., channels)
               Must have 'c' or 'channels' dimension

        Returns:
            PhiML Tensor with same shape as input
        """
        # Find channel dimension name
        channel_dim = None
        for dim_name in x.shape.names:
            if dim_name in ['c', 'channels', 'channel']:
                channel_dim = dim_name
                break

        if channel_dim is None:
            raise ValueError(f"Input must have channel dimension. Got shape: {x.shape}")

        # Split into dynamic and static channels
        dynamic = x[{channel_dim: slice(0, self.num_dynamic_fields)}]
        static = x[{channel_dim: slice(self.num_dynamic_fields, self.total_fields)}]

        # Predict residual for dynamic fields only
        residual = math.native_call(self.network, dynamic)

        # Residual learning: output = input + residual
        predicted_dynamic = dynamic + residual

        # Concatenate with unchanged static fields
        output = math.concat([predicted_dynamic, static], channel_dim)

        return output

    def get_network(self):
        """Get the underlying network (for optimization)"""
        return self.network

    def save(self, path: str):
        """Save network parameters"""
        math.save(self.network, path)
        self.logger.info(f"Saved model to {path}")

    def load(self, path: str):
        """Load network parameters"""
        self.network = math.load(path)
        self.logger.info(f"Loaded model from {path}")
```

### 1.3 Migrate UNet to PhiML

Create `src/models/synthetic/phiml_unet.py`:

```python
"""
UNet model using pure PhiML.
"""

from phiml import nn
from src.models.synthetic.phiml_base import PhiMLSyntheticModel


class PhiMLUNet(PhiMLSyntheticModel):
    """
    UNet architecture for field-to-field mapping.
    Pure PhiML implementation.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # Get architecture config
        arch_config = config["model"]["synthetic"]["architecture"]

        # Create U-Net using phiml.nn
        self.network = nn.u_net(
            in_channels=self.num_dynamic_fields,
            out_channels=self.num_dynamic_fields,
            levels=arch_config.get("levels", 4),
            filters=arch_config.get("filters", 32),
            batch_norm=arch_config.get("batch_norm", True),
            activation=arch_config.get("activation", "ReLU"),
            in_spatial=2  # 2D spatial dimensions
        )

        self.logger.info(
            f"Created U-Net: levels={arch_config.get('levels', 4)}, "
            f"filters={arch_config.get('filters', 32)}"
        )
```

### 1.4 Migrate ResNet to PhiML

Create `src/models/synthetic/phiml_resnet.py`:

```python
"""
ResNet model using pure PhiML.
"""

from phiml import nn
from src.models.synthetic.phiml_base import PhiMLSyntheticModel


class PhiMLResNet(PhiMLSyntheticModel):
    """
    ResNet architecture for field-to-field mapping.
    Pure PhiML implementation.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # Get architecture config (FIXED: correct path)
        arch_config = config["model"]["synthetic"]["architecture"]

        # Create ResNet using phiml.nn
        self.network = nn.res_net(
            in_channels=self.num_dynamic_fields,
            out_channels=self.num_dynamic_fields,
            layers=arch_config.get("layers", [16, 32, 64]),
            batch_norm=arch_config.get("batch_norm", True),
            activation=arch_config.get("activation", "ReLU")
        )

        self.logger.info(f"Created ResNet: layers={arch_config.get('layers', [16, 32, 64])}")
```

### 1.5 Migrate ConvNet to PhiML

Create `src/models/synthetic/phiml_convnet.py`:

```python
"""
ConvNet model using pure PhiML.
"""

from phiml import nn
from src.models.synthetic.phiml_base import PhiMLSyntheticModel


class PhiMLConvNet(PhiMLSyntheticModel):
    """
    Convolutional network for field-to-field mapping.
    Pure PhiML implementation.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # Get architecture config (FIXED: correct path)
        arch_config = config["model"]["synthetic"]["architecture"]

        # Create ConvNet using phiml.nn
        self.network = nn.conv_net(
            in_channels=self.num_dynamic_fields,
            out_channels=self.num_dynamic_fields,
            layers=arch_config.get("layers", [32, 64, 128]),
            batch_norm=arch_config.get("batch_norm", True),
            activation=arch_config.get("activation", "ReLU")
        )

        self.logger.info(f"Created ConvNet: layers={arch_config.get('layers', [32, 64, 128])}")
```

### 1.6 Update Model Factory

Modify `src/factories/model_factory.py`:

```python
# Add imports at top
from src.models.synthetic.phiml_unet import PhiMLUNet
from src.models.synthetic.phiml_resnet import PhiMLResNet
from src.models.synthetic.phiml_convnet import PhiMLConvNet

class ModelFactory:
    # ... existing code ...

    @staticmethod
    def create_synthetic_model(config: dict):
        """
        Create synthetic model.
        Now returns PhiML-native models.
        """
        model_name = config["model"]["synthetic"]["name"].lower()

        # Map to PhiML models
        model_map = {
            'unet': PhiMLUNet,
            'resnet': PhiMLResNet,
            'convnet': PhiMLConvNet,
        }

        if model_name not in model_map:
            raise ValueError(
                f"Unknown synthetic model: {model_name}. "
                f"Available: {list(model_map.keys())}"
            )

        model_class = model_map[model_name]
        model = model_class(config)

        return model
```

### 1.7 Add Conversion Layer in Trainer

Modify `src/training/synthetic/trainer.py`:

```python
# Add conversion utility at the top
def torch_to_phiml(torch_tensor):
    """Convert PyTorch tensor to PhiML tensor"""
    from phiml import math
    from phiml.math import batch, spatial, channel

    # Get tensor as numpy
    numpy_array = torch_tensor.detach().cpu().numpy()

    # Infer shape (assuming standard format: B, T, H, W, C)
    if numpy_array.ndim == 5:  # (batch, time, height, width, channels)
        B, T, H, W, C = numpy_array.shape
        phiml_tensor = math.wrap(
            numpy_array,
            batch(batch=B, time=T),
            spatial(x=H, y=W),
            channel(c=C)
        )
    elif numpy_array.ndim == 4:  # (batch, height, width, channels)
        B, H, W, C = numpy_array.shape
        phiml_tensor = math.wrap(
            numpy_array,
            batch(batch=B),
            spatial(x=H, y=W),
            channel(c=C)
        )
    else:
        raise ValueError(f"Unexpected tensor shape: {numpy_array.shape}")

    return phiml_tensor


def phiml_to_torch(phiml_tensor):
    """Convert PhiML tensor back to PyTorch tensor"""
    import torch

    # Get native array (numpy or torch)
    native = phiml_tensor.native()

    # Convert to torch if needed
    if not isinstance(native, torch.Tensor):
        native = torch.from_numpy(native)

    return native


class SyntheticTrainer(AbstractTrainer):
    # ... existing __init__ ...

    def _forward_pass(self, batch_tensor):
        """
        Forward pass with PhiML conversion.

        Args:
            batch_tensor: PyTorch tensor from dataloader

        Returns:
            PyTorch tensor (for loss computation)
        """
        # Convert to PhiML
        phiml_input = torch_to_phiml(batch_tensor)

        # Model forward (now expects PhiML tensor)
        phiml_output = self.model(phiml_input)

        # Convert back to PyTorch
        torch_output = phiml_to_torch(phiml_output)

        return torch_output

    def _train_epoch(self, dataloader, optimizer, scheduler):
        """Modified training loop with conversion"""
        self.model.train()  # This doesn't do anything for PhiML models, but keep for now
        epoch_loss = 0.0

        for batch_idx, batch_data in enumerate(dataloader):
            # batch_data is still PyTorch tensors from DataLoader
            inputs, targets = self._prepare_batch(batch_data)

            optimizer.zero_grad()

            # Use conversion wrapper
            predictions = self._forward_pass(inputs)

            # Compute loss (still PyTorch)
            loss = self._compute_loss(predictions, targets)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)
```

### 1.8 Testing Step 1

**Critical Test**:
```bash
python run.py --config-name=burgers.yaml
```

**Expected behavior**:
- ✅ Model creates successfully
- ✅ Training runs without errors
- ✅ Loss decreases over epochs
- ✅ Checkpoints save correctly
- ✅ Results are similar to old implementation

**Debug checklist if it fails**:
1. Check model creation in factory
2. Check tensor conversion shapes
3. Check dimension names are consistent
4. Print intermediate shapes to verify conversion

**Additional validation**:
```bash
# Run with synthetic mode
python run.py --config-name=burgers.yaml general.mode=synthetic trainer.synthetic.epochs=2

# Check model can be created and saved
python -c "
from src.factories.model_factory import ModelFactory
from omegaconf import OmegaConf
config = OmegaConf.load('conf/burgers.yaml')
model = ModelFactory.create_synthetic_model(config)
print('Model created successfully!')
"
```

### 1.9 Deliverables for Step 1

- ✅ `src/models/synthetic/phiml_base.py` created
- ✅ `src/models/synthetic/phiml_unet.py` created
- ✅ `src/models/synthetic/phiml_resnet.py` created
- ✅ `src/models/synthetic/phiml_convnet.py` created
- ✅ `src/factories/model_factory.py` updated
- ✅ `src/training/synthetic/trainer.py` updated with conversion layer
- ✅ `python run.py --config-name=burgers.yaml` works
- ✅ Old PyTorch model files kept as backup (don't delete yet)

---

## Step 2: Synthetic Trainer to PhiML

**Duration**: 2-3 days
**Risk**: Medium
**Goal**: Convert trainer to use phiml.nn training, convert data at trainer entry
**Validation**: `python run.py --config-name=burgers.yaml` must work

### 2.1 Understanding Current Trainer

Current trainer uses:
- `torch.optim.Adam` for optimization
- `torch.utils.data.DataLoader` for batching
- Manual `loss.backward()` and `optimizer.step()`
- PyTorch AMP for mixed precision

### 2.2 Create PhiML Synthetic Trainer

Create `src/training/synthetic/phiml_trainer.py`:

```python
"""
Synthetic trainer using pure PhiML.
No PyTorch training loop, uses phiml.nn.update_weights.
"""

from phiml import math, nn
from phiml.math import batch as batch_dim
import logging
from pathlib import Path
from typing import Optional


def torch_to_phiml(torch_tensor):
    """
    Convert PyTorch tensor to PhiML tensor.

    Args:
        torch_tensor: PyTorch tensor with shape (B, T, H, W, C) or (B, H, W, C)

    Returns:
        PhiML tensor with named dimensions
    """
    from phiml.math import batch, spatial, channel

    # Move to CPU and convert to numpy
    numpy_array = torch_tensor.detach().cpu().numpy()

    # Infer dimensions based on shape
    if numpy_array.ndim == 5:  # (batch, time, height, width, channels)
        B, T, H, W, C = numpy_array.shape
        return math.wrap(
            numpy_array,
            batch(batch=B, time=T),
            spatial(x=H, y=W),
            channel(c=C)
        )
    elif numpy_array.ndim == 4:  # (batch, height, width, channels)
        B, H, W, C = numpy_array.shape
        return math.wrap(
            numpy_array,
            batch(batch=B),
            spatial(x=H, y=W),
            channel(c=C)
        )
    else:
        raise ValueError(f"Unexpected tensor shape: {numpy_array.shape}")


class PhiMLSyntheticTrainer:
    """
    Trainer for synthetic models using pure PhiML.
    Converts data from PyTorch DataLoader at entry point.
    """

    def __init__(self, model, config: dict):
        self.model = model  # PhiML model
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Training config
        trainer_config = config["trainer"]["synthetic"]
        self.num_epochs = trainer_config["epochs"]
        self.learning_rate = trainer_config["learning_rate"]
        self.rollout_steps = config["trainer"]["rollout_steps"]

        # Create PhiML optimizer
        self.optimizer = nn.adam(
            self.model.get_network(),
            learning_rate=self.learning_rate
        )

        # Checkpointing
        checkpoint_dir = config["general"].get("checkpoint_dir", "checkpoints")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.best_loss = float('inf')

        self.logger.info(
            f"Initialized PhiML Synthetic Trainer: "
            f"lr={self.learning_rate}, epochs={self.num_epochs}"
        )

    def loss_function(self, prediction: 'Tensor', target: 'Tensor') -> 'Tensor':
        """
        Compute loss between prediction and target.

        Args:
            prediction: PhiML tensor
            target: PhiML tensor

        Returns:
            Scalar loss (PhiML tensor)
        """
        return math.l2_loss(prediction - target)

    def _compute_rollout_loss(self, initial_state: 'Tensor', target_sequence: 'Tensor') -> 'Tensor':
        """
        Compute autoregressive rollout loss.

        Args:
            initial_state: PhiML tensor (batch, x, y, c)
            target_sequence: PhiML tensor (batch, time, x, y, c)

        Returns:
            Scalar loss
        """
        num_steps = target_sequence.shape.get_size('time')

        # Autoregressive rollout
        current_state = initial_state
        predictions = []

        for t in range(num_steps):
            # Predict next state
            next_state = self.model(current_state)
            predictions.append(next_state)
            current_state = next_state

        # Stack predictions along time dimension
        pred_sequence = math.stack(predictions, batch_dim('time'))

        # Compute loss
        loss = self.loss_function(pred_sequence, target_sequence)

        return loss

    def train_epoch(self, dataloader) -> float:
        """
        Train for one epoch.

        Args:
            dataloader: PyTorch DataLoader (will convert data at entry)

        Returns:
            Average loss for epoch
        """
        epoch_losses = []

        for batch_idx, batch_data in enumerate(dataloader):
            # batch_data is PyTorch tensors from DataLoader
            # Shape: (batch, time, height, width, channels)

            # Convert to PhiML at entry point
            phiml_batch = torch_to_phiml(batch_data)

            # Split into initial state and target sequence
            initial_state = phiml_batch.time[0]  # First timestep
            target_sequence = phiml_batch.time[1:]  # Rest of sequence

            # Define loss function for this batch
            def batch_loss_fn(init, target):
                return self._compute_rollout_loss(init, target)

            # Update weights using PhiML (one line!)
            loss = nn.update_weights(
                self.model.get_network(),
                self.optimizer,
                batch_loss_fn,
                initial_state,
                target_sequence
            )

            epoch_losses.append(float(loss))

            if batch_idx % 10 == 0:
                self.logger.debug(f"Batch {batch_idx}/{len(dataloader)}, Loss: {float(loss):.6f}")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        return avg_loss

    def validate(self, dataloader) -> float:
        """
        Validate on validation set.

        Args:
            dataloader: PyTorch DataLoader

        Returns:
            Average validation loss
        """
        val_losses = []

        for batch_data in dataloader:
            # Convert to PhiML
            phiml_batch = torch_to_phiml(batch_data)

            initial_state = phiml_batch.time[0]
            target_sequence = phiml_batch.time[1:]

            # Forward pass only (no weight update)
            loss = self._compute_rollout_loss(initial_state, target_sequence)
            val_losses.append(float(loss))

        avg_loss = sum(val_losses) / len(val_losses)
        return avg_loss

    def train(self, train_dataloader, val_dataloader: Optional = None):
        """
        Main training loop.

        Args:
            train_dataloader: PyTorch DataLoader for training
            val_dataloader: Optional PyTorch DataLoader for validation
        """
        self.logger.info(f"Starting training for {self.num_epochs} epochs")

        for epoch in range(self.num_epochs):
            # Learning rate scheduling (cosine annealing)
            current_lr = self.learning_rate * 0.5 * (
                1 + math.cos(math.pi * epoch / self.num_epochs)
            )

            # Update optimizer with new learning rate
            self.optimizer = nn.adam(
                self.model.get_network(),
                learning_rate=current_lr
            )

            # Train
            train_loss = self.train_epoch(train_dataloader)

            # Validate
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
            else:
                val_loss = train_loss

            self.logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"LR: {current_lr:.6f}"
            )

            # Checkpointing
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint('best_model.phiml')
                self.logger.info(f"✓ Saved best model (val_loss={val_loss:.6f})")

            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.phiml')

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        path = self.checkpoint_dir / filename
        self.model.save(str(path))

    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = self.checkpoint_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        self.model.load(str(path))
        self.logger.info(f"Loaded checkpoint from {path}")
```

### 2.3 Update Trainer Factory

Modify `src/factories/trainer_factory.py`:

```python
# Add import at top
from src.training.synthetic.phiml_trainer import PhiMLSyntheticTrainer

class TrainerFactory:
    # ... existing code ...

    @staticmethod
    def create_synthetic_trainer(model, config: dict, mode: str = "train"):
        """
        Create synthetic trainer.
        Now returns PhiML trainer.
        """
        # Use PhiML trainer
        trainer = PhiMLSyntheticTrainer(model, config)
        return trainer
```

### 2.4 Update Run Script (if needed)

Verify `run.py` calls trainer correctly:

```python
# In run.py, synthetic training section should look like:

if mode == "synthetic":
    # Create model
    model = ModelFactory.create_synthetic_model(config)

    # Create dataloaders (still PyTorch DataLoader for now)
    train_loader = DataLoaderFactory.create_dataloader(config, mode="train")
    val_loader = DataLoaderFactory.create_dataloader(config, mode="val")

    # Create trainer (now PhiML trainer)
    trainer = TrainerFactory.create_synthetic_trainer(model, config)

    # Train (trainer handles conversion internally)
    trainer.train(train_loader, val_loader)
```

### 2.5 Testing Step 2

**Critical Test**:
```bash
python run.py --config-name=burgers.yaml general.mode=synthetic trainer.synthetic.epochs=5
```

**Expected behavior**:
- ✅ PhiML trainer loads correctly
- ✅ Data converts from PyTorch to PhiML at trainer entry
- ✅ Training loop uses `nn.update_weights` (one-line updates)
- ✅ Loss decreases smoothly
- ✅ Checkpoints save in PhiML format
- ✅ No PyTorch optimizer/backward calls in trainer

**Debug checklist**:
1. Check tensor conversion shapes match
2. Verify optimizer is PhiML optimizer
3. Check loss computation works with PhiML tensors
4. Verify checkpoint saving/loading

**Performance comparison**:
```bash
# Compare training time
# Old trainer (Step 1):
time python run.py --config-name=burgers.yaml general.mode=synthetic trainer.synthetic.epochs=2

# New trainer (Step 2):
time python run.py --config-name=burgers.yaml general.mode=synthetic trainer.synthetic.epochs=2

# Should be similar or slightly faster
```

### 2.6 Deliverables for Step 2

- ✅ `src/training/synthetic/phiml_trainer.py` created
- ✅ `src/factories/trainer_factory.py` updated
- ✅ `python run.py --config-name=burgers.yaml` works with PhiML trainer
- ✅ Training completes successfully
- ✅ Checkpoints save/load correctly
- ✅ Old trainer file kept as backup

---

## Step 3: Data Pipeline Refactor

**Duration**: 3-5 days
**Risk**: High (most complex step)
**Goal**: Refactor data pipeline to use PhiML tensors with dataclass caching
**Validation**: `python run.py --config-name=burgers.yaml` must work

### 3.1 Design: PhiML Dataclass-Based Caching

Following the pattern in [references/cached_parallel_example.py](references/cached_parallel_example.py):

```python
@dataclass
class Simulation:
    """
    Represents a single simulation.
    Uses @cached_property for lazy loading with disk caching.
    """
    sim_index: int
    config: dict

    @cached_property
    def trajectory(self):
        """Loads and caches the full simulation trajectory"""
        # This is computed once and cached to disk
        return load_simulation_from_scene(self.sim_index, self.config)

    @cached_property
    def windows(self):
        """Pre-computed windowed samples from this simulation"""
        # Depends on trajectory (will reuse cached value)
        return create_windows(self.trajectory, self.config["trainer"]["rollout_steps"])
```

### 3.2 Create Simulation Dataclass

Create `src/data/simulation.py`:

```python
"""
Simulation dataclass with cached loading.
Uses PhiML dataclasses for efficient caching.
"""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from phiml import math
from phiml.math import batch, spatial, channel
from phi.flow import Scene, CenteredGrid, StaggeredGrid
import logging


logger = logging.getLogger(__name__)


@dataclass
class Simulation:
    """
    Represents a single simulation with lazy loading and caching.

    Follows the pattern from references/cached_parallel_example.py
    """
    sim_index: int
    config: dict

    @cached_property
    def trajectory(self) -> 'Tensor':
        """
        Load full simulation trajectory as PhiML tensor.

        Returns:
            PhiML Tensor with shape (time=T, x=X, y=Y, channels=C)
        """
        data_config = self.config["data"]
        data_dir = Path(data_config["data_dir"])

        # Load from Scene
        scene_path = data_dir / f"sim_{self.sim_index:04d}"

        if not scene_path.exists():
            raise FileNotFoundError(f"Simulation not found: {scene_path}")

        scene = Scene.at(scene_path)

        logger.info(f"Loading simulation {self.sim_index} from {scene_path}")

        # Get field configuration
        field_names = data_config["fields"]
        trajectory_length = data_config["trajectory_length"]

        # Load all timesteps
        timestep_data = []

        for t in range(trajectory_length):
            frame = scene.read(frame=t)

            # Extract fields and convert to PhiML tensors
            field_tensors = []

            for field_name in field_names:
                if field_name not in frame:
                    raise ValueError(f"Field '{field_name}' not found in frame {t}")

                field = frame[field_name]

                # Get values as PhiML tensor
                # field.values is already a PhiML tensor!
                tensor = field.values

                # Ensure it's 2D spatial
                if 'x' not in tensor.shape or 'y' not in tensor.shape:
                    raise ValueError(f"Field must be 2D. Got shape: {tensor.shape}")

                field_tensors.append(tensor)

            # Stack fields along channel dimension
            frame_tensor = math.stack(field_tensors, channel('c'))

            timestep_data.append(frame_tensor)

        # Stack timesteps along time dimension
        trajectory = math.stack(timestep_data, batch('time'))

        logger.info(f"Loaded trajectory shape: {trajectory.shape}")

        return trajectory

    @cached_property
    def windows(self) -> list:
        """
        Pre-compute all valid sliding windows from this trajectory.

        Returns:
            List of PhiML tensors, each with shape (time=rollout_steps, x, y, c)
        """
        rollout_steps = self.config["trainer"]["rollout_steps"]
        traj = self.trajectory  # Reuses cached trajectory

        num_windows = traj.shape.get_size('time') - rollout_steps + 1

        windows = []
        for start_idx in range(num_windows):
            window = traj.time[start_idx:start_idx + rollout_steps]
            windows.append(window)

        logger.debug(f"Created {len(windows)} windows from simulation {self.sim_index}")

        return windows

    @property
    def num_windows(self) -> int:
        """Number of windowed samples in this simulation"""
        trajectory_length = self.config["data"]["trajectory_length"]
        rollout_steps = self.config["trainer"]["rollout_steps"]
        return trajectory_length - rollout_steps + 1
```

### 3.3 Create Unified PhiML Dataset

Create `src/data/phiml_dataset.py`:

```python
"""
Unified dataset using PhiML tensors.
Works with Simulation dataclasses for efficient caching.
"""

from typing import List, Optional
from phiml import math
from phiml.math import batch
import logging
from src.data.simulation import Simulation


logger = logging.getLogger(__name__)


class PhiMLDataset:
    """
    Unified dataset that works entirely with PhiML tensors.

    Replaces both TensorDataset and FieldDataset.
    Uses Simulation dataclasses for lazy loading and caching.
    """

    def __init__(
        self,
        config: dict,
        mode: str = "train",
        augmented_simulations: Optional[List[Simulation]] = None
    ):
        """
        Args:
            config: Full configuration dict
            mode: 'train', 'val', or 'test'
            augmented_simulations: Optional list of augmented Simulation objects
        """
        self.config = config
        self.mode = mode
        self.logger = logging.getLogger(self.__class__.__name__)

        # Determine which simulations to load
        if mode == "train":
            sim_indices = config["trainer"]["train_sim"]
        elif mode == "val":
            sim_indices = config["trainer"].get("val_sim", config["trainer"]["train_sim"])
        elif mode == "test":
            sim_indices = config["evaluation"]["test_sim"]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Create Simulation objects (lazy loading)
        self.real_simulations = [
            Simulation(sim_idx, config)
            for sim_idx in sim_indices
        ]

        # Augmented simulations (for hybrid training)
        self.augmented_simulations = augmented_simulations or []

        # Build index map for fast access
        self._build_index_map()

        self.logger.info(
            f"Created PhiMLDataset ({mode}): "
            f"{len(self.real_simulations)} real sims, "
            f"{len(self.augmented_simulations)} augmented sims, "
            f"{len(self)} total windows"
        )

    def _build_index_map(self):
        """
        Build mapping from global index to (simulation, window_index).
        This allows O(1) indexing.
        """
        self.index_map = []

        # Add real simulations
        for sim in self.real_simulations:
            for window_idx in range(sim.num_windows):
                self.index_map.append(('real', sim, window_idx))

        # Add augmented simulations
        for sim in self.augmented_simulations:
            for window_idx in range(sim.num_windows):
                self.index_map.append(('augmented', sim, window_idx))

    def __len__(self) -> int:
        """Total number of windowed samples"""
        return len(self.index_map)

    def __getitem__(self, idx: int) -> 'Tensor':
        """
        Get a single windowed sample.

        Args:
            idx: Global index

        Returns:
            PhiML Tensor with shape (time=rollout_steps, x, y, c)
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        source, sim, window_idx = self.index_map[idx]

        # Get window from simulation
        # This will trigger loading if not cached
        window = sim.windows[window_idx]

        return window

    def get_batch(self, indices: List[int]) -> 'Tensor':
        """
        Get multiple samples as a batch.

        Args:
            indices: List of global indices

        Returns:
            PhiML Tensor with shape (batch=len(indices), time, x, y, c)
        """
        windows = [self[idx] for idx in indices]

        # Stack along batch dimension
        batched = math.stack(windows, batch('batch'))

        return batched

    def enable_augmentation(self, augmented_sims: List[Simulation]):
        """
        Add augmented simulations to the dataset.

        Args:
            augmented_sims: List of Simulation objects
        """
        self.augmented_simulations = augmented_sims
        self._build_index_map()
        self.logger.info(f"Enabled augmentation: {len(augmented_sims)} augmented sims")

    def disable_augmentation(self):
        """Remove augmented simulations from dataset"""
        self.augmented_simulations = []
        self._build_index_map()
        self.logger.info("Disabled augmentation")

    def get_real_only_dataset(self):
        """Get a dataset with only real simulations"""
        return PhiMLDataset(self.config, self.mode, augmented_simulations=None)

    def get_augmented_only_dataset(self):
        """Get a dataset with only augmented simulations"""
        dataset = PhiMLDataset.__new__(PhiMLDataset)
        dataset.config = self.config
        dataset.mode = self.mode
        dataset.logger = self.logger
        dataset.real_simulations = []
        dataset.augmented_simulations = self.augmented_simulations
        dataset._build_index_map()
        return dataset
```

### 3.4 Create PhiML-Compatible DataLoader

Create `src/data/phiml_dataloader.py`:

```python
"""
DataLoader compatible with PhiML tensors.

Provides batching for PhiMLDataset without requiring PyTorch DataLoader.
"""

import random
from typing import Iterator
from phiml import math
from phiml.math import batch
import logging


logger = logging.getLogger(__name__)


class PhiMLDataLoader:
    """
    DataLoader for PhiMLDataset.

    Provides batching and shuffling without PyTorch dependency.
    Can also wrap a PyTorch-style interface for compatibility.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        """
        Args:
            dataset: PhiMLDataset instance
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle indices each epoch
            drop_last: Whether to drop incomplete last batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        """Number of batches"""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator:
        """Iterate over batches"""
        # Get all indices
        indices = list(range(len(self.dataset)))

        # Shuffle if requested
        if self.shuffle:
            random.shuffle(indices)

        # Yield batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]

            # Skip incomplete batch if drop_last
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            # Get batch from dataset
            batch_data = self.dataset.get_batch(batch_indices)

            yield batch_data


class TorchCompatibleDataLoader:
    """
    Wrapper that provides PyTorch DataLoader-compatible interface.

    This allows gradual migration - trainer can still use DataLoader interface,
    but data is PhiML underneath.
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        """
        Args:
            dataset: PhiMLDataset
            batch_size: Batch size
            shuffle: Whether to shuffle
        """
        self.phiml_loader = PhiMLDataLoader(dataset, batch_size, shuffle)

    def __len__(self):
        return len(self.phiml_loader)

    def __iter__(self):
        """
        Iterate and convert PhiML tensors to PyTorch for compatibility.
        """
        import torch

        for phiml_batch in self.phiml_loader:
            # Convert PhiML tensor to PyTorch tensor
            numpy_array = phiml_batch.native()
            torch_tensor = torch.from_numpy(numpy_array).float()

            yield torch_tensor
```

### 3.5 Update DataLoader Factory

Modify `src/factories/dataloader_factory.py`:

```python
"""
Factory for creating dataloaders.
Now supports PhiML-based dataloaders.
"""

from src.data.phiml_dataset import PhiMLDataset
from src.data.phiml_dataloader import PhiMLDataLoader, TorchCompatibleDataLoader


class DataLoaderFactory:
    """Factory for creating dataloaders"""

    @staticmethod
    def create_dataloader(config: dict, mode: str = "train"):
        """
        Create dataloader.

        For Step 3, we create PhiML-based dataloader but wrap it
        in TorchCompatibleDataLoader for backward compatibility.

        Args:
            config: Configuration dict
            mode: 'train', 'val', or 'test'

        Returns:
            DataLoader (TorchCompatible wrapper for now)
        """
        # Create PhiML dataset
        dataset = PhiMLDataset(config, mode=mode)

        # Get batch size
        batch_size = config["trainer"]["batch_size"]

        # Wrap in compatible loader for backward compatibility
        # This allows existing trainer code to work without changes
        loader = TorchCompatibleDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == "train")
        )

        return loader

    @staticmethod
    def create_phiml_dataloader(config: dict, mode: str = "train"):
        """
        Create pure PhiML dataloader (for future use).

        Args:
            config: Configuration dict
            mode: 'train', 'val', or 'test'

        Returns:
            PhiMLDataLoader
        """
        dataset = PhiMLDataset(config, mode=mode)

        batch_size = config["trainer"]["batch_size"]

        loader = PhiMLDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == "train")
        )

        return loader
```

### 3.6 Configure Parallel Caching

Create `src/data/cache_config.py`:

```python
"""
Configuration for PhiML dataclass caching.
Based on references/cached_parallel_example.py
"""

from phiml.dataclasses import set_cache_ttl, load_cache_as
import logging


logger = logging.getLogger(__name__)


def configure_caching(config: dict):
    """
    Configure PhiML caching for efficient data loading.

    Args:
        config: Full configuration dict
    """
    cache_config = config.get("cache", {})

    # Set cache time-to-live (how long to keep in memory)
    ttl = cache_config.get("ttl", 10.0)  # seconds
    set_cache_ttl(ttl)
    logger.info(f"Set cache TTL to {ttl}s")

    # Configure cache backend
    backend = cache_config.get("backend", "torch")  # or 'numpy', 'jax'
    worker_backend = cache_config.get("worker_backend", "numpy")

    load_cache_as(backend, worker_backend=worker_backend)
    logger.info(f"Cache backend: {backend}, worker backend: {worker_backend}")
```

Add to config YAML:

```yaml
cache:
  ttl: 10.0  # seconds
  backend: "torch"  # Main process backend
  worker_backend: "numpy"  # Worker backend (faster for CPU)
```

### 3.7 Update Run Script

Modify `run.py` to use new dataloader:

```python
# At the top, configure caching
from src.data.cache_config import configure_caching
configure_caching(config)

# Rest of code remains the same - factory handles the switch
train_loader = DataLoaderFactory.create_dataloader(config, mode="train")
val_loader = DataLoaderFactory.create_dataloader(config, mode="val")

# These now return TorchCompatibleDataLoader wrapping PhiML data
# Trainer code doesn't need to change!
```

### 3.8 Testing Step 3

**Critical Test**:
```bash
python run.py --config-name=burgers.yaml
```

**Expected behavior**:
- ✅ Data loads from Scene files
- ✅ Simulation dataclasses cache trajectories
- ✅ Dataset provides windowed samples
- ✅ DataLoader batches correctly
- ✅ Trainer receives data in expected format
- ✅ Training completes successfully

**Debug checklist**:
1. Check simulation files exist and load
2. Verify caching is working (check cache directory)
3. Check tensor shapes at each stage
4. Verify batching produces correct shapes
5. Test with and without augmentation

**Performance test**:
```bash
# First run (no cache)
time python run.py --config-name=burgers.yaml trainer.synthetic.epochs=1

# Second run (with cache)
time python run.py --config-name=burgers.yaml trainer.synthetic.epochs=1

# Second run should be significantly faster!
```

**Cache validation**:
```python
# Check cache is being used
import os
cache_dir = "cache"  # or from config
cache_files = os.listdir(cache_dir)
print(f"Cached files: {len(cache_files)}")
# Should see cached trajectory data
```

### 3.9 Optional: Remove PyTorch DataLoader Dependency

Once Step 3 works, optionally update trainer to use pure PhiML loader:

```python
# In phiml_trainer.py, update to accept PhiMLDataLoader directly
def train_epoch(self, dataloader) -> float:
    """
    Train for one epoch.

    Args:
        dataloader: PhiMLDataLoader (no conversion needed!)

    Returns:
        Average loss for epoch
    """
    epoch_losses = []

    for batch_data in dataloader:
        # batch_data is already a PhiML tensor!
        # No conversion needed anymore!

        initial_state = batch_data.time[0]
        target_sequence = batch_data.time[1:]

        # Rest is same...
```

Update factory:
```python
# In dataloader_factory.py
@staticmethod
def create_dataloader(config: dict, mode: str = "train"):
    """Now returns pure PhiMLDataLoader"""
    return DataLoaderFactory.create_phiml_dataloader(config, mode)
```

### 3.10 Deliverables for Step 3

- ✅ `src/data/simulation.py` created (dataclass with cached_property)
- ✅ `src/data/phiml_dataset.py` created (unified dataset)
- ✅ `src/data/phiml_dataloader.py` created
- ✅ `src/data/cache_config.py` created
- ✅ `src/factories/dataloader_factory.py` updated
- ✅ `run.py` updated with cache configuration
- ✅ `python run.py --config-name=burgers.yaml` works
- ✅ Caching is functional (verify second run is faster)
- ✅ Data pipeline is pure PhiML (no PyTorch tensors until compatibility layer)

---

## Future Work: Physical Models Integration

After Step 3 is complete, the physical models can be integrated:

### 4.1 Current State of Physical Models

Physical models already use PhiFlow/PhiML heavily:
- Use `phi.flow.Field` objects
- Use `@jit_compile` for performance
- Use `math.minimize` for parameter optimization

### 4.2 Integration Strategy

1. **Create PhysicalSimulation dataclass** (similar to Simulation)
2. **Update PhysicalTrainer** to work with PhiMLDataset
3. **Modify HybridTrainer** to use unified dataset

### 4.3 Hybrid Trainer Migration

Once data pipeline is pure PhiML:

```python
class PhiMLHybridTrainer:
    def train(self, dataset: PhiMLDataset):
        """
        Much simpler - no conversions needed!

        Everything is PhiML tensors:
        1. Real data from dataset (PhiML)
        2. Physical model generates (PhiML Fields)
        3. Synthetic model predicts (PhiML tensors)
        4. All compatible without conversion!
        """
```

---

## Testing Strategy

### Per-Step Testing

After each step:

1. **Smoke test**: `python run.py --config-name=burgers.yaml`
2. **Quick training**: Run 2-3 epochs, check loss decreases
3. **Checkpoint test**: Save and load model
4. **Shape verification**: Print tensor shapes at key points

### Integration Testing

After all steps:

```bash
# Full training run
python run.py --config-name=burgers.yaml trainer.synthetic.epochs=20

# Hybrid training
python run.py --config-name=burgers.yaml general.mode=hybrid trainer.hybrid.cycles=3

# Evaluation
python run.py --config-name=burgers.yaml general.mode=evaluate
```

### Performance Benchmarking

```python
# Create benchmark.py
import time

def benchmark():
    # Before migration
    start = time.time()
    run_old_version()
    old_time = time.time() - start

    # After migration
    start = time.time()
    run_new_version()
    new_time = time.time() - start

    print(f"Speedup: {old_time/new_time:.2f}x")
```

---

## Rollback Plan

### Per-Step Rollback

Each step keeps old files:

**Step 1**:
- Keep `src/models/synthetic/{base,unet,resnet,convnet}.py` as backup
- Keep `src/training/synthetic/trainer.py` as backup

**Step 2**:
- If issues, revert `trainer_factory.py` to use old trainer
- Old trainer still exists as backup

**Step 3**:
- Keep `src/data/{data_manager,tensor_dataset,field_dataset}.py`
- If issues, revert factory to use old data pipeline

### Git Strategy

```bash
# Create branches for each step
git checkout -b step-1-models
# ... implement step 1 ...
git commit -m "Step 1: Migrate synthetic models to PhiML"

git checkout -b step-2-trainer
# ... implement step 2 ...
git commit -m "Step 2: Migrate synthetic trainer to PhiML"

git checkout -b step-3-data
# ... implement step 3 ...
git commit -m "Step 3: Migrate data pipeline to PhiML"

# If step fails, can easily revert
git checkout step-2-trainer  # Go back to previous step
```

---

## Success Criteria

### Step 1 Success Criteria
- ✅ `python run.py --config-name=burgers.yaml` completes
- ✅ Models are PhiML-native (no nn.Module)
- ✅ Training loss decreases normally
- ✅ Results match old implementation (within tolerance)

### Step 2 Success Criteria
- ✅ Trainer uses `nn.update_weights` (no manual backward)
- ✅ Training completes without errors
- ✅ Checkpoints save in PhiML format
- ✅ Performance is similar or better

### Step 3 Success Criteria
- ✅ Data loads as PhiML tensors from Scene files
- ✅ Caching works (second run faster)
- ✅ Dataset provides correct windowed samples
- ✅ No PyTorch tensors in data pipeline (except compatibility layer)
- ✅ All training modes work (synthetic, physical, hybrid)

### Overall Success Criteria
- ✅ ~30% code reduction
- ✅ 15-20% performance improvement
- ✅ All tests pass
- ✅ Documentation updated

---

## Timeline

| Step | Duration | Cumulative | Complexity |
|------|----------|-----------|------------|
| Step 1: Models | 1-2 days | 2 days | Low |
| Step 2: Trainer | 2-3 days | 5 days | Medium |
| Step 3: Data | 3-5 days | 10 days | High |
| **Total** | **6-10 days** | **~1.5 weeks** | |

---

## Conclusion

This revised migration plan takes a **backward approach**:

1. **Start at the end (models)** - easiest to migrate, add conversion layer
2. **Move to middle (trainer)** - use PhiML training, convert at entry
3. **Finish at beginning (data)** - most complex, but only touch once

**Advantages of this approach**:
- ✅ Each step is testable independently
- ✅ Can stop at any step and still have working system
- ✅ Conversion boundaries move systematically backward
- ✅ Less risky than changing everything at once
- ✅ Uses PhiML dataclass caching pattern from references

**Key differences from V1**:
- Backward instead of forward migration
- Uses `@cached_property` dataclasses for caching
- Unified dataset from the start
- Physical models integrated later (after core pipeline stable)
- More incremental and safer

**Next step**: Begin Step 1 - migrate synthetic models to PhiML!
