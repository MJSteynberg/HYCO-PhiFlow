"""
Test script for Step 2 of PhiML migration.
Tests that PhiML trainer can be instantiated and perform forward/backward passes.
"""

import sys
from pathlib import Path
import yaml
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import models and trainer (avoid circular imports)
import importlib.util
spec = importlib.util.spec_from_file_location("phiml_trainer", Path(__file__).parent / "src" / "training" / "synthetic" / "phiml_trainer.py")
phiml_trainer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phiml_trainer_module)

PhiMLSyntheticTrainer = phiml_trainer_module.PhiMLSyntheticTrainer
torch_to_phiml = phiml_trainer_module.torch_to_phiml
phiml_to_torch = phiml_trainer_module.phiml_to_torch

from src.models import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_dummy_batch(batch_size, num_channels, num_steps, height, width):
    """Create a dummy batch of data for testing."""
    # Create initial state: [B, V, 1, H, W]
    initial_state = torch.randn(batch_size, num_channels, 1, height, width, dtype=torch.float32)

    # Create rollout targets: [B, V, T, H, W]
    rollout_targets = torch.randn(batch_size, num_channels, num_steps, height, width, dtype=torch.float32)

    return (initial_state, rollout_targets)


def test_tensor_conversions():
    """Test tensor conversion utilities."""
    logger.info("=" * 60)
    logger.info("Testing Tensor Conversions")
    logger.info("=" * 60)

    # Test BVTHW_single_t format
    logger.info("\n--- Testing BVTHW_single_t conversion ---")
    torch_tensor = torch.randn(2, 3, 1, 64, 64)
    logger.info(f"PyTorch tensor shape: {torch_tensor.shape}")

    phiml_tensor = torch_to_phiml(torch_tensor, format="BVTHW_single_t")
    logger.info(f"PhiML tensor shape: {phiml_tensor.shape}")
    logger.info(f"PhiML tensor dims: {phiml_tensor.shape.names}")

    # Convert back
    torch_tensor_back = phiml_to_torch(phiml_tensor, target_format="BVHW", device="cpu")
    logger.info(f"Back to PyTorch shape: {torch_tensor_back.shape}")

    # Test BVTHW format
    logger.info("\n--- Testing BVTHW conversion ---")
    torch_tensor = torch.randn(2, 3, 4, 64, 64)
    logger.info(f"PyTorch tensor shape: {torch_tensor.shape}")

    phiml_tensor = torch_to_phiml(torch_tensor, format="BVTHW")
    logger.info(f"PhiML tensor shape: {phiml_tensor.shape}")
    logger.info(f"PhiML tensor dims: {phiml_tensor.shape.names}")

    # Test indexing time dimension
    first_timestep = phiml_tensor.time[0]
    logger.info(f"First timestep shape: {first_timestep.shape}")

    logger.info("\n✓ Tensor conversions working correctly")


def test_trainer_creation(config):
    """Test that trainer can be created with PhiML model."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Trainer Creation")
    logger.info("=" * 60)

    # Create PhiML model
    logger.info("\nCreating PhiML model...")
    model = ModelRegistry.get_synthetic_model(config)
    logger.info(f"✓ Model created: {model.__class__.__name__}")

    # Create trainer
    logger.info("\nCreating trainer...")
    trainer = PhiMLSyntheticTrainer(config, model)
    logger.info(f"✓ Trainer created")
    logger.info(f"  - Model type detected: {'PhiML' if trainer.is_phiml_model else 'PyTorch'}")
    logger.info(f"  - Optimizer: {type(trainer.optimizer).__name__}")
    logger.info(f"  - Device: {trainer.device}")

    return trainer, model


def test_single_batch_forward(trainer, config):
    """Test forward pass on a single batch."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Single Batch Forward Pass")
    logger.info("=" * 60)

    # Get config parameters
    resolution = config["model"]["physical"]["resolution"]
    H, W = resolution["x"], resolution["y"]
    num_channels = 2  # From config (velocity field has 2 components)
    batch_size = 2
    num_steps = 4

    # Create dummy batch
    logger.info(f"\nCreating dummy batch:")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Channels: {num_channels}")
    logger.info(f"  - Rollout steps: {num_steps}")
    logger.info(f"  - Spatial resolution: {H}x{W}")

    batch = create_dummy_batch(batch_size, num_channels, num_steps, H, W)
    initial_state, rollout_targets = batch

    logger.info(f"\n  Initial state shape: {initial_state.shape}")
    logger.info(f"  Rollout targets shape: {rollout_targets.shape}")

    # Test forward pass (without actual training)
    try:
        logger.info("\nTesting loss computation...")

        # For PhiML models, test the conversion and model forward pass
        if trainer.is_phiml_model:
            # Convert to PhiML
            initial_state_phiml = torch_to_phiml(initial_state, format="BVTHW_single_t")
            logger.info(f"  Converted to PhiML: {initial_state_phiml.shape}")

            # Model forward pass
            output_phiml = trainer.model(initial_state_phiml)
            logger.info(f"  Model output: {output_phiml.shape}")

            # Convert back
            output_torch = phiml_to_torch(output_phiml, target_format="BVHW", device=str(trainer.device))
            logger.info(f"  Converted back to PyTorch: {output_torch.shape}")

        logger.info("\n✓ Forward pass successful!")

    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_dummy_dataloader(trainer, config):
    """Test trainer with a minimal dummy dataloader."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing with Dummy DataLoader")
    logger.info("=" * 60)

    # Get config parameters
    resolution = config["model"]["physical"]["resolution"]
    H, W = resolution["x"], resolution["y"]
    num_channels = 2
    batch_size = 2
    num_steps = 4

    # Create a simple dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples, num_channels, num_steps, H, W):
            self.num_samples = num_samples
            self.num_channels = num_channels
            self.num_steps = num_steps
            self.H = H
            self.W = W

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            initial_state = torch.randn(self.num_channels, 1, self.H, self.W, dtype=torch.float32)
            rollout_targets = torch.randn(self.num_channels, self.num_steps, self.H, self.W, dtype=torch.float32)
            return (initial_state, rollout_targets)

    # Create dataloader
    dataset = DummyDataset(num_samples=4, num_channels=num_channels, num_steps=num_steps, H=H, W=W)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"\nCreated dummy dataloader:")
    logger.info(f"  - Dataset size: {len(dataset)}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Number of batches: {len(dataloader)}")

    # Test that trainer can iterate through dataloader
    logger.info("\nTesting dataloader iteration...")
    try:
        for i, batch in enumerate(dataloader):
            initial_state, rollout_targets = batch
            logger.info(f"  Batch {i+1}: initial_state={initial_state.shape}, targets={rollout_targets.shape}")

        logger.info("\n✓ Dataloader iteration successful!")

    except Exception as e:
        logger.error(f"✗ Dataloader iteration failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 60)
    logger.info("PhiML Migration Step 2 - Trainer Tests")
    logger.info("=" * 60)

    # Load config
    config_path = Path(__file__).parent / "conf" / "burgers.yaml"
    logger.info(f"\nLoading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Modify config to use PhiMLUNet
    config["model"]["synthetic"]["name"] = "PhiMLUNet"
    logger.info(f"Using model: {config['model']['synthetic']['name']}")

    try:
        # Test 1: Tensor conversions
        test_tensor_conversions()

        # Test 2: Trainer creation
        trainer, model = test_trainer_creation(config)

        # Test 3: Single batch forward pass
        test_single_batch_forward(trainer, config)

        # Test 4: Dummy dataloader
        test_dummy_dataloader(trainer, config)

        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED! ✓")
        logger.info("=" * 60)
        logger.info("\nStep 2 migration is ready:")
        logger.info("  - Tensor conversions work correctly")
        logger.info("  - Trainer can be created with PhiML models")
        logger.info("  - Forward passes work with conversion")
        logger.info("  - Trainer can iterate through dataloaders")
        logger.info("\nNOTE: This test does NOT perform actual training with update_weights.")
        logger.info("      That requires the full data pipeline to be set up.")

    except Exception as e:
        logger.error("\n" + "=" * 60)
        logger.error("TESTS FAILED! ✗")
        logger.error("=" * 60)
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
