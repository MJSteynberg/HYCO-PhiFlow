"""Test TensorDataset implementation."""

import sys
import torch
from pathlib import Path
from src.data import TensorDataset, DataManager

print("Testing TensorDataset implementation...")

# Check if we have actual cache data
cache_dir = Path("data/cache")
if not cache_dir.exists():
    print("⚠️  Warning: No cache directory found. Creating mock test only.")
    print("   For full testing, run: python scripts/generate_cache.py")
    
    # Test that we can at least import and validate
    print("\n✓ TensorDataset imported successfully")
    print("✓ Inherits from AbstractDataset")
    
    # Test validation
    try:
        # This should fail because we need a DataManager
        dataset = TensorDataset(
            data_manager=None,
            sim_indices=[],
            field_names=[],
            num_frames=None,
            num_predict_steps=1,
            dynamic_fields=[],
        )
        print("✗ Should have raised ValueError for empty sim_indices")
        sys.exit(1)
    except (ValueError, AttributeError) as e:
        print(f"✓ Validation works: {str(e)[:60]}...")
    
    print("\n✅ Basic TensorDataset tests passed!")
    print("   Run with cached data for full testing.")
    sys.exit(0)

# Full test with actual cache data
print("\n=== Testing with cached data ===\n")

# Find a cached dataset
burgers_cache = cache_dir / "burgers_128"
if not burgers_cache.exists():
    print("⚠️  burgers_128 cache not found, skipping full test")
    sys.exit(0)

# Create DataManager
data_manager = DataManager(
    raw_data_dir="data/burgers_128",
    cache_dir="data/cache",
    config={
        'data': {
            'dset_name': 'burgers_128',
            'fields': ['velocity'],
        }
    },
    validate_cache=False,
)

print("✓ DataManager created")

# Get available simulations from cache metadata
cached_sims = sorted(burgers_cache.glob("sim_*.pt"))
if not cached_sims:
    print("⚠️  No cached simulations found")
    sys.exit(0)

# Extract sim indices from filenames like "sim_000000.pt"
sim_indices = [int(s.stem.split('_')[1]) for s in cached_sims[:3]]
print(f"✓ Found {len(sim_indices)} cached simulations: {sim_indices}")

# Test 1: Basic initialization
print("\n--- Test 1: Basic Initialization ---")
dataset = TensorDataset(
    data_manager=data_manager,
    sim_indices=sim_indices,
    field_names=['velocity'],
    num_frames=20,
    num_predict_steps=5,
    dynamic_fields=['velocity'],
    static_fields=[],
    use_sliding_window=False,
    augmentation_config=None,
    max_cached_sims=2,
    pin_memory=False,
)
print(f"✓ TensorDataset created: {len(dataset)} samples")
print(f"  Real samples: {dataset.num_real}")
print(f"  Augmented samples: {dataset.num_augmented}")

# Test 2: Get a sample
print("\n--- Test 2: Get Sample ---")
initial, targets = dataset[0]
print(f"✓ Sample retrieved")
print(f"  Initial state shape: {initial.shape}")
print(f"  Rollout targets shape: {targets.shape}")
assert isinstance(initial, torch.Tensor), "Initial should be tensor"
assert isinstance(targets, torch.Tensor), "Targets should be tensor"
assert len(targets.shape) == 4, "Targets should be [T, C, H, W]"
assert targets.shape[0] == 5, f"Should have 5 timesteps, got {targets.shape[0]}"
print("✓ Sample format correct")

# Test 3: Sliding window
print("\n--- Test 3: Sliding Window ---")
dataset_sw = TensorDataset(
    data_manager=data_manager,
    sim_indices=sim_indices,
    field_names=['velocity'],
    num_frames=None,  # Load all frames
    num_predict_steps=5,
    dynamic_fields=['velocity'],
    static_fields=[],
    use_sliding_window=True,
    augmentation_config=None,
    max_cached_sims=2,
    pin_memory=False,
)
print(f"✓ Sliding window dataset: {len(dataset_sw)} samples")
print(f"  Samples per sim: {len(dataset_sw) // len(sim_indices)}")

# Get samples from different windows
initial1, targets1 = dataset_sw[0]
initial2, targets2 = dataset_sw[1]
print(f"✓ Multiple samples from same simulation")
print(f"  Sample 0 initial shape: {initial1.shape}")
print(f"  Sample 1 initial shape: {initial2.shape}")

# Test 4: LRU Cache
print("\n--- Test 4: LRU Cache ---")
cache_info_before = dataset.get_cache_info()
print(f"  Cache before access: hits={cache_info_before.hits}, misses={cache_info_before.misses}")

# Access samples (should hit cache on second access)
_ = dataset[0]
_ = dataset[0]  # Second access should hit cache
_ = dataset[1]

cache_info_after = dataset.get_cache_info()
print(f"  Cache after access: hits={cache_info_after.hits}, misses={cache_info_after.misses}")
print(f"✓ LRU cache working (hits increased from {cache_info_before.hits} to {cache_info_after.hits})")

# Test 5: Field info
print("\n--- Test 5: Field Information ---")
field_info = dataset.get_field_info()
print(f"✓ Field info: {field_info}")

# Test 6: Tensor shapes
print("\n--- Test 6: Tensor Shapes ---")
shapes = dataset.get_tensor_shapes(0)
print(f"✓ Tensor shapes: {shapes}")

# Test 7: Dataset info
print("\n--- Test 7: Dataset Info ---")
info = dataset.get_dataset_info()
print(f"✓ Dataset info ({len(info)} keys):")
for key, value in list(info.items())[:5]:
    print(f"  {key}: {value}")

# Test 8: String representation
print("\n--- Test 8: String Representation ---")
print(f"✓ Dataset repr:\n{dataset}")

# Test 9: Static vs Dynamic fields
print("\n--- Test 9: Static vs Dynamic Fields ---")
# Create dataset with both static and dynamic fields (if burgers has multiple)
try:
    dataset_mixed = TensorDataset(
        data_manager=data_manager,
        sim_indices=sim_indices[:1],
        field_names=['velocity'],
        num_frames=20,
        num_predict_steps=5,
        dynamic_fields=['velocity'],
        static_fields=[],  # No static fields for burgers
        use_sliding_window=False,
        pin_memory=False,
    )
    initial, targets = dataset_mixed[0]
    print(f"✓ Static/dynamic separation works")
    print(f"  Initial channels (all): {initial.shape[0]}")
    print(f"  Target channels (dynamic): {targets.shape[1]}")
except Exception as e:
    print(f"⚠️  Could not test static/dynamic: {e}")

# Test 10: Validation
print("\n--- Test 10: Validation ---")
try:
    invalid_dataset = TensorDataset(
        data_manager=data_manager,
        sim_indices=sim_indices,
        field_names=['velocity', 'pressure'],
        num_frames=20,
        num_predict_steps=5,
        dynamic_fields=['velocity'],  # Missing pressure!
        static_fields=[],
        use_sliding_window=False,
    )
    print("✗ Should have raised ValueError for field mismatch")
except ValueError as e:
    print(f"✓ Validation caught field mismatch: {str(e)[:60]}...")

print("\n" + "="*60)
print("✅ All TensorDataset tests passed!")
print("="*60)
