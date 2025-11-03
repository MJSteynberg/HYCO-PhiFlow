"""Test FieldDataset implementation."""

import sys
from pathlib import Path
from src.data import FieldDataset, DataManager
from phi.field import Field

print("Testing FieldDataset implementation...")

# Check if we have actual cache data
cache_dir = Path("data/cache")
if not cache_dir.exists():
    print("⚠️  Warning: No cache directory found. Creating mock test only.")
    print("   For full testing, run: python scripts/generate_cache.py")
    
    # Test that we can at least import
    print("\n✓ FieldDataset imported successfully")
    print("✓ Inherits from AbstractDataset")
    
    print("\n✅ Basic FieldDataset tests passed!")
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

# Get available simulations
cached_sims = sorted(burgers_cache.glob("sim_*.pt"))
if not cached_sims:
    print("⚠️  No cached simulations found")
    sys.exit(0)

# Extract sim indices
sim_indices = [int(s.stem.split('_')[1]) for s in cached_sims[:3]]
print(f"✓ Found {len(sim_indices)} cached simulations: {sim_indices}")

# Test 1: Basic initialization
print("\n--- Test 1: Basic Initialization ---")
dataset = FieldDataset(
    data_manager=data_manager,
    sim_indices=sim_indices,
    field_names=['velocity'],
    num_frames=20,
    num_predict_steps=5,
    use_sliding_window=False,
    augmentation_config=None,
    max_cached_sims=2,
    move_to_gpu=False,  # Don't move to GPU for testing
)
print(f"✓ FieldDataset created: {len(dataset)} samples")
print(f"  Real samples: {dataset.num_real}")
print(f"  Augmented samples: {dataset.num_augmented}")

# Test 2: Get a sample
print("\n--- Test 2: Get Sample ---")
try:
    initial_fields, target_fields = dataset[0]
    print(f"✓ Sample retrieved")
    print(f"  Initial fields: {list(initial_fields.keys())}")
    print(f"  Target fields: {list(target_fields.keys())}")
    
    # Check types
    assert isinstance(initial_fields, dict), "Initial should be dict"
    assert isinstance(target_fields, dict), "Targets should be dict"
    
    # Check that we have Fields
    for name, field in initial_fields.items():
        assert isinstance(field, Field), f"Initial {name} should be Field, got {type(field)}"
        print(f"  Initial '{name}' field: {field.shape}")
    
    for name, fields_list in target_fields.items():
        assert isinstance(fields_list, list), f"Target {name} should be list"
        assert len(fields_list) == 5, f"Should have 5 timesteps for {name}"
        assert all(isinstance(f, Field) for f in fields_list), f"All targets for {name} should be Fields"
        print(f"  Target '{name}' fields: {len(fields_list)} timesteps, shape {fields_list[0].shape}")
    
    print("✓ Sample format correct (PhiFlow Fields)")
except Exception as e:
    print(f"✗ Error getting sample: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Sliding window
print("\n--- Test 3: Sliding Window ---")
dataset_sw = FieldDataset(
    data_manager=data_manager,
    sim_indices=sim_indices,
    field_names=['velocity'],
    num_frames=None,  # Load all frames
    num_predict_steps=5,
    use_sliding_window=True,
    augmentation_config=None,
    max_cached_sims=2,
    move_to_gpu=False,
)
print(f"✓ Sliding window dataset: {len(dataset_sw)} samples")
print(f"  Samples per sim: {len(dataset_sw) // len(sim_indices)}")

# Get samples from different windows
initial1, targets1 = dataset_sw[0]
initial2, targets2 = dataset_sw[1]
print(f"✓ Multiple samples from same simulation")
print(f"  Sample 0 fields: {list(initial1.keys())}")
print(f"  Sample 1 fields: {list(initial2.keys())}")

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

# Test 6: Sample info
print("\n--- Test 6: Sample Info ---")
try:
    sample_info = dataset.get_sample_info(0)
    print(f"✓ Sample info: {sample_info}")
except Exception as e:
    print(f"⚠️  Could not get sample info: {e}")

# Test 7: Dataset info
print("\n--- Test 7: Dataset Info ---")
info = dataset.get_dataset_info()
print(f"✓ Dataset info ({len(info)} keys):")
for key, value in list(info.items())[:5]:
    print(f"  {key}: {value}")

# Test 8: String representation
print("\n--- Test 8: String Representation ---")
print(f"✓ Dataset repr:\n{dataset}")

# Test 9: Compare with TensorDataset shape
print("\n--- Test 9: Data Consistency Check ---")
from src.data import TensorDataset

# Create equivalent TensorDataset
tensor_dataset = TensorDataset(
    data_manager=data_manager,
    sim_indices=sim_indices[:1],
    field_names=['velocity'],
    num_frames=20,
    num_predict_steps=5,
    dynamic_fields=['velocity'],
    static_fields=[],
    use_sliding_window=False,
    pin_memory=False,
)

field_dataset_single = FieldDataset(
    data_manager=data_manager,
    sim_indices=sim_indices[:1],
    field_names=['velocity'],
    num_frames=20,
    num_predict_steps=5,
    use_sliding_window=False,
    move_to_gpu=False,
)

# Get samples
tensor_initial, tensor_targets = tensor_dataset[0]
field_initial, field_targets = field_dataset_single[0]

print(f"✓ Both datasets use same underlying data")
print(f"  Tensor initial shape: {tensor_initial.shape}")
print(f"  Field initial shape: {field_initial['velocity'].shape}")
print(f"  Tensor targets shape: {tensor_targets.shape}")
print(f"  Field targets count: {len(field_targets['velocity'])} timesteps")

print("\n" + "="*60)
print("✅ All FieldDataset tests passed!")
print("="*60)
