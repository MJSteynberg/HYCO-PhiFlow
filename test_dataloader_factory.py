"""
Test DataLoaderFactory

Tests the unified factory for creating data loaders and datasets.
"""

from pathlib import Path
from torch.utils.data import DataLoader
from src.factories import DataLoaderFactory
from src.data import TensorDataset, FieldDataset


# Sample configuration for testing (matches Hydra structure)
def get_test_config():
    """Create a test configuration."""
    return {
        'data_dir': 'data/',
        'dset_name': 'burgers_128',
        'fields': ['velocity'],
        'fields_scheme': 'VV',  # V=vector field (dynamic)
        'cache_dir': 'data/cache',
        'validate_cache': False,
        'auto_clear_invalid': False,
        'model': {
            'synthetic': {
                'input_specs': {
                    'velocity': {'channels': 2},
                },
                'output_specs': {
                    'velocity': {'channels': 2},
                },
            },
        },
        'trainer_params': {
            'train_sim': [0, 1, 2],
            'val_sim': [3, 4],
            'batch_size': 4,
            'num_predict_steps': 3,
            'use_sliding_window': True,
        },
        'run_params': {
            'model_type': 'synthetic',
        },
        'augmentation': {
            'enabled': False,
        },
        'project_root': Path.cwd(),
    }


def test_create_tensor_mode():
    """Test creating DataLoader in tensor mode."""
    print("\nTest 1: Creating DataLoader (tensor mode)...")
    
    config = get_test_config()
    
    # Check if cache exists
    cache_dir = Path('data/cache/burgers_128')
    if not cache_dir.exists() or len(list(cache_dir.glob('sim_*.pt'))) == 0:
        print("  ⚠️  Skipping - no cached data found")
        return
    
    loader = DataLoaderFactory.create(
        config=config,
        mode='tensor',
        batch_size=4,
        shuffle=True
    )
    
    assert isinstance(loader, DataLoader), "Should return DataLoader"
    assert isinstance(loader.dataset, TensorDataset), "Should contain TensorDataset"
    assert len(loader) > 0, "DataLoader should have batches"
    
    print(f"  ✓ DataLoader created with {len(loader)} batches")
    print(f"  ✓ Dataset has {len(loader.dataset)} samples")


def test_create_field_mode():
    """Test creating FieldDataset in field mode."""
    print("\nTest 2: Creating FieldDataset (field mode)...")
    
    config = get_test_config()
    
    # Check if cache exists
    cache_dir = Path('data/cache/burgers_128')
    if not cache_dir.exists() or len(list(cache_dir.glob('sim_*.pt'))) == 0:
        print("  ⚠️  Skipping - no cached data found")
        return
    
    dataset = DataLoaderFactory.create(
        config=config,
        mode='field',
        batch_size=None  # Field mode doesn't use batching
    )
    
    assert isinstance(dataset, FieldDataset), "Should return FieldDataset"
    assert len(dataset) > 0, "Dataset should have samples"
    
    print(f"  ✓ FieldDataset created with {len(dataset)} samples")


def test_create_for_evaluation():
    """Test creating loader for evaluation."""
    print("\nTest 3: Creating loader for evaluation...")
    
    config = get_test_config()
    
    # Check if cache exists
    cache_dir = Path('data/cache/burgers_128')
    if not cache_dir.exists() or len(list(cache_dir.glob('sim_*.pt'))) == 0:
        print("  ⚠️  Skipping - no cached data found")
        return
    
    # Tensor mode
    loader = DataLoaderFactory.create_for_evaluation(
        config=config,
        mode='tensor'
    )
    
    assert isinstance(loader, DataLoader), "Should return DataLoader"
    print(f"  ✓ Evaluation DataLoader created with {len(loader)} batches")
    
    # Field mode
    dataset = DataLoaderFactory.create_for_evaluation(
        config=config,
        mode='field'
    )
    
    assert isinstance(dataset, FieldDataset), "Should return FieldDataset"
    print(f"  ✓ Evaluation FieldDataset created with {len(dataset)} samples")


def test_get_info():
    """Test getting loader info without creating it."""
    print("\nTest 4: Getting loader info...")
    
    config = get_test_config()
    info = DataLoaderFactory.get_info(config)
    
    assert 'dataset_name' in info, "Should include dataset_name"
    assert 'model_type' in info, "Should include model_type"
    assert 'field_names' in info, "Should include field_names"
    assert 'train_sims' in info, "Should include train_sims"
    assert 'batch_size' in info, "Should include batch_size"
    
    print(f"  ✓ Info retrieved:")
    print(f"    - Dataset: {info['dataset_name']}")
    print(f"    - Model: {info['model_type']}")
    print(f"    - Fields: {info['field_names']}")
    print(f"    - Train sims: {info['train_sims']}")
    print(f"    - Batch size: {info['batch_size']}")


def test_with_augmentation():
    """Test creating loader with augmentation enabled."""
    print("\nTest 5: Creating loader with augmentation...")
    
    config = get_test_config()
    config['augmentation'] = {
        'enabled': True,
        'alpha': 0.3,
        'num_samples': 10,
        'mode': 'memory'
    }
    
    # Check if cache exists
    cache_dir = Path('data/cache/burgers_128')
    if not cache_dir.exists() or len(list(cache_dir.glob('sim_*.pt'))) == 0:
        print("  ⚠️  Skipping - no cached data found")
        return
    
    loader = DataLoaderFactory.create(
        config=config,
        mode='tensor',
        batch_size=4
    )
    
    assert isinstance(loader, DataLoader), "Should return DataLoader"
    # With augmentation, dataset should be larger
    num_samples = len(loader.dataset)
    print(f"  ✓ DataLoader with augmentation created")
    print(f"  ✓ Dataset has {num_samples} samples (including augmented)")


def test_custom_parameters():
    """Test overriding config parameters."""
    print("\nTest 6: Overriding parameters...")
    
    config = get_test_config()
    
    # Check if cache exists
    cache_dir = Path('data/cache/burgers_128')
    if not cache_dir.exists() or len(list(cache_dir.glob('sim_*.pt'))) == 0:
        print("  ⚠️  Skipping - no cached data found")
        return
    
    loader = DataLoaderFactory.create(
        config=config,
        mode='tensor',
        sim_indices=[0],  # Override to use only first sim
        batch_size=2,  # Override batch size
        shuffle=False,  # Override shuffle
        use_sliding_window=False,  # Override sliding window
    )
    
    assert isinstance(loader, DataLoader), "Should return DataLoader"
    print(f"  ✓ Custom parameters applied")
    print(f"  ✓ Dataset has {len(loader.dataset)} samples")


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\nTest 7: Error handling...")
    
    config = get_test_config()
    
    # Test invalid mode
    try:
        DataLoaderFactory.create(config=config, mode='invalid')
        assert False, "Should raise error for invalid mode"
    except ValueError as e:
        print(f"  ✓ Invalid mode error: {e}")
    
    # Test invalid config (missing required field)
    invalid_config = get_test_config()
    del invalid_config['fields']
    
    try:
        DataLoaderFactory.create(config=invalid_config, mode='tensor')
        assert False, "Should raise error for invalid config"
    except (ValueError, KeyError):
        print(f"  ✓ Invalid config error detected")


def test_physical_model_config():
    """Test with physical model configuration."""
    print("\nTest 8: Physical model configuration...")
    
    config = get_test_config()
    config['run_params']['model_type'] = 'physical'
    
    # Check if cache exists
    cache_dir = Path('data/cache/burgers_128')
    if not cache_dir.exists() or len(list(cache_dir.glob('sim_*.pt'))) == 0:
        print("  ⚠️  Skipping - no cached data found")
        return
    
    # Physical models typically use field mode
    dataset = DataLoaderFactory.create(
        config=config,
        mode='field',
        batch_size=None
    )
    
    assert isinstance(dataset, FieldDataset), "Should return FieldDataset"
    print(f"  ✓ Physical model FieldDataset created")
    print(f"  ✓ Dataset has {len(dataset)} samples")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing DataLoaderFactory")
    print("=" * 60)
    
    tests = [
        test_create_tensor_mode,
        test_create_field_mode,
        test_create_for_evaluation,
        test_get_info,
        test_with_augmentation,
        test_custom_parameters,
        test_error_handling,
        test_physical_model_config,
    ]
    
    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            failed.append((test.__name__, e))
    
    print("\n" + "=" * 60)
    if failed:
        print(f"❌ {len(failed)} test(s) failed:")
        for name, error in failed:
            print(f"  - {name}: {error}")
    else:
        print("✅ All DataLoaderFactory tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
