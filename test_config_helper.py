"""Quick test to verify ConfigHelper is properly implemented."""

from pathlib import Path
from src.config import ConfigHelper

# Create a sample config structure
test_config = {
    'project_root': '.',
    'data': {
        'dset_name': 'burgers_128',
        'data_dir': 'data',
        'fields': ['velocity', 'density', 'inflow'],
        'validate_cache': True,
        'auto_clear_invalid': False,
    },
    'model': {
        'synthetic': {
            'input_specs': {
                'velocity': {'channels': 2},
                'density': {'channels': 1},
                'inflow': {'channels': 1},
            },
            'output_specs': {
                'velocity': {'channels': 2},
                'density': {'channels': 1},
            },
        },
    },
    'run_params': {
        'model_type': 'synthetic',
    },
    'trainer_params': {
        'train_sim': [0, 1, 2, 3, 4],
        'val_sim': [5, 6],
        'batch_size': 16,
        'num_predict_steps': 10,
        'use_sliding_window': True,
        'augmentation': {
            'enabled': True,
            'alpha': 0.1,
            'strategy': 'cached',
            'cache': {
                'experiment_name': 'burgers_hybrid',
            },
        },
    },
    'cache': {
        'root': 'data/cache',
    },
}

print("Testing ConfigHelper...")

# Initialize
cfg = ConfigHelper(test_config)
print("✓ ConfigHelper initialized")

# Test data config methods
assert cfg.get_dataset_name() == 'burgers_128'
assert cfg.get_field_names() == ['velocity', 'density', 'inflow']
assert cfg.get_raw_data_dir() == Path('data/burgers_128')
assert cfg.get_cache_dir() == Path('data/cache')
assert cfg.should_validate_cache() == True
print("✓ Data configuration methods work")

# Test field type extraction
dynamic, static = cfg.get_field_types()
assert dynamic == ['velocity', 'density']
assert static == ['inflow']
print(f"✓ Field types: dynamic={dynamic}, static={static}")

# Test training config methods
assert cfg.get_train_sim_indices() == [0, 1, 2, 3, 4]
assert cfg.get_val_sim_indices() == [5, 6]
assert cfg.get_batch_size() == 16
assert cfg.get_num_predict_steps() == 10
assert cfg.should_use_sliding_window() == True
assert cfg.get_num_frames(use_sliding_window=True) is None
assert cfg.get_num_frames(use_sliding_window=False) == 11
print("✓ Training configuration methods work")

# Test augmentation config methods
assert cfg.is_augmentation_enabled() == True
assert cfg.get_augmentation_alpha() == 0.1
assert cfg.get_augmentation_strategy() == 'cached'
assert cfg.get_augmentation_mode() == 'cache'
aug_config = cfg.get_augmentation_config()
assert aug_config is not None
assert aug_config['mode'] == 'cache'
assert aug_config['alpha'] == 0.1
assert 'burgers_hybrid' in aug_config['cache_dir']
print(f"✓ Augmentation config: {aug_config}")

# Test model config methods
assert cfg.get_model_type() == 'synthetic'
assert cfg.is_hybrid_training() == False
print("✓ Model configuration methods work")

# Test validation
issues = cfg.validate()
assert len(issues) == 0, f"Validation failed: {issues}"
print("✓ Config validation passed")

# Test summary
summary = cfg.get_summary()
assert summary['dataset_name'] == 'burgers_128'
assert summary['train_sims'] == 5
assert summary['augmentation_enabled'] == True
print(f"✓ Config summary generated: {len(summary)} keys")

# Test with disabled augmentation
test_config['trainer_params']['augmentation']['enabled'] = False
cfg2 = ConfigHelper(test_config)
assert cfg2.is_augmentation_enabled() == False
assert cfg2.get_augmentation_config() is None
print("✓ Disabled augmentation handled correctly")

# Test with physical model (all fields dynamic)
test_config['run_params']['model_type'] = 'physical'
cfg3 = ConfigHelper(test_config)
dynamic, static = cfg3.get_field_types()
assert dynamic == ['velocity', 'density', 'inflow']
assert static == []
print(f"✓ Physical model field types: all dynamic={dynamic}")

print("\n✅ All ConfigHelper tests passed!")
