# Step 4 Complete: DataLoaderFactory

## Overview
Successfully implemented **DataLoaderFactory** - the unified entry point for creating data loaders and datasets in HYCO-PhiFlow.

## Files Created/Modified

### New Files
1. **src/factories/dataloader_factory.py** (~310 lines)
   - Single unified factory with `.create()` method
   - Replaces 4 separate factory methods from TrainerFactory
   - Uses ConfigHelper for all config extraction
   - Clear mode parameter: 'tensor' or 'field'

2. **test_dataloader_factory.py** (~300 lines)
   - 8 comprehensive tests covering all use cases
   - Tests tensor and field modes
   - Tests evaluation mode
   - Tests parameter overrides
   - Tests error handling
   - All tests passing ✅

### Modified Files
1. **src/factories/__init__.py**
   - Added DataLoaderFactory to exports

2. **src/config/config_helper.py**
   - Updated `__init__()` to handle both nested and flat config structures
   - Now works with both test configs (`{data: {fields: ...}}`) and real Hydra configs (`{fields: ...}`)

## Key Features

### DataLoaderFactory.create()
```python
# Synthetic training (returns DataLoader)
loader = DataLoaderFactory.create(
    config, 
    mode='tensor',
    sim_indices=[0, 1, 2],
    batch_size=16,
    shuffle=True
)

# Physical training (returns FieldDataset)
dataset = DataLoaderFactory.create(
    config,
    mode='field',
    sim_indices=[0, 1, 2],
    batch_size=None
)
```

### DataLoaderFactory.create_for_evaluation()
```python
# Convenience method with evaluation defaults
loader = DataLoaderFactory.create_for_evaluation(
    config,
    mode='tensor'  # No shuffling, no augmentation
)
```

### DataLoaderFactory.get_info()
```python
# Get info without creating loader
info = DataLoaderFactory.get_info(config)
# Returns: dataset_name, model_type, fields, train_sims, etc.
```

## Test Results

All 8 tests passed successfully:

1. ✅ **test_create_tensor_mode**: DataLoader created with 74 batches, 294 samples
2. ✅ **test_create_field_mode**: FieldDataset created with 294 samples
3. ✅ **test_create_for_evaluation**: Both modes work for evaluation
4. ✅ **test_get_info**: Info retrieval without instantiation
5. ✅ **test_with_augmentation**: Augmentation config passed correctly
6. ✅ **test_custom_parameters**: Parameter overrides work correctly
7. ✅ **test_error_handling**: Invalid modes and configs raise appropriate errors
8. ✅ **test_physical_model_config**: Physical models use field mode correctly

## Architecture

```
DataLoaderFactory
├── Uses ConfigHelper for config extraction
├── Creates DataManager
├── Determines field types (dynamic/static)
├── Routes to appropriate dataset:
│   ├── TensorDataset (tensor mode)
│   │   └── Wraps in DataLoader
│   └── FieldDataset (field mode)
│       └── Returns dataset directly
```

## Simplifications Achieved

### Before (TrainerFactory)
```python
# 4 separate methods
TrainerFactory.create_data_loader_for_synthetic(config, use_sliding_window)
TrainerFactory.create_dataset_for_physical(config, use_sliding_window)
TrainerFactory._create_data_manager(config)
TrainerFactory._get_field_names(config)
# Direct config access scattered throughout
```

### After (DataLoaderFactory)
```python
# 1 unified method
DataLoaderFactory.create(config, mode='tensor'|'field')
# All config extraction via ConfigHelper
# Clear, consistent interface
```

## Benefits

1. **Single Responsibility**: Factory only creates loaders/datasets
2. **Clear Interface**: Single `create()` method with explicit mode parameter
3. **Reduced Coupling**: ConfigHelper handles all config access
4. **Easy Testing**: Simple to test with mock configs
5. **Maintainability**: Changes to config structure only affect ConfigHelper
6. **Flexibility**: Easy to add new modes or parameters

## Integration Points

### Current Usage (to be updated in Step 6)
```python
# Old way (TrainerFactory)
data_loader = TrainerFactory.create_data_loader_for_synthetic(config, True)

# New way (DataLoaderFactory)
data_loader = DataLoaderFactory.create(config, mode='tensor')
```

### Future Usage in Trainers
```python
class SyntheticTrainer:
    def _prepare_data(self):
        self.data_loader = DataLoaderFactory.create(
            self.config,
            mode='tensor',
            shuffle=True
        )

class PhysicalTrainer:
    def _prepare_data(self):
        self.dataset = DataLoaderFactory.create(
            self.config,
            mode='field',
            batch_size=None
        )
```

## Next Steps

**Step 5**: Remove old classes
- Mark HybridDataset as deprecated
- Remove AdaptiveAugmentedDataLoader wrapper
- Clean up augmentation wrappers

**Step 6**: Update trainers
- Modify SyntheticTrainer to use DataLoaderFactory
- Modify PhysicalTrainer to use DataLoaderFactory
- Modify HybridTrainer to use DataLoaderFactory

**Step 7**: Update run.py
- Replace TrainerFactory data methods with DataLoaderFactory

## Stats

- **Lines of Code**: ~310
- **Test Coverage**: 8/8 tests passing
- **Methods**: 4 (create, create_for_evaluation, get_info, _create_data_manager)
- **Modes Supported**: 2 (tensor, field)
- **Config Formats**: 2 (nested, flat)

---

**Status**: ✅ **COMPLETE** - DataLoaderFactory is fully implemented and tested
**Date**: November 3, 2025
**Branch**: feature/data-loading-simplification
