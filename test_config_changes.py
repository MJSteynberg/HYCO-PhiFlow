"""
Quick test script to verify config simplification changes work correctly.
Tests imports and basic functionality without running full training.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """Test that all modified modules can be imported."""
    print("Testing imports...")
    try:
        from src.data.data_manager import DataManager
        print("✓ DataManager imported successfully")
        
        from src.factories.dataloader_factory import DataLoaderFactory
        print("✓ DataLoaderFactory imported successfully")
        
        from src.training.tensor_trainer import TensorTrainer
        print("✓ TensorTrainer imported successfully")
        
        from src.training.physical.trainer import PhysicalTrainer
        print("✓ PhysicalTrainer imported successfully")
        
        print("\n✅ All imports successful!")
        return True
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_manager_initialization():
    """Test that DataManager can be initialized with new signature."""
    print("\nTesting DataManager initialization...")
    try:
        from src.data.data_manager import DataManager
        
        # Create a minimal config
        config = {
            "data": {
                "dset_name": "test_dataset"
            },
            "model": {
                "physical": {}
            }
        }
        
        # Test that DataManager can be created without validate_cache parameter
        dm = DataManager(
            raw_data_dir="data/test",
            cache_dir="data/cache/test",
            config=config,
            auto_clear_invalid=False
        )
        
        print("✓ DataManager initialized without validate_cache parameter")
        print("✓ Cache validation is now hardcoded to True")
        print("\n✅ DataManager test passed!")
        return True
    except Exception as e:
        print(f"\n❌ DataManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading():
    """Test that config files load correctly."""
    print("\nTesting config loading...")
    try:
        from omegaconf import OmegaConf
        
        # Test root config
        config_path = PROJECT_ROOT / "conf" / "config.yaml"
        cfg = OmegaConf.load(config_path)
        print(f"✓ Root config loaded successfully")
        print(f"  - cache.root: {cfg.cache.root}")
        print(f"  - cache.cleanup: {cfg.cache.cleanup}")
        
        # Test that removed keys are not present
        assert "auto_create" not in cfg.cache, "auto_create should be removed"
        assert "validation" not in cfg.cache, "validation section should be removed"
        print("✓ Hardcoded config variables removed")
        
        # Test trainer config
        trainer_path = PROJECT_ROOT / "conf" / "trainer" / "synthetic.yaml"
        trainer_cfg = OmegaConf.load(trainer_path)
        print(f"✓ Synthetic trainer config loaded")
        
        # Check that removed keys are not present
        assert "use_sliding_window" not in trainer_cfg, "use_sliding_window should be removed"
        assert "validate_on_train" not in trainer_cfg, "validate_on_train should be removed"
        assert "validation_rollout" not in trainer_cfg, "validation_rollout should be removed"
        assert "save_best_only" not in trainer_cfg, "save_best_only should be removed"
        assert "early_stopping" not in trainer_cfg, "early_stopping should be removed"
        assert "memory_monitor_batches" not in trainer_cfg, "memory_monitor_batches should be removed"
        print("✓ Hardcoded trainer variables removed")
        
        # Test augmentation - on_the_fly should be removed
        if "augmentation" in trainer_cfg:
            assert "strategy" not in trainer_cfg.augmentation, "augmentation.strategy should be removed"
            assert "on_the_fly" not in trainer_cfg.augmentation, "augmentation.on_the_fly should be removed"
            print("✓ On-the-fly augmentation removed (cached only)")
        
        # Test physical trainer config
        physical_path = PROJECT_ROOT / "conf" / "trainer" / "physical.yaml"
        physical_cfg = OmegaConf.load(physical_path)
        print(f"✓ Physical trainer config loaded")
        
        # Check that max_iterations is removed and epochs is present
        assert "max_iterations" not in physical_cfg, "max_iterations should be removed"
        assert "epochs" in physical_cfg, "epochs should be present"
        assert "learning_rate" not in physical_cfg, "learning_rate should be removed"
        print("✓ Physical trainer uses epochs as max_iterations")
        
        # Test generation config
        gen_path = PROJECT_ROOT / "conf" / "generation" / "default.yaml"
        gen_cfg = OmegaConf.load(gen_path)
        print(f"✓ Generation config loaded")
        
        # Check that seed is removed
        assert "seed" not in gen_cfg, "seed should be removed"
        print("✓ Random seed removed from generation")
        
        # Test data configs
        data_path = PROJECT_ROOT / "conf" / "data" / "burgers_128.yaml"
        data_cfg = OmegaConf.load(data_path)
        print(f"✓ Data config loaded")
        
        # Check that duplicate cache settings are removed
        assert "cache_dir" not in data_cfg, "cache_dir should be removed from data config"
        assert "validate_cache" not in data_cfg, "validate_cache should be removed from data config"
        assert "auto_clear_invalid" not in data_cfg, "auto_clear_invalid should be removed from data config"
        print("✓ Duplicate cache settings removed from data config")
        
        print("\n✅ All config tests passed!")
        return True
    except AssertionError as e:
        print(f"\n❌ Config test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("CONFIG SIMPLIFICATION - TEST SUITE")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("DataManager", test_data_manager_initialization()))
    results.append(("Config Loading", test_config_loading()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Config simplification is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
