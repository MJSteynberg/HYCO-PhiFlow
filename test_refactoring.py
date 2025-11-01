"""
Quick test script to verify the refactored trainer hierarchy works.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all new classes can be imported."""
    print("Testing imports...")
    
    try:
        from src.training.abstract_trainer import AbstractTrainer
        print("✓ AbstractTrainer imported successfully")
    except Exception as e:
        print(f"✗ Failed to import AbstractTrainer: {e}")
        return False
    
    try:
        from src.training.tensor_trainer import TensorTrainer
        print("✓ TensorTrainer imported successfully")
    except Exception as e:
        print(f"✗ Failed to import TensorTrainer: {e}")
        return False
    
    try:
        from src.training.field_trainer import FieldTrainer
        print("✓ FieldTrainer imported successfully")
    except Exception as e:
        print(f"✗ Failed to import FieldTrainer: {e}")
        return False
    
    try:
        from src.training.synthetic.trainer import SyntheticTrainer
        print("✓ SyntheticTrainer imported successfully")
    except Exception as e:
        print(f"✗ Failed to import SyntheticTrainer: {e}")
        return False
    
    try:
        from src.training.physical.trainer import PhysicalTrainer
        print("✓ PhysicalTrainer imported successfully")
    except Exception as e:
        print(f"✗ Failed to import PhysicalTrainer: {e}")
        return False
    
    try:
        from src.factories.trainer_factory import TrainerFactory
        print("✓ TrainerFactory imported successfully")
    except Exception as e:
        print(f"✗ Failed to import TrainerFactory: {e}")
        return False
    
    return True

def test_hierarchy():
    """Test that the inheritance hierarchy is correct."""
    print("\nTesting inheritance hierarchy...")
    
    from src.training.abstract_trainer import AbstractTrainer
    from src.training.tensor_trainer import TensorTrainer
    from src.training.field_trainer import FieldTrainer
    from src.training.synthetic.trainer import SyntheticTrainer
    from src.training.physical.trainer import PhysicalTrainer
    
    # Test TensorTrainer inherits from AbstractTrainer
    if issubclass(TensorTrainer, AbstractTrainer):
        print("✓ TensorTrainer inherits from AbstractTrainer")
    else:
        print("✗ TensorTrainer does not inherit from AbstractTrainer")
        return False
    
    # Test FieldTrainer inherits from AbstractTrainer
    if issubclass(FieldTrainer, AbstractTrainer):
        print("✓ FieldTrainer inherits from AbstractTrainer")
    else:
        print("✗ FieldTrainer does not inherit from AbstractTrainer")
        return False
    
    # Test SyntheticTrainer inherits from TensorTrainer
    if issubclass(SyntheticTrainer, TensorTrainer):
        print("✓ SyntheticTrainer inherits from TensorTrainer")
    else:
        print("✗ SyntheticTrainer does not inherit from TensorTrainer")
        return False
    
    # Test PhysicalTrainer inherits from FieldTrainer
    if issubclass(PhysicalTrainer, FieldTrainer):
        print("✓ PhysicalTrainer inherits from FieldTrainer")
    else:
        print("✗ PhysicalTrainer does not inherit from FieldTrainer")
        return False
    
    # Test that SyntheticTrainer is also an AbstractTrainer
    if issubclass(SyntheticTrainer, AbstractTrainer):
        print("✓ SyntheticTrainer inherits from AbstractTrainer (transitive)")
    else:
        print("✗ SyntheticTrainer does not inherit from AbstractTrainer")
        return False
    
    # Test that PhysicalTrainer is also an AbstractTrainer
    if issubclass(PhysicalTrainer, AbstractTrainer):
        print("✓ PhysicalTrainer inherits from AbstractTrainer (transitive)")
    else:
        print("✗ PhysicalTrainer does not inherit from AbstractTrainer")
        return False
    
    return True

def test_trainer_factory():
    """Test that TrainerFactory works with the new hierarchy."""
    print("\nTesting TrainerFactory...")
    
    from src.factories.trainer_factory import TrainerFactory
    
    # Check available trainers
    available = TrainerFactory.list_available_trainers()
    print(f"Available trainers: {available}")
    
    if 'synthetic' in available and 'physical' in available:
        print("✓ TrainerFactory has both synthetic and physical trainers registered")
    else:
        print("✗ TrainerFactory missing expected trainers")
        return False
    
    return True

def test_abstract_methods():
    """Test that abstract methods are properly defined."""
    print("\nTesting abstract methods...")
    
    from src.training.abstract_trainer import AbstractTrainer
    from src.training.tensor_trainer import TensorTrainer
    from src.training.field_trainer import FieldTrainer
    import inspect
    
    # Check AbstractTrainer has train() as abstract
    abstract_methods = AbstractTrainer.__abstractmethods__
    if 'train' in abstract_methods:
        print("✓ AbstractTrainer.train() is abstract")
    else:
        print("✗ AbstractTrainer.train() is not abstract")
        return False
    
    # Check TensorTrainer has expected abstract methods
    tensor_abstract = TensorTrainer.__abstractmethods__
    expected = {'_create_model', '_create_data_loader', '_train_epoch'}
    if expected.issubset(tensor_abstract):
        print(f"✓ TensorTrainer has expected abstract methods: {expected}")
    else:
        print(f"✗ TensorTrainer missing abstract methods. Has: {tensor_abstract}, Expected: {expected}")
        return False
    
    # Check FieldTrainer has expected abstract methods
    field_abstract = FieldTrainer.__abstractmethods__
    expected = {'_create_data_manager', '_create_model', '_setup_optimization'}
    if expected.issubset(field_abstract):
        print(f"✓ FieldTrainer has expected abstract methods: {expected}")
    else:
        print(f"✗ FieldTrainer missing abstract methods. Has: {field_abstract}, Expected: {expected}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("="*60)
    print("Testing Refactored Trainer Hierarchy")
    print("="*60)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_hierarchy():
        all_passed = False
    
    if not test_trainer_factory():
        all_passed = False
    
    if not test_abstract_methods():
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*60)
        return 1

if __name__ == "__main__":
    exit(main())
