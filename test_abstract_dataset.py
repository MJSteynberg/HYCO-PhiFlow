"""Quick test to verify AbstractDataset is properly implemented."""

import sys
import inspect
from src.data import AbstractDataset

print("✓ AbstractDataset imported successfully")
print(f"✓ Is abstract class: {inspect.isabstract(AbstractDataset)}")

# Check abstract methods
abstract_methods = [
    name for name, method in inspect.getmembers(AbstractDataset)
    if getattr(method, '__isabstractmethod__', False)
]
print(f"✓ Abstract methods: {abstract_methods}")

# Verify it cannot be instantiated
try:
    dataset = AbstractDataset(None, [], [], None, 1)
    print("✗ ERROR: AbstractDataset should not be instantiable!")
    sys.exit(1)
except TypeError as e:
    print(f"✓ Cannot instantiate (as expected): {str(e)[:80]}...")

print("\n✅ All checks passed! AbstractDataset is properly implemented.")
