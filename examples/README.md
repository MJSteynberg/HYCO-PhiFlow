# Examples

This folder contains example scripts demonstrating how to use various features of the HYCO-PhiFlow project.

## Available Examples

### Performance Monitoring Examples
**File**: `performance_monitoring_examples.py`

Demonstrates how to use the performance monitoring tools to track timing and memory usage:

- **Example 1**: Using `PerformanceMonitor` for multiple operations
- **Example 2**: Aggregated operation tracking (repeated operations with statistics)
- **Example 3**: Using the `track_performance` context manager
- **Example 4**: Using the `@monitor_performance` decorator
- **Example 5**: Using `EpochPerformanceMonitor` in training loops
- **Example 6**: Bottleneck identification with percentage analysis

**Run it**:
```bash
python examples\performance_monitoring_examples.py
```

## Adding New Examples

When adding new example files:

1. Place them in this `examples/` folder
2. Add proper documentation at the top of the file
3. Include usage instructions in this README
4. Make sure they can be run from the project root directory
5. Use relative imports: `from src.module import ...`

## Example Template

```python
"""
Example Name

Brief description of what this example demonstrates.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.module import SomeClass

def main():
    # Your example code here
    pass

if __name__ == "__main__":
    main()
```
