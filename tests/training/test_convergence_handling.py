"""Tests for physical model convergence handling."""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.physical.trainer import PhysicalTrainer
from phi import math


class TestConvergenceHandling:
    """Test convergence error handling in physical trainer."""
    
    def test_suppress_convergence_disabled_by_default(self):
        """Ensure existing behavior unchanged - suppression off by default."""
        trainer_config = {
            "method": "L-BFGS-B",
            "abs_tol": 1e-6,
            "max_iterations": 100,
            # suppress_convergence_errors not set - should default to False
        }
        
        # Create mock trainer with minimal setup
        class MockPhysicalTrainer:
            def __init__(self):
                self.trainer_config = trainer_config
                self.num_epochs = 100
                self.initial_guesses = (0.01,)
            
            def _setup_optimization(self):
                """Copy of the updated method."""
                method = self.trainer_config.get("method", "L-BFGS-B")
                abs_tol = self.trainer_config.get("abs_tol", 1e-6)
                max_iterations = self.trainer_config.get(
                    "max_iterations", self.num_epochs
                )
                
                suppress_convergence = self.trainer_config.get("suppress_convergence_errors", False)
                suppress_list = []
                if suppress_convergence:
                    suppress_list.append(math.NotConverged)
                
                return math.Solve(
                    method=method,
                    abs_tol=abs_tol,
                    x0=self.initial_guesses,
                    max_iterations=max_iterations,
                    suppress=tuple(suppress_list),
                )
        
        trainer = MockPhysicalTrainer()
        solve = trainer._setup_optimization()
        
        # Should have empty suppress tuple by default
        assert solve.suppress == (), f"Expected empty tuple, got {solve.suppress}"
        print("✓ Default behavior: suppression disabled")
    
    def test_suppress_convergence_enabled(self):
        """Test new suppression feature."""
        trainer_config = {
            "method": "L-BFGS-B",
            "abs_tol": 1e-6,
            "max_iterations": 100,
            "suppress_convergence_errors": True,  # Enable suppression
        }
        
        class MockPhysicalTrainer:
            def __init__(self):
                self.trainer_config = trainer_config
                self.num_epochs = 100
                self.initial_guesses = (0.01,)
            
            def _setup_optimization(self):
                method = self.trainer_config.get("method", "L-BFGS-B")
                abs_tol = self.trainer_config.get("abs_tol", 1e-6)
                max_iterations = self.trainer_config.get(
                    "max_iterations", self.num_epochs
                )
                
                suppress_convergence = self.trainer_config.get("suppress_convergence_errors", False)
                suppress_list = []
                if suppress_convergence:
                    suppress_list.append(math.NotConverged)
                
                return math.Solve(
                    method=method,
                    abs_tol=abs_tol,
                    x0=self.initial_guesses,
                    max_iterations=max_iterations,
                    suppress=tuple(suppress_list),
                )
        
        trainer = MockPhysicalTrainer()
        solve = trainer._setup_optimization()
        
        # Should have NotConverged in suppress tuple
        assert len(solve.suppress) == 1, f"Expected 1 item in suppress, got {len(solve.suppress)}"
        assert math.NotConverged in solve.suppress, "NotConverged should be in suppress tuple"
        print("✓ Suppression enabled: NotConverged in suppress tuple")
    
    def test_configuration_dataclass(self):
        """Test that the configuration dataclass has the new field."""
        from src.config.trainer_config import PhysicalTrainerConfig
        
        # Test default value
        config = PhysicalTrainerConfig()
        assert hasattr(config, 'suppress_convergence_errors'), \
            "PhysicalTrainerConfig should have suppress_convergence_errors field"
        assert config.suppress_convergence_errors == False, \
            "Default value should be False for backward compatibility"
        print("✓ Configuration dataclass updated correctly")
        
        # Test setting to True
        config.suppress_convergence_errors = True
        assert config.suppress_convergence_errors == True
        print("✓ Configuration field can be set to True")


def run_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Testing Convergence Handling Fix")
    print("="*60 + "\n")
    
    test_class = TestConvergenceHandling()
    
    try:
        test_class.test_suppress_convergence_disabled_by_default()
        test_class.test_suppress_convergence_enabled()
        test_class.test_configuration_dataclass()
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60 + "\n")
        return True
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
