"""Test that diffusion coefficient value correctly affects diffusion rate."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import numpy as np
from src.factories.model_factory import ModelFactory
from src.utils.logger import setup_logger
from phi.math import math

# Setup logging
logger = setup_logger("test_diffusion_scaling", level="INFO")

def test_diffusion_scaling():
    """Test that different D values produce different diffusion rates."""
    logger.info("="*60)
    logger.info("Testing Diffusion Coefficient Scaling")
    logger.info("="*60)

    # Load config
    config_path = PROJECT_ROOT / "conf" / "burgers_1d.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create model
    model = ModelFactory.create_physical_model(config)

    # Get initial state
    initial_state = model.get_initial_state(batch_size=1)

    # Test with different diffusion coefficients
    test_coefficients = [0.001, 0.01, 0.1]
    final_states = []

    for D_scale in test_coefficients:
        # Modify config to use scalar diffusion with specific value
        config['model']['physical']['pde_params']['type'] = 'scalar'
        config['model']['physical']['pde_params']['value'] = str(D_scale)

        # Create model with this diffusion coefficient
        model = ModelFactory.create_physical_model(config)
        params = model.get_real_params()

        logger.info(f"\nTesting with D = {D_scale}")
        logger.info(f"  Params: {params}")

        # Run simulation for 10 steps
        trajectory = model.rollout(initial_state, params, num_steps=10)

        # Measure variance (diffusion should reduce variance)
        initial_var = float(math.std(initial_state))
        final_var = float(math.std(trajectory.time[-1]))
        variance_reduction = (initial_var - final_var) / initial_var

        logger.info(f"  Initial variance: {initial_var:.6f}")
        logger.info(f"  Final variance: {final_var:.6f}")
        logger.info(f"  Variance reduction: {variance_reduction*100:.2f}%")

        final_states.append((D_scale, variance_reduction))

    # Check that higher diffusion -> more variance reduction
    logger.info("\n" + "="*60)
    logger.info("Verification:")
    logger.info("  Higher diffusion should cause more smoothing (variance reduction)")

    for i in range(len(final_states) - 1):
        D1, var_red1 = final_states[i]
        D2, var_red2 = final_states[i + 1]

        if var_red2 > var_red1:
            logger.info(f"  ✓ D={D2} causes more smoothing than D={D1}")
        else:
            logger.error(f"  ✗ FAILED: D={D2} should cause more smoothing than D={D1}")
            logger.error(f"     But got {var_red2:.4f} vs {var_red1:.4f}")
            raise AssertionError("Diffusion scaling is not working correctly!")

    logger.info("="*60)
    logger.info("✓ Diffusion scaling test PASSED!")
    logger.info("  The diffusion coefficient value correctly affects the rate!")
    logger.info("="*60)

if __name__ == "__main__":
    try:
        test_diffusion_scaling()
    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
