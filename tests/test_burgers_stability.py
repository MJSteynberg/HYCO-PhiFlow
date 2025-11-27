
import os
import sys
import torch
from phi.flow import *
from phi import math

# Add src to path
sys.path.append(os.getcwd())

from src.models.physical.burgers import BurgersModel

def test_stability():
    # Config for 1D Burgers
    config = {
        "model": {
            "physical": {
                "domain": {
                    "dimensions": {
                        "x": {"size": 100, "resolution": 128}
                    }
                },
                "dt": 0.1,
                "pde_params": {
                    "type": "scalar",
                    "value": "1000.0" # Very large diffusion coefficient
                }
            }
        }
    }

    model = BurgersModel(config)
    
    # Create initial state
    state = model.get_initial_state(batch_size=1)
    
    # Get params with large diffusion
    params = model.get_initial_params()
    # Manually set diffusion to a large value to trigger clipping
    params = math.expand(math.wrap(1000.0), channel(field='diffusion_coeff'))
    
    print("Running step with large diffusion coefficient...")
    try:
        next_state, _ = model._jit_step(state, params)
        print("Step completed successfully.")
        
        # Check for NaNs or Infs
        is_finite = math.all(math.isfinite(next_state))
        if isinstance(is_finite, math.Tensor):
            is_finite = bool(is_finite.all) # Ensure we reduce to python bool if it's still a tensor with batch dim
        
        if not is_finite:
            print("FAILED: Simulation produced NaNs or Infs.")
        else:
            print("PASSED: Simulation output is finite.")
            
    except Exception as e:
        print(f"FAILED: Simulation crashed with error: {e}")

    # Test params setter clipping
    print("\nTesting params setter clipping...")
    large_val = 1000.0
    params_large = math.expand(math.wrap(large_val), channel(field='diffusion_coeff'))
    model.params = params_large
    
    current_params = model.params
    current_val = current_params.field['diffusion_coeff'].native()
    
    # We expect it to be clipped to max_diffusion
    # max_diffusion = 0.5 * (min_dx ** 2) / dt
    # dx = 100/128 = 0.78125
    # dt = 0.1
    # max = 0.5 * (0.78125^2) / 0.1 = 3.0517578125
    
    print(f"Set value: {large_val}")
    print(f"Clipped value: {current_val}")
    
    if current_val < large_val and current_val < 10.0:
        print("PASSED: Params setter clipped the value.")
    else:
        print(f"FAILED: Params setter did not clip correctly. Value: {current_val}")

if __name__ == "__main__":
    test_stability()
