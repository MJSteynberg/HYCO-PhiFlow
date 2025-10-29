# scripts/inverse_heat.py

import torch
import time
from typing import Dict, List
from phi.math import Shape
# --- PhiFlow Imports ---
from phi.torch.flow import *
from phi.math import jit_compile, batch, gradient, stop_gradient
import sys 
import os

# --- Add project root to path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# --- NEW: Import our UNet model ---
from src.models.physical.heat import HeatModel
# --- Phiml Imports ---
# scripts/inverse_heat.py

import torch
import time
from typing import Dict, List
import sys
import os

# --- Add project root to path ---
# This allows importing from 'src'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- PhiFlow Imports ---
from phi.torch.flow import *
from phi.math import jit_compile, batch, gradient, stop_gradient, math, spatial

# --- Repo Imports ---
# We now import our *real* model!
from src.models.physical.heat import HeatModel

print("--- Imports Complete ---")


# --- 1. Define Ground Truth Simulation Parameters ---
DOMAIN = Box(x=100, y=100)
RESOLUTION = spatial(x=64, y=64)
TRUE_DIFFUSIVITY = wrap([0.5], batch('true_diff')) # The "correct" answer
TIMESTEP_DT = 1.0
TOTAL_STEPS = 20

print(f"--- Generating Ground Truth Data ---")
print(f"True Diffusivity: {TRUE_DIFFUSIVITY.native('true_diff')}")

# --- 2. Create the "Ground Truth" Model ---
truth_model = HeatModel(
    domain=DOMAIN,
    resolution=RESOLUTION,
    dt=TIMESTEP_DT,
    diffusivity=stop_gradient(TRUE_DIFFUSIVITY),
    batch_size=1  # Explicitly pass batch_size
)

# --- 3. Run the Forward Simulation (Rollout) ---
ground_truth_states: List[CenteredGrid] = []

# Get initial state (t=0)
# This call now uses the default batch_size=1
initial_state_dict = truth_model.get_initial_state() 
current_state = initial_state_dict
ground_truth_states.append(current_state['temp'])

print(f"Initial state shape: {current_state['temp'].shape}")

# Run simulation for TOTAL_STEPS
for step in range(TOTAL_STEPS):
    # --- MODIFICATION ---
    # Use the new Dict signature
    next_state_dict = truth_model.step(current_state)
    
    # Update the 'current_state' for the next loop iteration
    current_state = next_state_dict
    
    # Store the result
    ground_truth_states.append(current_state['temp'])

print(f"Generation complete. Total states stored: {len(ground_truth_states)}")

# --- 4. Define the Optimization Problem ---
INITIAL_GUESS_DIFFUSIVITY = wrap([0.1], batch('optimize'))
LEARNING_RATE = 1e-5
OPTIMIZATION_STEPS = 50

print(f"--- Setting up Inverse Problem ---")

# --- 5. Define the Loss Function (the "Physics-based Loss") ---

# Create ONE "guess" model here, initializing it with the guess.
guess_model = HeatModel(
    domain=DOMAIN,
    resolution=RESOLUTION,
    dt=TIMESTEP_DT,
    diffusivity=INITIAL_GUESS_DIFFUSIVITY,
    batch_size=1
)


def calculate_loss(guess_diffusivity):
    """
    Runs a full simulation with the guessed diffusivity
    and compares it to the ground truth.
    """
    total_loss = 0.0
    
    # 1. --- MODIFICATION ---
    # Set the new parameter using the @property setter
    guess_model.diffusivity = guess_diffusivity
    
    # 2. Get the TRUE initial state from our data
    # We need the full dict, so we wrap the T=0 state
    current_state = {'temp': ground_truth_states[0]}
    
    # 3. Run the rollout
    for step in range(1, TOTAL_STEPS + 1):
        # A. --- MODIFICATION ---
        #    Use the new Dict signature
        next_state_dict = guess_model.step(current_state)
        predicted_temp = next_state_dict['temp']
        
        # B. Get the corresponding "true" state
        true_temp = ground_truth_states[step]
        
        # C. Calculate the error (L2 loss)
        loss_at_this_step = l2_loss(predicted_temp - stop_gradient(true_temp))
        total_loss += loss_at_this_step
        
        # D. Update for next loop
        current_state = next_state_dict
        
    return total_loss

print("--- Loss Function Defined ---")

# --- 6. Run the Optimization Loop ---
current_guess_tensor = INITIAL_GUESS_DIFFUSIVITY 
guess_history = [current_guess_tensor.native('optimize')]
loss_history = []

print(f"\n--- Starting Optimization (Manual Gradient Descent) ---")
print(f"Step | Loss       | Current Guess")
print(f"---------------------------------")

start_time = time.time()

# Get the gradient function once, specifying 'wrt' by name
sim_grad = gradient(calculate_loss, wrt='guess_diffusivity')

for step in range(OPTIMIZATION_STEPS):
    loss, (grads,) = sim_grad(guess_diffusivity=current_guess_tensor)
    current_guess_tensor = current_guess_tensor - LEARNING_RATE * grads 
    
    loss_history.append(loss)
    guess_history.append(current_guess_tensor) 
    
    if step % 5 == 0 or step == OPTIMIZATION_STEPS - 1:
        print(f"{step:<4} | {loss} | {current_guess_tensor}")

end_time = time.time()
print(f"---------------------------------")
print(f"Optimization complete in {end_time - start_time:.2f} seconds.")

print(f"\nFinal optimized guess: {current_guess_tensor}")
print(f"True value:              {TRUE_DIFFUSIVITY}")