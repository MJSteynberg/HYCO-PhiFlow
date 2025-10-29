# scripts/inverse_heat.py

import torch
import time
from typing import Dict, List
from phi.math import Shape
# --- PhiFlow Imports ---
from phi.torch.flow import *
from phi.math import jit_compile, batch, gradient, stop_gradient

# --- Phiml Imports ---

# --- Base Class (minimal version for this script) ---
from abc import ABC, abstractmethod

class PhysicalModel(ABC):
    """
    A minimal abstract base class, mimicking the structure in
    src/models/physical/base.py for this standalone script.
    """
    def __init__(self, domain: Box, resolution: Shape, dt: float, **pde_params):
        self.domain = domain
        self.resolution = resolution
        self.dt = dt
        for key, val in pde_params.items():
            setattr(self, key, val)
        print(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def get_initial_state(self) -> Dict[str, Field]:
        pass

    @abstractmethod
    def step(self, *current_state: Field) -> Dict[str, Field]:
        pass

print("--- Imports and Base Class Complete ---")

# --- JIT-Compiled Physics Step ---
@jit_compile
def _heat_step(
    temp: CenteredGrid, 
    diffusivity: math.Tensor, 
    dt: float
) -> CenteredGrid:
    """
    Performs one step of the heat equation (diffusion).

    Args:
        temp (CenteredGrid): The current temperature field.
        diffusivity (math.Tensor): The diffusion coefficient. This is
                                   what we will optimize for.
        dt (float): The time step.

    Returns:
        CenteredGrid: The temperature field at the next time step.
    """
    # Use Fourier-space diffusion, which is efficient
    # and unconditionally stable for constant coefficients.
    return diffuse.explicit(u = temp, diffusivity=diffusivity, dt=dt)


# --- Model Class Implementation ---
class HeatModel(PhysicalModel):
    """
    Physical model for the heat equation (diffusion).
    Stores diffusivity as an internal parameter.
    """
    def __init__(self,
                 domain: Box,
                 resolution: Shape,
                 dt: float,
                 diffusivity: math.Tensor): # <-- Back as an init arg
        """
        Initializes the Heat model.

        Args:
            domain (Box): The simulation domain.
            resolution (Shape): The grid resolution.
            dt (float): Time step size.
            diffusivity (math.Tensor): The diffusion coefficient.
        """
        # Call the parent's init
        super().__init__(
            domain=domain,
            resolution=resolution,
            dt=dt,
            diffusivity=diffusivity  # <-- Stored on self
        )

    def get_initial_state(self) -> Dict[str, Field]:
        """
        Returns an initial state with a "hot spot" in the middle.
        (Unchanged)
        """
        temp_0 = CenteredGrid(
            Noise(scale=1, smoothness=5), # Noisy initial temperature
            extrapolation=extrapolation.PERIODIC, # Periodic boundaries
            bounds=self.domain,
            resolution=self.resolution
        )
        return {"temp": temp_0}

    def step(self, temp: CenteredGrid) -> Dict[str, Field]:
        """
        Performs a single simulation step using the model's
        internal diffusivity.
        """
        new_temp = _heat_step(
            temp=temp,
            diffusivity=self.diffusivity, # <-- Use self.diffusivity
            dt=self.dt
        )
        return {"temp": new_temp}

    # --- NEW METHOD ---
    def set_diffusivity(self, new_diffusivity: math.Tensor):
        """
        Updates the model's internal diffusivity parameter.
        """
        self.diffusivity = new_diffusivity

print("--- HeatModel (Forward Simulator) Defined ---")

# --- 1. Define Ground Truth Simulation Parameters ---
DOMAIN = Box(x=100, y=100)
RESOLUTION = spatial(x=64, y=64)
TRUE_DIFFUSIVITY = wrap([0.5], batch('true_diff')) # The "correct" answer
TIMESTEP_DT = 1.0
TOTAL_STEPS = 20

print(f"--- Generating Ground Truth Data ---")
print(f"True Diffusivity: {TRUE_DIFFUSIVITY.native('true_diff')}")

# --- 2. Create the "Ground Truth" Model ---
# We use 'stop_gradient' here to be explicit that this
# model's parameters should never be changed.
truth_model = HeatModel(
    domain=DOMAIN,
    resolution=RESOLUTION,
    dt=TIMESTEP_DT,
    diffusivity=stop_gradient(TRUE_DIFFUSIVITY) # Prevent gradient tracking
)

# --- 3. Run the Forward Simulation (Rollout) ---
ground_truth_states: List[CenteredGrid] = []

# Get initial state (t=0)
initial_state_dict = truth_model.get_initial_state()
current_temp = initial_state_dict['temp']
ground_truth_states.append(current_temp)

print(f"Initial state shape: {current_temp.shape}")

# Run simulation for TOTAL_STEPS
for step in range(TOTAL_STEPS):
    # Get the next state (as a dict)
    next_state_dict = truth_model.step(temp=current_temp)
    
    # Update the 'current_temp' for the next loop iteration
    current_temp = next_state_dict['temp']
    
    # Store the result
    ground_truth_states.append(current_temp)

print(f"Generation complete. Total states stored: {len(ground_truth_states)}")
print(f"Final state (t={TOTAL_STEPS}) shape: {ground_truth_states[-1].shape}")

# --- 4. Define the Optimization Problem ---
# (Parameters are unchanged)
INITIAL_GUESS_DIFFUSIVITY = wrap([0.1], batch('optimize'))
LEARNING_RATE = 1e-5
OPTIMIZATION_STEPS = 50

print(f"--- Setting up Inverse Problem ---")
print(f"Initial Guess: {INITIAL_GUESS_DIFFUSIVITY.native('optimize')}")
print(f"True Value: {TRUE_DIFFUSIVITY.native('true_diff')}")


# --- 5. Define the Loss Function (the "Physics-based Loss") ---

# --- MODIFICATION ---
# Create ONE "guess" model here, initializing it with the guess.
guess_model = HeatModel(
    domain=DOMAIN,
    resolution=RESOLUTION,
    dt=TIMESTEP_DT,
    diffusivity=INITIAL_GUESS_DIFFUSIVITY  # <-- Initialize with the guess
)
# --- END MODIFICATION ---


def calculate_loss(guess_diffusivity: math.Tensor) -> math.Tensor:
    """
    Runs a full simulation with the guessed diffusivity
    and compares it to the ground truth.
    
    This function UPDATES the 'guess_model' from the outer
    scope via its setter.
    """
    total_loss = 0.0
    
    # 1. --- MODIFICATION ---
    # Set the new parameter on the *existing* model object.
    # PhiFlow's gradient trace will see this.
    guess_model.set_diffusivity(guess_diffusivity)
    # --- END MODIFICATION ---
    
    # 2. Get the TRUE initial state from our data
    current_temp = ground_truth_states[0] # <-- Use the T=0 state from the true data
    
    # 3. Run the rollout
    for step in range(1, TOTAL_STEPS + 1):
        # A. Simulate one step.
        #    The step() method now implicitly uses the diffusivity
        #    we just set.
        next_state_dict = guess_model.step(temp=current_temp)
        predicted_temp = next_state_dict['temp']
        
        # B. Get the corresponding "true" state
        true_temp = ground_truth_states[step]
        
        # C. Calculate the error (L2 loss)
        loss_at_this_step = l2_loss(predicted_temp - stop_gradient(true_temp))
        total_loss += loss_at_this_step
        
        # D. Update for next loop
        current_temp = predicted_temp
        
    return total_loss

print("--- Loss Function Defined ---")

# --- 6. Run the Optimization Loop ---

# (The loop itself is now identical to your final version)
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
    
    # B. Calculate loss and gradient
    loss, (grads,) = sim_grad(guess_diffusivity=current_guess_tensor)
    
    # C. Manual gradient descent step
    current_guess_tensor = current_guess_tensor - LEARNING_RATE * grads 
    
    # Store history
    loss_history.append(loss)
    guess_history.append(current_guess_tensor) 
    
    if step % 5 == 0 or step == OPTIMIZATION_STEPS - 1:
        print(f"{step:<4} | {loss} | {current_guess_tensor}")

end_time = time.time()
print(f"---------------------------------")
print(f"Optimization complete in {end_time - start_time:.2f} seconds.")

print(f"\nFinal optimized guess: {current_guess_tensor}")
print(f"True value:              {TRUE_DIFFUSIVITY}")