# src/models/physical/heat.py

from typing import Dict

# --- PhiFlow Imports ---
from phi.torch.flow import *
from phi.math import Shape, Tensor, batch, math
from matplotlib import pyplot as plt

# --- Repo Imports ---
from .base import PhysicalModel  # <-- Import from your repo's base class
from src.models.registry import ModelRegistry


# --- JIT-Compiled Physics Step ---
@jit_compile
def _heat_step(
    temp: CenteredGrid, 
    diffusivity: Tensor, 
    dt: float
) -> CenteredGrid:
    """
    Performs one step of the heat equation (diffusion).

    Args:
        temp (CenteredGrid): The current temperature field.
        diffusivity (Tensor): The diffusion coefficient.
        dt (float): The time step.

    Returns:
        CenteredGrid: The temperature field at the next time step.
    """
    return diffuse.explicit(temp, diffusivity=diffusivity, dt=dt)


# --- Model Class Implementation ---
@ModelRegistry.register_physical('HeatModel')
class HeatModel(PhysicalModel):
    """
    Physical model for the heat equation (diffusion).
    Stores diffusivity as an internal parameter.
    """
    
    # Declare PDE-specific parameters
    PDE_PARAMETERS = {
        'diffusivity': {
            'type': float,
            'default': 0.1,
        }
    }

    def get_initial_state(self) -> Dict[str, Field]:
        """
        Returns a batched initial state with a "hot spot" in the middle.
        """
        # Create a batch shape
        b = batch(batch=self.batch_size)

        def initial(x):
            return (math.sum(math.cos(2*np.pi*x/100), 'vector'))
        
        temp_0 = CenteredGrid(
            initial, # Noisy initial temperature
            extrapolation=extrapolation.PERIODIC, # Periodic boundaries
            x=self.resolution.get_size('x'),
            y=self.resolution.get_size('y'),
            bounds=self.domain
        )
        temp_0 = math.expand(temp_0, b) # Expand to batch size
        return {"temp": temp_0}

    def step(self, current_state: Dict[str, Field]) -> Dict[str, Field]:
        """
        Performs a single simulation step using the model's
        internal diffusivity.

        Args:
            *current_state (Field): A tuple containing the (temp,) field.
        """
        new_temp = _heat_step(
            temp=current_state["temp"],
            diffusivity=self.diffusivity, # <-- Use self.diffusivity
            dt=self.dt
        )
        return {"temp": new_temp}