# src/models/physical/heat.py

from typing import Dict

# --- PhiFlow Imports ---
from phi.torch.flow import *
from phi.math import Shape, Tensor, batch, math

# --- Repo Imports ---
from .base import PhysicalModel  # <-- Import from your repo's base class


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
                 diffusivity: Tensor,
                 batch_size: int = 1):  # <-- Added batch_size
        """
        Initializes the Heat model.

        Args:
            domain (Box): The simulation domain.
            resolution (Shape): The grid resolution.
            dt (float): Time step size.
            diffusivity (Tensor): The diffusion coefficient.
            batch_size (int): The batch size (from base class).
        """
        # Set private attributes
        self._diffusivity = diffusivity
        # Call the parent's init
        super().__init__(
            domain=domain,
            resolution=resolution,
            dt=dt,
            batch_size=batch_size, # <-- Pass to base
            diffusivity=diffusivity  # <-- Stored on self via **pde_params
        )

    @property
    def diffusivity(self) -> Tensor:
        return self._diffusivity
    
    @diffusivity.setter
    def diffusivity(self, value: Tensor):
        self._diffusivity = value

    def get_initial_state(self, batch_size: int = 1) -> Dict[str, Field]:
        """
        Returns a batched initial state with a "hot spot" in the middle.
        
        Args:
            batch_size (int): Number of parallel states to create.
        """
        # Create a batch shape
        batch_shape = batch(batch=batch_size)
        
        temp_0 = CenteredGrid(
            Noise(scale=1, smoothness=5), # Noisy initial temperature
            extrapolation=extrapolation.PERIODIC, # Periodic boundaries
            x=self.resolution.get_size('x'),
            y=self.resolution.get_size('y'),
            bounds=self.domain
        )
        return {"temp": temp_0}

    def step(self, current_state: Dict[str, Field]) -> Dict[str, Field]:
        """
        Performs a single simulation step using the model's
        internal diffusivity.

        Args:
            *current_state (Field): A tuple containing the (temp,) field.
        """
        # Unpack the state tuple, as required by the base class
        
        new_temp = _heat_step(
            temp=current_state["temp"],
            diffusivity=self.diffusivity, # <-- Use self.diffusivity
            dt=self.dt
        )
        return {"temp": new_temp}