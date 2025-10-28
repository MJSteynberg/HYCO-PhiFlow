import torch
from phi.torch.flow import *
from phi.math import jit_compile, batch
from .base import PhysicalModel  # <-- Assuming this base class exists

# --- JIT-Compiled Physics Function ---
@jit_compile
def _burgers_physics_step(velocity: StaggeredGrid, dt: float, nu: float) -> StaggeredGrid:
    """
    Performs one physics-based Burgers' equation step.

    Args:
        velocity (StaggeredGrid): Current velocity field.
        dt (float): Time step.
        nu (float): Viscosity.

    Returns:
        StaggeredGrid: new_velocity
    """
    # Advect velocity (self-advection: u * grad(u))
    velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)
    
    # Diffuse velocity (viscosity: nu * laplace(u))
    if nu > 0:
        velocity = diffuse.explicit(velocity, nu, dt)
    
    return velocity

# --- Model Class Implementation ---

class BurgersModel(PhysicalModel):
    """
    Physical model for the Burgers' equation.
    Implements the PhysicalModel interface.
    """
    
    def __init__(self,
                 domain: Box,
                 resolution: Shape,
                 dt: float,
                 batch_size: int = 1,
                 nu: float = 0.1,
                 **pde_params): # To catch any other unused params
        """
        Initializes the Burgers' model.
        
        Args:
            domain (Box): The simulation domain.
            resolution (Shape): The grid resolution.
            dt (float): Time step size.
            batch_size (int): Number of simulations to run in parallel.
            nu (float): Viscosity.
        """
        # Call the parent's init with params it knows about
        super().__init__(
            domain=domain,
            resolution=resolution,
            dt=dt,
            batch_size=batch_size,
            nu=nu
        )
        
        # No inflow or buoyancy parameters needed
        print(f"BurgersModel created with nu={self.nu}")

    def get_initial_state(self) -> StaggeredGrid:
        """
        Returns an initial state of (noisy velocity).
        We use periodic boundaries as they are common for Burgers.
        """
        b = batch(batch=self.batch_size)

        velocity_0 = StaggeredGrid(
            Noise(scale=20), # Initialize with noise
            extrapolation.PERIODIC,    # Use periodic boundaries
            x=self.resolution.get_size('x'),
            y=self.resolution.get_size('y'),
            bounds=self.domain,
        )
        velocity_0 = math.expand(velocity_0, b)

        return {"velocity": velocity_0}

    def step(self, velocity: StaggeredGrid) -> StaggeredGrid:
        """
        Performs a single simulation step.
        """
        new_velocity = _burgers_physics_step(
            velocity=velocity,
            dt=self.dt,
            nu=self.nu
        )
        return {"velocity": new_velocity}