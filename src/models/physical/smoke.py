# src/models/physical/smoke_model.py

import torch
from phi.torch.flow import *
from phi.math import jit_compile, batch
from .base import PhysicalModel  # <-- Import our new base class
import random
from typing import Dict


# --- JIT-Compiled Physics Function ---
# (This is kept separate for performance and clarity)
@jit_compile
def _smoke_physics_step(velocity, density, inflow, domain, dt, buoyancy_factor, nu):
    """
    Performs one physics-based smoke simulation step.
    
    Args:
        velocity (StaggeredGrid): Current velocity field.
        density (CenteredGrid): Current density field.
        inflow (CenteredGrid): Inflow mask.
        domain (Box): Simulation domain.
        dt (float): Time step.
        buoyancy_factor (float): Strength of buoyancy.
        nu (float): Viscosity.

    Returns:
        tuple: (new_velocity, new_density)
    """
    # Advect density and add inflow
    density = advect.mac_cormack(density, velocity, dt=dt) + dt * inflow
    
    # Apply forces
    buoyancy_force = (density * (0, buoyancy_factor)).at(velocity)
    velocity = velocity + dt * buoyancy_force

    # Advect velocity
    velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)
    
    # Apply viscosity !!!!!!!!!!!!!! This is giving errors !!!!!!!!!!!!!!!!
    # print(f"Applying viscosity with dt={dt}")
    # if nu > 0:
    #     velocity = diffuse.implicit(velocity, nu, dt=dt)
    
    # Make incompressible
    velocity, pressure = fluid.make_incompressible(velocity, solve=Solve('CG', 1e-3, rank_deficiency=0, suppress=[phi.math.NotConverged]))
    
    return velocity, density


# --- Model Class Implementation ---

class SmokeModel(PhysicalModel):
    """
    Physical model for the smoke simulation.
    Implements the PhysicalModel interface.
    """
    
    def __init__(self,
                 domain: Box,
                 resolution: Shape,
                 dt: float,
                 batch_size: int = 1,
                 nu: float = 0.0,
                 buoyancy: float = 1.0,
                 inflow_center = None,
                 inflow_radius: float = 10.0,
                 inflow_rate: float = 0.1,
                 inflow_rand_x_range: list = [0.2, 0.8],
                 inflow_rand_y_range: list = [0.15, 0.25],
                 **pde_params): # To catch any other unused params
        """
        Initializes the smoke model.
        
        Args:
            domain (Box): The simulation domain.
            resolution (Shape): The grid resolution.
            dt (float): Time step size.
            nu (float): Viscosity.
            buoyancy (float): Buoyancy factor.
            inflow_center (tuple): (x, y) center of the inflow.
            inflow_radius (float): Radius of the inflow.
        """
        # Call the parent's init with params it knows about
        super().__init__(
            domain=domain,
            resolution=resolution,
            dt=dt,
            batch_size=batch_size,
            nu=nu,
            buoyancy=buoyancy
        )
        
        if inflow_center is None:
            rand_x = self.domain.size[0] * (inflow_rand_x_range[0] + inflow_rand_x_range[1] * random.random())
            rand_y = self.domain.size[1] * (inflow_rand_y_range[0] + inflow_rand_y_range[1] * random.random())
            inflow_center = (rand_x, rand_y)
            # This print statement will now come from the model itself
            print(f"Generated new inflow position: ({rand_x:.1f}, {rand_y:.1f})")


        self.inflow_center = math.tensor(inflow_center, channel(vector='x,y'))
        self.inflow_radius = inflow_radius
        self.inflow_rate = inflow_rate

        # --- FIX: Create the inflow mask internally ---
        INFLOW_SHAPE = Sphere(center=self.inflow_center, radius=self.inflow_radius)
        self.inflow = self.inflow_rate * CenteredGrid(
            INFLOW_SHAPE, 
            extrapolation.BOUNDARY, 
            x=self.resolution.get_size('x'),
            y=self.resolution.get_size('y'),
            bounds=self.domain
        )
        
        # Add a batch dimension for broadcasting
        self.inflow = math.expand(self.inflow, batch(batch=self.batch_size))

        print(f"SmokeModel created inflow at {inflow_center} "
              f"with shape {self.inflow.shape}")

    def get_initial_state(self) -> Dict[str, Field]:
        """
        Returns an initial state of (zero velocity, zero density).
        """
        b = batch(batch=self.batch_size)

        velocity_0 = StaggeredGrid(
            0,
            extrapolation.ZERO,
            x=self.resolution.get_size('x'),
            y=self.resolution.get_size('y'),
            bounds=self.domain,
        )
        velocity_0 = math.expand(velocity_0, b)
        density_0 = CenteredGrid(
            0,
            extrapolation.BOUNDARY,
            x=self.resolution.get_size('x'),
            y=self.resolution.get_size('y'),
            bounds=self.domain,
        )
        density_0 = math.expand(density_0, b)
        
        # No more batch size check against inflow,
        # as self.inflow (batch=1) will broadcast to batch=N
        
        return {"velocity": velocity_0, "density": density_0}

    def step(self, velocity: StaggeredGrid, density: CenteredGrid) -> Dict[str, Field]:
        """
        Performs a single simulation step.
        """
        # This is unchanged and will now use the internally created self.inflow
        new_velocity, new_density =_smoke_physics_step(
                velocity=velocity,
                density=density,
                inflow=self.inflow,
                domain=self.domain,
                dt=self.dt,
                buoyancy_factor=self.buoyancy,
                nu=self.nu
            )
        return {"velocity": new_velocity, "density": new_density}