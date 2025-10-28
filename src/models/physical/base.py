# src/models/physical/base.py

from abc import ABC, abstractmethod
from phi.flow import Field, Box
from phi.math import Shape

class PhysicalModel(ABC):
    """
    Abstract Base Class for all physical PDE models.

    This interface guarantees that all physical models can:
    1. Be initialized from a configuration.
    2. Generate a batched initial state (t=0).
    3. Be advanced one time step.
    """
    
    def __init__(self,
                 domain: Box,
                 resolution: Shape,  # <-- ADDED
                 dt: float,
                 batch_size: int = 1,
                 **pde_params):
        """
        Initializes the model with its core configuration.
        
        Args:
            domain (Box): The simulation domain.
            resolution (Shape): The grid resolution (e.g., spatial(x=64, y=80)).
            dt (float): The time step dt.
            **pde_params: Any other PDE-specific parameters 
                          (e.g., 'nu' for viscosity, 'inflow' for smoke).
        """
        self.domain = domain
        self.resolution = resolution  # <-- ADDED
        self.dt = dt
        self.batch_size = batch_size
        # Store other params
        for key, val in pde_params.items():
            setattr(self, key, val)
        
        print(f"Initialized {self.__class__.__name__} with resolution={resolution}, dt={dt}")

    @abstractmethod
    def get_initial_state(self, batch_size: int = 1) -> tuple[Field, ...]:
        """
        Generates a batched initial state (t=0) for the simulation.
        
        The batch dimension should be named 'batch'.

        Args:
            batch_size (int): The number of parallel simulations.

        Returns:
            tuple[Field, ...]: A tuple of PhiFlow Fields representing
                               the initial state (e..g, (velocity_0, density_0)).
        """
        pass

    @abstractmethod
    def step(self, *current_state: Field) -> tuple[Field, ...]:
        """
        Advances the simulation by one time step (dt).

        Args:
            *current_state (Field): A variable number of fields that
                                    make up the current state, passed in
                                    the same order as returned by 
                                    get_initial_state().

        Returns:
            tuple[Field, ...]: A tuple of Fields representing the next state.
        """
        pass
    
    def __call__(self, *args) -> tuple[Field, ...]:
        """Convenience wrapper for the step method."""
        return self.step(*args)