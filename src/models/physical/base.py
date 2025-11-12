# src/models/physical/base.py

from abc import ABC, abstractmethod
from phi.flow import Field, Box, math, batch, plot
from phi.math import Shape, spatial
import matplotlib.pyplot as plt
from src.utils.logger import get_logger
from typing import Dict, Any, Callable, Optional, List
import torch
import logging


class PhysicalModel(ABC):
    """
    Abstract Base Class for all physical PDE models.

    This interface guarantees that all physical models can:
    1. Be initialized from a configuration.
    2. Generate a batched initial state (t=0).
    3. Be advanced one time step.

    Child classes should declare their PDE-specific parameters by overriding
    the PDE_PARAMETERS class variable:

    Example:
        class BurgersModel(PhysicalModel):
            PDE_PARAMETERS = {
                'nu': {'type': float, 'default': 0.01}
            }
    """

    # Child classes override this to declare their PDE parameters
    PDE_PARAMETERS: Dict[str, Dict[str, Any]] = {}

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the model from a configuration dictionary.

        Args:
            config (Dict): Configuration dictionary containing:
                - domain: Dict with 'size_x' and 'size_y'
                - resolution: Dict with 'x' and 'y'
                - dt: float time step
                - pde_params: Dict with model-specific parameters
        """
        # Parse common configuration
        self.domain = self._parse_domain(config["domain"])
        self.resolution = self._parse_resolution(config["resolution"])
        self.dt = float(config["dt"])

        # Parse batch_size from pde_params (default: 1)
        pde_params = config["pde_params"]
        # Parse and validate PDE-specific parameters
        self._parse_pde_parameters(pde_params)

        self.logger = get_logger(__name__)
        # Simple string representation to avoid Unicode superscript issues
        self.logger.info(
            f"Initialized {self.__class__.__name__} with resolution={tuple(self.resolution.sizes)}, dt={self.dt}"
        )

    def _parse_domain(self, domain_config: Dict[str, Any]) -> Box:
        """Parse domain configuration into a Box object."""
        size_x = domain_config["size_x"]
        size_y = domain_config["size_y"]
        return Box(x=size_x, y=size_y)

    def _parse_resolution(self, resolution_config: Dict[str, Any]) -> Shape:
        """Parse resolution configuration into a Shape object."""
        x = resolution_config["x"]
        y = resolution_config["y"]
        return spatial(x=x, y=y)

    def _parse_pde_parameters(self, pde_params: Dict[str, Any]):
        """
        Parse and validate PDE-specific parameters based on PDE_PARAMETERS declaration.

        Creates properties for each parameter with automatic getter/setter.
        """
        for param_name, param_spec in self.PDE_PARAMETERS.items():
            # Extract parameter specification
            param_type = param_spec["type"]
            default_value = param_spec["default"]
            # Get value from config or use default
            if param_name in pde_params:
                value = pde_params[param_name]
                # Convert to appropriate type
                if param_type is not None:
                    value = param_type(value)
            elif default_value is not None:
                value = default_value
            else:
                self.logger.error(
                    f"Missing required PDE parameter '{param_name}'"
                )

            # Store as private attribute
            private_name = f"_{param_name}"
            setattr(self, private_name, value)

            # Create property dynamically
            self._create_property(param_name)

    def _create_property(self, param_name: str):
        """
        Dynamically create a property for a PDE parameter.

        This creates a getter and setter that access the private attribute.
        """
        private_name = f"_{param_name}"

        # Create getter and setter functions
        def getter(self):
            return getattr(self, private_name)

        def setter(self, value):
            # Re-validate if validator exists
            setattr(self, private_name, value)

        # Set property on the class (not instance)
        prop = property(getter, setter)
        setattr(self.__class__, param_name, prop)

    @abstractmethod
    def get_initial_state(self, batch_size: int = 1) -> Dict[str, Field]:
        """
        Generates a batched initial state (t=0) for the simulation.

        The batch dimension should be named 'batch'.

        Args:
            batch_size (int): The number of parallel simulations.

        Returns:
            Dict[str, Field]: A dictionary mapping field names to their
                              initial Field values.
        """
        pass

    @abstractmethod
    def get_random_state(self, batch_size: int = 1) -> Dict[str, Field]:
        """
        Generates a random state for the simulation.

        The batch dimension should be named 'batch'.

        Args:
            batch_size (int): The number of parallel simulations.

        Returns:
            Dict[str, Field]: A dictionary mapping field names to their
                              random Field values.
        """
        pass

    @abstractmethod
    def forward(self, current_state: Dict[str, Field]) -> Dict[str, Field]:
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

    @abstractmethod
    def rollout(self, initial_state: Dict[str, Field], num_steps: int) -> Dict[str, Field]:
        """
        Roll out the simulation for a specified number of time steps.

        Args:
            initial_state (Dict[str, Field]): The initial state at t=0.
            num_steps (int): The number of time steps to roll out.
        Returns:
            Dict[str, Field]: A dictionary mapping field names to their
                                states after num_steps.
        """
        pass

    def __call__(self, *args) -> tuple[Field, ...]:
        """Convenience wrapper for the forward method."""
        return self.forward(*args)
    
    def generate_synthetic_trajectories(
        self,
        num_trajectories: int,
        trajectory_length: int,
        warmup_steps: int = 5,
    ) -> List[List[Dict[str, Field]]]:
        """
        Generate complete synthetic trajectories from random initial conditions.
        
        This creates PURE synthetic data without sampling from real dataset.
        Each trajectory is a complete simulation from random ICs.
        
        Strategy:
        1. Generate random initial states (NOT from real data)
        2. Apply warmup evolution to settle physics
        3. Continue rollout to generate full trajectory
        4. Return complete trajectories for dataset windowing
        
        Args:
            num_trajectories: Number of independent trajectories to generate
            trajectory_length: Number of time steps in each trajectory (after warmup)
            warmup_steps: Number of steps to stabilize random initial conditions
            
        Returns:
            List of trajectories, where each trajectory is a List[Dict[str, Field]]
            Each trajectory has length `trajectory_length`
            Each state Dict has fields with NO batch dimension (single trajectory)
        """
        
        all_trajectories = []
        
        for traj_idx in range(num_trajectories):
            if (traj_idx + 1) % 10 == 0:
                self.logger.debug(
                    f"  Generating trajectory {traj_idx + 1}/{num_trajectories}"
                )
            
            # 1. Generate random initial condition (batch_size=1 for single trajectory)
            initial_state = self.get_initial_state(batch_size=1)
            
            # 2. Warmup phase - let physics settle the random state
            current_state = initial_state
            for _ in range(warmup_steps):
                current_state = self.forward(current_state)
            
            # 3. Generate trajectory by rolling out from settled state
            trajectory = []
            for step in range(trajectory_length):
                current_state = self.forward(current_state)
                
                # Store a copy, removing batch dimension since batch=1
                # PhiFlow Fields with batch dimension need to be unbatched
                trajectory.append({
                    name: field.batch[0] if 'batch' in field.shape else field
                    for name, field in current_state.items()
                })
            
            all_trajectories.append(trajectory)
        rollout = self.rollout(initial_state, num_steps=trajectory_length)
        return all_trajectories, rollout

    @staticmethod
    def _select_proportional_indices(total_count: int, sample_count: int):
        """
        Select indices proportionally across the dataset.

        Ensures diverse sampling rather than just taking the first N samples.
        """
        if sample_count >= total_count:
            return list(range(total_count))
        elif sample_count <= 0:
            return []

        # Calculate step size for proportional sampling
        step = total_count / sample_count

        # Select indices evenly distributed
        indices = [int(i * step) for i in range(sample_count)]

        # Ensure no duplicates and within bounds
        indices = sorted(list(set(indices)))[:sample_count]

        return indices
