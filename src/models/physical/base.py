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
    """

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

        self._parse_config(config)


    def _parse_config(self, config: Dict[str, Any]):
        """
        Parse configuration dictionary to setup model.
        """
        # Setup domain 
        size_x = config["model"]["physical"]["domain"]["size_x"]
        size_y = config["model"]["physical"]["domain"]["size_y"]
        self.domain = Box(x=size_x, y=size_y)

        # Setup resolution
        res_x = config["model"]["physical"]["resolution"]["x"]
        res_y = config["model"]["physical"]["resolution"]["y"]
        self.resolution = spatial(x=res_x, y=res_y)

        # Setup PDE parameters
        pde_params = config["model"]["physical"]["pde_params"]
        for param_name, value in pde_params.items():         
            self._create_property(param_name, value)
        self.dt = float(config["model"]["physical"]["dt"])
        

    def _create_property(self, param_name: str, value: Any):
        """
        Dynamically create a property for a PDE parameter.

        This creates a getter and setter that access the private attribute.
        """
        private_name = f"_{param_name}"
        setattr(self, private_name, value)
        # Create getter and setter functions
        def getter(self):
            return getattr(self, private_name)

        def setter(self, value):
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
