# src/models/physical/base.py

from abc import ABC, abstractmethod
from phi.flow import Field, Box
from phi.math import Shape, spatial

from typing import Dict, Any, Callable, Optional


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
                'nu': {'type': float, 'default': 0.01, 'validator': lambda x: x > 0}
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
        self.domain = self._parse_domain(config.get("domain", {}))
        self.resolution = self._parse_resolution(config.get("resolution", {}))
        self.dt = float(config.get("dt", 0.1))

        # Parse batch_size from pde_params (default: 1)
        pde_params = config.get("pde_params", {})
        self.batch_size = int(pde_params.get("batch_size", 1))

        # Parse and validate PDE-specific parameters
        self._parse_pde_parameters(pde_params)

        print(
            f"Initialized {self.__class__.__name__} with resolution={self.resolution}, dt={self.dt}"
        )

    def _parse_domain(self, domain_config: Dict[str, Any]) -> Box:
        """Parse domain configuration into a Box object."""
        size_x = domain_config.get("size_x", 100)
        size_y = domain_config.get("size_y", 100)
        return Box(x=size_x, y=size_y)

    def _parse_resolution(self, resolution_config: Dict[str, Any]) -> Shape:
        """Parse resolution configuration into a Shape object."""
        x = resolution_config.get("x", 64)
        y = resolution_config.get("y", 64)
        return spatial(x=x, y=y)

    def _parse_pde_parameters(self, pde_params: Dict[str, Any]):
        """
        Parse and validate PDE-specific parameters based on PDE_PARAMETERS declaration.

        Creates properties for each parameter with automatic getter/setter.
        """
        for param_name, param_spec in self.PDE_PARAMETERS.items():
            # Extract parameter specification
            param_type = param_spec.get("type", float)
            default_value = param_spec.get("default", None)
            # Get value from config or use default
            if param_name in pde_params:
                value = pde_params[param_name]
                # Convert to appropriate type
                if param_type is not None:
                    value = param_type(value)
            elif default_value is not None:
                value = default_value
            else:
                raise ValueError(
                    f"Required parameter '{param_name}' not found in pde_params"
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
    def get_initial_state(self) -> Dict[str, Field]:
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
    def step(self, current_state: Dict[str, Field]) -> Dict[str, Field]:
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
