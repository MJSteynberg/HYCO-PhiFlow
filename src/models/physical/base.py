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

        # Use logger instead of print to avoid Unicode issues on Windows
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        # Simple string representation to avoid Unicode superscript issues
        logger.info(
            f"Initialized {self.__class__.__name__} with resolution={tuple(self.resolution.sizes)}, dt={self.dt}"
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

    def generate_predictions(
        self,
        real_dataset,
        alpha: float,
        device: str = 'cpu',
        num_rollout_steps: int = 10
    ):
        """
        Generate predictions for data augmentation.
        
        This method generates rollout predictions on real Field samples for use
        in hybrid training augmentation. The number of predictions is proportional
        to alpha.
        
        Args:
            real_dataset: Dataset of real field samples (FieldDataset)
            alpha: Proportion of generated samples (e.g., 0.1 = 10%)
            device: Device to run model on ('cpu' or 'cuda')
            num_rollout_steps: Number of rollout steps for prediction
            
        Returns:
            Tuple of (initial_fields_list, target_fields_list) where:
            - initial_fields_list: List of initial field states (Dict[str, Field])
            - target_fields_list: List of predicted rollout states (List[Dict[str, Field]])
        """
        import torch
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Calculate number of samples to generate
        num_real = len(real_dataset)
        num_generate = int(num_real * alpha)
        
        logger.info(
            f"Generating {num_generate} physical predictions "
            f"(alpha={alpha:.2f} * {num_real} real samples)"
        )
        
        if num_generate == 0:
            logger.warning("Alpha too small, no samples will be generated")
            return [], []
        
        # Select proportional indices
        indices = self._select_proportional_indices(num_real, num_generate)
        
        # Generate predictions
        initial_fields_list = []
        target_fields_list = []
        
        with torch.no_grad():
            for idx in indices:
                # Get sample from dataset
                initial_fields, _ = real_dataset[idx]
                
                # Perform rollout prediction
                predictions = self._perform_rollout(initial_fields, num_rollout_steps)
                
                # Store results
                initial_fields_list.append(initial_fields)
                target_fields_list.append(predictions)
        
        logger.info(f"Generated {len(initial_fields_list)} physical predictions")
        
        return initial_fields_list, target_fields_list
    
    def _perform_rollout(self, initial_fields: Dict[str, Field], num_steps: int):
        """
        Perform rollout prediction.
        
        Args:
            initial_fields: Initial field state (dict of PhiFlow Fields)
            num_steps: Number of rollout steps
            
        Returns:
            List of predicted field states, one for each step (List[Dict[str, Field]])
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.debug(f"Performing {num_steps}-step rollout with physical model")
        
        # Start from initial state
        current_state = initial_fields
        
        # Store all rollout states
        rollout_states = []
        
        # Perform rollout steps
        for step_num in range(num_steps):
            try:
                # Call model's step method
                current_state = self.step(current_state)
                # Store the state at this step
                rollout_states.append(current_state)
            except Exception as e:
                logger.warning(f"Rollout failed at step {step_num}/{num_steps}: {e}")
                # Return states collected so far
                return rollout_states if rollout_states else [current_state]
        
        return rollout_states
    
    @staticmethod
    def _select_proportional_indices(total_count: int, sample_count: int):
        """
        Select indices proportionally across the dataset.
        
        Ensures diverse sampling rather than just taking the first N samples.
        """
        if sample_count >= total_count:
            return list(range(total_count))
        
        # Calculate step size for proportional sampling
        step = total_count / sample_count
        
        # Select indices evenly distributed
        indices = [int(i * step) for i in range(sample_count)]
        
        # Ensure no duplicates and within bounds
        indices = sorted(list(set(indices)))[:sample_count]
        
        return indices
