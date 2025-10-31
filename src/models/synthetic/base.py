# In src/models/synthetic/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List

import torch.nn as nn
from phi.field import Field, StaggeredGrid, CenteredGrid, stack, native_call
from phi.math import math, channel
from phi import field as phi_field
from phi.field import native_call
class SyntheticModel(nn.Module, ABC):
    """
    Abstract base class for all synthetic models (neural networks).

    This class handles the boilerplate logic for:
    1.  Pre-processing: Converting a state dict of Phiflow Fields (including
        StaggeredGrids) into a single, multi-channel CenteredGrid tensor.
    2.  Post-processing: Converting the network's output CenteredGrid back
        into a state dict of individual Fields, restoring original
        StaggeredGrid types where appropriate.

    Subclasses are only required to implement the `_predict` method.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the synthetic model.

        Args:
            config: A dictionary containing model-specific configurations.
                    Expected to contain 'input_specs' and 'output_specs'
                    dictionaries, e.g., {'density': 1, 'velocity': 2}.
        """
        super().__init__()
        self.config = config

        # Get specs from config, default to empty dict if not provided
        self.INPUT_SPECS: Dict[str, int] = config.get('input_specs', {})
        self.OUTPUT_SPECS: Dict[str, int] = config.get('output_specs', {})

        # Derive the field lists directly from the specs
        self.INPUT_FIELDS: List[str] = list(self.INPUT_SPECS.keys())
        self.OUTPUT_FIELDS: List[str] = list(self.OUTPUT_SPECS.keys())

    @abstractmethod
    def forward(self, state: Dict[str, Field], dt: float = 0.0) -> Dict[str, Field]:
        """Forward pass through the model. """
        raise NotImplementedError("Subclasses must implement the forward method.")
    