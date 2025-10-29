# In src/models/synthetic/base.py
# (This is the updated file)

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from phiml.math import Tensor

import torch.nn as nn

class SyntheticModel(nn.Module, ABC):
    """
    Abstract base class for all synthetic models (neural networks).

    Inherits from `nn.Module` to be compatible with PyTorch optimizers
    (i.e., provide .parameters()) and to correctly handle PhiFlow Tensors
    during training.
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

        # --- NEW: Store the full specs ---
        # Get specs from config, default to empty dict if not provided
        self.INPUT_SPECS: Dict[str, int] = config.get('input_specs', {})
        self.OUTPUT_SPECS: Dict[str, int] = config.get('output_specs', {})

        # --- Derive the field lists directly from the specs ---
        self.INPUT_FIELDS: List[str] = list(self.INPUT_SPECS.keys())
        self.OUTPUT_FIELDS: List[str] = list(self.OUTPUT_SPECS.keys())

    @abstractmethod
    def forward(self, state: Dict[str, Tensor], dt: float) -> Dict[str, Tensor]:
        """
        Performs one prediction step.

        In PyTorch, the `forward` method is automatically called when you
        execute the model instance (e.g., `model(state, dt)`).

        Args:
            state: A dictionary of Tensors representing the current state.
                   Must contain all fields listed in self.INPUT_FIELDS.
            dt: The time step duration (float).

        Returns:
            A dictionary of Tensors representing the predicted next state.
            Must contain all fields listed in self.OUTPUT_FIELDS.
        """
        pass