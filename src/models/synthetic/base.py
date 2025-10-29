# In src/models/synthetic/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from phiml.math import Tensor

# --- The Fix ---
# We inherit from torch.nn.Module directly, not a phiml wrapper
import torch.nn as nn
# ---

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
        """
        super().__init__()  # This is required to initialize the torch.nn.Module
        self.config = config

        # Define which fields are used for input and output.
        self.INPUT_FIELDS: List[str] = config.get('input_fields', [])
        self.OUTPUT_FIELDS: List[str] = config.get('output_fields', [])

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