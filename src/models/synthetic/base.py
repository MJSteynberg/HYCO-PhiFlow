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
        self.INPUT_SPECS: Dict[str, int] = config.get("input_specs", {})
        self.OUTPUT_SPECS: Dict[str, int] = config.get("output_specs", {})

        # Derive the field lists directly from the specs
        self.INPUT_FIELDS: List[str] = list(self.INPUT_SPECS.keys())
        self.OUTPUT_FIELDS: List[str] = list(self.OUTPUT_SPECS.keys())

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass through the model.
        
        Subclasses should implement this to handle their specific input/output formats.
        For tensor-based models: forward(x: torch.Tensor) -> torch.Tensor
        For field-based models: forward(state: Dict[str, Field], dt: float) -> Dict[str, Field]
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    def generate_predictions(
        self,
        real_dataset,
        alpha: float,
        device: str = 'cpu',
        batch_size: int = 32,
        num_workers: int = 0
    ):
        """
        Generate predictions for data augmentation.
        
        This method generates predictions on real data samples for use in
        hybrid training augmentation. The number of predictions is proportional
        to alpha.
        
        Args:
            real_dataset: Dataset of real samples (TensorDataset)
            alpha: Proportion of generated samples (e.g., 0.1 = 10%)
            device: Device to run model on ('cpu' or 'cuda')
            batch_size: Batch size for generation
            num_workers: Number of DataLoader workers
            
        Returns:
            Tuple of (inputs_list, targets_list) where:
            - inputs_list: List of input tensors [C_all, H, W]
            - targets_list: List of predicted target tensors [T, C_all, H, W]
        """
        import torch
        from torch.utils.data import DataLoader
        import logging
        
        logger = logging.getLogger(__name__)
        
        self.eval()
        self.to(device)
        
        # Calculate number of samples to generate
        num_real = len(real_dataset)
        num_generate = int(num_real * alpha)
        
        logger.debug(
            f"Generating {num_generate} synthetic predictions "
            f"(alpha={alpha:.2f} * {num_real} real samples)"
        )
        
        if num_generate == 0:
            logger.warning("Alpha too small, no samples will be generated")
            return [], []
        
        # Select proportional indices for diverse sampling
        indices = self._select_proportional_indices(num_real, num_generate)
        subset = torch.utils.data.Subset(real_dataset, indices)
        
        # Create DataLoader
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        # Generate predictions
        inputs_list = []
        targets_list = []
        
        with torch.no_grad():
            for batch_inputs, _ in loader:
                # Move to device
                batch_inputs = batch_inputs.to(device)
                
                # Generate predictions
                batch_predictions = self(batch_inputs)
                
                # Debug: log shapes
                if len(inputs_list) == 0:  # Log only for first batch
                    logger.debug(f"  Batch inputs shape: {batch_inputs.shape}")
                    logger.debug(f"  Batch predictions shape: {batch_predictions.shape}")
                
                # Store results (move back to CPU for storage)
                for i in range(batch_inputs.shape[0]):
                    inputs_list.append(batch_inputs[i].cpu())
                    targets_list.append(batch_predictions[i].cpu())
        
        logger.debug(f"Generated {len(inputs_list)} synthetic predictions")
        
        # Debug: log sample shapes
        if len(inputs_list) > 0:
            logger.debug(f"  Sample input shape: {inputs_list[0].shape}")
            logger.debug(f"  Sample target shape: {targets_list[0].shape}")
        
        return inputs_list, targets_list
    
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
