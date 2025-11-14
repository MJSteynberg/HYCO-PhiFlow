# In src/training/physical/trainer.py

import os
import sys
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

# --- PhiFlow Imports ---
from phi.torch.flow import *
from phi.field import l2_loss
from phi.math import math, Tensor

# --- Repo Imports ---
from src.models import ModelRegistry
from src.training.field_trainer import FieldTrainer
from src.utils.logger import get_logger
from torch.utils.data import DataLoader

logger = get_logger(__name__)


class PhysicalTrainer(FieldTrainer):
    """
    Solves an inverse problem for a PhysicalModel using cached data
    from DataManager/FieldDataset.

    This trainer uses math.minimize for optimization and leverages
    the efficient DataLoader pipeline with field conversion.

    Inherits from FieldTrainer to get PhiFlow-specific functionality.

    Phase 1 Migration: Now receives model and learnable params externally,
    data passed via train(). Always uses sliding window.
    """

    def __init__(self, config: Dict[str, Any], model, learnable_params: Dict[str, Tensor]):
        """
        Initializes the trainer with external model and learnable parameters.

        Args:
            config: Full configuration dictionary
            model: Pre-created physical model
            learnable_params: List of PhiFlow Tensors for learnable parameters
        """
        # Initialize base trainer with model and params
        super().__init__(config, model, learnable_params)
        
        # Placeholder
        self.batch_size = 1000

    def _train_epoch(self, data_source: DataLoader) -> float:
        """
        Train using a DataLoader that provides pre-collated batches.
        """
        total_loss = 0.0

        # The data_source is now a DataLoader
        for stacked_initial, stacked_targets in data_source:
            # Optimize over the entire batch
            batch_loss = self._optimize_batch(stacked_initial, stacked_targets)

            total_loss += batch_loss


        return total_loss

    def _optimize_batch(
        self,
        initial_fields: Dict[str, Field],
        target_fields: Dict[str, Field],
    ) -> float:
        """
        Optimize parameters over a batch of samples.
        
        Key insight: The loss function computes loss for ALL samples in parallel,
        and math.minimize finds parameters that minimize the AVERAGE loss.
        
        Args:
            initial_fields: Fields with shape [samples, x, y, ...]
            target_fields: Fields with shape [samples, time, x, y, ...]
        
        Returns:
            Average loss across batch
        """
        
        def loss_function(*learnable_tensors: Tensor) -> Tensor:
            """
            Compute loss across entire batch.
            
            Returns:
                Scalar loss (averaged over batch dimension)
            """
            # Update model parameters
            self._update_model_params(learnable_tensors)
            
            # Initialize loss accumulator (scalar)
            total_loss = math.tensor(0.0)
            
            # Run batched simulation
            # initial_fields already has batch dimension [samples, x, y]
            current_state = initial_fields
            
            for step in range(self.num_predict_steps):
                # Forward step operates on batch dimension automatically!
                # current_state: [samples, x, y] → [samples, x, y]
                current_state = self.model.forward(current_state)
                
                # Compute loss for this timestep
                step_loss = self._compute_step_loss(
                    current_state, target_fields, step
                )
                total_loss += step_loss
            
            # Average over timesteps
            avg_loss = total_loss / self.num_predict_steps
            # Loss is already averaged over batch dimension by PhiML operations
            return avg_loss
        
        # Run optimization
        try:
            estimated_params = minimize(loss_function, self.optimizer)
        except Exception as e:
            logger.error(f"Batched optimization failed: {e}")
            estimated_params = tuple(self.learnable_params)
        
        # Compute final loss
        final_loss = loss_function(*estimated_params)
        # Update parameters
        self.update_optimizer_params(list(estimated_params))
        return self._tensor_to_float(final_loss)

    def _compute_step_loss(
        self,
        current_state: Dict[str, Field],
        target_fields: Dict[str, Field],
        step: int,
    ) -> Tensor:
        """
        Compute L2 loss for a timestep across batch.
        
        Args:
            current_state: Current predictions [samples, x, y]
            target_fields: Ground truth [samples, time, x, y]
            step: Current timestep index
        
        Returns:
            Scalar loss (automatically averaged over batch)
        """
        step_loss = math.tensor(0.0)
        
        for field_name, gt_field in target_fields.items():
            # Extract target for this timestep
            # gt_field has shape [samples, time, x, y]
            # We want [samples, x, y]
            target = gt_field.time[step]
            
            # Current prediction
            prediction = current_state[field_name]
            
            # Compute difference
            # Both have shape [samples, x, y]
            diff = prediction - target
            
            # L2 loss (automatically handles batch dimension)
            field_loss = l2_loss(diff)
            
            # Sum over spatial dimensions, average over samples
            # PhiML automatically reduces over batch dimensions in math.sum
            field_loss = math.mean(field_loss, dim="batch,time")
            step_loss += field_loss
        return step_loss    

    
    def _update_model_params(self, learnable_tensors: Tuple[Tensor, ...]):
        """
        Update model's learnable parameters.

        Args:
            learnable_tensors: Current parameter values from optimizer
        """
        for param_name, param_value in zip(self.param_names, learnable_tensors):
            setattr(self.model, param_name, param_value)
    
    def _run_optimization(self, loss_function: callable) -> Tuple[Tensor, ...]:
        """
        Run PhiML optimization with error handling.

        Args:
            loss_function: Loss function to minimize

        Returns:
            Tuple of optimized parameter tensors
        """
        try:
            # Run optimization with optional monitoring
            estimated_params = minimize(loss_function, self.optimizer)
            return estimated_params

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            # Return initial parameters as fallback
            return tuple(self.learnable_params)
