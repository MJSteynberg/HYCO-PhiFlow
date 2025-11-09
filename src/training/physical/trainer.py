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
        self.batch_size = config.get("batch_size", 32)


    def _train_epoch(self, data_source: DataLoader) -> float:
        """
        Train using batched optimization (MUCH FASTER).
        
        Key insight: PhiML's math.minimize can optimize over batch dimensions,
        meaning we optimize parameters that minimize loss across ALL samples
        simultaneously.
        """
        total_loss = 0.0
        num_batches = 0
        
        # Collect samples into batches
        batch_initial = []
        batch_targets = []
        
        for sample_idx, (initial_fields, target_fields) in enumerate(data_source):
            batch_initial.append(initial_fields)
            batch_targets.append(target_fields)
            
            # Process batch when full or at end of data
            if len(batch_initial) >= self.batch_size or sample_idx == len(data_source) - 1:
                # Stack samples along batch dimension
                stacked_initial = self._stack_samples(batch_initial)
                stacked_targets = self._stack_target_sequences(batch_targets)
                
                # Optimize over entire batch
                batch_loss = self._optimize_batch(stacked_initial, stacked_targets)
                
                total_loss += batch_loss
                num_batches += 1
                
                # Clear batch
                batch_initial = []
                batch_targets = []
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
        return avg_loss 

    def _stack_samples(
        self, samples: List[Dict[str, Field]]
    ) -> Dict[str, Field]:
        """
        Stack multiple samples along batch dimension.
        
        Args:
            samples: List of sample dicts, each containing initial fields
        
        Returns:
            Single dict with fields stacked along batch('samples') dimension
        
        Example:
            Input: [{'velocity': Field[x, y]}, {'velocity': Field[x, y]}]
            Output: {'velocity': Field[samples, x, y]}
        """
        if not samples:
            raise ValueError("Cannot stack empty sample list")
        
        # Get field names from first sample
        field_names = samples[0].keys()
        
        stacked = {}
        for field_name in field_names:
            # Collect fields across samples
            field_list = [sample[field_name] for sample in samples]
            
            # Stack along new batch dimension named 'samples'
            stacked[field_name] = stack(field_list, batch('samples'))
        
        return stacked

    def _stack_target_sequences(
        self, target_sequences: List[Dict[str, List[Field]]]
    ) -> Dict[str, Field]:
        """
        Stack target sequences from multiple samples.
        
        Args:
            target_sequences: List of target dicts, each containing field sequences
        
        Returns:
            Dict with fields stacked along batch('samples') and batch('time')
        
        Example:
            Input: [
                {'velocity': [Field[x,y], Field[x,y], ...]},  # Sample 1
                {'velocity': [Field[x,y], Field[x,y], ...]},  # Sample 2
            ]
            Output: {'velocity': Field[samples, time, x, y]}
        """
        if not target_sequences:
            raise ValueError("Cannot stack empty target list")
        
        field_names = target_sequences[0].keys()
        stacked = {}
        
        for field_name in field_names:
            # First stack each sample's time sequence
            sample_sequences = []
            for sample_targets in target_sequences:
                field_list = sample_targets[field_name]
                # Stack time dimension for this sample
                time_stacked = stack(field_list, batch('time'))
                sample_sequences.append(time_stacked)
            
            # Then stack samples dimension
            stacked[field_name] = stack(sample_sequences, batch('samples'))
        
        return stacked

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
            field_loss = math.mean(field_loss, dim="samples,time")
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
