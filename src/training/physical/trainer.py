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
from src.training.abstract_trainer import AbstractTrainer
from src.utils.logger import get_logger
from torch.utils.data import DataLoader

from tqdm import tqdm

logger = get_logger(__name__)


class PhysicalTrainer():
    """
    Solves an inverse problem for a PhysicalModel using cached data
    from DataManager/FieldDataset.

    This trainer uses math.minimize for optimization and leverages
    the efficient DataLoader pipeline with field conversion.

    Inherits from FieldTrainer to get PhiFlow-specific functionality.

    Phase 1 Migration: Now receives model and learnable params externally,
    data passed via train(). Always uses sliding window.
    """

    def __init__(self, config: Dict[str, Any], model):
        """
        Initializes the trainer with external model and learnable parameters.

        Args:
            config: Full configuration dictionary
            model: Pre-created physical model
            learnable_parameters: List of PhiFlow Tensors for learnable parameters
        """

        # --- Derive all parameters from config ---
        self.model = model

        self._parse_config(config)
        self._setup()
 
        # Results storage
        self.final_loss: float = 0.0
        self.training_history: List[float] = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def _parse_config(self, config: Dict[str, Any]):
        """
        Parse configuration dictionary to setup trainer parameters.

        Args:
            config: Full configuration dictionary
        """

        # --- Data specifications ---
        self.field_names: List[str] = config['data']["fields"]
        self.data_dir = config['data']["data_dir"]
        self.dset_name = config['data']["dset_name"]
        
        # -- Store model and parameters ---
        learnable_parameters = config['trainer']['physical']['learnable_parameters']
        self.learnable_parameters = []
        for param_config in learnable_parameters:
            param_name = param_config["name"]
            param_type = param_config["type"]
            setattr(self.model, param_name, param_config.get("initial_guess"))
            param_value = getattr(self.model, param_name)
            if param_type == "field":
                
                self.learnable_parameters.append(param_value.values)
            else:
                # Use scalar as-is
                self.learnable_parameters.append(param_value)
        self.param_names = [p["name"] for p in learnable_parameters]
        self.param_types = [p["type"] for p in learnable_parameters]
        self.rollout_steps = config['trainer']['rollout_steps']

        # --- Trainer specifications ---
        self.method = config['trainer']['physical']['method']
        self.abs_tol = config['trainer']['physical']['abs_tol']
        self.max_iterations = config['trainer']['physical']['max_iterations']

    def _setup(self):
        """
        Create optimizer for learnable parameters.

        Returns:
            PhiFlow optimizer instance
        """
        self.optimizer = math.Solve(
            method=self.method,
            abs_tol=self.abs_tol,
            x0=self.learnable_parameters,
            max_iterations=self.max_iterations,
            suppress=(math.NotConverged,),
        )


    #########################
    # Training Loop Methods #
    #########################

    def train(self, data_source: DataLoader, num_epochs: int, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute training for specified number of epochs with provided data.

        Args:
            data_source: Iterable yielding (initial_fields, target_fields) tuples
                        Note: NO weights - all samples treated equally!
            num_epochs: Number of epochs to train

        Returns:
            Dictionary with training results including losses and metrics
        """

        results = {
            "train_losses": [],
            "epochs": [],
            "num_epochs": num_epochs,
            "best_epoch": 0,
            "best_val_loss": float("inf"),
        }

        disable_tqdm = not verbose
        pbar = tqdm(
            range(num_epochs), desc="Training", unit="epoch", disable=disable_tqdm
        )

        for epoch in pbar:
            start_time = time.time()
            train_loss = 0.0

            for stacked_initial, stacked_targets in data_source:
                batch_loss = self._optimize_batch(stacked_initial, stacked_targets)
                train_loss += batch_loss

            results["train_losses"].append(train_loss)
            results["epochs"].append(epoch + 1)

            if train_loss < self.best_val_loss:
                self.best_val_loss = train_loss
                self.best_epoch = epoch + 1
                results["best_epoch"] = self.best_epoch
                results["best_val_loss"] = self.best_val_loss

            epoch_time = time.time() - start_time

            # Update progress bar
            postfix_dict = {
                "train_loss": f"{train_loss:.6f}",
                "time": f"{epoch_time:.2f}s",
            }

            postfix_dict["best_epoch"] = self.best_epoch

            pbar.set_postfix(postfix_dict)

        final_loss = results["train_losses"][-1]
        results["final_loss"] = final_loss

        return results

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
            self._update_params(learnable_tensors)
            
            # Initialize loss accumulator (scalar)
            total_loss = math.tensor(0.0)
            
            # Run batched simulation
            # initial_fields already has batch dimension [samples, x, y]
            current_state = initial_fields
            
            for step in range(self.rollout_steps):
                # Forward step operates on batch dimension automatically!
                # current_state: [samples, x, y] → [samples, x, y]
                current_state = self.model.forward(current_state)
                
                # Compute loss for this timestep
                step_loss = math.tensor(0.0)
                for field_name, gt_field in target_fields.items():
                    target = gt_field.time[step]
                    prediction = current_state[field_name]
                    field_loss = l2_loss(prediction - target)
                    field_loss = math.mean(field_loss, dim="batch,time")
                    step_loss += field_loss
                total_loss += step_loss
            
            # Average over timesteps
            avg_loss = total_loss / self.rollout_steps
            # Loss is already averaged over batch dimension by PhiML operations
            return avg_loss
        
        # Run optimization
        try:
            estimated_params = minimize(loss_function, self.optimizer)
        except Exception as e:
            logger.error(f"Batched optimization failed: {e}")
            estimated_params = tuple(self.learnable_parameters)
        
        # Compute final loss
        final_loss = loss_function(*estimated_params)
        return float(final_loss)

    #############
    # Utilities #
    #############
    def _update_params(self, learnable_tensors: Tuple[Tensor, ...]):
        """Update model parameters (scalars or fields)."""
        for param_name, param_value, param_type in zip(
            self.param_names, learnable_tensors, self.param_types
        ):
            if param_type == "field":
                # Wrap tensor in CenteredGrid
                original_field = getattr(self.model, param_name)
                updated_field = CenteredGrid(
                    param_value,
                    extrapolation=original_field.extrapolation,
                    bounds=original_field.bounds,
                )
                setattr(self.model, param_name, updated_field)
            else:
                # Scalar - set directly
                setattr(self.model, param_name, param_value)

            self.optimizer.x0 = list(learnable_tensors)
            self.learnable_parameters = list(learnable_tensors)


    
    

    def get_current_params(self) -> Dict[str, Tensor]:
        """
        Get current learnable parameters as a dictionary.

        Returns:
            Dictionary mapping parameter names to current values
        """
        return {name: param for name, param in zip(self.param_names, self.learnable_parameters)}
    

