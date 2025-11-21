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

from pathlib import Path
import torch

from tqdm import tqdm

logger = get_logger(__name__)


class PhysicalTrainer():
    """
    Solves an inverse problem for a PhysicalModel using PhiML Dataset.

    This trainer uses math.minimize for optimization and works with
    the pure PhiML data pipeline. Converts PhiML tensors to Fields
    as needed for physics operations.

    Phase 3 Migration: Works with pure PhiML Dataset, converts tensors
    to Fields internally.
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

        # --- Try to load checkpoint if exists ---
        if Path(self.checkpoint_path).exists():
            try:
                checkpoint = self.load_checkpoint(self.checkpoint_path)
                logger.info(
                    f"Loaded checkpoint from {self.checkpoint_path} at epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}"
                )
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
    def _parse_config(self, config: Dict[str, Any]):
        """
        Parse configuration dictionary to setup trainer parameters.

        Args:
            config: Full configuration dictionary
        """

        # --- Backend configuration for GPU acceleration ---
        device = config['trainer'].get('device', 'cpu')
        if device.startswith('cuda'):
            logger.info(f"Setting PhiML backend to PyTorch with device: {device}")
            math.use('torch')
            import torch
            torch.set_default_device(device)
        else:
            logger.info("Using CPU backend")

        # --- Data specifications ---
        self.field_names: List[str] = config['data']["fields"]
        self.data_dir = config['data']["data_dir"]
        self.dset_name = config['data']["dset_name"]

        # --- Checkpoint path ---
        model_path_dir = config["model"]["physical"]["model_path"]
        model_save_name = config["model"]["physical"]["model_save_name"]
        
        self.checkpoint_path = Path(model_path_dir) / f"{model_save_name}.pth"
        os.makedirs(model_path_dir, exist_ok=True)
        
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

    def _prepare_batch(self, batch) -> Tuple[Dict[str, Tensor], Dict[str, Field]]:
        """
        Prepare batch data for model - convert target tensors to Fields.

        No splitting needed! Dataset already provides fields separately.

        Args:
            batch: Batch dataclass with:
                  - initial_state: Dict[field_name, Tensor(batch, x, y, vector)]
                  - targets: Dict[field_name, Tensor(batch, time, x, y, vector)]

        Returns:
            Tuple of (initial_tensors, target_fields):
                - initial_tensors: Dict[field_name, Tensor] (passthrough from batch)
                - target_fields: Dict[field_name, Field] for loss computation
        """
        # Initial tensors are already in the right format (dict of tensors)
        initial_tensors = batch['initial_state']

        # Create bounds for target Fields (PhiFlow 2.2+ syntax)
        bounds = self.model.domain

        # Convert target tensors to Fields for loss computation
        target_fields = {}
        for field_name, target_tensor in batch['targets'].items():
            target_fields[field_name] = CenteredGrid(
                target_tensor,
                bounds=bounds,
                extrapolation=math.extrapolation.ZERO
            )

        return initial_tensors, target_fields


    #########################
    # Training Loop Methods #
    #########################

    def train(self, dataset, num_epochs: int, batch_size: int = 1, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute training for specified number of epochs with PhiML Dataset.

        Args:
            dataset: PhiML Dataset yielding tensor batches
            num_epochs: Number of epochs to train
            batch_size: Batch size for training (default: 1 for physical models)
            verbose: Whether to show progress bars

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
            num_batches = 0
            # Iterate through dataset batches
            for batch in dataset.iterate_batches(batch_size=batch_size, shuffle=True):
                if num_batches < 5:
                    # Prepare batch (convert targets to Fields)
                    initial_tensors, target_fields = self._prepare_batch(batch)

                    # Optimize batch (model will update from tensors)
                    batch_loss = self._optimize_batch(initial_tensors, target_fields)
                    train_loss += batch_loss
                    num_batches += 1

            # Average loss over batches
            avg_train_loss = train_loss / num_batches if num_batches > 0 else train_loss

            results["train_losses"].append(avg_train_loss)
            results["epochs"].append(epoch + 1)

            if avg_train_loss < self.best_val_loss:
                self.best_val_loss = avg_train_loss
                self.best_epoch = epoch + 1
                results["best_epoch"] = self.best_epoch
                results["best_val_loss"] = self.best_val_loss

                self.save_checkpoint(epoch, avg_train_loss)

            epoch_time = time.time() - start_time

            # Update progress bar
            postfix_dict = {
                "train_loss": f"{avg_train_loss:.6f}",
                "best": f"{self.best_val_loss:.6f}",
                "time": f"{epoch_time:.2f}s",
            }

            postfix_dict["best_epoch"] = self.best_epoch

            pbar.set_postfix(postfix_dict)


        final_loss = results["train_losses"][-1]
        results["final_loss"] = final_loss
        return results

    def _optimize_batch(
        self,
        initial_tensors: Dict[str, Tensor],
        target_fields: Dict[str, Field],
    ) -> float:
        """
        Optimize parameters over a batch of samples.

        Model updates its internal Fields from tensors, then steps forward.

        Args:
            initial_tensors: Dict of tensors {field_name: Tensor(batch, x, y, vector)}
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

            # Update model's internal fields from tensors
            self.model.update_from_tensors(initial_tensors)

            # Initialize loss accumulator (scalar)
            total_loss = math.tensor(0.0)

            for step in range(self.rollout_steps):
                # Forward step using internal fields
                # Model updates its internal state and returns new fields
                current_state = self.model.forward()

                # Compute loss for this timestep
                step_loss = math.tensor(0.0)
                for field_name, gt_field in target_fields.items():
                    target = gt_field.time[step]
                    prediction = current_state[field_name]

                    # Compute L2 loss and reduce properly
                    field_loss = l2_loss(prediction - target)
                    field_loss = mean(field_loss, 'batch')
                    step_loss += field_loss

                total_loss += step_loss

            # Average over timesteps
            avg_loss = total_loss / self.rollout_steps
            return avg_loss

        # Run optimization
        
        estimated_params = minimize(loss_function, self.optimizer)
        
        # Compute final loss
        final_loss = loss_function(*estimated_params)
        return float(final_loss)

    #############
    # Utilities #
    #############
    def _update_params(self, learnable_tensors: Tuple[Tensor, ...]):
        """
        Update model parameters (scalars or fields) from optimizer.

        Args:
            learnable_tensors: Tuple of updated parameter values from optimizer
        """
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

        # Update optimizer state
        self.optimizer.x0 = list(learnable_tensors)
        self.learnable_parameters = list(learnable_tensors)

    def save_checkpoint(self, epoch: int, loss: float):
        """
        Save model checkpoint to specified path.

        Args:
            path: File path to save the checkpoint
            epoch: Current training epoch
            loss: Loss value at the checkpoint

        """
        
        params = [param.native('x,y') if isinstance(param, Tensor) else param for param in self.learnable_parameters]
        # Convert them to native tensors for saving
        checkpoint = {
            "learnable_parameters": params,
            "epoch": epoch,
            "loss": loss,
        }
        torch.save(checkpoint, self.checkpoint_path)
    
    def load_checkpoint(self, path: str, strict: bool = True) -> Dict[str, Any]:
        """
        Load model checkpoint from specified path.

        Args:
            path: File path to load the checkpoint from
            strict: Whether to strictly enforce that the keys in state_dict match the model
        Returns:
            Loaded checkpoint dictionary
        """
        checkpoint = torch.load(path)
        self.learnable_parameters = [math.tensor(param, spatial("x,y")) for param in checkpoint["learnable_parameters"]]
        return checkpoint

    def get_current_params(self) -> Dict[str, Tensor]:
        """
        Get current learnable parameters as a dictionary.

        Returns:
            Dictionary mapping parameter names to current values
        """
        return {name: param for name, param in zip(self.param_names, self.learnable_parameters)}
    

