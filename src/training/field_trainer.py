"""
Field Trainer

This module provides the base class for PhiFlow field-based trainers.
All PhiFlow-specific functionality lives here, including:
- Field-based data management
- Optimization-based parameter inference
- Physical model simulation
- Result saving and visualization

This separates field-based concerns from tensor-based concerns,
providing a cleaner architecture for physical model training.
"""

from abc import abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from phi.field import Field
from phi.math import Solve, Tensor, minimize
from tqdm import tqdm
import time
import os
from phi.torch.flow import *
from src.training.abstract_trainer import AbstractTrainer
from src.models.physical.base import PhysicalModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FieldTrainer(AbstractTrainer):
    """
    Base class for PhiFlow field-based trainers.

    NEW ARCHITECTURE (Phase 1):
    - Model and learnable_params are passed in __init__, not created internally
    - Data is passed to train() method, not held internally
    - Trainers are persistent across training calls
    - Optimizer state is preserved

    Provides all PhiFlow-specific functionality:
    - Field-based optimization
    - Physical model simulation and evaluation
    - Result saving in appropriate formats

    Unlike TensorTrainer which uses epoch-based training, FieldTrainer
    uses sample-by-sample iteration (since field operations don't batch well).

    Subclasses should implement:
    - _train_sample(): Train on a single sample

    The train() method accepts data explicitly and should not be overridden
    in most cases.

    Attributes:
        config: Full configuration dictionary
        model: PhysicalModel instance (passed in __init__)
        learnable_params: List of parameters to optimize (passed in __init__)
        optimizer: PhiML/PyTorch optimizer
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: Any,  # PhysicalModel instance
        learnable_params: List[Dict[str, Tensor]],
    ):
        """
        Initialize field trainer with model and learnable parameters.

        Args:
            config: Full configuration dictionary containing all settings
            model: Pre-created PhysicalModel instance
            learnable_params: List of parameters to optimize
        """
        super().__init__(config)

        # --- Derive all parameters from config ---
        self.data_config = config["data"]
        self.model_config = config["model"]["physical"]
        self.trainer_config = config["trainer_params"]

        # --- Data specifications ---
        self.field_names: List[str] = self.data_config["fields"]
        self.dset_name = self.data_config["dset_name"]
        self.data_dir = self.data_config["data_dir"]

        # # --- Checkpoint path ---
        # model_save_name = self.model_config["model_save_name"]
        # model_path_dir = self.model_config["model_path"]
        # self.checkpoint_path = Path(model_path_dir) / f"{model_save_name}.pth"
        # os.makedirs(model_path_dir, exist_ok=True)

        # -- Store model and parameters ---
        self.model = model
        self.learnable_params = [p['initial_guess'] for p in learnable_params]
        self.param_names = [p["name"] for p in learnable_params]

        # Create optimizer
        self.optimizer = self._create_optimizer()
 
        # Results storage
        self.final_loss: float = 0.0
        self.training_history: List[float] = []

        self.best_val_loss = float("inf")
        self.best_epoch = 0

        # --- Get parameters ---
        self.num_predict_steps= self.trainer_config["num_predict_steps"]

    def _create_optimizer(self):
        """
        Create optimizer for learnable parameters.

        Returns:
            PhiFlow optimizer instance
        """
        method = self.trainer_config['method']
        abs_tol = self.trainer_config['abs_tol']
        max_iterations = self.trainer_config['max_iterations']

        optimizer = math.Solve(
            method=method,
            abs_tol=abs_tol,
            x0=self.learnable_params,
            max_iterations=max_iterations,
            suppress=(math.NotConverged,),
        )
        return optimizer

    @abstractmethod
    def _train_epoch(
        self, data_source: DataLoader
    ) -> float:
        """
        Train on a single sample.

        This method should:
        1. Run simulation from initial_fields
        2. Compute loss against target_fields
        3. Perform backward pass and optimization step
        4. Return loss value

        Args:
            initial_fields: Dict[field_name, Field] for initial state
            target_fields: Dict[field_name, List[Field]] for target trajectory

        Returns:
            Loss value for this sample
        """
        pass

    def train(self, data_source: DataLoader, num_epochs: int, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute training for specified number of epochs with provided data.

        NEW SIGNATURE: Data is passed explicitly, not held internally.

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
            train_loss = self._train_epoch(data_source)

            results["train_losses"].append(train_loss)
            results["epochs"].append(epoch + 1)

            if train_loss < self.best_val_loss:
                self.best_val_loss = train_loss
                self.best_epoch = epoch + 1
                results["best_epoch"] = self.best_epoch
                results["best_val_loss"] = self.best_val_loss

                # self.save_checkpoint(
                #     epoch=epoch,
                #     loss=train_loss,
                #     optimizer_state=(
                #         self.optimizer.state_dict() if self.optimizer else None
                #     )
                # )

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

    # =========================================================================
    # PhiFlow-Specific Utilities (kept for backward compatibility)
    # =========================================================================

    def update_optimizer_params(self, new_params: List[Tensor]):
        """
        Update optimizer with new parameter values.

        This is useful for continuing optimization with updated initial guesses.

        Args:
            new_params: New parameter values
        """
        self.learnable_params = new_params
        self.optimizer.x0 = new_params

    def get_current_params(self) -> Dict[str, float]:
        """
        Get current parameter values.

        Returns:
            Dictionary mapping parameter names to values
        """
        return {
            name: self._tensor_to_float(param)
            for name, param in zip(self.param_names, self.learnable_params)
        }
    
    @staticmethod
    def _tensor_to_float(tensor: Tensor) -> float:
        """
        Extract Python float from PhiFlow tensor.

        Args:
            tensor: PhiFlow Tensor

        Returns:
            Python float value
        """
        return float(tensor)
