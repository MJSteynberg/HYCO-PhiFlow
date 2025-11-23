"""PhiML trainer for physical models using inverse problem optimization."""

import os
import time
from typing import Dict, Any, List, Tuple
from pathlib import Path
from tqdm import tqdm

from phi.torch.flow import *
from phi.math import math, Tensor
from phiml import nn as phiml_nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PhysicalTrainer:
    """Inverse problem solver for physical models using math.minimize."""

    def __init__(self, config: Dict[str, Any], model):
        self.model = model
        self._parse_config(config)

        # Setup optimizer inline
        self.optimizer = math.Solve(
            method=self.method,
            abs_tol=self.abs_tol,
            x0=self.learnable_parameters,
            max_iterations=self.max_iterations,
            suppress=(math.NotConverged,),
        )

        self.best_val_loss = float("inf")
        self.best_epoch = 0

        # Try to load checkpoint if exists
        if self.checkpoint_path.exists():
            try:
                self.load_checkpoint()
                logger.info(f"Loaded checkpoint from {self.checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

    def _parse_config(self, config: Dict[str, Any]):
        """Parse training configuration."""
        # Device setup
        device = config['trainer'].get('device', 'cpu')
        if device.startswith('cuda'):
            math.use('torch')
            import torch
            torch.set_default_device(device)

        # Checkpoint path
        model_path_dir = config["model"]["physical"]["model_path"]
        model_save_name = config["model"]["physical"]["model_save_name"]
        self.checkpoint_path = Path(model_path_dir) / f"{model_save_name}.npz"
        os.makedirs(model_path_dir, exist_ok=True)

        # Learnable parameters - store as tensors only
        learnable_config = config['trainer']['physical']['learnable_parameters']
        self.learnable_parameters = []
        self.param_names = []

        for param_config in learnable_config:
            param_name = param_config["name"]
            initial_value = param_config.get("initial_guess")

            # Convert to tensor if it's a field
            if hasattr(initial_value, 'values'):
                self.learnable_parameters.append(initial_value.values)
            else:
                self.learnable_parameters.append(math.tensor(initial_value))

            self.param_names.append(param_name)

        # Training params
        physical_config = config['trainer']['physical']
        self.rollout_steps = physical_config.get('rollout_steps', config['trainer'].get('rollout_steps', 4))
        self.batch_size = physical_config.get('batch_size', 1)
        self.method = physical_config['method']
        self.abs_tol = physical_config['abs_tol']
        self.max_iterations = physical_config['max_iterations']

    def train(self, dataset, num_epochs: int, verbose: bool = True) -> Dict[str, Any]:
        """Execute training for specified number of epochs."""
        results = {
            "train_losses": [],
            "epochs": [],
            "num_epochs": num_epochs,
            "best_epoch": 0,
            "best_val_loss": float("inf"),
        }

        pbar = tqdm(range(num_epochs), desc="Training", unit="epoch", disable=not verbose)

        for epoch in pbar:
            start_time = time.time()
            train_loss = 0.0
            num_batches = 0

            for batch in dataset.iterate_batches(batch_size=self.batch_size, shuffle=True):
                if num_batches < 5:
                    batch_loss = self._optimize_batch(batch.initial_state, batch.targets)
                    train_loss += batch_loss
                    num_batches += 1

            avg_train_loss = train_loss / num_batches if num_batches > 0 else train_loss
            results["train_losses"].append(avg_train_loss)
            results["epochs"].append(epoch + 1)

            if avg_train_loss < self.best_val_loss:
                self.best_val_loss = avg_train_loss
                self.best_epoch = epoch + 1
                results["best_epoch"] = self.best_epoch
                results["best_val_loss"] = self.best_val_loss
                self.save_checkpoint()

            epoch_time = time.time() - start_time
            pbar.set_postfix({
                "loss": f"{avg_train_loss:.6f}",
                "best": f"{self.best_val_loss:.6f}",
                "time": f"{epoch_time:.2f}s",
            })

        results["final_loss"] = results["train_losses"][-1]
        return results

    def _optimize_batch(self, initial_state: Tensor, targets: Tensor) -> float:
        """Optimize parameters over a batch."""
        def loss_function(*learnable_tensors: Tensor) -> Tensor:
            self._update_params(learnable_tensors)

            total_loss = math.tensor(0.0)
            current_state = initial_state

            for step in range(self.rollout_steps):
                current_state = self.model.forward(current_state)
                target = targets.time[step]
                step_loss = math.mean((current_state - target) ** 2)
                total_loss += step_loss

            return math.mean(total_loss, 'batch') / self.rollout_steps

        estimated_params = minimize(loss_function, self.optimizer)
        return float(loss_function(*estimated_params))

    def _update_params(self, learnable_tensors: Tuple[Tensor, ...]):
        """Update model parameters from optimizer (tensors only)."""
        for param_name, param_value in zip(self.param_names, learnable_tensors):
            setattr(self.model, param_name, param_value)

        self.optimizer.x0 = list(learnable_tensors)
        self.learnable_parameters = list(learnable_tensors)

    def save_checkpoint(self):
        """Save learnable parameters using phi native format."""
        params_dict = {name: param for name, param in zip(self.param_names, self.learnable_parameters)}
        math.save(str(self.checkpoint_path), **params_dict)

    def load_checkpoint(self):
        """Load learnable parameters using phi native format."""
        loaded = math.load(str(self.checkpoint_path))
        self.learnable_parameters = [loaded[name] for name in self.param_names]
        for param_name, param_value in zip(self.param_names, self.learnable_parameters):
            setattr(self.model, param_name, param_value)

    def get_current_params(self) -> Dict[str, Tensor]:
        """Get current learnable parameters as a dictionary."""
        return {name: param for name, param in zip(self.param_names, self.learnable_parameters)}
