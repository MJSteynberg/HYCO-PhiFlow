"""PhiML trainer for physical models using inverse problem optimization."""

import os
import time
from typing import Dict, Any, List, Tuple
from pathlib import Path
from tqdm import tqdm

from phi.torch.flow import *
from phi.math import math, Tensor, minimize
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
            x0=self.model.params,
            max_iterations=self.max_iterations,
            suppress=(math.NotConverged,),
        )

        self.best_val_loss = float("inf")
        self.best_epoch = 0

        # Loss scaling for hybrid training (NEW)
        self.real_loss_weight = 1.0
        self.interaction_loss_weight = 1.0
        self.proportional_scaling = False

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

        # Training params
        physical_config = config['trainer']['physical']
        self.rollout_steps = physical_config.get('rollout_steps', config['trainer'].get('rollout_steps', 4))
        self.batch_size = physical_config.get('batch_size', 1)
        self.method = physical_config['method']
        self.abs_tol = float(physical_config['abs_tol'])
        self.max_iterations = int(physical_config['max_iterations'])

    def set_loss_scaling(self, real_weight: float, interaction_weight: float,
                         proportional: bool = False):
        """Configure loss scaling for hybrid training."""
        self.real_loss_weight = real_weight
        self.interaction_loss_weight = interaction_weight
        self.proportional_scaling = proportional
    
    def _optimize_batch_separated(self, separated_batch, params: Tensor) -> float:
        """Optimize with separated real/generated losses."""
        
        def loss_function(params: Tensor) -> Tensor:
            def compute_loss(init_state, targets):
                total_loss = math.tensor(0.0)
                current_state = init_state
                for step in range(self.rollout_steps):
                    current_state = self.model.forward(current_state, params)
                    target = targets.time[step]
                    step_loss = math.mean((current_state - target) ** 2)
                    total_loss += step_loss
                return math.mean(total_loss, 'batch') / self.rollout_steps
            
            real_loss = math.tensor(0.0)
            interaction_loss = math.tensor(0.0)
            
            if separated_batch.has_real:
                real_loss = compute_loss(
                    separated_batch.real_initial_state,
                    separated_batch.real_targets
                )
            
            if separated_batch.has_generated:
                interaction_loss = compute_loss(
                    separated_batch.generated_initial_state,
                    separated_batch.generated_targets
                )
            
            # Proportional scaling
            i_weight = self.interaction_loss_weight
            if self.proportional_scaling and separated_batch.has_real and separated_batch.has_generated:
                ratio = math.stop_gradient(real_loss / (interaction_loss + 1e-8))
                i_weight = self.interaction_loss_weight * ratio
            
            return self.real_loss_weight * real_loss + i_weight * interaction_loss
        
        try:
            estimated_params = minimize(loss_function, self.optimizer)
            self._update_params(estimated_params)
        except Diverged:
            estimated_params = self.optimizer.params
            logger.warning("Optimization diverged, using last parameters")
        
        return float(loss_function(estimated_params))

    def train_separated(self, dataset, num_epochs: int, verbose: bool = True) -> Dict[str, Any]:
        """Execute training with separated real/generated batches for loss scaling."""
        results = {
            "train_losses": [],
            "epochs": [],
            "num_epochs": num_epochs,
            "best_epoch": 0,
            "best_val_loss": float("inf"),
        }

        pbar = tqdm(range(num_epochs), desc="Training (separated)", unit="epoch", disable=not verbose)
        for epoch in pbar:
            start_time = time.time()
            train_loss = 0.0
            num_batches = 0

            for separated_batch in dataset.iterate_batches_separated(batch_size=self.batch_size, shuffle=True):
                batch_loss = self._optimize_batch_separated(separated_batch, self.model.params)
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
                "loss": f"{avg_train_loss}",
                "best": f"{self.best_val_loss}",
                "time": f"{epoch_time}",
            })

        results["final_loss"] = results["train_losses"][-1]
        return results

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
                # Use current model params (updated after each batch)
                batch_loss = self._optimize_batch(batch.initial_state, batch.targets, self.model.params)
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

    def _optimize_batch(self, initial_state: Tensor, targets: Tensor, params: Tensor) -> float:
        """Optimize parameters over a batch."""
        def loss_function(params: Tensor) -> Tensor:
            total_loss = math.tensor(0.0)
            current_state = initial_state

            for step in range(self.rollout_steps):
                current_state = self.model.forward(current_state, params)
                target = targets.time[step]
                step_loss = math.mean((current_state - target) ** 2)
                total_loss += step_loss

            return math.mean(total_loss, 'batch') / self.rollout_steps
        
        estimated_params = minimize(loss_function, self.optimizer)
        
        # Update model with optimized params
        self._update_params(estimated_params)
        
        return float(loss_function(estimated_params))

    def _update_params(self, params: Tensor):
        """Update model parameters from optimizer (tensors only)."""
        self.model.params = params
        self.optimizer.x0 = params
        self.learnable_parameters = params

    def save_checkpoint(self):
        self.model.save(self.checkpoint_path)

    def load_checkpoint(self):
        path = self.checkpoint_path
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        self.model.load(str(path))
        # Update optimizer's initial point to loaded params
        self.optimizer.x0 = self.model.params
        logger.info(f"Loaded checkpoint from {path}")