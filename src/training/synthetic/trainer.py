"""PhiML trainer for synthetic models with LR scheduling."""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR

from phi.torch.flow import *
from phiml import math as phimath
from phiml import nn as phiml_nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SyntheticTrainer:
    """PhiML trainer for synthetic models with LR scheduling support."""

    def __init__(self, config: Dict[str, Any], model):
        self.model = model
        self.config = config

        self._parse_config(config)
        self._setup_optimizer()
        self._setup_scheduler()

        self.best_val_loss = float("inf")
        self.best_epoch = 0

        logger.info(f"Trainer ready: lr={self.learning_rate}, scheduler={self.scheduler_type}")

    def _parse_config(self, config):
        """Parse training configuration."""
        synthetic_config = config['trainer']['synthetic']
        self.epochs = synthetic_config['epochs']
        self.learning_rate = synthetic_config['learning_rate']
        self.batch_size = config['trainer']['batch_size']
        self.rollout_steps = synthetic_config.get('rollout_steps', config['trainer'].get('rollout_steps', 4))
        self.scheduler_type = synthetic_config.get('scheduler', 'cosine')

        model_path_dir = config["model"]["synthetic"]["model_path"]
        model_save_name = config["model"]["synthetic"]["model_save_name"]
        self.checkpoint_path = Path(model_path_dir) / f"{model_save_name}"
        os.makedirs(model_path_dir, exist_ok=True)

    def _setup_optimizer(self):
        """Setup PhiML optimizer."""
        self.optimizer = phiml_nn.adam(self.model.network, learning_rate=self.learning_rate)

    def _setup_scheduler(self):
        """Setup LR scheduler (torch scheduler computes LR, applied to PhiML)."""
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        dummy_optimizer = torch.optim.Adam([dummy_param], lr=self.learning_rate)

        if self.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(dummy_optimizer, T_max=self.epochs)
        elif self.scheduler_type == 'step':
            self.scheduler = StepLR(dummy_optimizer, step_size=self.epochs // 3, gamma=0.1)
        elif self.scheduler_type == 'exponential':
            self.scheduler = ExponentialLR(dummy_optimizer, gamma=0.99)
        else:
            self.scheduler = None

    def _update_learning_rate(self):
        """Update PhiML optimizer learning rate from scheduler."""
        if self.scheduler is not None:
            self.scheduler.step()
            new_lr = self.scheduler.get_last_lr()[0]
            # Update existing optimizer's learning rate to preserve momentum
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

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
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataset.iterate_batches(self.batch_size, shuffle=True):
                batch_loss = self._train_batch(batch)
                epoch_loss += batch_loss
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else epoch_loss
            results["train_losses"].append(float(avg_epoch_loss))
            results["epochs"].append(epoch + 1)

            loss_value = float(avg_epoch_loss)
            if loss_value < self.best_val_loss:
                self.best_val_loss = loss_value
                self.best_epoch = epoch + 1
                results["best_epoch"] = self.best_epoch
                results["best_val_loss"] = self.best_val_loss

            self.save_checkpoint(epoch=epoch, loss=avg_epoch_loss)
            self._update_learning_rate()

            epoch_time = time.time() - start_time
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.learning_rate
            pbar.set_postfix({
                "loss": f"{loss_value:.6f}",
                "best": f"{self.best_val_loss:.6f}",
                "lr": f"{current_lr:.2e}",
                "time": f"{epoch_time:.2f}s"
            })

        results["final_loss"] = results["train_losses"][-1]
        return results

    def _train_batch(self, batch):
        """Train on a single batch using autoregressive rollout."""
        initial_state = batch.initial_state
        targets = batch.targets

        def loss_function(init_state, rollout_targets):
            current_state = init_state
            total_loss = phimath.tensor(0.0)

            for t in range(self.rollout_steps):
                next_state = self.model(current_state)
                target_t = rollout_targets.time[t]
                step_loss = phimath.mean((next_state - target_t)**2)
                total_loss += step_loss
                current_state = next_state

            total_loss = phimath.mean(total_loss, 'batch')
            return total_loss / float(self.rollout_steps)

        self.optimizer.zero_grad()
        
        loss = loss_function(initial_state, targets)
        # Convert to native torch tensor for backward
        if hasattr(loss, 'native'):
            native_loss = loss.native()
        else:
            native_loss = loss
            
        native_loss.backward()
        self.optimizer.step()
        
        return loss

    def save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint."""
        self.model.save(str(self.checkpoint_path))

    def load_checkpoint(self, path: Path = None):
        """Load model checkpoint."""
        path = path or self.checkpoint_path
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        self.model.load(str(path))
        logger.info(f"Loaded checkpoint from {path}")
