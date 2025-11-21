# src/training/synthetic/trainer.py

"""
Pure PhiML trainer for synthetic models.
Uses phiml.nn.update_weights for training following PhiML best practices.

NO tensor conversions - works directly with PhiML tensors from Dataset.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Union
from tqdm import tqdm

from phiml import math as phimath
from phiml import nn as phiml_nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SyntheticTrainer:
    """
    Pure PhiML trainer for synthetic models.

    Works directly with PhiML tensors - no conversions needed!
    Accepts Dataset which yields PhiML tensor batches.
    """

    def __init__(self, config: Dict[str, Any], model):
        """
        Initialize trainer with PhiML model.

        Args:
            config: Full configuration dictionary
            model: PhiML synthetic model (must have get_network() method)
        """
        super().__init__()

        self.model = model
        self.config = config

        # Parse configuration
        self._parse_config(config)

        # Setup PhiML optimizer
        self._setup_optimizer()

        # Validation state tracking
        self.best_val_loss = float("inf")
        self.best_epoch = 0

        logger.info(f"PhiML trainer ready: lr={self.learning_rate}, epochs={self.epochs}")

    def _parse_config(self, config):
        """Parse configuration for trainer."""
        # Training parameters
        self.epochs = config['trainer']['synthetic']['epochs']
        self.learning_rate = config['trainer']['synthetic']['learning_rate']
        self.batch_size = config['trainer']['batch_size']
        self.rollout_steps = config['trainer']['rollout_steps']

        # Checkpoint path
        model_path_dir = config["model"]["synthetic"]["model_path"]
        model_save_name = config["model"]["synthetic"]["model_save_name"]
        self.checkpoint_path = Path(model_path_dir) / f"{model_save_name}_phiml.npz"
        os.makedirs(model_path_dir, exist_ok=True)

    def _setup_optimizer(self):
        """Setup PhiML optimizer."""
        self.optimizer = phiml_nn.adam(
            self.model.get_network(),
            learning_rate=self.learning_rate
        )
        logger.debug(f"Created PhiML Adam optimizer with lr={self.learning_rate}")

    def train(self, dataset, num_epochs: int, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute training for specified number of epochs.

        Args:
            dataset: Dataset yielding PhiML tensor batches
            num_epochs: Number of epochs to train
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

        # Create progress bar
        pbar = tqdm(
            range(num_epochs),
            desc="Training",
            unit="epoch",
            disable=not verbose
        )

        for epoch in pbar:
            start_time = time.time()

            # Train one epoch using PhiML dataset iterator
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataset.iterate_batches(self.batch_size, shuffle=True):
                # Compute loss and update weights using PhiML
                batch_loss = self._train_batch(batch)
                epoch_loss += batch_loss
                num_batches += 1

            # Average loss over batches
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else epoch_loss

            # Update results
            results["train_losses"].append(float(avg_epoch_loss))
            results["epochs"].append(epoch + 1)

            # Track best model (convert PhiML tensor to float)
            loss_value = float(avg_epoch_loss)
            if loss_value < self.best_val_loss:
                self.best_val_loss = loss_value
                self.best_epoch = epoch + 1
                results["best_epoch"] = self.best_epoch
                results["best_val_loss"] = self.best_val_loss

                # Save checkpoint
                self.save_checkpoint(epoch=epoch, loss=avg_epoch_loss)

            epoch_time = time.time() - start_time

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_value:.6f}",
                "best": f"{self.best_val_loss:.6f}",
                "epoch": f"{self.best_epoch}",
                "time": f"{epoch_time:.2f}s"
            })

        results["final_loss"] = results["train_losses"][-1]

        logger.info(f"Training complete! Best loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}")

        return results

    def _train_batch(self, batch):
        """
        Train on a single batch using PhiML's update_weights.

        Args:
            batch: Batch dataclass with:
                   - initial_state: Dict[field_name, Tensor]
                   - targets: Dict[field_name, Tensor]

        Returns:
            Loss value (scalar)
        """
        # Extract dicts of PhiML tensors (one tensor per field)
        initial_state = batch['initial_state']  # Dict[str, Tensor]
        targets = batch['targets']              # Dict[str, Tensor]

        # Define loss function for this batch (PhiML best practice)
        def loss_function(init_state, rollout_targets):
            """
            Compute autoregressive rollout loss.

            This function is passed to nn.update_weights which handles
            gradient computation and optimization automatically.

            Args:
                init_state: Dict[field_name, Tensor(batch, x, y, vector)]
                rollout_targets: Dict[field_name, Tensor(batch, time, x, y, vector)]
            """
            current_state = init_state  # Dict of tensors
            total_loss = 0.0

            # Autoregressive rollout
            for t in range(self.rollout_steps):
                # Predict next state (returns dict of tensors)
                next_state = self.model(current_state)

                # Compute loss for each field and sum
                step_loss = 0.0
                for field_name in rollout_targets.keys():
                    # Get target for this field at timestep t
                    target_t = rollout_targets[field_name].time[t]
                    predicted_t = next_state[field_name]

                    # Compute loss for this field
                    field_loss = phimath.l2_loss(predicted_t - target_t)
                    step_loss += phimath.mean(field_loss, 'batch')

                total_loss += step_loss

                # Update current state (dict of tensors)
                current_state = next_state

            return total_loss / float(self.rollout_steps)

        # Use PhiML's update_weights (one-line training!)
        # This handles:
        # - Gradient computation (backward)
        # - Optimizer step
        # - Returns loss value
        loss = phiml_nn.update_weights(
            self.model.get_network(),
            self.optimizer,
            loss_function,
            initial_state,
            targets
        )

        return loss

    def save_checkpoint(self, epoch: int, loss: float):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            loss: Current loss value
        """
        self.model.save(str(self.checkpoint_path))
        logger.debug(f"Checkpoint saved: epoch={epoch}, loss={loss:.6f} to {self.checkpoint_path}")

    def load_checkpoint(self, path: Path = None):
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        if path is None:
            path = self.checkpoint_path

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        self.model.load(str(path))
        logger.info(f"Loaded checkpoint from {path}")

    def print_model_summary(self):
        """Print model summary information."""
        logger.info("=" * 60)
        logger.info("MODEL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Framework: PhiML (Pure - No PyTorch)")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info("=" * 60)
