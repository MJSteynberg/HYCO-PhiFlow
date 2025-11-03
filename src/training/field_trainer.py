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

from phi.field import Field
from phi.math import Solve, Tensor, minimize

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
        learnable_params: List[torch.nn.Parameter]
    ):
        """
        Initialize field trainer with model and learnable parameters.

        Args:
            config: Full configuration dictionary containing all settings
            model: Pre-created PhysicalModel instance
            learnable_params: List of parameters to optimize
        """
        super().__init__(config)

        # Store model and parameters
        self.model = model
        self.learnable_params = learnable_params

        # Results storage
        self.final_loss: float = 0.0
        self.training_history: List[float] = []

    @abstractmethod
    def _train_sample(
        self, 
        initial_fields: Dict[str, Field], 
        target_fields: Dict[str, Field]
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

    def train(self, data_source, num_epochs: int) -> Dict[str, Any]:
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
        if self.model is None:
            raise RuntimeError("Model must be initialized before training")

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Physical Model Training")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"{'='*60}\n")

        results = {
            "train_losses": [],
            "epochs": [],
            "num_epochs": num_epochs
        }

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_samples = 0
            
            # Iterate through data source
            for sample in data_source:
                # Unpack sample (2-tuple: no weights!)
                initial_fields, target_fields = sample
                
                # Train on this sample
                loss = self._train_sample(initial_fields, target_fields)
                
                epoch_loss += loss
                num_samples += 1
            
            # Compute average loss for epoch
            avg_loss = epoch_loss / num_samples if num_samples > 0 else float('inf')
            results["train_losses"].append(avg_loss)
            results["epochs"].append(epoch + 1)
            self.training_history.append(avg_loss)
            
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.6f}")
        
        self.final_loss = results["train_losses"][-1]
        results["final_loss"] = self.final_loss
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Complete!")
        logger.info(f"Final Loss: {self.final_loss:.6f}")
        logger.info(f"{'='*60}\n")

        return results


    # =========================================================================
    # PhiFlow-Specific Utilities (kept for backward compatibility)
    # =========================================================================

    def save_results(self, path: Path, results: Dict[str, Any]):
        """
        Save optimization results to file.

        Unlike PyTorch models, we don't save a state_dict.
        Instead, we save the optimized parameter values and loss history.

        Args:
            path: Path to save results
            results: Results dictionary from train()
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save as PyTorch file for consistency, but content is different
        torch.save(results, path)
        logger.info(f"Saved training results to {path}")

    def load_results(self, path: Path) -> Dict[str, Any]:
        """
        Load training results from file.

        Args:
            path: Path to results file

        Returns:
            Results dictionary
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")

        results = torch.load(path)
        logger.info(f"Loaded training results from {path}")

        return results

