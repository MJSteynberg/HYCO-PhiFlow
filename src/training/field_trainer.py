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
from src.data import DataManager
from src.models.physical.base import PhysicalModel


class FieldTrainer(AbstractTrainer):
    """
    Base class for PhiFlow field-based trainers.

    Provides all PhiFlow-specific functionality:
    - Field-based data management via DataManager
    - Optimization-based parameter inference
    - Physical model simulation and evaluation
    - Result saving in appropriate formats

    Unlike TensorTrainer which uses epoch-based training, FieldTrainer
    uses optimization-based training (e.g., math.minimize) which is more
    appropriate for physical parameter inference.

    Subclasses must implement:
    - _create_data_manager(): Create DataManager for loading fields
    - _create_model(): Create and initialize the physical model
    - _setup_optimization(): Setup optimization parameters and solve config

    The train() method can be overridden if needed, but a default
    implementation is provided for standard optimization-based training.

    Attributes:
        config: Full configuration dictionary
        data_manager: DataManager for loading field data
        model: PhysicalModel instance
        learnable_params: List of parameters to optimize
        learnable_params_config: Configuration for learnable parameters
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize field trainer.

        Args:
            config: Full configuration dictionary containing all settings
        """
        super().__init__(config)

        # Field-specific components (to be set by subclasses)
        self.data_manager: Optional[DataManager] = None
        self.model: Optional[PhysicalModel] = None
        self.learnable_params: List[Tensor] = []
        self.learnable_params_config: List[Dict[str, Any]] = []

        # Results storage
        self.final_loss: float = 0.0
        self.optimization_history: List[float] = []

    @abstractmethod
    def _create_data_manager(self) -> DataManager:
        """
        Create and return DataManager for loading field data.

        This method should:
        1. Get data specifications from self.config
        2. Set up paths for raw data and cache
        3. Create DataManager instance with appropriate settings
        4. Set self.data_manager
        5. Return the DataManager

        Returns:
            DataManager instance
        """
        pass

    @abstractmethod
    def _create_model(self) -> PhysicalModel:
        """
        Create and return the physical model.

        This method should:
        1. Get model specifications from self.config
        2. Instantiate the appropriate PhysicalModel subclass
        3. Initialize learnable parameters to their initial guesses
        4. Set self.model
        5. Return the model

        Returns:
            PhysicalModel instance
        """
        pass

    @abstractmethod
    def _setup_optimization(self) -> Solve:
        """
        Setup optimization configuration.

        This method should:
        1. Get optimization settings from self.config
        2. Determine which parameters to optimize
        3. Create math.Solve configuration
        4. Set self.learnable_params and self.learnable_params_config
        5. Return the Solve configuration

        Returns:
            math.Solve configuration for optimization
        """
        pass

    def train(self) -> Dict[str, Any]:
        """
        Execute optimization-based training.

        Default implementation of optimization-based parameter inference.
        Uses math.minimize to optimize physical model parameters.

        Subclasses can override for custom training logic.

        Returns:
            Dictionary with training results including optimized parameters
            and loss values
        """
        if self.model is None or self.data_manager is None:
            raise RuntimeError(
                "Model and data manager must be initialized before training"
            )

        print(f"\n{'='*60}")
        print(f"Starting Physical Model Optimization")
        print(f"{'='*60}\n")

        # Get optimization configuration
        solve_config = self._setup_optimization()

        # Get ground truth data for this training run
        gt_data = self._load_ground_truth()

        # Define loss function for optimization
        def loss_fn(*params):
            """
            Loss function for optimization.

            Args:
                *params: Learnable parameter values

            Returns:
                Loss value (scalar)
            """
            # Update model parameters
            self._update_model_parameters(params)

            # Run simulation
            predictions = self._run_simulation(gt_data)

            # Compute loss against ground truth
            loss = self._compute_loss(predictions, gt_data)

            # Track history
            loss_value = float(loss)
            self.optimization_history.append(loss_value)

            return loss

        # Run optimization
        print(f"Optimizing {len(self.learnable_params)} parameter(s)...")
        optimized_params = minimize(loss_fn, solve=solve_config, *self.learnable_params)

        # Update model with optimized parameters
        self._update_model_parameters(optimized_params)

        # Compute final loss
        final_predictions = self._run_simulation(gt_data)
        self.final_loss = float(self._compute_loss(final_predictions, gt_data))

        # Build results
        results = {
            "final_loss": self.final_loss,
            "optimization_history": self.optimization_history,
            "optimized_parameters": {},
            "iterations": len(self.optimization_history),
        }

        # Store optimized parameter values
        for param_config, param_value in zip(
            self.learnable_params_config, optimized_params
        ):
            param_name = param_config["name"]
            results["optimized_parameters"][param_name] = float(param_value)
            print(f"  {param_name}: {param_value}")

        print(f"\n{'='*60}")
        print(f"Optimization Complete!")
        print(f"Final Loss: {self.final_loss:.6f}")
        print(f"Iterations: {len(self.optimization_history)}")
        print(f"{'='*60}\n")

        return results

    # =========================================================================
    # PhiFlow-Specific Utilities
    # =========================================================================

    def _load_ground_truth(self) -> Dict[str, Field]:
        """
        Load ground truth field data for training.

        This method can be overridden by subclasses for custom loading logic.

        Returns:
            Dictionary mapping field names to Field objects with time dimension
        """
        raise NotImplementedError("Subclass must implement _load_ground_truth()")

    def _run_simulation(self, initial_data: Dict[str, Field]) -> Dict[str, Field]:
        """
        Run physical simulation with current model parameters.

        Args:
            initial_data: Dictionary of Fields with initial conditions

        Returns:
            Dictionary of Fields with simulation predictions
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        # Get initial state (first timestep)
        initial_state = {name: field.time[0] for name, field in initial_data.items()}

        # Get number of prediction steps from config
        num_steps = self.config.get("trainer_params", {}).get("num_predict_steps", 4)

        # Run simulation
        predictions = {name: [initial_state[name]] for name in initial_state.keys()}
        current_state = initial_state

        for t in range(num_steps):
            current_state = self.model.step(current_state)
            for name, field in current_state.items():
                predictions[name].append(field)

        # Stack predictions along time dimension
        from phi.field import stack
        from phi.math import batch

        stacked_predictions = {}
        for name, fields in predictions.items():
            stacked_predictions[name] = stack(fields, batch("time"))

        return stacked_predictions

    def _compute_loss(
        self, predictions: Dict[str, Field], ground_truth: Dict[str, Field]
    ) -> Tensor:
        """
        Compute loss between predictions and ground truth.

        Args:
            predictions: Dictionary of predicted Fields
            ground_truth: Dictionary of ground truth Fields

        Returns:
            Loss value as scalar Tensor
        """
        from phi.field import l2_loss

        total_loss = 0.0

        for field_name in predictions.keys():
            pred = predictions[field_name]
            gt = ground_truth[field_name]

            # Compute L2 loss for this field
            field_loss = l2_loss(pred - gt)
            total_loss = total_loss + field_loss

        return total_loss

    def _update_model_parameters(self, params):
        """
        Update model parameters during optimization.

        Args:
            params: Tuple or list of parameter values
        """
        if not isinstance(params, (tuple, list)):
            params = [params]

        for param_config, param_value in zip(self.learnable_params_config, params):
            param_name = param_config["name"]
            setattr(self.model, param_name, param_value)

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
        print(f"Saved optimization results to {path}")

    def load_results(self, path: Path) -> Dict[str, Any]:
        """
        Load optimization results from file.

        Args:
            path: Path to results file

        Returns:
            Results dictionary
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")

        results = torch.load(path)

        # Apply optimized parameters to model
        if "optimized_parameters" in results:
            for param_name, param_value in results["optimized_parameters"].items():
                if hasattr(self.model, param_name):
                    setattr(self.model, param_name, param_value)
                    print(f"Loaded parameter {param_name} = {param_value}")

        return results
