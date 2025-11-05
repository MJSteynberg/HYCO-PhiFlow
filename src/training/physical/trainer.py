# In src/training/physical/trainer.py

import os
import sys
import time
import logging
from typing import Dict, Any, List

# --- PhiFlow Imports ---
from phi.torch.flow import *
from phi.field import l2_loss
from phi.math import math, Tensor

# --- Repo Imports ---
from src.models import ModelRegistry
from src.training.field_trainer import FieldTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Suppress PhiFlow's verbose ML_LOGGER to avoid Unicode errors on Windows
# PhiFlow uses its own logging that outputs Unicode characters
import warnings
warnings.filterwarnings('ignore')
try:
    # Suppress PhiFlow's internal loggers
    for logger_name in ['phi', 'phiml', 'phi.math', 'phiml.math']:
        phi_logger = logging.getLogger(logger_name)
        phi_logger.setLevel(logging.CRITICAL)
        phi_logger.disabled = True
except Exception:
    pass


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

    def __init__(self, config: Dict[str, Any], model, learnable_params: List[Tensor]):
        """
        Initializes the trainer with external model and learnable parameters.
        
        Args:
            config: Full configuration dictionary
            model: Pre-created physical model
            learnable_params: List of PhiFlow Tensors for learnable parameters
        """
        # Initialize base trainer with model and params
        super().__init__(config, model, learnable_params)

        self.project_root = config.get("project_root", ".")

        # --- Parse Configs ---
        self.data_config = config["data"]
        self.model_config = config["model"]["physical"]
        self.trainer_config = config["trainer_params"]

        # --- Get parameters ---
        self.num_predict_steps: int = self.trainer_config["num_predict_steps"]

        # --- Get Ground Truth field names ---
        self.gt_fields: List[str] = self.data_config["fields"]

        # --- Optimization settings ---
        self.method = self.trainer_config.get("method", "L-BFGS-B")
        self.abs_tol = self.trainer_config.get("abs_tol", 1e-6)
        # Note: rel_tol is not supported by minimize(), only abs_tol
        # Note: epochs now controls max_iterations per simulation (semantic change)
        self.max_iterations = self.trainer_config.get("max_iterations", self.trainer_config.get("epochs", 50))
        
        # Configure error suppression for hybrid training
        self.suppress_convergence = self.trainer_config.get("suppress_convergence_errors", True)
        
        # --- Store parameter names for logging ---
        learnable_params_config = self.trainer_config.get("learnable_parameters", [])
        self.param_names = [p["name"] for p in learnable_params_config]
        
        # --- Memory monitoring (optional, enabled by config) ---
        enable_memory_monitoring = self.trainer_config.get(
            "enable_memory_monitoring", False
        )
        if enable_memory_monitoring:
            try:
                from src.utils.memory_monitor import PerformanceMonitor

                self.memory_monitor = PerformanceMonitor(
                    enabled=True, device=0 if torch.cuda.is_available() else -1
                )
                self.verbose_iterations = self.trainer_config.get(
                    "memory_monitor_batches", 5
                )
                logger.info(
                    f"Performance monitoring enabled (verbose for first {self.verbose_iterations} iterations)"
                )
            except ImportError:
                logger.warning(
                    "Could not import PerformanceMonitor. Monitoring disabled."
                )
                self.memory_monitor = None
        else:
            self.memory_monitor = None

        # Only log initialization if not suppressed
        if not self.config.get("trainer_params", {}).get("suppress_training_logs", False):
            logger.info(
                f"PhysicalTrainer initialized with {len(learnable_params)} learnable parameter(s)."
            )

    def _train_sample(self, initial_fields: Dict[str, Any], target_fields: Dict[str, Any]) -> float:
        """
        Trains on a single sample by optimizing learnable parameters.
        
        Args:
            initial_fields: Dictionary of initial field states
            target_fields: Dictionary of target field trajectories (list of fields per timestep)
        
        Returns:
            Final loss value (float)
        """
        # Track loss function calls for monitoring
        loss_call_count = [0]
        
        # Get learnable parameter names from config
        learnable_params_config = self.trainer_config.get("learnable_parameters", [])
        param_names = [p["name"] for p in learnable_params_config]
        
        # Extract rollout targets - assume target_fields is a dict with lists of fields
        gt_rollout_dict = {}
        for field_name, field_list in target_fields.items():
            # Stack the list of fields along time dimension
            if field_list:  # If not empty
                stacked_field = stack(field_list, batch("time"))
                gt_rollout_dict[field_name] = stacked_field
        
        def loss_function(*learnable_tensors):
            """
            Calculates L2 loss for a rollout using current parameter guesses.
            """
            loss_call_count[0] += 1
            iteration_num = loss_call_count[0]

            # 1. Update the model's parameters with the current guess
            for i, param_tensor in enumerate(learnable_tensors):
                param_name = param_names[i]
                # Set the parameter on the model
                setattr(self.model, param_name, param_tensor)

            # 2. Simulate forward from initial state
            total_loss = math.tensor(0.0)
            current_state = initial_fields

            for step in range(self.num_predict_steps):
                current_state = self.model.step(current_state)

                # 3. Calculate L2 loss for this step
                step_loss = 0.0
                for field_name, gt_rollout in gt_rollout_dict.items():
                    if step < gt_rollout.shape.get_size('time'):
                        target = gt_rollout.time[step]
                        pred = current_state[field_name]

                        # Compute L2 loss
                        diff = pred - target
                        field_loss = l2_loss(diff)
                        field_loss = math.sum(field_loss)
                        step_loss += field_loss

                total_loss += step_loss

            final_loss = total_loss / self.num_predict_steps
            # Print loss for first few iterations (if monitoring enabled and not suppressed)
            suppress_logs = self.config.get("trainer_params", {}).get("suppress_training_logs", False)
            if not suppress_logs and hasattr(self, "memory_monitor") and self.memory_monitor and iteration_num <= self.verbose_iterations:
                logger.info(f"  Iteration {iteration_num}: loss={final_loss}")

            return final_loss

        # Setup optimization
        suppress_list = []
        if self.suppress_convergence:
            suppress_list.append(math.NotConverged)
        
        solve_params = math.Solve(
            method=self.method,
            abs_tol=self.abs_tol,
            x0=self.learnable_params,  # Use params from base class
            max_iterations=self.max_iterations,
            suppress=tuple(suppress_list),
        )
        # Run optimization with detailed error tracking
        try:
            # Suppress PhiFlow's internal logger that causes Unicode errors on Windows
            if hasattr(self, "memory_monitor") and self.memory_monitor:
                with self.memory_monitor.track("optimization"):
                    estimated_tensors = math.minimize(loss_function, solve_params)
            else:
                estimated_tensors = math.minimize(loss_function, solve_params)
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            estimated_tensors = tuple(self.learnable_params)

        # Compute final loss and return as float
        final_loss = loss_function(*estimated_tensors)
        
        # Log optimization results at DEBUG level
        logger.debug(f"\n{'='*60}")
        logger.debug(f"OPTIMIZATION RESULTS")
        logger.debug(f"{'='*60}")
        for i, param_name in enumerate(param_names):
            initial_val = self.learnable_params[i]
            final_val = estimated_tensors[i]
            logger.debug(f"Parameter: {param_name}")
            logger.debug(f"  Initial guess: {initial_val}")
            logger.debug(f"  Optimized value: {final_val}")
            
            # Try to get true value from model if available
            if hasattr(self.model, f"_true_{param_name}"):
                true_val = getattr(self.model, f"_true_{param_name}")
                error = abs(float(final_val) - float(true_val))
                rel_error = (error / abs(float(true_val))) * 100 if abs(float(true_val)) > 1e-10 else 0
                logger.debug(f"  True value: {true_val}")
                logger.debug(f"  Absolute error: {error:.6f}")
                logger.debug(f"  Relative error: {rel_error:.2f}%")
        logger.debug(f"{'='*60}\n")
        
        # Update learnable_params with optimized values for next sample
        self.learnable_params = list(estimated_tensors)
        
        # Extract native Python float from PhiFlow tensor
        if hasattr(final_loss, 'numpy'):
            final_loss_float = float(final_loss.numpy())
        else:
            final_loss_float = float(final_loss)
        
        return final_loss_float

