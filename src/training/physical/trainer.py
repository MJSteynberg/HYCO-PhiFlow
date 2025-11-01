# In src/training/physical/trainer.py

import os
import time
from typing import Dict, Any, List

# --- PhiFlow Imports ---
from phi.torch.flow import *
from phi.field import l2_loss
from phi.math import math, Tensor

# --- Repo Imports ---
from src.models import ModelRegistry
from src.data import DataManager, HybridDataset
from src.training.field_trainer import FieldTrainer


class PhysicalTrainer(FieldTrainer):
    """
    Solves an inverse problem for a PhysicalModel using cached data
    from DataManager/HybridDataset.

    This trainer uses math.minimize for optimization and leverages
    the efficient DataLoader pipeline with field conversion.
    
    Inherits from FieldTrainer to get PhiFlow-specific functionality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the trainer from a unified configuration dictionary.

        Args:
            config (Dict[str, Any]): The experiment configuration.
        """
        # Initialize base trainer
        super().__init__(config)
        
        self.project_root = config.get('project_root', '.')

        # --- Parse Configs ---
        self.data_config = config['data']
        self.model_config = config['model']['physical']
        self.trainer_config = config['trainer_params']
        
        # --- Get parameters ---
        self.train_sims: List[int] = self.trainer_config['train_sim']
        self.num_epochs: int = self.trainer_config['epochs']
        self.num_predict_steps: int = self.trainer_config['num_predict_steps']
        
        self.learnable_params_config = self.trainer_config.get('learnable_parameters', [])
        
        # --- Get Ground Truth field names ---
        self.gt_fields: List[str] = self.data_config['fields']
        
        # --- Setup DataManager and Dataset ---
        self.data_manager = self._create_data_manager()
        
        # --- Setup Learnable Parameters ---
        self.initial_guesses = self._get_initial_guesses()

        # --- Setup Model ---
        self.model = self._setup_physical_model()
        
        # --- Memory monitoring (optional, enabled by config) ---
        enable_memory_monitoring = self.trainer_config.get('enable_memory_monitoring', False)
        if enable_memory_monitoring:
            try:
                from src.utils.memory_monitor import PerformanceMonitor
                self.memory_monitor = PerformanceMonitor(
                    enabled=True,
                    device=0 if torch.cuda.is_available() else -1
                )
                self.verbose_iterations = self.trainer_config.get('memory_monitor_batches', 5)
                print(f"Performance monitoring enabled (verbose for first {self.verbose_iterations} iterations)")
            except ImportError:
                print("Warning: Could not import PerformanceMonitor. Monitoring disabled.")
                self.memory_monitor = None
        else:
            self.memory_monitor = None
        
        print(f"PhysicalTrainer initialized. Will optimize for {len(self.initial_guesses)} parameter(s).")
    
    def _create_data_manager(self) -> DataManager:
        """Create DataManager for loading cached field data."""
        raw_data_dir = os.path.join(
            self.project_root,
            self.data_config['data_dir'],
            self.data_config['dset_name']
        )
        cache_dir = os.path.join(
            self.project_root,
            self.data_config.get('cache_dir', 'data/cache')
        )
        
        return DataManager(
            raw_data_dir=raw_data_dir,
            cache_dir=cache_dir,
            config=self.config,  # Pass full config for validation
            validate_cache=self.data_config.get('validate_cache', True),
            auto_clear_invalid=self.data_config.get('auto_clear_invalid', False)
        )

    def _create_model(self):
        """
        Stub for BaseTrainer abstract method.
        Physical trainer uses _setup_physical_model() instead.
        """
        pass

    def _setup_physical_model(self):
        """
        Instantiates the physical model from the config, setting
        learnable parameters to their initial guesses.
        """
        model_name = self.model_config['name']
        
        # Start with a copy of the model config
        model_config_copy = self.model_config.copy()
        
        # Start with a copy of PDE params from config
        pde_params = model_config_copy.get('pde_params', {}).copy()
        
        # --- IMPORTANT ---
        # Overwrite/set the *initial guess* for the learnable parameters
        # on the model instance. The optimizer will update these.
        for param in self.learnable_params_config:
            pde_params[param['name']] = param['initial_guess']
        
        # Update the pde_params in the config copy
        model_config_copy['pde_params'] = pde_params

        # Use the model registry to create the model
        print(f"Creating physical model: {model_name}...")
        model = ModelRegistry.get_physical_model(model_name, model_config_copy)
        return model
    
    def _get_initial_guesses(self) -> List[Tensor]:
        """
        Extracts the initial guesses for the learnable parameters
        and wraps them as named PhiFlow Tensors.
        """
        guesses = []
        print("Setting up learnable parameters:")
        for param in self.learnable_params_config:
            name = param['name']
            guess_val = param['initial_guess']
            print(f"  - {name}: initial_guess={guess_val}")
            # Wrap the guess in a named Tensor
            guesses.append(math.tensor(guess_val))
        
        if not guesses:
            raise ValueError("No 'learnable_parameters' defined in trainer_params.")
        
        return guesses
    
    def _load_ground_truth_data(self, sim_index: int) -> Dict[str, Tensor]:
        """
        Loads ground truth data for one simulation using HybridDataset with field conversion.
        
        Args:
            sim_index (int): The simulation index to load.

        Returns:
            Dict[str, Tensor]: A dict mapping field names to their
                              full time-batched (batchᵇ=1, time, ...) tensors.
        """
        print(f"Loading ground truth for sim {sim_index} from cache...")
        
        # Create dataset with return_fields=True for this single simulation
        dataset = HybridDataset(
            data_manager=self.data_manager,
            sim_indices=[sim_index],
            field_names=self.gt_fields,
            num_frames=self.num_predict_steps + 1,
            num_predict_steps=self.num_predict_steps,
            return_fields=True
        )
        
        # Get the data (initial_fields and target_fields)
        initial_fields, target_fields = dataset[0]
        
        # Combine into full time sequence for each field
        data = {}
        for field_name in self.gt_fields:
            # Stack: [initial_field at t=0] + [target_fields at t=1..T]
            if field_name in target_fields:
                all_frames = [initial_fields[field_name]] + target_fields[field_name]
                stacked_field = stack(all_frames, batch('time'))
                
                # Add batch dimension for consistency with model output
                stacked_field = math.expand(stacked_field, batch(batch=1))
                
                data[field_name] = stacked_field
                print(f"  Loaded '{field_name}' with shape {data[field_name].shape}")
        
        # Load true PDE params from scene metadata if available
        scene_path = os.path.join(
            self.project_root,
            self.data_config['data_dir'],
            self.data_config['dset_name'],
            f"sim_{sim_index:06d}"
        )
        
        # Try alternative name format
        if not os.path.exists(scene_path):
            scene_path = os.path.join(
                self.project_root,
                self.data_config['data_dir'],
                self.data_config['dset_name'],
                f"sim_{sim_index}"
            )
        
        if os.path.exists(scene_path):
            scene = Scene.at(scene_path)
            self.true_pde_params = scene.properties.get('PDE_Params', {})
            print(f"  True PDE Parameters from metadata: {self.true_pde_params}")
        else:
            self.true_pde_params = {}
            print(f"  Warning: Could not load scene metadata from {scene_path}")
        
        return data

    def _create_model(self):
        """
        Required by FieldTrainer.
        Returns the physical model (already created in __init__).
        """
        return self.model

    def _setup_optimization(self):
        """
        Required by FieldTrainer.
        Setup optimization configuration for math.minimize.
        """
        return math.Solve(
            method='L-BFGS-B',
            abs_tol=1e-6,
            x0=self.initial_guesses,
            max_iterations=self.num_epochs,
        )

    def train(self):
        """
        Runs the full inverse problem optimization.
        """
        print(f"\n--- Starting Physical Parameter Optimization ---")
        
        # 1. --- Load Ground Truth Data ---
        sim_to_train = self.train_sims[0]
        if len(self.train_sims) > 1:
            print(f"Warning: Training on sim {sim_to_train}. "
                  f"Multi-sim training not yet implemented.")
        
        # Track data loading performance
        if hasattr(self, 'memory_monitor') and self.memory_monitor:
            with self.memory_monitor.track("load_ground_truth"):
                gt_data_dict = self._load_ground_truth_data(sim_to_train)
        else:
            gt_data_dict = self._load_ground_truth_data(sim_to_train)
        
        print(gt_data_dict)
        # Get the initial state (t=0) for all fields
        initial_state_dict = {
            name: field.time[0]
            for name, field in gt_data_dict.items()
            if name in self.model.get_initial_state() # Only fields the model knows
        }

        # Get the ground truth *rollout* (t=1 to T)
        gt_rollout_dict = {
            name: field.time[1:]
            for name, field in gt_data_dict.items()
            if name in initial_state_dict # Must be a predicted field
        }
        
        # 2. --- Define the Loss Function ---
        
        # Track loss function calls for monitoring
        loss_call_count = [0]  # Use list to allow modification in nested function
        
        def loss_function(*learnable_tensors):
            """
            Calculates L2 loss for a rollout.
            Properly handles batch dimensions.
            """
            loss_call_count[0] += 1
            iteration_num = loss_call_count[0]
            
            # 1. Update the model's parameters with the current guess
            for i, param_config in enumerate(self.learnable_params_config):
                param_name = param_config['name']
                setattr(self.model, param_name, learnable_tensors[i])
            
            # 2. Simulate forward
            total_loss = math.tensor(0.0)
            current_state = initial_state_dict
            
            for step in range(self.num_predict_steps):
                current_state = self.model.step(current_state)
                
                # 3. Calculate L2 loss for this step
                step_loss = 0.0
                for field_name, gt_rollout in gt_rollout_dict.items():
                    target = gt_rollout.time[step]  # Has batchᵇ=1
                    pred = current_state[field_name]  # Has batchᵇ=1
                    
                    # Both pred and target should now have matching batch dims
                    # Compute L2 loss - will reduce over spatial dims, keep batch
                    diff = pred - target
                    
                    field_loss = l2_loss(diff)
                    
                    # Sum over any remaining dimensions (including batch)
                    field_loss = math.sum(field_loss)
                    
                    step_loss += field_loss
                
                total_loss += step_loss
            
            final_loss = total_loss / self.num_predict_steps
            
            # Print loss for first few iterations (if monitoring enabled)
            if hasattr(self, 'memory_monitor') and self.memory_monitor:
                if iteration_num <= self.verbose_iterations:
                    print(f"  Iteration {iteration_num}: loss={final_loss}, time since start: "
                          f"{time.perf_counter() - self._optimization_start_time:.1f}s")
            
            return final_loss

        # 3. --- Run Optimization ---
        print("\nStarting optimization with math.minimize (L-BFGS-B)...")
        
        # Track optimization start time
        if hasattr(self, 'memory_monitor') and self.memory_monitor:
            self._optimization_start_time = time.perf_counter()
        
        # Disable validation during optimization to allow exploration of negative values
        
        # Initial loss
        initial_loss = loss_function(*self.initial_guesses)

        initial_guess_strs = [
            f"{cfg['name']}={self.initial_guesses[i]:.4f}" 
            for i, cfg in enumerate(self.learnable_params_config)
        ]
        print(f"Initial guess: {', '.join(initial_guess_strs)}")
        print(f"Initial loss: {initial_loss}")
        
        solve_params = math.Solve(
            method='L-BFGS-B',
            abs_tol=1e-6,
            x0=self.initial_guesses,
            max_iterations=self.num_epochs,
        )
        
        try:
            # math.minimize returns a TUPLE of optimized tensors
            if hasattr(self, 'memory_monitor') and self.memory_monitor:
                with self.memory_monitor.track("optimization"):
                    estimated_tensors = math.minimize(loss_function, solve_params)
            else:
                estimated_tensors = math.minimize(loss_function, solve_params)
            
            print(f"\nOptimization completed!")
            print(f"Total loss function evaluations: {loss_call_count[0]}")
        except Exception as e:
            print(f"Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            estimated_tensors = tuple(self.initial_guesses) # Return guess on failure
        
        # Print performance summary
        if hasattr(self, 'memory_monitor') and self.memory_monitor:
            self.memory_monitor.print_summary()
        
        final_loss = loss_function(*estimated_tensors)
        print(f"Final loss: {final_loss}")

        # 4. --- Report Results ---
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        for i, param_config in enumerate(self.learnable_params_config):
            name = param_config['name']
            true_val = self.true_pde_params.get(name, 'N/A')
            
            estimated_val = estimated_tensors[i]
            
            print(f"\nParameter: {name}")
            print(f"  True value:      {true_val}")
            print(f"  Estimated value: {estimated_val}")
            
            if isinstance(true_val, (int, float)):
                error = abs(estimated_val - true_val)
                rel_error = (error / abs(true_val)) * 100 if abs(true_val) > 1e-6 else 0
                print(f"  Absolute error:  {error}")
                print(f"  Relative error:  {rel_error:.2f}%")
        
        print("="*60)