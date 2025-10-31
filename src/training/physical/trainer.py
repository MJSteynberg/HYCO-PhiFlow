# In src/training/physical/trainer.py

import os
import time
from typing import Dict, Any, List

# --- PhiFlow Imports ---
from phi.torch.flow import *
from phi.field import l2_loss
from phi.math import math, Tensor

# --- Repo Imports ---
import src.models.physical as physical_models
from src.data import DataManager, HybridDataset


class PhysicalTrainer:
    """
    Solves an inverse problem for a PhysicalModel using cached data
    from DataManager/HybridDataset.

    This trainer uses math.minimize for optimization and leverages
    the efficient DataLoader pipeline with field conversion.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the trainer from a unified configuration dictionary.

        Args:
            config (Dict[str, Any]): The experiment configuration.
        """
        self.config = config
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
        self.model = self._create_model()
        
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
            config={'dset_name': self.data_config['dset_name']}
        )


    def _create_model(self) -> physical_models.PhysicalModel:
        """
        Instantiates the physical model from the config, setting
        learnable parameters to their initial guesses.
        """
        model_name = self.model_config['name']
        domain_cfg = self.model_config['domain']
        res_cfg = self.model_config['resolution']
        
        domain = Box(x=domain_cfg['size_x'], y=domain_cfg['size_y'])
        resolution = spatial(x=res_cfg['x'], y=res_cfg['y'])
        
        # Start with a copy of PDE params from config
        pde_params = self.model_config.get('pde_params', {}).copy()
        
        # --- IMPORTANT ---
        # Overwrite/set the *initial guess* for the learnable parameters
        # on the model instance. The optimizer will update these.
        for param in self.learnable_params_config:
            pde_params[param['name']] = param['initial_guess']

        try:
            ModelClass = getattr(physical_models, model_name)
        except AttributeError:
            raise ImportError(f"Model '{model_name}' not found in src/models/physical/__init__.py")
        print(self.model_config['dt'])
        model = ModelClass(
            domain=domain,
            resolution=resolution,
            dt=self.model_config['dt'],
            **pde_params
        )
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
    
    # Add this 'train' method to the PhysicalTrainerScene class:

    # In PhysicalTrainerScene class

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
        # print(f"Loaded initial state {initial_state_dict}")
        # print(f"Loaded ground truth rollout {gt_rollout_dict}")
        # print(f"Loaded ground truth rollout {mean(gt_rollout_dict['temp'])}")
        # print('upper = ', self.model.domain.upper, 'lower = ',self.model.domain.lower)
        # 2. --- Define the Loss Function ---
        
        # --- FIX 2: Re-enable JIT ---
        # Pass dicts as non-differentiable auxiliary variables
        # @jit_compile(auxiliary_args="initial_state_dict, gt_rollout_dict")
        def loss_function(*learnable_tensors):
            """
            Calculates L2 loss for a rollout.
            Now properly handles batch dimensions.
            """
            # 1. Update the model's parameters with the current guess
            for i, param_config in enumerate(self.learnable_params_config):
                param_name = param_config['name']
                setattr(self.model, param_name, learnable_tensors[i])
            
            # Debug: check if diffusivity is nan
            if hasattr(self.model, 'nu'):
                diff_val = self.model.nu
                if hasattr(diff_val, 'native'):
                    diff_native = float(diff_val.native())
                    print(f"Current nu value: {diff_native}")
            
            # 2. Simulate forward
            total_loss = math.tensor(0.0)
            current_state = initial_state_dict
            
            for step in range(self.num_predict_steps):
                print(f"After step {step}, current_state: {current_state['velocity'].batch[0].x[0].y[0].values}")
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
            return 1e-3*final_loss

        # 3. --- Run Optimization ---
        print("\nStarting optimization with math.minimize (L-BFGS-B)...")
        
        # Initial loss
        initial_loss = loss_function(*self.initial_guesses)

        # --- FIX 3: Correct print logic ---
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
            estimated_tensors = math.minimize(loss_function, solve_params)
            print(f"\nOptimization completed!")
        except Exception as e:
            print(f"Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            estimated_tensors = tuple(self.initial_guesses) # Return guess on failure
        
        final_loss = loss_function(*estimated_tensors)
        print(f"Final loss: {final_loss}")

        # 4. --- Report Results ---
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        for i, param_config in enumerate(self.learnable_params_config):
            name = param_config['name']
            true_val = self.true_pde_params.get(name, 'N/A')
            
            # --- FIX 3: Extract Python value ---
            estimated_val = estimated_tensors[i]
            
            print(f"\nParameter: {name}")
            print(f"  True value:      {true_val}")
            print(f"  Estimated value: {estimated_val}")
            
            if isinstance(true_val, (int, float)):
                # Now this math is between floats
                error = abs(estimated_val - true_val)
                rel_error = (error / abs(true_val)) * 100 if abs(true_val) > 1e-6 else 0
                print(f"  Absolute error:  {error}")
                print(f"  Relative error:  {rel_error:.2f}%")
        
        print("="*60)