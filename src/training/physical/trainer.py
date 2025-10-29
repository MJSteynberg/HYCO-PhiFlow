# src/training/physical/trainer.py

import time
from pathlib import Path
from typing import Dict, Any, List

from phi.torch.flow import *
from phi.field import CenteredGrid, l2_loss
import os
from phi.math import jit_compile, batch, gradient, stop_gradient
# --- Repo Imports ---
# We'll import the HeatModel, just like inverse_heat.py does.
# This can be generalized later if needed.
import src.models.physical as physical_models


class PhysicalTrainer:
    """
    Handles the optimization loop for physical parameters,
    mirroring the structure of SyntheticTrainer but using
    manual gradient descent from inverse_heat.py.
    """
    
    def __init__(self, config: Dict[str, Any], log_dir: str):
        """
        Initializes the trainer from a unified configuration dictionary.

        Args:
            config: The main configuration dictionary.
            log_dir: The path to the logging directory for this run.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- 1. Parse Configs ---
        self.data_config = config['data']
        self.model_config = config['model']['physical']
        self.trainer_config = config['trainer_params']
        
        self.data_loader_fields: List[str] = self.data_config['fields']
        self.dset_name = self.data_config['dset_name'] # Also get dset_name from data_config
        self.data_dir = self.data_config['data_dir'] # And data_dir

        self.input_specs = self.model_config['input_specs']
        self.output_specs = self.model_config['output_specs']

        model_save_name = self.model_config['model_save_name']
        model_path_dir = self.model_config['model_path']
        self.checkpoint_path = os.path.join(model_path_dir, f"{model_save_name}.pth")
        os.makedirs(model_path_dir, exist_ok=True)

        # --- Training parameters ---
        self.learning_rate = self.trainer_config['learning_rate']
        self.epochs = self.trainer_config['epochs']
        self.batch_size = self.trainer_config['batch_size']
        self.train_sim = self.trainer_config['train_sim']
        self.total_steps = config['generation_params']['total_steps']  # Total sim steps from generation config

        self.train_loader = self._create_data_loader()
        self.model = self._create_model()

        self.grad_function = self._create_gradient_function()
        
        # --- 6. Create Learnable Parameters ---
        self.learnable_params = self._create_learnable_params()
        param_names = [p.shape.name for p in self.learnable_params]
        
        # --- 8. Create Gradient Function ---
        # self.grad_function = self._create_gradient_function()

    def _create_data_loader(self):
        pass 

    def _create_model(self):
        domain = Box(x=self.model_config['domain']['size_x'], y=self.model_config['domain']['size_y'])
        resolution = spatial(x=self.model_config['resolution']['x'], y=self.model_config['resolution']['y'])

        # 2. Get PDE params
        pde_params = self.model_config.get('pde_params', {}).copy()

        # 3. Get the Model Class
        try:
            ModelClass = getattr(physical_models, self.model_config['name'])
        except AttributeError:
            raise ImportError(f"Model '{self.model_config['name']}' not found in src/models/physical/__init__.py")

        # 4. Instantiate the model
        model = ModelClass(
            domain=domain,
            resolution=resolution,
            dt=self.model_config['dt'], # <-- Use new schema
            **pde_params
        )
        return model

    def _create_learnable_params(self) -> List[Tensor]:
        """
        Creates the list of phiml Tensors that will be optimized.
        """
        initial_guess_cfg = self.trainer_config['initial_guess']
        
        # We wrap the guess in a tensor with a batch name.
        # The name 'diffusivity' is critical for the gradient function.
        diffusivity_guess = wrap(
            [initial_guess_cfg['diffusivity']],
            batch('diffusivity')
        )
        
        return [diffusivity_guess]

    
    def _calculate_loss(self, **learnable_params_kwargs) -> Tensor:
        """
        The loss function to be differentiated.
        
        It takes the current guess for learnable parameters as kwargs,
        runs a full simulation, and compares to ground truth.
        
        Args:
            **learnable_params_kwargs: e.g., diffusivity=Tensor(...)
        
        Returns:
            Total L2 loss over the simulation rollout.
        """
        
        # 1. Update the guess model with the new parameter values
        # This uses the @property.setter we defined in HeatModel
        self.model.diffusivity = learnable_params_kwargs['diffusivity']
        
        total_loss = 0.0
        
        # 2. Get the *true* initial state (t=0)
        current_state_dict = {'temp': self.ground_truth_states[0]}
        
        # 3. Run the simulation rollout
        for step in range(1, self.total_steps + 1):
            
            # A. Get prediction from the 'guess' model
            next_state_dict = self.model.step(current_state_dict)
            predicted_temp = next_state_dict['temp']
            
            # B. Get the corresponding ground truth state
            true_temp = self.ground_truth_states[step]
            
            # C. Calculate L2 loss for this step
            # We stop_gradient on the ground truth
            loss_at_step = l2_loss(predicted_temp - stop_gradient(true_temp))
            total_loss += loss_at_step
            
            # D. Update state for next loop iteration
            current_state_dict = next_state_dict
            
        return total_loss

    def _create_gradient_function(self) -> callable:
        """
        Creates the JIT-compiled gradient function.
        """
        # Get the names of the parameters to differentiate with respect to (wrt)
        param_names = [p.shape.name for p in self.learnable_params]

        # Get the gradient function
        # get_value=True returns (value, (grads...))
        grad_fn = gradient(
            self._calculate_loss,
            wrt=param_names,
            get_value=True
        )
        
        # JIT-compile for performance
        return jit_compile(grad_fn)
    
    def train(self):
        """
        Runs the full optimization loop (manual gradient descent).
        """
        start_time = time.time()
        
        # Initialize with the guess tensors from __init__
        current_params_list = self.learnable_params
        
        for step in range(self.epochs):
            
            # 1. Prepare kwargs for the gradient function
            param_kwargs = {p.shape.name: p for p in current_params_list}
            
            # 2. Calculate loss and gradients
            loss, grads_tuple = self.grad_function(**param_kwargs)
            grads_list = list(grads_tuple)
            
            # 3. Manual Gradient Descent Step
            new_params_list = []
            for param, grad in zip(current_params_list, grads_list):
                # Ensure grad is valid, replace NaN/Inf with 0
                grad = math.where(math.is_finite(grad), grad, 0.0)
                new_param = param - self.learning_rate * grad
                new_params_list.append(new_param)
            
            # Update the parameters for the next iteration
            current_params_list = new_params_list
            
            # 4. Logging
            print(f"{step:<4} | {loss} | " + " | ".join([f"{p}" for p in current_params_list]))

        end_time = time.time()
        
        final_params_str = ", ".join(
            [f"{p.shape.name}: {p}" for p in current_params_list]
        )
        print(f"Final optimized parameters: {final_params_str}")