# src/training/physical/trainer.py

# src/training/physical/trainer.py
import time
from pathlib import Path
from typing import Dict, Any, List
import os

# --- PhiFlow Imports ---
from phi.torch.flow import *
from phi.field import CenteredGrid, l2_loss
from phi.math import jit_compile, batch, gradient, stop_gradient, math

# --- Repo Imports ---
import src.models.physical as physical_models

# --- PBDL Dataloader Import ---
from pbdl.torch.loader import Dataloader


class PhysicalTrainer:
    """
    Handles the optimization loop for physical parameters,
    mirroring the structure of SyntheticTrainer but using
    manual gradient descent from inverse_heat.py.
    """
    
    def __init__(self, config: Dict[str, Any]):
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

        model_save_name = self.model_config['model_save_name']
        model_path_dir = self.model_config['model_path']
        self.checkpoint_path = os.path.join(model_path_dir, f"{model_save_name}.pth")
        os.makedirs(model_path_dir, exist_ok=True)

        # --- Training parameters ---
        self.learning_rate = self.trainer_config['learning_rate']
        self.epochs = self.trainer_config['epochs']
        self.batch_size = self.trainer_config['batch_size']
        self.train_sim = self.trainer_config['train_sim']
        self.num_predict_steps = self.trainer_config['num_predict_steps']

        self.train_loader = self._create_data_loader()
        self.model = self._create_model()
        self.learnable_params = self._create_learnable_params()
        self.grad_function = self._create_gradient_function()

        print(f"PhysicalTrainer initialized with model '{self.model_config['name']}'.")

    def _create_data_loader(self):
        """Creates the pbdl.Dataloader."""
        print(f"Setting up data loader for '{self.dset_name}'...")
        hdf5_filepath = os.path.join(self.data_dir, f"{self.dset_name}.hdf5")
        if not os.path.exists(hdf5_filepath):
            print(f"Error: Dataset not found at {hdf5_filepath}")
            raise FileNotFoundError(f"Dataset not found at {hdf5_filepath}")

        # This is very similar to SyntheticTrainer
        loader = Dataloader(
            self.dset_name,
            load_fields=self.data_loader_fields, # e.g., ['temp']
            time_steps=self.num_predict_steps,   # Use short rollout steps
            intermediate_time_steps=True,
            batch_size=self.batch_size,
            shuffle=True,
            sel_sims=self.train_sim,
            local_datasets_dir=self.data_dir
        )
        print("Data loader created.")
        return loader 
    
    def _tensor_to_grid_dict(self, tensor_batch: torch.Tensor) -> Dict[str, CenteredGrid]:
        """
        Converts a raw (B, C, Y, X) tensor from the data loader into a
        dictionary of PhiFlow CenteredGrids.
        
        Note: This implementation assumes a single field (e.g., 'temp').
              It can be generalized if you have more fields.
        """
        if 'temp' not in self.data_loader_fields:
             raise ValueError("This trainer is hard-coded for 'temp' field.")
             
        # The data tensor from the loader has shape (B, C, Y, X)
        # where B is *dynamic* (e.g., 16, or 12 for the last batch)

        # 1. Define the 4D shape that *matches* the data tensor's order
        
        # --- FIX: Get batch size *dynamically* from the tensor ---
        dynamic_batch_size = tensor_batch.shape[0]
        batch_dim = batch(batch=dynamic_batch_size) # <-- Was batch(batch=self.batch_size)
        
        # Channel dimension
        channel_dim = channel(vector=1)

        # Spatial dimensions in (Y, X) order to match data
        spatial_dims_names = self.model.resolution.names[::-1] 
        # Leave sizes as None to be inferred by wrap()
        spatial_dim = spatial(*spatial_dims_names)

        # Combine all 4 dimensions in the correct order: (B, C, Y, X)
        full_4d_shape = batch_dim & channel_dim & spatial_dim
        
        # 2. Wrap the tensor with the full 4D shape
        phiml_tensor = wrap(tensor_batch, full_4d_shape)
        
        # 3. Create the CenteredGrid
        grid = CenteredGrid(
            values=phiml_tensor,
            extrapolation=extrapolation.PERIODIC,
            bounds=self.model.domain,
            resolution=self.model.resolution 
        )
        
        return {'temp': grid}

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

    
    def _calculate_loss(self, 
                        initial_state_tensor: torch.Tensor, 
                        true_rollout_tensor: torch.Tensor, 
                        **learnable_params_kwargs) -> Tensor:
        """
        The loss function to be differentiated.
        
        It takes the current guess for learnable parameters,
        runs a short simulation rollout, and compares to ground truth
        data from the dataloader.
        
        Args:
            initial_state_tensor: (B, C, Y, X) tensor for t=0.
            true_rollout_tensor: (B, T, C, Y, X) tensor for t=1 to t=T.
            **learnable_params_kwargs: e.g., diffusivity=Tensor(...)
        
        Returns:
            Average L2 loss over the short rollout.
        """
        
        # 1. Update the guess model with the new parameter values
        self.model.diffusivity = learnable_params_kwargs['diffusivity']
        
        total_loss = 0.0
        
        # 2. Convert the initial (t=0) tensor to a PhiFlow grid dict
        current_state_dict = self._tensor_to_grid_dict(initial_state_tensor)
        
        # 3. Run the simulation rollout for num_predict_steps
        for step in range(self.num_predict_steps):
            
            # A. Get prediction from the 'guess' model
            next_state_dict = self.model.step(current_state_dict)
            predicted_temp = next_state_dict['temp']
            
            # B. Get the corresponding ground truth tensor for this step
            true_tensor_this_step = true_rollout_tensor[:, step, ...] # (B, C, Y, X)
            
            # C. Convert ground truth tensor to a PhiFlow grid dict
            true_grid_dict = self._tensor_to_grid_dict(true_tensor_this_step)
            
            # D. Calculate L2 loss for this step
            loss_at_step = l2_loss(predicted_temp - stop_gradient(true_grid_dict['temp']))
            total_loss += loss_at_step
            
            # E. Update state for next loop iteration
            current_state_dict = next_state_dict

        batched_loss = total_loss / self.num_predict_steps
        scalar_loss = math.mean(batched_loss, dim='batch')
        # Return the *average* loss over the rollout
        return scalar_loss

    def _create_gradient_function(self) -> callable:
        """
        Creates the JIT-compiled gradient function.
        """
        return gradient(
            self._calculate_loss, wrt=[name.shape.name for name in self.learnable_params])
    
    def train(self):
        """
        Runs the full optimization loop (manual gradient descent)
        using batch-based, short-rollout training.
        """
        print(f"\n--- Starting Physical Parameter Optimization ---")
        print(f"Epoch | Avg. Loss  | Current Guess")
        print(f"-------------------------------------------")
        
        start_time = time.time()
        
        current_params_list = self.learnable_params
        
        for epoch in range(1, self.epochs + 1):
            total_epoch_loss = 0.0
            
            # --- MODIFIED: Loop over the data loader ---
            for x_batch, y_batch in self.train_loader:
                
                # Move data to device (Phiml-PyTorch usually handles this,
                # but explicit is fine)
                x_batch = x_batch.to(self.device) # (B, C, Y, X)
                y_batch = y_batch.to(self.device) # (B, T, C, Y, X)
                
                # 1. Prepare kwargs for the gradient function
                param_kwargs = {p.shape.name: p for p in current_params_list}
                
                # 2. Calculate loss and gradients for this batch
                loss, grads_tuple = self.grad_function(
                    initial_state_tensor=x_batch,
                    true_rollout_tensor=y_batch,
                    **param_kwargs
                )
                grads_list = list(grads_tuple)
                
                # 3. Manual Gradient Descent Step
                new_params_list = []
                for param, grad in zip(current_params_list, grads_list):
                    grad = math.where(math.is_finite(grad), grad, 0.0) # Handle NaN/Inf grads
                    new_param = param - self.learning_rate * grad
                    new_params_list.append(new_param)
                
                current_params_list = new_params_list
                
                # Add batch loss to epoch loss
                # .native() gets the raw scalar from the phiml Tensor
                total_epoch_loss += loss
            
            # --- End of data loader loop ---
            
            avg_epoch_loss = total_epoch_loss / len(self.train_loader)
            
            # 4. Logging
            if epoch % 5 == 0 or epoch == 1 or epoch == self.epochs:
                 print(f"{epoch:<5} | {avg_epoch_loss:<10.6f} | " + " | ".join([f"{p}" for p in current_params_list]))

        end_time = time.time()
        print(f"-------------------------------------------")
        print(f"Optimization complete in {end_time - start_time:.2f} seconds.")
        
        final_params_str = ", ".join(
            [f"{p.shape.name}: {p}" for p in current_params_list]
        )
        print(f"Final optimized parameters: {final_params_str}")