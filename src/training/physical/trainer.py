# src/training/physical/trainer.py
import time
from pathlib import Path
from typing import Dict, Any, List
import os

# --- PhiFlow Imports ---
from phi.torch.flow import *
from phi.field import CenteredGrid, StaggeredGrid, l2_loss 
from phi.math import jit_compile, batch, gradient, stop_gradient, math, dual 
# --- 1. ADD THIS IMPORT ---
from phi.physics._boundaries import Domain, PERIODIC, ZERO
# --- Repo Imports ---
import src.models.physical as physical_models

# --- PBDL Dataloader Import ---
from pbdl.torch.loader import Dataloader


class PhysicalTrainer:
    """
    (Class docstring)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        (init docstring)
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- 1. Parse Configs ---
        self.data_config = config['data']
        self.model_config = config['model']['physical']
        self.trainer_config = config['trainer_params']
        
        self.data_loader_fields_config: List[str] = self.data_config['fields'] 
        self.dset_name = self.data_config['dset_name'] 
        self.data_dir = self.data_config['data_dir'] 

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

        self.pbdl_load_fields_list = []
        for field_name in self.data_loader_fields_config:
            field_name_lower = field_name.lower()
            if field_name_lower == 'velocity':
                self.pbdl_load_fields_list.append('velocity') # For x-component
                self.pbdl_load_fields_list.append('velocity') # For y-component
            else:
                self.pbdl_load_fields_list.append(field_name)
        
        self.train_loader = self._create_data_loader()
        self.model = self._create_model()
        
        # --- 2. IMPLEMENT YOUR SUGGESTION ---
        # Create a physics.Domain object from the model's Box and Shape.
        # We assume PERIODIC boundaries as this matches the models. !!!!!!!!!!!!!!!!!!!!!!!!!!
        BNDS = {
            'y':(ZERO, ZERO),
            'x':(ZERO, ZERO)
        }
        self.physics_domain = Domain(
            x=self.model.resolution.get_size('x'),
            y=self.model.resolution.get_size('y'),
            boundaries = BNDS,
            bounds = self.model.domain
        )
        
        
        self.learnable_params = self._create_learnable_params()
        self.grad_function = self._create_gradient_function()

        print(f"PhysicalTrainer initialized with model '{self.model_config['name']}'.")

    def _create_data_loader(self):
        """Creates the pbdl.Dataloader."""
        print(f"Setting up data loader for '{self.dset_name}'...")
        hdf5_filepath = os.path.join(self.data_dir, f"{self.dset_name}.hdf5")
        if not os.path.exists(hdf5_filepath):
            raise FileNotFoundError(f"Dataset not found at {hdf5_filepath}")

        loader = Dataloader(
            self.dset_name,
            load_fields=self.pbdl_load_fields_list, 
            time_steps=self.num_predict_steps,
            intermediate_time_steps=True,
            batch_size=self.batch_size,
            shuffle=True,
            sel_sims=self.train_sim,
            local_datasets_dir=self.data_dir
        )
        print(f"Data loader created. Loading channels: {self.pbdl_load_fields_list}")
        return loader 
    
    
    def _tensor_to_grid_dict(self, tensor_batch: torch.Tensor) -> Dict[str, Field]:
        """
        Converts a raw (B, C, Y, X) tensor from the data loader into a
        dictionary of PhiFlow Fields, correctly reconstructing
        StaggeredGrids by PADDING the sliced components.
        """
        
        num_channels_actual = tensor_batch.shape[1]
        if num_channels_actual != len(self.pbdl_load_fields_list):
             raise ValueError(
                f"Channel mismatch: Dataloader was supposed to load "
                f"{len(self.pbdl_load_fields_list)} channels, but tensor "
                f"has {num_channels_actual} channels."
            )

        dynamic_batch_size = tensor_batch.shape[0]
        batch_dim = batch(batch=dynamic_batch_size)
        channel_dim = channel(channels=num_channels_actual)
        spatial_dims_names = self.model.resolution.names[::-1] 
        spatial_dim = spatial(*spatial_dims_names) 
        full_4d_shape = batch_dim & channel_dim & spatial_dim
        
        phiml_tensor = wrap(tensor_batch, full_4d_shape)
        
        grid_dict = {}
        channel_idx = 0
        for field_name in self.data_loader_fields_config:
            field_name_lower = field_name.lower()
            
            if field_name_lower == 'velocity':
                # Note: Order matches generator (x-comp first, then y-comp)
                vx = phiml_tensor.channels[channel_idx]
                vy = phiml_tensor.channels[channel_idx + 1]

                grid = self.physics_domain.staggered_grid( math.stack( [
                            math.tensor(vy, math.batch('batch'), math.spatial('x,y')),
                            math.tensor(vx, math.batch('batch'), math.spatial('x,y')),
                        ], math.dual(vector="x,y")
                    ) )
                
                grid_dict[field_name] = grid
                channel_idx += 2 

            else:
                # --- Standard CenteredGrid path (temp, density) ---
                field_values = phiml_tensor.channels[channel_idx]
                
                # --- 3. THE FIX (for consistency) ---
                grid = self.physics_domain.scalar_grid(
                    field_values
                )
                # --- END FIX ---
                
                grid_dict[field_name] = grid
                channel_idx += 1
            
        return grid_dict

    def _create_model(self):
        """(This function is general)"""
        domain = Box(x=self.model_config['domain']['size_x'], y=self.model_config['domain']['size_y'])
        resolution = spatial(x=self.model_config['resolution']['x'], y=self.model_config['resolution']['y'])
        pde_params = self.model_config.get('pde_params', {}).copy()
        try:
            ModelClass = getattr(physical_models, self.model_config['name'])
        except AttributeError:
            raise ImportError(f"Model '{self.model_config['name']}' not found in src/models/physical/__init__.py")
        model = ModelClass(
            domain=domain,
            resolution=resolution,
            dt=self.model_config['dt'],
            **pde_params
        )
        return model

    def _create_learnable_params(self) -> List[Tensor]:
        """(This function is general)"""
        learnable_params_cfg = self.trainer_config.get('learnable_parameters')
        if not learnable_params_cfg:
            raise ValueError("Config missing `trainer_params.learnable_parameters` list.")
        learnable_tensors = []
        for param_info in learnable_params_cfg:
            name = param_info['name']
            guess = param_info['initial_guess']
            param_tensor = wrap([guess], batch(name))
            learnable_tensors.append(param_tensor)
        print(f"Created learnable parameters: {[p.shape.name for p in learnable_tensors]}")
        return learnable_tensors

    
    def _calculate_loss(self, 
                        initial_state_tensor: torch.Tensor, 
                        true_rollout_tensor: torch.Tensor, 
                        **learnable_params_kwargs) -> Tensor:
        """(This function is general)"""
        
        for param_name, param_value in learnable_params_kwargs.items():
            if not hasattr(self.model, param_name):
                raise AttributeError(f"Model {type(self.model)} does not have attribute '{param_name}'")
            setattr(self.model, param_name, param_value)
        
        total_loss = 0.0
        
        current_state_dict = self._tensor_to_grid_dict(initial_state_tensor)
        
        for step in range(self.num_predict_steps):
            
            next_state_dict = self.model.step(current_state_dict)
            
            true_tensor_this_step = true_rollout_tensor[:, step, ...] # (B, C, Y, X)
            
            true_grid_dict = self._tensor_to_grid_dict(true_tensor_this_step)
            
            loss_at_step = 0.0
            for field_name, true_grid in true_grid_dict.items():
                if field_name not in next_state_dict:
                    continue 
                
                predicted_grid = next_state_dict[field_name]
                
                loss_at_step += l2_loss(predicted_grid - stop_gradient(true_grid))
            
            total_loss += loss_at_step
            
            current_state_dict = next_state_dict

        batched_loss = total_loss / self.num_predict_steps
        scalar_loss = math.mean(batched_loss, dim='batch')
        return scalar_loss

    def _create_gradient_function(self) -> callable:
        """(This function is general)"""
        return gradient(
            self._calculate_loss, 
            wrt=[p.shape.name for p in self.learnable_params]
        )
    
    def train(self):
        """(This function is general)"""
        print(f"\n--- Starting Physical Parameter Optimization ---")
        print(f"Epoch | Avg. Loss  | Current Guess(es)")
        print(f"-------------------------------------------")
        
        start_time = time.time()
        
        current_params_list = self.learnable_params
        
        for epoch in range(1, self.epochs + 1):
            total_epoch_loss = 0.0
            
            for x_batch, y_batch in self.train_loader:
                
                x_batch = x_batch.to(self.device) 
                y_batch = y_batch.to(self.device)
                
                param_kwargs = {p.shape.name: p for p in current_params_list}
                
                loss, grads_tuple = self.grad_function(
                    initial_state_tensor=x_batch,
                    true_rollout_tensor=y_batch,
                    **param_kwargs
                )
                grads_list = list(grads_tuple)
                
                new_params_list = []
                for param, grad in zip(current_params_list, grads_list):
                    grad = math.where(math.is_finite(grad), grad, 0.0) 
                    new_param = param - self.learning_rate * grad
                    new_params_list.append(new_param)
                
                current_params_list = new_params_list
                total_epoch_loss += loss
            
            avg_epoch_loss = total_epoch_loss / len(self.train_loader)
            
            if epoch % 5 == 0 or epoch == 1 or epoch == self.epochs:
                 param_strs = [f"{p.shape.name}={p.native()[0]:.4f}" for p in current_params_list]
                 print(f"{epoch:<5} | {avg_epoch_loss:<10.6f} | " + ", ".join(param_strs))

        end_time = time.time()
        print(f"-------------------------------------------")
        print(f"Optimization complete in {end_time - start_time:.2f} seconds.")
        
        final_params_str = ", ".join(
            [f"{p.shape.name}: {p.native()[0]:.6f}" for p in current_params_list]
        )
        print(f"Final optimized parameters: {final_params_str}")