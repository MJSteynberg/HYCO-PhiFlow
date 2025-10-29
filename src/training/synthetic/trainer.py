# src/training/synthetic/trainer.py

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pbdl.torch.loader import Dataloader
from typing import Dict, Any, List

# Import our model and loss
from src.models.synthetic.unet import UNet
from src.training.synthetic.losses import DictL2Loss

class SyntheticTrainer:
    """
    An input-agnostic trainer class for synthetic models.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the trainer from a unified configuration dictionary.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Derive all parameters from config ---
        self.data_config = config['data']
        self.model_config = config['model']['synthetic']
        self.trainer_config = config['trainer_params']
        
        # --- Data and Model specifications ---
        # This list *must* match the channel order in the HDF5 file.
        # It is the single source of truth for data loading.
        self.data_loader_fields: List[str] = self.data_config['fields']
        self.dset_name = self.data_config['dset_name'] # Also get dset_name from data_config
        self.data_dir = self.data_config['data_dir'] # And data_dir

        self.input_specs = self.model_config['input_specs']
        self.output_specs = self.model_config['output_specs']

        # --- Define field types from specs ---
        # Dynamic fields are predicted by the model (in output_specs)
        self.dynamic_fields: List[str] = list(self.output_specs.keys())
        
        # Static fields are inputs, but not predicted (in input_specs but not output_specs)
        self.static_fields: List[str] = [
            f for f in self.input_specs.keys() if f not in self.output_specs
        ]
        
        # Build a spec dict {field_name: channel_count} for *all* loaded fields
        all_specs = {**self.input_specs, **self.output_specs}
        self.data_loader_channels: Dict[str, int] = {
            f: all_specs[f] for f in self.data_loader_fields
        }

        # --- Paths ---
        model_save_name = self.model_config['model_save_name']
        model_path_dir = self.model_config['model_path']
        self.checkpoint_path = os.path.join(model_path_dir, f"{model_save_name}.pth")
        os.makedirs(model_path_dir, exist_ok=True)

        # --- Training parameters ---
        self.learning_rate = self.trainer_config['learning_rate']
        self.epochs = self.trainer_config['epochs']
        self.batch_size = self.trainer_config['batch_size']
        self.num_predict_steps = self.trainer_config['num_predict_steps']
        self.train_sim = self.trainer_config['train_sim']

        # --- Setup Components ---
        self.train_loader = self._create_data_loader()
        self.model = self._create_model()
        self.loss_fn = DictL2Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs * len(self.train_loader)
        )

    def _create_data_loader(self):
        """Creates the pbdl.Dataloader."""
        print(f"Setting up data loader for '{self.dset_name}'...")
        hdf5_filepath = os.path.join(self.data_dir, f"{self.dset_name}.hdf5")
        if not os.path.exists(hdf5_filepath):
            print(f"Error: Dataset not found at {hdf5_filepath}")
            raise FileNotFoundError(f"Dataset not found at {hdf5_filepath}")

        loader = Dataloader(
            self.dset_name,
            load_fields=self.data_loader_fields, # --- NEW: Use generic list ---
            time_steps=self.num_predict_steps,
            intermediate_time_steps=True,
            batch_size=self.batch_size,
            shuffle=True,
            sel_sims=self.train_sim,
            local_datasets_dir=self.data_dir
        )
        return loader

    def _create_model(self):
        """Creates the UNet model from the config."""
        print("Creating U-Net model...")
        # This is already agnostic, just passes the config
        model = UNet(config=self.model_config) 

        try:
            model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
            print(f"Loaded model weights from {self.checkpoint_path}")
        except FileNotFoundError:
            print("No pre-existing model weights found. Training from scratch.")
        
        model = model.to(self.device)
        print("Model created successfully and moved to device.")
        return model

    def _unpack_tensor_to_dict(self, tensor_batch: torch.Tensor, 
                               fields_to_unpack: List[str]) -> Dict[str, torch.Tensor]:
        """
        Helper function to slice a concatenated tensor into a field dictionary.
        
        Args:
            tensor_batch: The (B, C, Y, X) tensor from the data loader.
            fields_to_unpack: A list of field names to extract.

        Returns:
            A dictionary {field_name: tensor}
        """
        output_dict = {}
        start_channel = 0
        
        # Iterate in the *exact order* of the loaded data
        for field_name in self.data_loader_fields:
            num_channels = self.data_loader_channels[field_name]
            end_channel = start_channel + num_channels
            
            # If this field is one we want, slice and add it
            if field_name in fields_to_unpack:
                output_dict[field_name] = tensor_batch[:, start_channel:end_channel, ...]
                
            start_channel = end_channel
            
        return output_dict

    def _train_epoch(self):
        """
        Runs one epoch of autoregressive training in an input-agnostic way.
        """
        self.model.train()
        total_loss = 0.0

        for x_batch, y_batch in self.train_loader:
            x_batch = x_batch.to(self.device) # (B, C, Y, X)
            y_batch = y_batch.to(self.device) # (B, T, C, Y, X)
            
            # --- NEW: Generic Unpacking ---
            
            # Unpack all fields from the initial state (t=0)
            initial_state_all_fields = self._unpack_tensor_to_dict(x_batch, self.data_loader_fields)
            
            # Get static fields (e.g., 'inflow'). Constant for the rollout.
            static_field_dict = {f: initial_state_all_fields[f] for f in self.static_fields}
            
            # Get initial dynamic state (e.g., 'density', 'velocity')
            current_state_dict = {f: initial_state_all_fields[f] for f in self.dynamic_fields}
            
            batch_rollout_loss = 0.0
            self.optimizer.zero_grad()

            for t_step in range(self.num_predict_steps):
                
                # 1. Prepare model input dict
                model_input_dict = {**current_state_dict, **static_field_dict}
                
                # 2. Forward pass: predicts only dynamic fields
                pred_dict = self.model(model_input_dict)
                
                # 3. Get ground truth *dynamic* fields for this step
                gt_tensor_this_step = y_batch[:, t_step, ...] # (B, C, Y, X)
                gt_dict = self._unpack_tensor_to_dict(gt_tensor_this_step, self.dynamic_fields)
                
                # 4. Loss is computed on dynamic fields
                step_loss = self.loss_fn(pred_dict, gt_dict)
                batch_rollout_loss = batch_rollout_loss + step_loss
                
                # 5. Prediction becomes the next input
                current_state_dict = pred_dict
            
            # --- End of Generic Loop ---
            
            avg_rollout_loss = batch_rollout_loss / self.num_predict_steps
            avg_rollout_loss.backward()
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += avg_rollout_loss.item()
            
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def train(self):
        """Runs the full training loop."""
        print(f"\nStarting autoregressive training for {self.epochs} epochs...")
        best_loss = float('inf')

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            
            train_loss = self._train_epoch()
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch}/{self.epochs} | Train Loss: {train_loss:.6f} | Time: {epoch_time:.2f}s")

            if train_loss < best_loss and epoch % 10 == 0:
                best_loss = train_loss
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"  -> New best model saved to {self.checkpoint_path}")

        torch.save(self.model.state_dict(), self.checkpoint_path)
        print("Autoregressive training complete.")