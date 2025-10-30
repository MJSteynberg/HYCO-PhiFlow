# src/training/synthetic/trainer.py

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, List

# Import our data pipeline
from src.data import DataManager, HybridDataset

# Import our model and loss
from src.models.synthetic.unet import UNet
from src.training.synthetic.losses import DictL2Loss


class SyntheticTrainer:
    """
    Tensor-based trainer for synthetic models using DataManager pipeline.
    
    Uses HybridDataset for efficient cached data loading with no runtime
    Field conversions. All conversions happen once during caching.
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
        
        # --- Data specifications ---
        self.field_names: List[str] = self.data_config['fields']
        self.dset_name = self.data_config['dset_name']
        self.data_dir = self.data_config['data_dir']
        
        self.input_specs = self.model_config['input_specs']
        self.output_specs = self.model_config['output_specs']

        # --- Define field types from specs ---
        self.dynamic_fields: List[str] = list(self.output_specs.keys())
        self.static_fields: List[str] = [
            f for f in self.input_specs.keys() if f not in self.output_specs
        ]
        
        # Calculate channel indices for unpacking
        self._build_channel_map()

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
        
        # Calculate total frames needed
        self.num_frames = self.num_predict_steps + 1  # Initial state + rollout

        # --- Setup Components ---
        self.train_loader = self._create_data_loader()
        self.model = self._create_model()
        self.loss_fn = nn.MSELoss()  # Simple MSE for tensor-based training
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs * len(self.train_loader)
        )

    def _build_channel_map(self):
        """Build channel indices for slicing concatenated tensors."""
        self.channel_map = {}
        channel_offset = 0
        
        for field_name in self.field_names:
            # Get channel count from input or output specs
            if field_name in self.input_specs:
                num_channels = self.input_specs[field_name]
            elif field_name in self.output_specs:
                num_channels = self.output_specs[field_name]
            else:
                raise ValueError(f"Field '{field_name}' not found in specs")
            
            self.channel_map[field_name] = (channel_offset, channel_offset + num_channels)
            channel_offset += num_channels
        
        self.total_channels = channel_offset

    def _create_data_loader(self):
        """Creates DataManager and HybridDataset with PyTorch DataLoader."""
        print(f"Setting up DataManager for '{self.dset_name}'...")
        
        # Paths
        project_root = Path(self.config.get('project_root', '.'))
        raw_data_dir = project_root / self.data_dir / self.dset_name
        cache_dir = project_root / self.data_dir / 'cache'
        
        # Create DataManager
        data_manager = DataManager(
            raw_data_dir=str(raw_data_dir),
            cache_dir=str(cache_dir),
            config={'dset_name': self.dset_name}
        )
        
        # Create HybridDataset
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=self.train_sim,
            field_names=self.field_names,
            num_frames=self.num_frames,
            num_predict_steps=self.num_predict_steps,
            dynamic_fields=self.dynamic_fields,
            static_fields=self.static_fields
        )
        
        # Create PyTorch DataLoader
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Keep simple for now
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"DataLoader created: {len(dataset)} samples, batch_size={self.batch_size}")
        return loader

    def _create_model(self):
        """Creates the UNet model from the config."""
        print("Creating U-Net model...")
        model = UNet(config=self.model_config)

        try:
            model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
            print(f"Loaded model weights from {self.checkpoint_path}")
        except FileNotFoundError:
            print("No pre-existing model weights found. Training from scratch.")
        
        model = model.to(self.device)
        print("Model created successfully and moved to device.")
        return model

    def _unpack_tensor_to_dict(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Slice a concatenated tensor into individual field tensors.
        
        Args:
            tensor: Tensor with all fields concatenated on channel dimension
                   Shape: [B, C, H, W] or [B, T, C, H, W]
            
        Returns:
            Dictionary mapping field names to their tensor slices
        """
        output_dict = {}
        for field_name, (start_ch, end_ch) in self.channel_map.items():
            if tensor.dim() == 4:  # [B, C, H, W]
                output_dict[field_name] = tensor[:, start_ch:end_ch, :, :]
            elif tensor.dim() == 5:  # [B, T, C, H, W]
                output_dict[field_name] = tensor[:, :, start_ch:end_ch, :, :]
        return output_dict

    def _train_epoch(self):
        """
        Runs one epoch of autoregressive training.
        """
        self.model.train()
        total_loss = 0.0

        for initial_state, rollout_targets in self.train_loader:
            initial_state = initial_state.to(self.device)  # [B, C_all, H, W] - all fields
            rollout_targets = rollout_targets.to(self.device)  # [B, T, C_dynamic, H, W] - only dynamic
            
            # Start with initial state containing all fields
            current_state = initial_state  # [B, C_all, H, W]
            
            batch_rollout_loss = 0.0
            self.optimizer.zero_grad()

            for t_step in range(self.num_predict_steps):
                # Forward pass - model handles static field preservation internally
                prediction = self.model(current_state)  # [B, C_all, H, W] - returns all fields
                
                # Extract only dynamic fields from prediction for loss computation
                pred_dynamic_tensors = []
                channel_offset = 0
                for field_name in self.field_names:
                    if field_name in self.dynamic_fields:
                        start, end = self.channel_map[field_name]
                        pred_dynamic_tensors.append(prediction[:, start:end, :, :])
                
                pred_dynamic = torch.cat(pred_dynamic_tensors, dim=1)  # [B, C_dynamic, H, W]
                
                # Get ground truth for this timestep (already only dynamic fields)
                gt_this_step = rollout_targets[:, t_step, :, :, :]  # [B, C_dynamic, H, W]
                
                # Compute loss on dynamic fields only
                step_loss = self.loss_fn(pred_dynamic, gt_this_step)
                batch_rollout_loss += step_loss
                
                # Update current state with full prediction (includes static + dynamic)
                current_state = prediction
            
            # Average loss over rollout steps
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