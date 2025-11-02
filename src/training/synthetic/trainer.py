# src/training/synthetic/trainer.py

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm

# Import our data pipeline
from src.data import DataManager, HybridDataset

# Import tensor trainer (new hierarchy)
from src.training.tensor_trainer import TensorTrainer

# Import the model registry
from src.models import ModelRegistry

# Import logging
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SyntheticTrainer(TensorTrainer):
    """
    Tensor-based trainer for synthetic models using DataManager pipeline.

    Uses HybridDataset for efficient cached data loading with no runtime
    Field conversions. All conversions happen once during caching.

    Inherits from TensorTrainer to get PyTorch-specific functionality.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the trainer from a unified configuration dictionary.
        """
        # Initialize base trainer
        super().__init__(config)

        # --- Derive all parameters from config ---
        self.data_config = config["data"]
        self.model_config = config["model"]["synthetic"]
        self.trainer_config = config["trainer_params"]

        # --- Data specifications ---
        self.field_names: List[str] = self.data_config["fields"]
        self.dset_name = self.data_config["dset_name"]
        self.data_dir = self.data_config["data_dir"]

        self.input_specs = self.model_config["input_specs"]
        self.output_specs = self.model_config["output_specs"]

        # --- Define field types from specs ---
        self.dynamic_fields: List[str] = list(self.output_specs.keys())
        self.static_fields: List[str] = [
            f for f in self.input_specs.keys() if f not in self.output_specs
        ]

        # Calculate channel indices for unpacking
        self._build_channel_map()

        # --- Paths ---
        model_save_name = self.model_config["model_save_name"]
        model_path_dir = self.model_config["model_path"]
        self.checkpoint_path = os.path.join(model_path_dir, f"{model_save_name}.pth")
        os.makedirs(model_path_dir, exist_ok=True)

        # --- Training parameters ---
        self.learning_rate = self.trainer_config["learning_rate"]
        self.epochs = self.trainer_config["epochs"]
        self.batch_size = self.trainer_config["batch_size"]
        self.num_predict_steps = self.trainer_config["num_predict_steps"]
        self.train_sim = self.trainer_config["train_sim"]
        self.val_sim = self.trainer_config.get("val_sim", [])
        self.use_sliding_window = self.trainer_config.get("use_sliding_window", False)

        # Calculate total frames needed
        if self.use_sliding_window:
            # Load all available frames for maximum data augmentation
            self.num_frames = None  # None means load all available frames
        else:
            # Load only what's needed for one sample
            self.num_frames = self.num_predict_steps + 1  # Initial state + rollout

        # --- Setup Components ---
        self._create_data_loaders()  # Creates both train_loader and val_loader
        self.model = self._create_model()
        self.loss_fn = nn.MSELoss()  # Simple MSE for tensor-based training
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs * len(self.train_loader)
        )

        # --- Memory monitoring (optional, enabled by config) ---
        enable_memory_monitoring = self.trainer_config.get(
            "enable_memory_monitoring", False
        )
        if enable_memory_monitoring:
            try:
                from src.utils.memory_monitor import EpochPerformanceMonitor

                verbose_batches = self.trainer_config["memory_monitor_batches"]
                self.memory_monitor = EpochPerformanceMonitor(
                    enabled=True,
                    verbose_batches=verbose_batches,
                    device=0 if torch.cuda.is_available() else -1,
                )
                logger.info(
                    f"Performance monitoring enabled (verbose for first {verbose_batches} batches)"
                )
            except ImportError:
                logger.warning(
                    "Could not import EpochPerformanceMonitor. Monitoring disabled."
                )
                self.memory_monitor = None
        else:
            self.memory_monitor = None

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

            self.channel_map[field_name] = (
                channel_offset,
                channel_offset + num_channels,
            )
            channel_offset += num_channels

        self.total_channels = channel_offset

    def _create_data_loaders(self):
        """Creates DataManager and train/validation DataLoaders."""
        logger.debug(f"Setting up DataManager for '{self.dset_name}'")

        # Paths
        project_root = Path(self.config.get("project_root", "."))
        raw_data_dir = project_root / self.data_dir / self.dset_name
        cache_dir = project_root / self.data_dir / "cache"

        # Create DataManager with validation settings
        data_manager = DataManager(
            raw_data_dir=str(raw_data_dir),
            cache_dir=str(cache_dir),
            config=self.config,  # Pass full config for validation
            validate_cache=self.data_config.get("validate_cache", True),
            auto_clear_invalid=self.data_config.get("auto_clear_invalid", False),
        )

        # Create Training Dataset
        train_dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=self.train_sim,
            field_names=self.field_names,
            num_frames=self.num_frames,
            num_predict_steps=self.num_predict_steps,
            dynamic_fields=self.dynamic_fields,
            static_fields=self.static_fields,
            use_sliding_window=self.use_sliding_window,
        )

        # Create Training DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle training data
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        mode_desc = (
            "sliding window" if self.use_sliding_window else "single starting point"
        )
        logger.debug(
            f"Train DataLoader: {len(train_dataset)} samples ({mode_desc}), batch_size={self.batch_size}"
        )

        # Create Validation Dataset (if val_sim specified)
        if self.val_sim:
            # Use validation_rollout_steps if specified, otherwise use num_predict_steps
            val_predict_steps = self.trainer_config.get("validation_rollout_steps", self.num_predict_steps)
            if val_predict_steps is None:
                val_predict_steps = self.num_predict_steps
            
            # Calculate frames needed for validation (no sliding window)
            val_num_frames = val_predict_steps + 1  # Initial state + rollout targets
            
            val_dataset = HybridDataset(
                data_manager=data_manager,
                sim_indices=self.val_sim,
                field_names=self.field_names,
                num_frames=val_num_frames,
                num_predict_steps=val_predict_steps,
                dynamic_fields=self.dynamic_fields,
                static_fields=self.static_fields,
                use_sliding_window=False,  # Always False for validation - use full rollouts
            )

            # Create Validation DataLoader
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,  # Don't shuffle validation data
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False,
            )
            
            logger.debug(
                f"Val DataLoader: {len(val_dataset)} samples (full rollouts, no sliding window), batch_size={self.batch_size}"
            )
        else:
            self.val_loader = None
            logger.warning("No validation data specified (val_sim is empty)")

    def _create_model(self):
        """Creates the synthetic model using the registry."""
        model_name = self.model_config.get("name", "UNet")
        logger.debug(f"Creating synthetic model: {model_name}")

        model = ModelRegistry.get_synthetic_model(model_name, config=self.model_config)

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            # Handle both direct state_dict and nested checkpoint formats
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            logger.debug(f"Loaded model weights from {self.checkpoint_path}")
        except FileNotFoundError:
            logger.warning("No pre-existing model weights found. Training from scratch.")

        model = model.to(self.device)
        logger.debug("Model created successfully and moved to device.")
        return model

    def _compute_batch_loss(self, batch) -> torch.Tensor:
        """
        Compute loss for a single batch.
        
        Used by both training and validation through parent class.
        
        Args:
            batch: Tuple of (initial_state, rollout_targets) from HybridDataset
            
        Returns:
            Loss tensor for the batch
        """
        initial_state, rollout_targets = batch
        initial_state = initial_state.to(self.device)
        rollout_targets = rollout_targets.to(self.device)
        
        # Autoregressive rollout
        batch_size = initial_state.shape[0]
        num_steps = rollout_targets.shape[1]
        
        current_state = initial_state  # [B, C_all, H, W] - all fields
        total_step_loss = 0.0
        
        for t in range(num_steps):
            # Predict next state (model returns all fields)
            prediction = self.model(current_state)  # [B, C_all, H, W]
            
            # Extract only dynamic fields from prediction for loss computation
            pred_dynamic_tensors = []
            for field_name in self.field_names:
                if field_name in self.dynamic_fields:
                    start, end = self.channel_map[field_name]
                    pred_dynamic_tensors.append(prediction[:, start:end, :, :])
            
            pred_dynamic = torch.cat(pred_dynamic_tensors, dim=1)  # [B, C_dynamic, H, W]
            
            # Get ground truth for this timestep (already only dynamic fields)
            target = rollout_targets[:, t, :, :, :]  # [B, C_dynamic, H, W]
            
            # Compute loss on dynamic fields only
            step_loss = self.loss_fn(pred_dynamic, target)
            total_step_loss += step_loss
            
            # Use full prediction (all fields) as input for next timestep
            current_state = prediction
        
        # Average loss over timesteps
        avg_rollout_loss = total_step_loss / num_steps
        return avg_rollout_loss

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

        # Use memory monitor if available
        if self.memory_monitor is not None:
            self.memory_monitor.on_epoch_start()

        for batch_idx, batch in enumerate(self.train_loader):
            # Track batch start time
            if self.memory_monitor is not None:
                self.memory_monitor.on_batch_start(batch_idx)

            self.optimizer.zero_grad()

            # Compute loss using shared method
            avg_rollout_loss = self._compute_batch_loss(batch)
            avg_rollout_loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            # Track batch completion with memory monitor
            if self.memory_monitor is not None:
                self.memory_monitor.on_batch_end(batch_idx, avg_rollout_loss.item())

            total_loss += avg_rollout_loss.item()

        # Track epoch completion
        if self.memory_monitor is not None:
            self.memory_monitor.on_epoch_end()

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def _validate_epoch_rollout(self) -> float:
        """
        Perform rollout-based validation.
        
        For each validation simulation, starts from t=0 and rolls out
        until the end, computing loss over the entire trajectory.
        This gives a better measure of model performance on real rollouts.
        
        Returns:
            Average validation loss across all validation simulations
        """
        if self.val_loader is None or len(self.val_loader) == 0:
            return float('inf')
        
        logger.debug("Running rollout-based validation (full simulation trajectories)")
        
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        num_batches = 0
        
        # Get validation rollout steps from config (None means use full available length)
        val_rollout_steps = self.config.get("trainer_params", {}).get("validation_rollout_steps", None)
        
        with torch.no_grad():
            for batch in self.val_loader:
                initial_state, rollout_targets = batch
                initial_state = initial_state.to(self.device)
                rollout_targets = rollout_targets.to(self.device)
                
                batch_size = initial_state.shape[0]
                max_steps = rollout_targets.shape[1]
                
                # Use configured rollout steps or full length
                num_steps = min(val_rollout_steps, max_steps) if val_rollout_steps is not None else max_steps
                
                # Log rollout details for first batch
                if num_batches == 0:
                    if val_rollout_steps is not None and num_steps < val_rollout_steps:
                        logger.debug(f"  Rollout settings: {num_steps} timesteps per trajectory (limited by available data, requested {val_rollout_steps}), batch_size={batch_size}")
                    elif val_rollout_steps is not None:
                        logger.debug(f"  Rollout settings: {num_steps} timesteps per trajectory (configured), batch_size={batch_size}")
                    else:
                        logger.debug(f"  Rollout settings: {num_steps} timesteps per trajectory (full length), batch_size={batch_size}")
                
                # Perform full autoregressive rollout for each sample
                current_state = initial_state  # [B, C_all, H, W]
                batch_loss = 0.0
                
                for t in range(num_steps):
                    # Predict next state (model returns all fields)
                    prediction = self.model(current_state)  # [B, C_all, H, W]
                    
                    # Extract dynamic fields from prediction for loss computation
                    pred_dynamic_tensors = []
                    for field_name in self.field_names:
                        if field_name in self.dynamic_fields:
                            start, end = self.channel_map[field_name]
                            pred_dynamic_tensors.append(prediction[:, start:end, :, :])
                    
                    pred_dynamic = torch.cat(pred_dynamic_tensors, dim=1)  # [B, C_dynamic, H, W]
                    
                    # Get ground truth for this timestep
                    target = rollout_targets[:, t, :, :, :]  # [B, C_dynamic, H, W]
                    
                    # Compute step loss
                    step_loss = self.loss_fn(pred_dynamic, target)
                    batch_loss += step_loss.item()
                    
                    # Use full prediction (all fields) as input for next timestep
                    current_state = prediction
                
                # Average loss over timesteps for this batch
                avg_batch_loss = batch_loss / num_steps
                total_loss += avg_batch_loss * batch_size
                num_samples += batch_size
                num_batches += 1
        
        logger.debug(f"  Completed {num_batches} validation rollouts ({num_samples} total trajectories)")
        avg_val_loss = total_loss / num_samples if num_samples > 0 else float('inf')
        return avg_val_loss
