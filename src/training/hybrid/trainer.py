"""
Hybrid Trainer

Implements the HYCO (Hybrid Corrector) approach that alternates between 
training synthetic and physical models with cross-model data augmentation.

The hybrid training cycle:
1. Generate predictions from physical model → augment synthetic training data
2. Train synthetic model with augmented data
3. Generate predictions from synthetic model → augment physical training data  
4. Train physical model with augmented data
5. Repeat for specified number of cycles

This enables both models to learn from each other's strengths.
"""
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
import gc
from contextlib import contextmanager
from src.training.abstract_trainer import AbstractTrainer
from src.training.synthetic.trainer import SyntheticTrainer
from src.training.physical.trainer import PhysicalTrainer
from src.factories.dataloader_factory import DataLoaderFactory
from src.data import TensorDataset, FieldDataset
from src.utils.logger import get_logger
from src.data.dataset_utilities import field_collate_fn
# === 2. Batched conversion ===
from src.utils.field_conversion import make_converter
from src.config import ConfigHelper
import torch
from phi.math import math, Tensor
import logging
# Setup dataset creation
from torch.utils.data import DataLoader
from src.config import ConfigHelper
from src.data import DataManager
from phi.flow import plot
import matplotlib.pyplot as plt

logger = get_logger(__name__)


class HybridTrainer(AbstractTrainer):
    """
    Hybrid trainer that alternates between synthetic and physical model training
    with cross-model data augmentation.

    The trainer orchestrates:
    - Alternating training cycles
    - Cross-model prediction generation
    - Data augmentation with generated predictions
    - Model checkpointing and evaluation

    Args:
        config: Full configuration dictionary
        synthetic_model: Pre-created synthetic model (e.g., UNet)
        physical_model: Pre-created physical model (e.g., BurgersModel)
        learnable_params: List of learnable physical parameters
    """

    def __init__(
        self,
        config: Dict[str, Any],
        synthetic_model: nn.Module,
        physical_model,
        learnable_params: Dict[str, Tensor],
    ):
        """Initialize hybrid trainer with both models."""
        super().__init__(config)

        # Store models
        self.synthetic_model = synthetic_model
        self.physical_model = physical_model
        self.learnable_params = learnable_params

        # Parse configuration
        self._parse_config()

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pre-create base datasets (created once, reused)
        self._base_tensor_dataloader = None
        self._base_field_dataloader = None
        self._initialize_datasets()

        # Create component trainers
        self.synthetic_trainer = SyntheticTrainer(config, synthetic_model)
        self.physical_trainer = PhysicalTrainer(config, physical_model, learnable_params)

        # Training state
        self.current_cycle = 0
        self.best_synthetic_loss = float("inf")
        self.best_physical_loss = float("inf")

    # =======================================
    # INITIALIZATION
    # =======================================

    def _parse_config(self):
        """Extract all configuration parameters."""
        self.trainer_config = self.config["trainer_params"]
        self.hybrid_config = self.trainer_config["hybrid"]
        self.aug_config = self.trainer_config["augmentation"]
        self.generation_config = self.config["generation_params"]
        
        # Training parameters
        self.num_cycles = self.hybrid_config["num_cycles"]
        self.synthetic_epochs_per_cycle = self.hybrid_config["synthetic_epochs_per_cycle"]
        self.physical_epochs_per_cycle = self.hybrid_config["physical_epochs_per_cycle"]
        self.warmup_synthetic_epochs = self.hybrid_config["warmup_synthetic_epochs"]
        
        # Augmentation parameters
        self.alpha = self.aug_config["alpha"]
        
        # Data access control
        self.real_data_access = self.hybrid_config["real_data_access"]
        self._validate_real_data_access()

    def _validate_real_data_access(self):
        """Validate real_data_access parameter."""
        valid_options = ["both", "synthetic_only", "physical_only", "neither"]
        if self.real_data_access not in valid_options:
            raise ValueError(
                f"Invalid real_data_access: '{self.real_data_access}'. "
                f"Must be one of {valid_options}"
            )

    def _initialize_datasets(self):
        """
        Pre-create base datasets that will be reused throughout training.
        These are created ONCE and then updated in-place.
        """
        logger.info("Initializing reusable datasets...")

        train_sim_indices = self.trainer_config["train_sim"]
        percentage_real_data = self.trainer_config["percentage_real_data"]
        # Create base tensor dataloader (for synthetic training)
        self._base_tensor_dataloader = self._create_dataset(
            train_sim_indices,
            return_fields=False,
            percentage_real_data=percentage_real_data
        )
        
        # Create base field dataloader (for physical training)  
        self._base_field_dataloader = self._create_dataset(
            train_sim_indices,
            return_fields=True,
            percentage_real_data=percentage_real_data
        )

    # =======================================
    # MAIN TRAINING LOOP
    # =======================================

    def train(self):
        """
        FULLY OPTIMIZED training loop with all improvements:
        - Step 1: Batched physical predictions
        - Step 2: Optimized synthetic predictions  
        - Step 3: Dataset reuse
        - Step 4: Memory management
        
        Replace train() with this method.
        """
        # Optional warmup phase
        if self.warmup_synthetic_epochs > 0:
            self._run_warmup()
        
        # Main hybrid training loop
        pbar = tqdm(range(self.num_cycles), desc="Hybrid Cycles", unit="cycle")
        
        for cycle in pbar:
            self.current_cycle = cycle
            
            synthetic_loss, physical_loss = self._run_cycle()
            
            # Update progress bar and save checkpoints
            pbar.set_postfix({
                'syn_loss': f"{synthetic_loss:.6f}",
                'phy_loss': f"{physical_loss:.6f}",
                'params': f"{self.physical_trainer.get_current_params()}"
            })
            self._save_if_best(synthetic_loss, physical_loss)

    def _run_warmup(self):
        """
        OPTIMIZED warmup using pre-created dataset.
        """
        # Use base dataset directly (no augmentation)
        self._base_tensor_dataloader.dataset.access_policy = "real_only"
        
        self.synthetic_trainer.train(
            data_source=self._base_tensor_dataloader,
            num_epochs=self.warmup_synthetic_epochs,
            verbose=True
        )

        # Clean GPU memory after warmup
        self._clear_gpu_memory()

    def _run_cycle(self) -> Tuple[float, float]:
        """
        Execute one complete hybrid training cycle.
        
        Returns:
            Tuple of (synthetic_loss, physical_loss)
        """
        # Phase 1: Generate physical predictions
        physical_preds = self._generate_physical_predictions()
        
        # Phase 2: Train synthetic model with augmentation
        synthetic_loss = self._train_synthetic_model(physical_preds)
        
        # Clean up
        del physical_preds
        
        # Phase 3: Generate synthetic predictions
        synthetic_preds = self._generate_synthetic_predictions()
        
        # Phase 4: Train physical model with augmentation
        physical_loss = self._train_physical_model(synthetic_preds)
        
        # Clean up
        del synthetic_preds
        self._clear_gpu_memory()

        
        
        return synthetic_loss, physical_loss
    
    # =======================================
    # Prediction Generation
    # =======================================

    def _generate_physical_predictions(self) -> List[Tuple]:
        """
        Generate predictions using physical model with SYNTHETIC trajectories.
        
        NEW APPROACH:
        1. Physical model generates complete trajectories from random ICs
        2. Trajectories are passed to dataset for windowing
        3. Dataset handles conversion to appropriate format (tensors)
        
        Returns:
            List of (input_tensor, target_tensor) tuples
        """
        with self.managed_memory_phase("Physical Prediction"):
            # Move physical model to GPU if not already there
            if hasattr(self.physical_model, 'to'):
                self.physical_model.to(self.device)
            
            # Calculate how many trajectories to generate based on alpha
            num_real = len(self._base_tensor_dataloader.dataset.sim_indices) * \
                    (self._base_tensor_dataloader.dataset.num_frames - self._base_tensor_dataloader.dataset.num_predict_steps)
            num_samples_needed = int(num_real * self.alpha)
            
            # Calculate trajectory parameters
            # We need enough trajectory length to create sliding windows
            trajectory_length = self.generation_config["total_steps"] # Extra for windowing
            
            # Calculate how many trajectories we need
            # Each trajectory produces (trajectory_length - num_predict_steps) samples
            samples_per_trajectory = trajectory_length - self.trainer_config["num_predict_steps"]
            num_trajectories = max(1, (num_samples_needed + samples_per_trajectory - 1) // samples_per_trajectory)
            
            
            # Generate synthetic trajectories from random ICs
            trajectories, rollout = self.physical_model.generate_synthetic_trajectories(
                num_trajectories=num_trajectories,
                trajectory_length=trajectory_length,
                warmup_steps=5,  # Let physics settle
            )
            
            print(rollout)
            return trajectories
        
    def _generate_synthetic_predictions(self) -> List[Tuple]:
        """
        Generate predictions using synthetic model for physical augmentation.
        
        Returns:
            List of (input_tensor, target_tensor) tuples
        """
        with self.managed_memory_phase("Synthetic Prediction"):
            # Move model to GPU
            self.synthetic_model.to(self.device)
            self.synthetic_model.eval()
            
            # Use synthetic model's optimized generation method
            tensor_dataloader =  self._create_dataset(self.trainer_config["train_sim"])
            batch_size = self.trainer_config["batch_size"]
            
            inputs_tensor, targets_tensor = self.synthetic_model.generate_predictions(
                real_dataset=tensor_dataloader.dataset,
                alpha=self.alpha,
                device=str(self.device),
                batch_size=batch_size,
            )


            
            # Convert to list of tuples
            tensor_predictions = list(zip(inputs_tensor, targets_tensor))
            
            return tensor_predictions
        
    def _convert_fields_to_tensors(
        self, 
        initial_fields, 
        predictions, 
        field_dataset
    ) -> List[Tuple]:
        """
        Convert PhiFlow fields to PyTorch tensors (batched for efficiency).
        
        Args:
            initial_fields: Initial field states
            predictions: Predicted field trajectories
            field_dataset: Field dataset (for field names)
        
        Returns:
            List of (input_tensor, target_tensor) tuples
        """
        cfg = ConfigHelper(self.config)
        field_names_input = field_dataset.field_names
        field_names_target = cfg.get_field_names()
        
        # Pre-create converters (reused for all samples)
        input_converters = {
            name: make_converter(initial_fields[name]) 
            for name in field_names_input
        }
        target_converters = {
            name: make_converter(predictions[0][name]) 
            for name in field_names_target
        }
        
        # Batched conversion: all inputs at once
        batched_input = torch.cat([
            input_converters[name].field_to_tensor(
                initial_fields[name], 
                ensure_cpu=False
            )
            for name in field_names_input
        ], dim=1)
        
        # Batched conversion: all targets at once
        batched_targets = torch.stack([
            torch.cat([
                target_converters[name].field_to_tensor(
                    pred_t[name], 
                    ensure_cpu=False
                )
                for name in field_names_target
            ], dim=1)
            for pred_t in predictions
        ], dim=1)
        
        # Move to CPU once (not per-sample)
        batched_input_cpu = batched_input.cpu()
        batched_targets_cpu = batched_targets.cpu()
        
        # Split into list of tuples
        B = batched_input.shape[0]
        tensor_predictions = [
            (batched_input_cpu[i], batched_targets_cpu[i])
            for i in range(B)
        ]
        
        return tensor_predictions
    

    # =======================================
    # MODEL TRAINING
    # =======================================

    def _train_synthetic_model(self, generated_data: List[Tuple]) -> float:
        """
        OPTIMIZED: Synthetic training with memory management.
        """
        with self.managed_memory_phase("Synthetic Training", clear_cache=False):
            # Update dataset (from Step 3)
            access_policy = self._get_access_policy(for_synthetic=True)
            
            self._update_dataset_augmentation(
                self._base_tensor_dataloader.dataset,
                generated_data,
                access_policy
            )

            
            # Train
            result = self.synthetic_trainer.train(
                data_source=self._base_tensor_dataloader,
                num_epochs=self.synthetic_epochs_per_cycle,
                verbose=False
            )
            
            return result['final_loss']
        
    def _train_physical_model(self, generated_data: List[Tuple]) -> float:
        """
        OPTIMIZED: Physical training with memory management.
        """
        if len(self.physical_trainer.learnable_params) == 0:
            return 0.0
        
        with self.managed_memory_phase("Physical Training", clear_cache=False):
            # Update dataset (from Step 3)
            access_policy = self._get_access_policy(for_synthetic=False)
            
            self._update_dataset_augmentation(
                self._base_field_dataloader.dataset,
                generated_data,
                access_policy
            )

            # Pass the DataLoader to the trainer
            sample_loss = self.physical_trainer.train(
                data_source=self._base_field_dataloader, # Pass the loader, not the dataset
                num_epochs=self.physical_epochs_per_cycle,
                verbose=False
            )
            
            return float(sample_loss['final_loss'])
        
    # =======================================
    # DATASET MANAGEMENT
    # =======================================

    def _create_dataset(
        self, sim_indices: List[int], return_fields: bool = False, percentage_real_data: float = 1.0
    ):
        """Create dataset for training using new DataLoaderFactory.

        Args:
            sim_indices: List of simulation indices to include
            return_fields: If True, return FieldDataset (for physical model).
                          If False, return TensorDataset (for synthetic model).
        """
        # Use the new DataLoaderFactory
        mode = "field" if return_fields else "tensor"

        # For physical model (field mode), we get a FieldDataset directly
        # For synthetic model (tensor mode), we get a DataLoader, so extract the dataset
        result = DataLoaderFactory.create(
            config=self.config,
            mode=mode,
            sim_indices=sim_indices,
            enable_augmentation=False,  # We handle augmentation separately in hybrid training
            batch_size=(self.trainer_config["batch_size"]),
            percentage_real_data=percentage_real_data
        )

        return result
    
    def _update_dataset_augmentation(
        self, 
        dataset, 
        augmented_data: List[Tuple],
        access_policy: str
    ):
        """
        OPTIMIZED: Update dataset's augmented samples in-place.
        
        This avoids recreating the entire dataset - we just swap out
        the augmented_samples list and update counters.
        
        Args:
            dataset: TensorDataset or FieldDataset to update
            augmented_data: New augmented samples
            access_policy: 'both', 'real_only', or 'generated_only'
        """ 
        # Process augmented samples through dataset's converter
        # (this ensures format consistency)
        processed_samples = [
            dataset._process_augmented_sample(sample)
            for sample in augmented_data
        ]
        
        # Update in-place (no dataset recreation!)
        dataset.augmented_samples = processed_samples
        dataset.num_augmented = len(processed_samples)
        dataset.access_policy = access_policy
     
    def _get_access_policy(self, for_synthetic: bool) -> str:
        """
        Determine data access policy based on configuration.
        
        Args:
            for_synthetic: If True, get policy for synthetic model; else physical
        
        Returns:
            Access policy string
        """
        if for_synthetic:
            return "both" if self.real_data_access in ["both", "synthetic_only"] else "generated_only"
        else:
            return "both" if self.real_data_access in ["both", "physical_only"] else "generated_only"

    # =======================================
    # UTILITIES
    # =======================================

    @staticmethod
    @contextmanager
    def managed_memory_phase(phase_name: str, clear_cache: bool = True):
        """
        Context manager for memory-efficient training phases.
        
        Usage:
            with self.managed_memory_phase("physical_prediction"):
                predictions = self._generate_physical_predictions()
        
        Args:
            phase_name: Name of the phase (for logging)
            clear_cache: Whether to clear GPU cache after phase
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            yield
        finally:
            if torch.cuda.is_available() and clear_cache:
                gc.collect()
                torch.cuda.empty_cache()

    def _clear_gpu_memory(self):
        """Force GPU memory cleanup."""
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
     
    def _save_if_best(self, synthetic_loss: float, physical_loss: float):
        """
        Save checkpoints if losses improved.

        Args:
            synthetic_loss: Current synthetic model loss
            physical_loss: Current physical model loss
        """
        # Save synthetic model if improved
        if synthetic_loss < self.best_synthetic_loss:
            self.best_synthetic_loss = synthetic_loss
            checkpoint_path = Path(self.synthetic_trainer.checkpoint_path)
            checkpoint_path = (
                checkpoint_path.parent
                / f"{checkpoint_path.stem}_hybrid_best{checkpoint_path.suffix}"
            )
            torch.save(self.synthetic_model.state_dict(), checkpoint_path)

        # Save physical parameters if improved
        if physical_loss < self.best_physical_loss:
            self.best_physical_loss = physical_loss

    

    
            
    
    
    
    

