"""
REFACTORED HybridTrainer - Clean API with new dataset methods

Key Changes:
- Uses set_augmented_trajectories() for TensorDataset (physical → synthetic training)
- Uses set_augmented_predictions() for FieldDataset (synthetic → physical training)
- Clear distinction between trajectory-based and prediction-based augmentation
- Datasets maintain source tracking internally
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
import gc
from contextlib import contextmanager

from src.training.abstract_trainer import AbstractTrainer
from src.training.synthetic.trainer import SyntheticTrainer
from src.training.physical.trainer import PhysicalTrainer
from src.data import TensorDataset, FieldDataset
from src.utils.logger import get_logger, logging
from src.config import ConfigHelper
from phi.math import Tensor
from phi.flow import Field
from torch.utils.data import DataLoader
from src.factories.dataloader_factory import DataLoaderFactory

logger = get_logger(__name__)


class HybridTrainer(AbstractTrainer):
    """
    Hybrid trainer with clean separation of concerns.
    
    Training Flow:
    1. Physical model generates trajectories (Fields) → TensorDataset augmentation
    2. Synthetic model trains on real + physical trajectories
    3. Synthetic model generates predictions (tensors) → FieldDataset augmentation
    4. Physical model trains on real + synthetic predictions
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

        # Pre-create base datasets
        self._base_tensor_dataset = None
        self._base_field_dataset = None
        self._initialize_datasets()

        # Create component trainers
        self.synthetic_trainer = SyntheticTrainer(config, synthetic_model)
        self.physical_trainer = PhysicalTrainer(config, physical_model, learnable_params)

        # Training state
        self.current_cycle = 0
        self.best_synthetic_loss = float("inf")
        self.best_physical_loss = float("inf")

    def _parse_config(self):
        """Extract all configuration parameters."""
        self.trainer_config = self.config["trainer_params"]
        self.hybrid_config = self.trainer_config["hybrid"]
        self.aug_config = self.trainer_config["augmentation"]
        self.generation_config = self.config["generation_params"]
        
        self.num_cycles = self.hybrid_config["num_cycles"]
        self.synthetic_epochs_per_cycle = self.hybrid_config["synthetic_epochs_per_cycle"]
        self.physical_epochs_per_cycle = self.hybrid_config["physical_epochs_per_cycle"]
        self.warmup_synthetic_epochs = self.hybrid_config["warmup_synthetic_epochs"]
        
        self.alpha = self.aug_config["alpha"]
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
        """Pre-create base datasets that will be reused."""
        logger.info("Initializing reusable datasets...")

        train_sim_indices = self.trainer_config["train_sim"]
        percentage_real_data = self.trainer_config["percentage_real_data"]
        
        # Create base datasets (without augmentation)
        tensor_result = self._create_dataset(
            train_sim_indices,
            return_fields=False,
            percentage_real_data=percentage_real_data
        )
        self._base_tensor_dataset = tensor_result.dataset
        
        field_result = self._create_dataset(
            train_sim_indices,
            return_fields=True,
            percentage_real_data=percentage_real_data
        )
        self._base_field_dataset = field_result.dataset

    def train(self):
        """Main training loop."""
        if self.warmup_synthetic_epochs > 0:
            self._run_warmup()
        
        pbar = tqdm(range(self.num_cycles), desc="Hybrid Cycles", unit="cycle")
        
        for cycle in pbar:
            self.current_cycle = cycle
            
            synthetic_loss, physical_loss = self._run_cycle()
            
            pbar.set_postfix({
                'syn_loss': f"{synthetic_loss:.6f}",
                'phy_loss': f"{physical_loss:.6f}",
                'params': f"{self.physical_trainer.get_current_params()}"
            })
            self._save_if_best(synthetic_loss, physical_loss)

    def _run_warmup(self):
        """Warmup phase - train synthetic model on real data only."""
        logger.info(f"Running warmup for {self.warmup_synthetic_epochs} epochs...")
        
        self._base_tensor_dataset.access_policy = "real_only"
        
        warmup_dataloader = DataLoader(
            self._base_tensor_dataset,
            batch_size=self.trainer_config["batch_size"],
            shuffle=True,
            num_workers=self.trainer_config.get("num_workers", 0),
            pin_memory=True
        )
        
        self.synthetic_trainer.train(
            data_source=warmup_dataloader,
            num_epochs=self.warmup_synthetic_epochs,
            verbose=True
        )

        self._clear_gpu_memory()

    def _run_cycle(self) -> Tuple[float, float]:
        """Execute one complete hybrid training cycle."""
        logger.debug(f"\n{'='*60}")
        logger.debug(f"CYCLE {self.current_cycle + 1}/{self.num_cycles}")
        logger.debug(f"{'='*60}")
        
        # Phase 1: Generate physical trajectories (as Field rollouts)
        logger.debug("Phase 1: Generating physical trajectories...")
        physical_rollouts = self._generate_physical_rollouts()
        
        # Phase 2: Add to tensor dataset and train synthetic model
        logger.debug("Phase 2: Training synthetic model on physical trajectories...")
        # Ensure dataset access policy is set before we recompute totals in the dataset
        access_policy = self._get_access_policy(for_synthetic=True)
        self._base_tensor_dataset.access_policy = access_policy

        logger.debug(f"  Set TensorDataset access_policy={access_policy} before adding augmented trajectories")
        self._base_tensor_dataset.set_augmented_trajectories(physical_rollouts)
        synthetic_loss = self._train_synthetic_model()
        
        # Phase 3: Generate synthetic predictions (as tensor windows)
        logger.debug("Phase 3: Generating synthetic predictions...")
        synthetic_predictions = self._generate_synthetic_predictions()
        
        # Phase 4: Add to field dataset and train physical model
        logger.debug("Phase 4: Training physical model on synthetic predictions...")
        physical_loss = self._train_physical_model(synthetic_predictions)
        
        # Cleanup
        del physical_rollouts
        del synthetic_predictions
        self._clear_gpu_memory()
        
        return synthetic_loss, physical_loss
    
    # ==================== PHASE 1: PHYSICAL ROLLOUT GENERATION ====================
    
    def _generate_physical_rollouts(self) -> List[Dict[str, Field]]:
        """
        Generate physical model rollouts as Field trajectories.
        
        Returns:
            List of rollout dictionaries: [{'field_name': Field[time, x, y]}]
        """
        with self.managed_memory_phase("Physical Generation"):
            if hasattr(self.physical_model, 'to'):
                self.physical_model.to(self.device)
            
            # Calculate requirements
            num_real_samples = self._calculate_num_real_samples()
            num_synthetic_samples = int(num_real_samples * self.alpha)
            
            trajectory_length = self.generation_config["total_steps"]
            samples_per_trajectory = trajectory_length - self.trainer_config["num_predict_steps"]
            num_trajectories = max(1, (num_synthetic_samples + samples_per_trajectory - 1) // samples_per_trajectory)
            
            logger.debug(
                f"  Generating {num_trajectories} trajectories "
                f"(~{num_synthetic_samples} samples after windowing)"
            )
            
            # Generate batched rollout
            initial_state = self.physical_model.get_random_state(batch_size=num_trajectories)
            rollout = self.physical_model.rollout(initial_state, num_steps=trajectory_length)
            
            # Split into list of individual trajectories
            field_names = ConfigHelper(self.config).get_field_names()
            rollouts = []
            
            for traj_idx in range(num_trajectories):
                trajectory = {}
                for field_name in field_names:
                    trajectory[field_name] = rollout[field_name].batch[traj_idx]
                rollouts.append(trajectory)
            
            logger.debug(f"  Generated {len(rollouts)} physical trajectories")
            return rollouts
    
    # ==================== PHASE 2: SYNTHETIC MODEL TRAINING ====================
    
    def _train_synthetic_model(self) -> float:
        """
        Train synthetic model on real + physical trajectories.
        
        TensorDataset handles windowing of physical trajectories internally.
        """
        with self.managed_memory_phase("Synthetic Training", clear_cache=False):
            # Set access policy
            access_policy = self._get_access_policy(for_synthetic=True)
            self._base_tensor_dataset.access_policy = access_policy
            
            logger.debug(
                f"  TensorDataset: {self._base_tensor_dataset.num_real} real + "
                f"{self._base_tensor_dataset.num_augmented} augmented = "
                f"{len(self._base_tensor_dataset)} total samples"
            )
            
            # Create dataloader
            dataloader = DataLoader(
                self._base_tensor_dataset,
                batch_size=self.trainer_config["batch_size"],
                shuffle=True,
                num_workers=self.trainer_config.get("num_workers", 0),
                pin_memory=True
            )
            
            # Train
            result = self.synthetic_trainer.train(
                data_source=dataloader,
                num_epochs=self.synthetic_epochs_per_cycle,
                verbose=False
            )
            
            logger.debug(f"  Synthetic loss: {result['final_loss']:.6f}")
            return result['final_loss']
    
    # ==================== PHASE 3: SYNTHETIC PREDICTION GENERATION ====================
    
# In src/training/hybrid/trainer.py
# Update the _generate_synthetic_predictions method

# In src/training/hybrid/trainer.py
# Update the _generate_synthetic_predictions method

    def _generate_synthetic_predictions(self) -> List[torch.Tensor]:
        """
        Generate windowed synthetic predictions as trajectories.
        
        NEW BEHAVIOR:
        - Passes physical trajectories directly to synthetic model
        - No dataset indexing involved - uses raw augmented_samples
        - Returns trajectories in BVTS format that can be windowed
        - Format: [1, V, T, H, W] where T is trajectory length
        
        Returns:
            List of prediction trajectory tensors in BVTS format
        """
        with self.managed_memory_phase("Synthetic Prediction"):
            self.synthetic_model.to(self.device)
            self.synthetic_model.eval()
            
            # Get physical trajectories directly from TensorDataset
            # These are stored in augmented_samples as cache-format dicts
            physical_trajectories = self._base_tensor_dataset.augmented_samples
            
            if not physical_trajectories:
                logger.warning("No physical trajectories available for synthetic prediction")
                return []
            
            logger.debug(f"  Using {len(physical_trajectories)} physical trajectories as input")
            
            # Use model's built-in generation method
            # Pass trajectories directly, not through dataset indexing
            prediction_trajectories = self.synthetic_model.generate_predictions(
                trajectories=physical_trajectories,
                device=str(self.device),
                batch_size=1,  # Could batch multiple trajectories if needed
            )
            
            logger.debug(f"  Generated {len(prediction_trajectories)} synthetic prediction trajectories")
            return prediction_trajectories


    def _train_physical_model(
        self,
        synthetic_predictions: List[torch.Tensor]
    ) -> float:
        """
        Train physical model on real + synthetic predictions.
        
        UPDATED: Now handles prediction trajectories instead of pre-windowed samples.
        FieldDataset will handle windowing using set_augmented_trajectories.
        """
        if len(self.physical_trainer.learnable_params) == 0:
            logger.info("No learnable parameters, skipping physical training")
            return 0.0

        with self.managed_memory_phase("Physical Training", clear_cache=False):
            # Set access policy
            access_policy = self._get_access_policy(for_synthetic=False)
            
            # NEW: Convert BVTS trajectories to cache-format dicts for FieldDataset
            cache_format_trajectories = []
            cfg_helper = ConfigHelper(self.config)
            field_names = cfg_helper.get_field_names()
            
            for traj_tensor in synthetic_predictions:
                # traj_tensor: [1, V, T, H, W] in BVTS format
                # Squeeze batch dimension
                traj_tensor = traj_tensor.squeeze(0)  # [V, T, H, W]
                
                # Split into per-field tensors based on channel counts
                field_trajectories = {}
                channel_idx = 0
                
                for field_name in field_names:
                    # Get number of channels for this field
                    if field_name in self.synthetic_model.output_specs:
                        num_channels = self.synthetic_model.output_specs[field_name]
                    else:
                        num_channels = self.synthetic_model.input_specs[field_name]
                    
                    # Extract field channels
                    field_tensor = traj_tensor[channel_idx:channel_idx + num_channels]  # [C, T, H, W]
                    field_trajectories[field_name] = field_tensor
                    channel_idx += num_channels
                
                # Create cache-format dict
                cache_format_trajectories.append({'tensor_data': field_trajectories})
            
            # Set augmented trajectories (now in proper cache format)
            self._base_field_dataset.set_augmented_trajectories(cache_format_trajectories)
            self._base_field_dataset.access_policy = access_policy
            
            logger.debug(
                f"  FieldDataset: {self._base_field_dataset.num_real} real + "
                f"{self._base_field_dataset.num_augmented} augmented = "
                f"{len(self._base_field_dataset)} total samples"
            )
            
            # Create dataloader
            from src.data.dataset_utilities import field_collate_fn
            dataloader = DataLoader(
                self._base_field_dataset,
                batch_size=self.trainer_config["batch_size"],
                shuffle=True,
                num_workers=0,
                collate_fn=field_collate_fn
            )
            
            # Train
            result = self.physical_trainer.train(
                data_source=dataloader,
                num_epochs=self.physical_epochs_per_cycle,
                verbose=False
            )
            
            logger.debug(f"  Physical loss: {result['final_loss']:.6f}")
            return float(result['final_loss'])
        
    # ==================== UTILITIES ====================
    
    def _calculate_num_real_samples(self) -> int:
        """
        Calculate number of truly real samples (excluding any augmentation).
        
        For TensorDataset: Need to count only the original real data, not
        any previously added physical trajectories.
        """
        # Check if dataset has original real count stored
        if hasattr(self._base_tensor_dataset, '_num_real_only'):
            # This exists if we've added trajectories before
            return self._base_tensor_dataset._num_real_only
        
        # Otherwise, num_real is the true real count
        return self._base_tensor_dataset.num_real
    
    def _create_dataset(
        self,
        sim_indices: List[int],
        return_fields: bool = False,
        percentage_real_data: float = 1.0
    ):
        """Create dataset using DataLoaderFactory."""
        mode = "field" if return_fields else "tensor"
        
        result = DataLoaderFactory.create(
            config=self.config,
            mode=mode,
            sim_indices=sim_indices,
            enable_augmentation=False,
            batch_size=self.trainer_config["batch_size"],
            percentage_real_data=percentage_real_data
        )
        
        return result
    
    def _get_access_policy(self, for_synthetic: bool) -> str:
        """Determine data access policy based on configuration."""
        if for_synthetic:
            # Synthetic model training
            if self.real_data_access in ["both", "synthetic_only"]:
                return "both"  # Use real + physical trajectories
            else:
                return "generated_only"  # Only physical trajectories
        else:
            # Physical model training
            if self.real_data_access in ["both", "physical_only"]:
                return "both"  # Use real + synthetic predictions
            else:
                return "generated_only"  # Only synthetic predictions
    
    @staticmethod
    @contextmanager
    def managed_memory_phase(phase_name: str, clear_cache: bool = True):
        """Context manager for memory-efficient training phases."""
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
        """Save checkpoints if losses improved."""
        if synthetic_loss < self.best_synthetic_loss:
            self.best_synthetic_loss = synthetic_loss
            checkpoint_path = Path(self.synthetic_trainer.checkpoint_path)
            checkpoint_path = (
                checkpoint_path.parent
                / f"{checkpoint_path.stem}_hybrid_best{checkpoint_path.suffix}"
            )
            torch.save(self.synthetic_model.state_dict(), checkpoint_path)
            logger.debug(f"  Saved best synthetic model (loss: {synthetic_loss:.6f})")
        
        if physical_loss < self.best_physical_loss:
            self.best_physical_loss = physical_loss
            logger.debug(f"  New best physical loss: {physical_loss:.6f}")