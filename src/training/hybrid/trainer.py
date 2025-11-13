"""
REFACTORED HybridTrainer - Clean separation of concerns

The trainer ONLY:
1. Generates data (Fields from physical, tensors from synthetic)
2. Passes data to datasets
3. Calls train methods

Datasets handle ALL conversions internally via their _process_augmented_sample methods.
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
from src.utils.logger import get_logger
from src.config import ConfigHelper
from phi.math import Tensor
from phi.flow import Field
from torch.utils.data import DataLoader
from src.factories.dataloader_factory import DataLoaderFactory

logger = get_logger(__name__)


class HybridTrainer(AbstractTrainer):
    """
    Hybrid trainer with clean separation of concerns.
    
    Generates data and passes to datasets - datasets handle all conversions.
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
        self._base_tensor_dataset.access_policy = "real_only"
        
        # Create dataloader for warmup
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
        # Phase 1: Generate physical trajectories (as Field rollouts)
        physical_rollouts = self._generate_physical_rollouts()
        
        # Phase 2: Add to tensor dataset and train synthetic model
        self._base_tensor_dataset.clear_synthetic_simulations()
        self._base_tensor_dataset.add_synthetic_simulations(physical_rollouts)
        synthetic_loss = self._train_synthetic_model()
        
        # Phase 3: Generate synthetic predictions (as tensor windows)
        synthetic_predictions = self._generate_synthetic_predictions()
        
        # Phase 4: Add to field dataset and train physical model
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
            Each rollout is a full trajectory in Field format.
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
            
            
            # Generate batched rollout
            initial_state = self.physical_model.get_random_state(batch_size=num_trajectories)
            rollout = self.physical_model.rollout(initial_state, num_steps=trajectory_length)
            # rollout: Dict[field_name, Field] with shape [batch, time, x, y]
            
            # Split into list of individual trajectories
            field_names = ConfigHelper(self.config).get_field_names()
            rollouts = []
            
            for traj_idx in range(num_trajectories):
                trajectory = {}
                for field_name in field_names:
                    # Extract single trajectory [time, x, y]
                    trajectory[field_name] = rollout[field_name].batch[traj_idx]
                rollouts.append(trajectory)
            
            return rollouts
    
    # ==================== PHASE 2: SYNTHETIC MODEL TRAINING ====================
    
    def _train_synthetic_model(self) -> float:
        """
        Train synthetic model on real + synthetic trajectory windows.
        
        TensorDataset handles conversion from Field trajectories to tensor windows.
        """
        
        with self.managed_memory_phase("Synthetic Training", clear_cache=False):
            # Set access policy
            access_policy = self._get_access_policy(for_synthetic=True)
            self._base_tensor_dataset.access_policy = access_policy
            
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
            
            return result['final_loss']
    
    # ==================== PHASE 3: SYNTHETIC PREDICTION GENERATION ====================
    
    def _generate_synthetic_predictions(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate windowed synthetic predictions as tensors.
        
        Returns:
            List of (initial_state, targets) tuples where:
            - initial_state: [C_all, H, W]
            - targets: [T, C_all, H, W]
        """
        
        with self.managed_memory_phase("Synthetic Prediction"):
            self.synthetic_model.to(self.device)
            self.synthetic_model.eval()
            
            batch_size = self.trainer_config["batch_size"]
            
            # Use model's built-in generation method
            inputs_tensor, targets_tensor = self.synthetic_model.generate_predictions(
                real_dataset=self._base_tensor_dataset,
                alpha=self.alpha,
                device=str(self.device),
                batch_size=batch_size,
            )
            
            
            return list(zip(inputs_tensor, targets_tensor))
    
    # ==================== PHASE 4: PHYSICAL MODEL TRAINING ====================
    
    def _train_physical_model(
        self,
        synthetic_predictions: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> float:
        """
        Train physical model on real data + synthetic windowed predictions.
        
        FieldDataset handles conversion from tensor windows to Field windows.
        
        Args:
            synthetic_predictions: List of (input, target) tensor tuples
        
        Returns:
            Final loss value
        """
        if len(self.physical_trainer.learnable_params) == 0:
            logger.info("No learnable parameters, skipping physical training")
            return 0.0
        
        
        with self.managed_memory_phase("Physical Training", clear_cache=False):
            # Set access policy
            access_policy = self._get_access_policy(for_synthetic=False)
            
            # OPTIMIZED: Pre-convert all predictions to Fields at once
            # This happens once instead of per-access, eliminating repeated conversions
            self._base_field_dataset.set_augmented_predictions(synthetic_predictions)
            self._base_field_dataset.access_policy = access_policy
            self._base_field_dataset._total_length = self._base_field_dataset._compute_length()
            
            # Create dataloader with custom collate function
            from src.data.dataset_utilities import field_collate_fn
            dataloader = DataLoader(
                self._base_field_dataset,
                batch_size=self.trainer_config["batch_size"],
                shuffle=True,
                num_workers=0,  # Field conversion doesn't work well with multiprocessing
                collate_fn=field_collate_fn
            )
            
            # Train
            result = self.physical_trainer.train(
                data_source=dataloader,
                num_epochs=self.physical_epochs_per_cycle,
                verbose=False
            )
            
            return float(result['final_loss'])
    
    # ==================== UTILITIES ====================
    
    def _calculate_num_real_samples(self) -> int:
        """Calculate number of real samples in dataset."""
        # Use _num_real_only if it exists (excludes synthetic sims)
        if hasattr(self._base_tensor_dataset, '_num_real_only'):
            return self._base_tensor_dataset._num_real_only
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
                return "both"  # Use real + synthetic trajectories
            else:
                return "generated_only"  # Only synthetic trajectories
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
        
        if physical_loss < self.best_physical_loss:
            self.best_physical_loss = physical_loss