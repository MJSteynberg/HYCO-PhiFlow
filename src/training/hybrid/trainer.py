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

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm

from src.training.abstract_trainer import AbstractTrainer
from src.training.synthetic.trainer import SyntheticTrainer
from src.training.physical.trainer import PhysicalTrainer
from src.factories.dataloader_factory import DataLoaderFactory
from src.data import TensorDataset, FieldDataset
from src.utils.logger import get_logger

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
        learnable_params: List,
    ):
        """Initialize hybrid trainer with both models."""
        super().__init__(config)
        
        # Store models
        self.synthetic_model = synthetic_model
        self.physical_model = physical_model
        self.learnable_params = learnable_params
        
        # Parse configuration
        self.trainer_config = config.get("trainer_params", {})
        self.hybrid_config = self.trainer_config.get("hybrid", {})
        self.aug_config = self.trainer_config.get("augmentation", {})
        
        # Hybrid training parameters
        self.num_cycles = self.hybrid_config.get("num_cycles", 10)
        self.synthetic_epochs_per_cycle = self.hybrid_config.get("synthetic_epochs_per_cycle", 5)
        self.physical_epochs_per_cycle = self.hybrid_config.get("physical_epochs_per_cycle", 3)
        self.alpha = self.aug_config.get("alpha", 0.1)
        self.warmup_synthetic_epochs = self.hybrid_config.get("warmup_synthetic_epochs", 10)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Hybrid trainer using device: {self.device}")
        
        # Create component trainers
        self.synthetic_trainer = SyntheticTrainer(config, synthetic_model)
        self.physical_trainer = PhysicalTrainer(config, physical_model, learnable_params)
        
        # Note: CacheManager removed in new architecture
        # Augmentation is now handled directly by TensorDataset/FieldDataset
        # via augmentation_config parameter
        
        # Training state
        self.current_cycle = 0
        self.best_synthetic_loss = float('inf')
        self.best_physical_loss = float('inf')
        
        logger.info("="*60)
        logger.info("HYBRID TRAINER INITIALIZED")
        logger.info("="*60)
        logger.info(f"  Cycles: {self.num_cycles}")
        logger.info(f"  Synthetic epochs per cycle: {self.synthetic_epochs_per_cycle}")
        logger.info(f"  Physical epochs per cycle: {self.physical_epochs_per_cycle}")
        logger.info(f"  Augmentation alpha: {self.alpha}")
        logger.info(f"  Warmup epochs: {self.warmup_synthetic_epochs}")
        logger.info("="*60)
    
    def train(self):
        """
        Execute hybrid training for specified number of cycles.
        
        Training flow:
        1. Optional warmup: Train synthetic model without augmentation
        2. For each cycle:
           a. Generate physical predictions
           b. Train synthetic model with augmented data
           c. Generate synthetic predictions
           d. Train physical model with augmented data
           e. Evaluate and checkpoint
        """
        logger.info("\n" + "="*60)
        logger.info("STARTING HYBRID TRAINING")
        logger.info("="*60)
        
        # Optional warmup phase
        if self.warmup_synthetic_epochs > 0:
            self._warmup_synthetic()
        
        # Main hybrid training loop
        for cycle in range(self.num_cycles):
            self.current_cycle = cycle
            logger.info("\n" + "="*60)
            logger.info(f"CYCLE {cycle + 1}/{self.num_cycles}")
            logger.info("="*60)
            
            # Phase 1: Synthetic training with physical predictions
            logger.info(f"\n--- Phase 1: Synthetic Training (Cycle {cycle + 1}) ---")
            physical_preds = self._generate_physical_predictions()
            synthetic_loss = self._train_synthetic_with_augmentation(physical_preds)
            
            # Phase 2: Physical training with synthetic predictions
            logger.info(f"\n--- Phase 2: Physical Training (Cycle {cycle + 1}) ---")
            synthetic_preds = self._generate_synthetic_predictions()
            physical_loss = self._train_physical_with_augmentation(synthetic_preds)
            
            # Log cycle summary
            logger.info(f"\n--- Cycle {cycle + 1} Summary ---")
            logger.info(f"  Synthetic loss: {synthetic_loss:.6f}")
            logger.info(f"  Physical loss: {physical_loss:.6f}")
            
            # Save checkpoints if improved
            self._save_if_best(synthetic_loss, physical_loss)
        
        logger.info("\n" + "="*60)
        logger.info("HYBRID TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"  Best synthetic loss: {self.best_synthetic_loss:.6f}")
        logger.info(f"  Best physical loss: {self.best_physical_loss:.6f}")
        logger.info("="*60)
    
    def _warmup_synthetic(self):
        """
        Warm up the synthetic model with standard training (no augmentation).
        
        This gives the synthetic model a head start before hybrid training begins.
        """
        logger.info("\n" + "="*60)
        logger.info("WARMUP PHASE: Training Synthetic Model")
        logger.info("="*60)
        logger.info(f"  Epochs: {self.warmup_synthetic_epochs}")
        logger.info("  Data: Real data only (no augmentation)")
        
        # Create standard dataset (no augmentation)
        # Use TensorDataset directly for the base data
        base_dataset = self._create_hybrid_dataset(
            self.trainer_config["train_sim"]
        )
        
        # Create dataloader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            base_dataset,
            batch_size=self.trainer_config.get("batch_size", 16),
            shuffle=True,
        )
        
        # Train synthetic model
        self.synthetic_trainer.train(
            data_source=dataloader,
            num_epochs=self.warmup_synthetic_epochs
        )
        
        logger.info("Warmup complete")
    
    def _create_hybrid_dataset(self, sim_indices: List[int], return_fields: bool = False):
        """Create dataset for training using new DataLoaderFactory.
        
        Args:
            sim_indices: List of simulation indices to include
            return_fields: If True, return FieldDataset (for physical model).
                          If False, return TensorDataset (for synthetic model).
        """
        # Use the new DataLoaderFactory
        mode = 'field' if return_fields else 'tensor'
        
        # For physical model (field mode), we get a FieldDataset directly
        # For synthetic model (tensor mode), we get a DataLoader, so extract the dataset
        result = DataLoaderFactory.create(
            config=self.config,
            mode=mode,
            sim_indices=sim_indices,
            use_sliding_window=True,
            enable_augmentation=False,  # We handle augmentation separately in hybrid training
            batch_size=None if return_fields else self.trainer_config.get("batch_size", 16),
        )
        
        # Extract dataset if we got a DataLoader
        if return_fields:
            # Field mode returns FieldDataset directly
            return result
        else:
            # Tensor mode returns DataLoader, extract the dataset
            return result.dataset
    
    def _generate_physical_predictions(self) -> List[Tuple]:
        """
        Generate predictions using physical model for synthetic training.
        
        Returns:
            List of (input_tensor, target_tensor) tuples
        """
        logger.info("Generating predictions from physical model...")
        
        # Create field dataset for physical model (returns PhiFlow Fields)
        field_dataset = self._create_hybrid_dataset(
            self.trainer_config["train_sim"],
            return_fields=True  # Physical model needs Fields not tensors
        )
        
        # Generate predictions using the physical model's method
        # Use num_predict_steps to match dataset format
        num_predict_steps = self.trainer_config["num_predict_steps"]
        
        initial_fields_list, target_fields_list = self.physical_model.generate_predictions(
            real_dataset=field_dataset,
            alpha=self.alpha,
            device=str(self.device),
            num_rollout_steps=num_predict_steps,
        )
        
        # Convert Fields to tensors for synthetic model training
        # Need to convert each Field dict to tensor format matching TensorDataset output
        from src.utils.field_conversion import make_converter
        from src.config import ConfigHelper
        import torch
        
        # Get dynamic fields from config (for synthetic model training)
        cfg = ConfigHelper(self.config)
        dynamic_fields, static_fields = cfg.get_field_types()
        
        tensor_predictions = []
        for initial_fields, target_fields_list in zip(initial_fields_list, target_fields_list):
            # Convert initial fields to input tensor
            # Format: [C_all, H, W] - concatenate all fields
            input_tensors = []
            for field_name in field_dataset.field_names:
                converter = make_converter(initial_fields[field_name])
                field_tensor = converter.field_to_tensor(initial_fields[field_name], ensure_cpu=False)
                # Move to correct device
                field_tensor = field_tensor.to(self.device)
                # Remove batch dimension if present [B, C, H, W] -> [C, H, W]
                while field_tensor.dim() > 3:
                    field_tensor = field_tensor.squeeze(0)
                # Ensure minimum 3 dimensions [C, H, W]
                if field_tensor.dim() == 2:  # [H, W] -> [1, H, W]
                    field_tensor = field_tensor.unsqueeze(0)
                input_tensors.append(field_tensor)
            
            input_tensor = torch.cat(input_tensors, dim=0)  # [C_all, H, W]
            
            # Convert target fields to output tensor
            # Format: [T, C_all, H, W] - ALL fields to match UNet output structure
            # This ensures consistency between synthetic and physical training
            timestep_tensors = []
            for target_fields in target_fields_list:
                # For each timestep, concatenate ALL fields in the same order as input
                field_tensors = []
                for field_name in cfg.get_field_names():
                    converter = make_converter(target_fields[field_name])
                    field_tensor = converter.field_to_tensor(target_fields[field_name], ensure_cpu=False)
                    # Move to correct device
                    field_tensor = field_tensor.to(self.device)
                    # Remove batch dimension if present [B, C, H, W] -> [C, H, W]
                    while field_tensor.dim() > 3:
                        field_tensor = field_tensor.squeeze(0)
                    # Ensure minimum 3 dimensions [C, H, W]
                    if field_tensor.dim() == 2:  # [H, W] -> [1, H, W]
                        field_tensor = field_tensor.unsqueeze(0)
                    field_tensors.append(field_tensor)
                
                # Concatenate all fields for this timestep [C_all, H, W]
                timestep_tensor = torch.cat(field_tensors, dim=0)
                timestep_tensors.append(timestep_tensor)
            
            # Stack timesteps to create [T, C_all, H, W]
            target_tensor = torch.stack(timestep_tensors, dim=0)
            
            # Move tensors to CPU for storage (DataLoader will handle device placement)
            input_tensor = input_tensor.cpu()
            target_tensor = target_tensor.cpu()
            
            tensor_predictions.append((input_tensor, target_tensor))
        
        logger.info(f"Generated {len(tensor_predictions)} physical predictions")
        return tensor_predictions
    
    def _generate_synthetic_predictions(self) -> List[Tuple]:
        """
        Generate predictions using synthetic model for physical training.
        
        Returns:
            List of (initial_fields, target_fields) tuples
        """
        logger.info("Generating predictions from synthetic model...")
        
        # Create tensor dataset for synthetic model
        tensor_dataset = self._create_hybrid_dataset(
            self.trainer_config["train_sim"]
        )
        
        # Generate predictions using the synthetic model's method
        batch_size = self.trainer_config.get("batch_size", 32)
        
        inputs_list, targets_list = self.synthetic_model.generate_predictions(
            real_dataset=tensor_dataset,
            alpha=self.alpha,
            device=str(self.device),
            batch_size=batch_size,
            num_workers=0,
        )
        
        # Convert to list of tuples: [(input1, target1), (input2, target2), ...]
        predictions = list(zip(inputs_list, targets_list))
        
        logger.info(f"Generated {len(predictions)} synthetic predictions")
        return predictions
    
    def _train_synthetic_with_augmentation(
        self,
        generated_data: List[Tuple]
    ) -> float:
        """
        Train synthetic model with augmented data (real + physical predictions).
        
        Args:
            generated_data: Physical model predictions as (input, target) tuples
            
        Returns:
            Final training loss
        """
        logger.info("Training synthetic model with augmented data...")
        
        # In the new architecture, augmentation is handled via augmentation_config
        # We create a TensorDataset with augmentation_config using 'memory' mode
        augmentation_config = {
            'mode': 'memory',
            'alpha': self.alpha,
            'data': generated_data,  # Pre-loaded augmented data
        }
        
        # Manually create the dataset with augmentation (can't use DataLoaderFactory
        # because we need to pass pre-generated data in memory)
        from torch.utils.data import DataLoader
        from src.config import ConfigHelper
        from src.data import DataManager
        
        cfg = ConfigHelper(self.config)
        
        # Create DataManager
        project_root = cfg.get_project_root()
        raw_data_dir = project_root / cfg.get_raw_data_dir()
        cache_dir = project_root / cfg.get_cache_dir()
        
        data_manager = DataManager(
            raw_data_dir=str(raw_data_dir),
            cache_dir=str(cache_dir),
            config=self.config,
        )
        
        # Get field specs
        dynamic_fields, static_fields = cfg.get_field_types()
        
        # Create TensorDataset with augmentation
        augmented_dataset = TensorDataset(
            data_manager=data_manager,
            sim_indices=self.trainer_config["train_sim"],
            field_names=cfg.get_field_names(),
            num_frames=None,  # Load all frames for sliding window
            num_predict_steps=cfg.get_num_predict_steps(),
            dynamic_fields=dynamic_fields,
            static_fields=static_fields,
            use_sliding_window=True,
            augmentation_config=augmentation_config,  # Pass generated data
        )
        
        # Create dataloader
        train_loader = DataLoader(
            augmented_dataset,
            batch_size=self.trainer_config.get("batch_size", 16),
            shuffle=True,
            num_workers=0,
        )
        
        # Train using synthetic trainer's train method
        result = self.synthetic_trainer.train(
            data_source=train_loader, 
            num_epochs=self.synthetic_epochs_per_cycle
        )
        
        # Get final loss from training result
        final_loss = result.get('final_loss', 0.0)
        
        logger.info(f"Synthetic training complete (loss: {final_loss:.6f})")
        return final_loss
    
    def _train_physical_with_augmentation(
        self,
        generated_data: List[Tuple]
    ) -> float:
        """
        Train physical model with augmented data (real + synthetic predictions).
        
        Args:
            generated_data: Synthetic model predictions as (initial, target) tuples
            
        Returns:
            Final training loss
        """
        logger.info("Training physical model with augmented data...")
        
        # Check if there are any learnable parameters
        if len(self.physical_trainer.learnable_params) == 0:
            logger.info("Skipping physical training - no learnable parameters")
            return 0.0
        
        # In the new architecture, create FieldDataset with augmentation
        augmentation_config = {
            'mode': 'memory',
            'alpha': self.alpha,
            'data': generated_data,  # Pre-loaded augmented data
        }
        
        # Create FieldDataset with augmentation
        from src.config import ConfigHelper
        from src.data import DataManager
        
        cfg = ConfigHelper(self.config)
        
        # Create DataManager
        project_root = cfg.get_project_root()
        raw_data_dir = project_root / cfg.get_raw_data_dir()
        cache_dir = project_root / cfg.get_cache_dir()
        
        data_manager = DataManager(
            raw_data_dir=str(raw_data_dir),
            cache_dir=str(cache_dir),
            config=self.config,
        )
        
        # Create FieldDataset with augmentation
        augmented_dataset = FieldDataset(
            data_manager=data_manager,
            sim_indices=self.trainer_config["train_sim"],
            field_names=cfg.get_field_names(),
            num_frames=None,  # Load all frames for sliding window
            num_predict_steps=cfg.get_num_predict_steps(),
            use_sliding_window=True,
            augmentation_config=augmentation_config,  # Pass generated data
        )
        
        # Physical trainer doesn't have a high-level train method yet
        # For now, manually iterate through samples
        total_loss = 0.0
        num_samples = min(len(augmented_dataset), self.physical_epochs_per_cycle * 10)  # Limit samples
        
        # Track initial and final parameter values for summary
        initial_params = {}
        if len(self.physical_trainer.learnable_params) > 0:
            param_names = self.physical_trainer.param_names
            for i, name in enumerate(param_names):
                initial_params[name] = float(self.physical_trainer.learnable_params[i])
        
        logger.info(f"Training on {num_samples} samples...")
        for i, (initial_fields, target_fields) in enumerate(augmented_dataset):
            if i >= num_samples:
                break
            sample_loss = self.physical_trainer._train_sample(initial_fields, target_fields)
            total_loss += sample_loss
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        
        # Log parameter summary at INFO level
        if len(self.physical_trainer.learnable_params) > 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"PHYSICAL TRAINING SUMMARY ({num_samples} samples)")
            logger.info(f"{'='*60}")
            logger.info(f"Average loss: {avg_loss:.6f}")
            logger.info(f"\nLearned Parameters:")
            for i, name in enumerate(param_names):
                initial_val = initial_params[name]
                final_val = float(self.physical_trainer.learnable_params[i])
                change = final_val - initial_val
                change_pct = (change / abs(initial_val) * 100) if abs(initial_val) > 1e-10 else 0
                logger.info(f"  {name}:")
                logger.info(f"    Initial: {initial_val:.6f}")
                logger.info(f"    Final:   {final_val:.6f}")
                logger.info(f"    Change:  {change:+.6f} ({change_pct:+.2f}%)")
                
                # Show true value if available
                if hasattr(self.physical_trainer.model, f"_true_{name}"):
                    true_val = float(getattr(self.physical_trainer.model, f"_true_{name}"))
                    error = abs(final_val - true_val)
                    rel_error = (error / abs(true_val) * 100) if abs(true_val) > 1e-10 else 0
                    logger.info(f"    True:    {true_val:.6f}")
                    logger.info(f"    Error:   {error:.6f} ({rel_error:.2f}%)")
            logger.info(f"{'='*60}\n")
        else:
            logger.info(f"Physical training complete (avg loss: {avg_loss:.6f})")
        
        return avg_loss
    
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
            checkpoint_path = checkpoint_path.parent / f"{checkpoint_path.stem}_hybrid_best{checkpoint_path.suffix}"
            torch.save(self.synthetic_model.state_dict(), checkpoint_path)
            logger.info(f"Saved best synthetic model (loss: {synthetic_loss:.6f})")
        
        # Save physical parameters if improved  
        if physical_loss < self.best_physical_loss:
            self.best_physical_loss = physical_loss
            # Physical model saving would go here
            logger.info(f"New best physical loss: {physical_loss:.6f}")
    
    def evaluate(self):
        """
        Evaluate both models on validation/test data.
        
        This can be called after training to assess final performance.
        """
        logger.info("\n" + "="*60)
        logger.info("HYBRID TRAINER EVALUATION")
        logger.info("="*60)
        
        # TODO: Implement evaluation logic
        # Could call evaluate() on both component trainers
        
        logger.info("Evaluation not yet fully implemented")
        pass
