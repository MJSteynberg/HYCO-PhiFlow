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

from src.training.abstract_trainer import AbstractTrainer
from src.training.synthetic.trainer import SyntheticTrainer
from src.training.physical.trainer import PhysicalTrainer
from src.factories.dataloader_factory import DataLoaderFactory
from src.data import TensorDataset, FieldDataset
from src.utils.logger import get_logger
# === 2. Batched conversion ===
from src.utils.field_conversion import make_converter
from src.config import ConfigHelper
import torch

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
        self.synthetic_epochs_per_cycle = self.hybrid_config.get(
            "synthetic_epochs_per_cycle", 5
        )
        self.physical_epochs_per_cycle = self.hybrid_config.get(
            "physical_epochs_per_cycle", 3
        )
        self.alpha = self.aug_config.get("alpha", 0.1)
        self.warmup_synthetic_epochs = self.hybrid_config.get(
            "warmup_synthetic_epochs", 10
        )

        # Data access control: which models are allowed to see real data
        # Options: 'both', 'synthetic_only', 'physical_only', 'neither'
        self.real_data_access = self.hybrid_config.get("real_data_access", "both")

        # Validate real_data_access parameter
        valid_options = ["both", "synthetic_only", "physical_only", "neither"]
        if self.real_data_access not in valid_options:
            raise ValueError(
                f"Invalid real_data_access: '{self.real_data_access}'. "
                f"Must be one of {valid_options}"
            )

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Hybrid trainer using device: {self.device}")

        # Suppress sub-trainer logging during hybrid training
        # Save original setting and set suppression flag
        self._original_suppress_logs = config.get("trainer_params", {}).get(
            "suppress_training_logs", False
        )
        config["trainer_params"]["suppress_training_logs"] = True

        # Reduce logging verbosity for data/model modules during hybrid training
        # This needs to be done early to suppress warmup logs
        import logging

        logging.getLogger("data.abstract_dataset").setLevel(logging.WARNING)
        logging.getLogger("factories.dataloader_factory").setLevel(logging.WARNING)
        logging.getLogger("physical.base").setLevel(logging.WARNING)
        logging.getLogger("src.models.synthetic.base").setLevel(logging.WARNING)

        # Create component trainers
        self.synthetic_trainer = SyntheticTrainer(config, synthetic_model)
        self.physical_trainer = PhysicalTrainer(
            config, physical_model, learnable_params
        )

        # Note: CacheManager removed in new architecture
        # Augmentation is now handled directly by TensorDataset/FieldDataset
        # via augmentation_config parameter

        # Training state
        self.current_cycle = 0
        self.best_synthetic_loss = float("inf")
        self.best_physical_loss = float("inf")

        logger.info("=" * 60)
        logger.info("HYBRID TRAINER INITIALIZED")
        logger.info(
            f"  Cycles: {self.num_cycles}, Synthetic: {self.synthetic_epochs_per_cycle}e/c, Physical: {self.physical_epochs_per_cycle}e/c, Warmup: {self.warmup_synthetic_epochs}e"
        )
        logger.info(f"  Alpha: {self.alpha}, Real data access: {self.real_data_access}")
        logger.info("=" * 60)

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
        logger.info("=" * 60)
        logger.info("HYBRID TRAINING")
        logger.info("=" * 60)

        # Optional warmup phase
        if self.warmup_synthetic_epochs > 0:
            logger.info(f"Warmup: {self.warmup_synthetic_epochs} epoch(s)")
            self._warmup_synthetic()

        # Main hybrid training loop with tqdm progress bar
        pbar = tqdm(range(self.num_cycles), desc="Hybrid Cycles", unit="cycle")

        for cycle in pbar:
            self.current_cycle = cycle

            # Phase 1: Synthetic training with physical predictions
            physical_preds = self._generate_physical_predictions()
            synthetic_loss = self._train_synthetic_with_augmentation(physical_preds)

            # Phase 2: Physical training with synthetic predictions
            synthetic_preds = self._generate_synthetic_predictions()
            physical_loss = self._train_physical_with_augmentation(synthetic_preds)

            # Update progress bar with losses
            pbar.set_postfix(
                {
                    "syn_loss": f"{synthetic_loss:.6f}",
                    "phy_loss": f"{physical_loss:.6f}",
                }
            )

            # Save checkpoints if improved
            self._save_if_best(synthetic_loss, physical_loss)

        # Print final summary
        self._print_training_summary()

    def _print_training_summary(self):
        """Print comprehensive training summary including optimized parameters."""
        logger.info("\n" + "=" * 60)
        logger.info("HYBRID TRAINING COMPLETE")
        logger.info("=" * 60)

        # Loss summary
        logger.info("Loss Summary:")
        logger.info(f"  Best Synthetic Loss: {self.best_synthetic_loss:.6f}")
        logger.info(f"  Best Physical Loss:  {self.best_physical_loss:.6f}")

        # Physical model parameters summary
        if len(self.physical_trainer.learnable_params) > 0:
            logger.info("\nOptimized Physical Parameters:")
            param_names = self.physical_trainer.param_names

            for i, name in enumerate(param_names):
                final_val = float(self.physical_trainer.learnable_params[i])

                # Get initial value from config
                learnable_params_config = self.trainer_config.get(
                    "learnable_parameters", []
                )
                initial_val = None
                for param_config in learnable_params_config:
                    if param_config["name"] == name:
                        initial_val = param_config.get("initial_guess", 1.0)
                        break

                if initial_val is not None:
                    change = final_val - initial_val
                    change_pct = (
                        (change / abs(initial_val) * 100)
                        if abs(initial_val) > 1e-10
                        else 0
                    )
                    logger.info(f"  {name}:")
                    logger.info(f"    Initial:  {initial_val:.6f}")
                    logger.info(f"    Final:    {final_val:.6f}")
                    logger.info(f"    Change:   {change:+.6f} ({change_pct:+.2f}%)")
                else:
                    logger.info(f"  {name}: {final_val:.6f}")

                # Show true value if available (for validation/debugging)
                if hasattr(self.physical_trainer.model, f"_true_{name}"):
                    true_val = float(
                        getattr(self.physical_trainer.model, f"_true_{name}")
                    )
                    error = abs(final_val - true_val)
                    rel_error = (
                        (error / abs(true_val) * 100) if abs(true_val) > 1e-10 else 0
                    )
                    logger.info(f"    True:     {true_val:.6f}")
                    logger.info(f"    Error:    {error:.6f} ({rel_error:.2f}%)")

        logger.info("=" * 60)

    def _warmup_synthetic(self):
        """
        Warm up the synthetic model with standard training (no augmentation).

        This gives the synthetic model a head start before hybrid training begins.
        """
        logger.debug(
            f"Warmup: training synthetic model ({self.warmup_synthetic_epochs} epochs)"
        )

        # Create standard dataset (no augmentation)
        # Use TensorDataset directly for the base data
        base_dataset = self._create_hybrid_dataset(self.trainer_config["train_sim"])

        # Create dataloader
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            base_dataset,
            batch_size=self.trainer_config.get("batch_size", 16),
            shuffle=True,
        )

        # Train synthetic model
        self.synthetic_trainer.train(
            data_source=dataloader, num_epochs=self.warmup_synthetic_epochs
        )

    def _create_hybrid_dataset(
        self, sim_indices: List[int], return_fields: bool = False
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
            use_sliding_window=True,
            enable_augmentation=False,  # We handle augmentation separately in hybrid training
            batch_size=(
                None if return_fields else self.trainer_config.get("batch_size", 16)
            ),
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
        t1 = time.time()
        # Create field dataset for physical model (returns PhiFlow Fields)
        field_dataset = self._create_hybrid_dataset(
            self.trainer_config["train_sim"],
            return_fields=True,  # Physical model needs Fields not tensors
        )

        num_predict_steps = self.trainer_config["num_predict_steps"]

        t3 = time.time()
        # === 1. Batched rollout (already tested above) ===
        initial_fields, predictions = self.physical_model.generate_predictions(
            real_dataset=field_dataset,
            alpha=self.alpha,
            num_rollout_steps=num_predict_steps,
        )
        t4 = time.time()


        cfg = ConfigHelper(self.config)
        field_names_input = field_dataset.field_names  # e.g. ['density', 'velocity']
        field_names_target = cfg.get_field_names()     # same order for targets

        t5 = time.time()
        # Pre-create converters for all input and target fields
        input_converters = {name: make_converter(initial_fields[name]) for name in field_names_input}
        target_converters = {name: make_converter(predictions[0][name]) for name in field_names_target}

        # Batched input tensor: [B, C_all, H, W]
        batched_input = torch.cat(
            [input_converters[name].field_to_tensor(initial_fields[name], ensure_cpu=False)
            for name in field_names_input],
            dim=1
        )

        # Batched targets: [B, T, C_all, H, W]
        batched_targets = torch.stack([
            torch.cat([
                target_converters[name].field_to_tensor(pred_t[name], ensure_cpu=False)
                for name in field_names_target
            ], dim=1)
            for pred_t in predictions
        ], dim=1)

        # === 3. Split batch into individual samples for return ===
        B = batched_input.shape[0]
        tensor_predictions = []
        for i in range(B):
            input_tensor = batched_input[i].cpu()            # [C_all, H, W]
            target_tensor = batched_targets[i].cpu()         # [T, C_all, H, W]
            tensor_predictions.append((input_tensor, target_tensor))

        t6 = time.time()

        t2 = time.time()
        print("Time taken for one run: ", t2 - t1)
        print("Time taken for batched rollout: ", t4 - t3)
        print("Time taken for tensor conversion: ", t6 - t5)
        input("Pausing here...")
        return tensor_predictions


    def _generate_synthetic_predictions(self) -> List[Tuple]:
        """
        Generate predictions using synthetic model for physical training.

        Returns:
            List of (initial_fields, target_fields) tuples
        """
        # Create tensor dataset for synthetic model
        tensor_dataset = self._create_hybrid_dataset(self.trainer_config["train_sim"])

        # Generate predictions using the synthetic model's method
        batch_size = self.trainer_config.get("batch_size", 32)

        inputs_list, targets_list = self.synthetic_model.generate_predictions(
            real_dataset=tensor_dataset,
            alpha=self.alpha,
            device=str(self.device),
            batch_size=batch_size,
        )

        # Convert to list of tuples: [(input1, target1), (input2, target2), ...]
        predictions = list(zip(inputs_list, targets_list))

        return predictions

    def _train_synthetic_with_augmentation(self, generated_data: List[Tuple]) -> float:
        """
        Train synthetic model with data access controlled by dataset.

        The dataset's access_policy parameter controls whether the synthetic model
        sees real data, generated data, or both.

        Args:
            generated_data: Physical model predictions as (input, target) tuples

        Returns:
            Final training loss
        """
        # Determine access policy based on real_data_access config
        # 'both' or 'synthetic_only' -> synthetic sees both real and generated
        # 'physical_only' or 'neither' -> synthetic sees only generated
        if self.real_data_access in ["both", "synthetic_only"]:
            access_policy = "both"
        else:
            access_policy = "generated_only"

        # Prepare augmentation config
        augmentation_config = {
            "mode": "memory",
            "alpha": self.alpha,
            "data": generated_data,  # Pre-loaded augmented data
        }

        # Setup dataset creation
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

        # Create TensorDataset with augmentation and access policy
        augmented_dataset = TensorDataset(
            data_manager=data_manager,
            sim_indices=self.trainer_config["train_sim"],
            field_names=cfg.get_field_names(),
            num_frames=None,  # Load all frames for sliding window
            num_predict_steps=cfg.get_num_predict_steps(),
            dynamic_fields=dynamic_fields,
            static_fields=static_fields,
            use_sliding_window=True,
            augmentation_config=augmentation_config,
            access_policy=access_policy,  # Dataset controls data access
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
            data_source=train_loader, num_epochs=self.synthetic_epochs_per_cycle
        )

        # Get final loss from training result
        final_loss = result.get("final_loss", 0.0)

        return final_loss

    def _train_physical_with_augmentation(self, generated_data: List[Tuple]) -> float:
        """
        Train physical model with data access controlled by dataset.

        The dataset's access_policy parameter controls whether the physical model
        sees real data, generated data, or both.

        Args:
            generated_data: Synthetic model predictions as (initial, target) tuples

        Returns:
            Final training loss
        """
        # Check if there are any learnable parameters
        if len(self.physical_trainer.learnable_params) == 0:
            return 0.0

        # Determine access policy based on real_data_access config
        # 'both' or 'physical_only' -> physical sees both real and generated
        # 'synthetic_only' or 'neither' -> physical sees only generated
        if self.real_data_access in ["both", "physical_only"]:
            access_policy = "both"
        else:
            access_policy = "generated_only"

        # Prepare augmentation config
        augmentation_config = {
            "mode": "memory",
            "alpha": self.alpha,
            "data": generated_data,  # Pre-loaded augmented data
        }

        # Create FieldDataset with augmentation and access policy
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

        # Create FieldDataset with augmentation and access policy
        augmented_dataset = FieldDataset(
            data_manager=data_manager,
            sim_indices=self.trainer_config["train_sim"],
            field_names=cfg.get_field_names(),
            num_frames=None,  # Load all frames for sliding window
            num_predict_steps=cfg.get_num_predict_steps(),
            use_sliding_window=True,
            augmentation_config=augmentation_config,
            access_policy=access_policy,  # Dataset controls data access
        )

        # Physical trainer doesn't have a high-level train method yet
        # For now, manually iterate through samples
        total_loss = 0.0
        num_samples = min(
            len(augmented_dataset), self.physical_epochs_per_cycle * 10
        )  # Limit samples

        # Track initial and final parameter values for summary
        initial_params = {}
        if len(self.physical_trainer.learnable_params) > 0:
            param_names = self.physical_trainer.param_names
            for i, name in enumerate(param_names):
                initial_params[name] = float(self.physical_trainer.learnable_params[i])

        logger.debug(f"Training on {num_samples} samples...")
        for i, (initial_fields, target_fields) in enumerate(augmented_dataset):
            if i >= num_samples:
                break
            sample_loss = self.physical_trainer._train_sample(
                initial_fields, target_fields
            )
            total_loss += sample_loss

        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0

        # Log parameter summary at DEBUG level
        if len(self.physical_trainer.learnable_params) > 0:
            logger.debug(f"\n{'='*60}")
            logger.debug(f"PHYSICAL TRAINING SUMMARY ({num_samples} samples)")
            logger.debug(f"{'='*60}")
            logger.debug(f"Average loss: {avg_loss:.6f}")
            logger.debug(f"\nLearned Parameters:")
            for i, name in enumerate(param_names):
                initial_val = initial_params[name]
                final_val = float(self.physical_trainer.learnable_params[i])
                change = final_val - initial_val
                change_pct = (
                    (change / abs(initial_val) * 100) if abs(initial_val) > 1e-10 else 0
                )
                logger.debug(
                    f"  {name}: {initial_val:.6f} -> {final_val:.6f} ({change:+.6f}, {change_pct:+.2f}%)"
                )

                # Show true value if available
                if hasattr(self.physical_trainer.model, f"_true_{name}"):
                    true_val = float(
                        getattr(self.physical_trainer.model, f"_true_{name}")
                    )
                    error = abs(final_val - true_val)
                    rel_error = (
                        (error / abs(true_val) * 100) if abs(true_val) > 1e-10 else 0
                    )
                    logger.debug(
                        f"    True: {true_val:.6f}, Error: {error:.6f} ({rel_error:.2f}%)"
                    )
            logger.debug(f"{'='*60}\n")

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
            checkpoint_path = (
                checkpoint_path.parent
                / f"{checkpoint_path.stem}_hybrid_best{checkpoint_path.suffix}"
            )
            torch.save(self.synthetic_model.state_dict(), checkpoint_path)

        # Save physical parameters if improved
        if physical_loss < self.best_physical_loss:
            self.best_physical_loss = physical_loss

    def evaluate(self):
        """
        Evaluate both models on validation/test data.

        This can be called after training to assess final performance.
        """
        logger.info("\n" + "=" * 60)
        logger.info("HYBRID TRAINER EVALUATION")
        logger.info("=" * 60)

        # TODO: Implement evaluation logic
        # Could call evaluate() on both component trainers

        logger.info("Evaluation not yet fully implemented")
        pass
