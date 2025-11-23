"""Hybrid trainer alternating between physical and synthetic model training."""

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
from src.utils.logger import get_logger
from phi.math import Tensor
from phi.flow import Field
from src.data.dataset import Dataset, AccessPolicy

logger = get_logger(__name__)


class HybridTrainer(AbstractTrainer):
    """Alternates between physical rollout generation and synthetic model training."""

    def __init__(
        self,
        config: Dict[str, Any],
        synthetic_model: nn.Module,
        physical_model,
    ):
        """Initialize hybrid trainer with both models."""
        super().__init__(config)

        # Store models
        self.synthetic_model = synthetic_model
        self.physical_model = physical_model
        # Parse configuration
        self._parse_config(config)

        # Create single unified dataset
        self._base_dataset = None
        self._initialize_dataset()

        # Create component trainers
        self.synthetic_trainer = SyntheticTrainer(config, synthetic_model)
        self.physical_trainer = PhysicalTrainer(config, physical_model)

        # Training state
        self.current_cycle = 0
        self.best_synthetic_loss = float("inf")
        self.best_physical_loss = float("inf")

    def _parse_config(self, config: Dict[str, Any]):
        """Extract all configuration parameters."""
        self.config = config
        # Hybrid training parameters
        self.cycles = config['trainer']['hybrid']['cycles']
        self.warmup = config['trainer']['hybrid']['warmup']
        self.synthetic_epochs = config['trainer']['synthetic']['epochs']
        self.physical_epochs = config['trainer']['physical']['epochs']
        self.alpha = config['trainer']['hybrid']['augmentation']['alpha']
        self.real_data_access = config['trainer']['data_access']
        self.device = torch.device(config['trainer']['device'] if torch.cuda.is_available() else "cpu")

        # Data config
        self.train_sim = config['trainer']['train_sim']
        self.alpha = config['trainer']["hybrid"]["augmentation"]['alpha']
        # field_names will be set after dataset initialization
        self.field_names = None
        self.batch_size = config['trainer']['batch_size']
        self.trajectory_length = config['data']['trajectory_length']

        # Separate rollout steps for physical and synthetic models
        self.physical_rollout_steps = config['trainer']['physical'].get(
            'rollout_steps',
            config['trainer'].get('rollout_steps', 4)
        )
        self.synthetic_rollout_steps = config['trainer']['synthetic'].get(
            'rollout_steps',
            config['trainer'].get('rollout_steps', 4)
        )
        # Default rollout_steps for backward compatibility (uses synthetic)
        self.rollout_steps = self.synthetic_rollout_steps

        self.learnable_parameters = config['trainer']['physical']['learnable_parameters']


    def _initialize_dataset(self):
        """Create unified PhiML Dataset.

        Uses the maximum of physical and synthetic rollout steps to ensure
        enough target frames are available for both training phases.
        """
        # Use max rollout steps to ensure enough targets for both models
        max_rollout_steps = max(self.physical_rollout_steps, self.synthetic_rollout_steps)
        self._base_dataset = Dataset(
            config=self.config,
            train_sim=self.train_sim,
            rollout_steps=max_rollout_steps
        )
        # Get field_names from dataset (tensor shape is self-describing)
        self.field_names = list(self._base_dataset.field_names)
        logger.info(
            f"Initialized dataset with {len(self._base_dataset)} samples "
            f"(physical_rollout={self.physical_rollout_steps}, synthetic_rollout={self.synthetic_rollout_steps}), "
            f"fields={self.field_names}"
        )

    def train(self):
        """Main training loop."""
        if self.warmup > 0:
            self._run_warmup()
        
        pbar = tqdm(range(self.cycles), desc="Hybrid Cycles", unit="cycle")
        
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
        logger.info(f"Running warmup for {self.warmup} epochs...")

        # Set dataset to use only real data
        self._base_dataset.access_policy = AccessPolicy.REAL_ONLY

        # Train synthetic model directly with dataset
        self.synthetic_trainer.train(
            dataset=self._base_dataset,
            num_epochs=self.warmup,
            verbose=True
        )

        self._clear_gpu_memory()

    def _run_cycle(self) -> Tuple[float, float]:
        """Execute one complete hybrid training cycle."""
        logger.debug(f"\n{'='*60}")
        logger.debug(f"CYCLE {self.current_cycle + 1}/{self.cycles}")
        logger.debug(f"{'='*60}")

        # Phase 1: Generate physical trajectories (as Field rollouts)
        logger.debug("Phase 1: Generating physical trajectories...")
        physical_rollouts = self._generate_physical_rollouts()

        # Phase 2: Add to dataset and train synthetic model
        logger.debug("Phase 2: Training synthetic model on physical trajectories...")
        self._base_dataset.set_augmented_trajectories(physical_rollouts)
        synthetic_loss = self._train_synthetic_model()

        # Phase 3: Generate synthetic predictions (as tensor trajectories)
        logger.debug("Phase 3: Generating synthetic predictions...")
        synthetic_predictions = self._generate_synthetic_predictions()

        # Phase 4: Add to dataset and train physical model
        logger.debug("Phase 4: Training physical model on synthetic predictions...")
        self._base_dataset.set_augmented_trajectories(synthetic_predictions)
        physical_loss = self._train_physical_model()

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
            
            samples_per_trajectory = self.trajectory_length - self.rollout_steps
            num_trajectories = max(1, (num_synthetic_samples + samples_per_trajectory - 1) // samples_per_trajectory)
            
            # Generate batched rollout
            initial_state = self.physical_model.get_initial_state(batch_size=num_trajectories)
            rollout = self.physical_model.rollout(initial_state, num_steps=self.trajectory_length)
            
            # Split into list of individual trajectories
            rollouts = []
            
            for traj_idx in range(num_trajectories):
                trajectory = {}
                for field_name in self.field_names:
                    trajectory[field_name] = rollout[field_name].batch[traj_idx]
                rollouts.append(trajectory)
            
            logger.debug(f"  Generated {len(rollouts)} physical trajectories")
            return rollouts
    
    # ==================== PHASE 2: SYNTHETIC MODEL TRAINING ====================

    def _train_synthetic_model(self) -> float:
        """
        Train synthetic model on real + physical trajectories.

        Dataset handles windowing of physical trajectories internally.
        """
        with self.managed_memory_phase("Synthetic Training", clear_cache=False):
            # Set access policy
            access_policy = self._get_access_policy(for_synthetic=True)
            self._base_dataset.access_policy = access_policy

            logger.debug(
                f"  Dataset: {self._base_dataset.num_real_samples} real + "
                f"{len(self._base_dataset.augmented_samples)} augmented = "
                f"{len(self._base_dataset)} total samples"
            )

            # Train directly with dataset
            result = self.synthetic_trainer.train(
                dataset=self._base_dataset,
                num_epochs=self.synthetic_epochs,
                verbose=False
            )

            logger.debug(f"  Synthetic loss: {result['final_loss']:.6f}")
            return result['final_loss']
    
    # ==================== PHASE 3: SYNTHETIC PREDICTION GENERATION ====================

    def _generate_synthetic_predictions(self) -> List[Dict[str, Tensor]]:
        """
        Generate synthetic predictions using autoregressive rollout.

        Uses the same number of trajectories as physical model generated,
        starting from initial states in the dataset.

        Returns:
            List of prediction dicts with PhiML tensors: [{'field_name': Tensor[time, x, y, vector]}, ...]
        """
        with self.managed_memory_phase("Synthetic Prediction"):
            from phi.math import math

            # Calculate how many predictions to generate
            num_real_samples = self._base_dataset.num_real_samples
            num_synthetic_samples = int(num_real_samples * self.alpha)
            samples_per_trajectory = self.trajectory_length - self.rollout_steps
            num_trajectories = max(1, (num_synthetic_samples + samples_per_trajectory - 1) // samples_per_trajectory)

            logger.debug(f"  Generating {num_trajectories} synthetic prediction trajectories")

            predictions = []

            # Generate predictions starting from random initial states from dataset
            for _ in range(num_trajectories):
                # Get a random sample from the dataset
                sample_idx = torch.randint(0, self._base_dataset.num_real_samples, (1,)).item()
                sample = self._base_dataset._get_sample(sample_idx)
                current_state = sample.initial_state  # Dict[field_name, Tensor]

                # Autoregressive rollout using synthetic model
                # Collect trajectory as list of dicts per timestep
                trajectory_steps = []

                for step in range(self.trajectory_length):
                    # Call model with dict of fields (matches dataset format!)
                    next_state = self.synthetic_model(current_state)
                    trajectory_steps.append(next_state)
                    current_state = next_state

                # Stack each field's trajectory along time dimension
                trajectory_dict = {}
                for field_name in self.field_names:
                    # Stack this field across all timesteps: [time, x, y, vector]
                    trajectory_dict[field_name] = math.stack(
                        [step[field_name] for step in trajectory_steps],
                        math.batch('time')
                    )

                predictions.append(trajectory_dict)

            logger.debug(f"  Generated {len(predictions)} synthetic predictions")
            return predictions


    def _train_physical_model(self) -> float:
        """
        Train physical model on real + synthetic predictions.

        Dataset already has augmented predictions set, just need to train.
        """
        if len(self.physical_trainer.learnable_parameters) == 0:
            logger.info("No learnable parameters, skipping physical training")
            return 0.0

        with self.managed_memory_phase("Physical Training", clear_cache=False):
            # Set access policy
            access_policy = self._get_access_policy(for_synthetic=False)
            self._base_dataset.access_policy = access_policy

            logger.debug(
                f"  Dataset: {self._base_dataset.num_real_samples} real + "
                f"{len(self._base_dataset.augmented_samples)} augmented = "
                f"{len(self._base_dataset)} total samples"
            )

            # Train directly with dataset
            result = self.physical_trainer.train(
                dataset=self._base_dataset,
                num_epochs=self.physical_epochs,
                verbose=False
            )

            logger.debug(f"  Physical loss: {result['final_loss']:.6f}")
            return float(result['final_loss'])
        
    # ==================== UTILITIES ====================

    def _calculate_num_real_samples(self) -> int:
        """
        Calculate number of truly real samples (excluding any augmentation).

        Returns the count of original simulation data samples.
        """
        return self._base_dataset.num_real_samples
    
    def _get_access_policy(self, for_synthetic: bool) -> AccessPolicy:
        """Determine data access policy based on configuration."""
        if for_synthetic:
            # Synthetic model training
            if self.real_data_access in ["both", "synthetic_only"]:
                return AccessPolicy.BOTH  # Use real + physical trajectories
            else:
                return AccessPolicy.GENERATED_ONLY  # Only physical trajectories
        else:
            # Physical model training
            if self.real_data_access in ["both", "physical_only"]:
                return AccessPolicy.BOTH  # Use real + synthetic predictions
            else:
                return AccessPolicy.GENERATED_ONLY  # Only synthetic predictions
    
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
            self.synthetic_trainer.save_checkpoint(epoch=self.current_cycle, loss=synthetic_loss)
            logger.debug(f"Saved best synthetic model: loss={synthetic_loss:.6f}")

        if physical_loss < self.best_physical_loss:
            self.best_physical_loss = physical_loss
            self.physical_trainer.save_checkpoint()
            logger.debug(f"Saved best physical model: loss={physical_loss:.6f}")