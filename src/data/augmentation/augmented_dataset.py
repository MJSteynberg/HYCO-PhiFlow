"""
Augmented Dataset Classes for Hybrid Training

Implements count-based data augmentation by combining real simulation data
with model-generated predictions.

Key Design Principles:
1. Count-based weighting: Generate int(len(real) * alpha) samples
2. All samples have weight = 1.0 (NO weight-based scaling)
3. Random shuffling of real and generated samples
4. No weights returned from __getitem__ (2-tuple only)
"""

from typing import List, Tuple, Dict, Optional, Any
import torch
from torch.utils.data import Dataset
from phi.field import Field

from src.data import HybridDataset
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AugmentedTensorDataset(Dataset):
    """
    Combines real tensor data with generated predictions for synthetic training.
    
    This dataset implements count-based augmentation where a proportional number
    of generated samples are added to the real data. All samples have equal weight
    (1.0) to avoid double-scaling in loss calculation.
    
    Key Features:
    - Count-based: generates int(len(real) * alpha) samples
    - All samples have weight = 1.0 (no weight-based scaling)
    - Random shuffling via DataLoader
    - Returns 2-tuple (input, target) - NO weights!
    
    Args:
        real_dataset: HybridDataset with return_fields=False (tensors)
        generated_data: List of (input_tensor, target_tensor) tuples
        alpha: Proportion for generated data (e.g., 0.1 = 10%)
        validate_count: If True, validates generated count matches expected
        
    Returns:
        Tuple of (input_tensor, target_tensor) - no weights!
        
    Example:
        >>> real_dataset = HybridDataset(..., return_fields=False)
        >>> generated_data = generate_physical_predictions(...)
        >>> augmented = AugmentedTensorDataset(real_dataset, generated_data, alpha=0.1)
        >>> loader = DataLoader(augmented, batch_size=32, shuffle=True)
        >>> for input_batch, target_batch in loader:
        ...     # Train with equal weighting for all samples
        ...     loss = loss_fn(model(input_batch), target_batch)
    """
    
    def __init__(
        self,
        real_dataset: HybridDataset,
        generated_data: List[Tuple[torch.Tensor, torch.Tensor]],
        alpha: float = 0.1,
        validate_count: bool = True,
        device: str = 'cpu',
    ):
        """Initialize augmented tensor dataset.
        
        Args:
            real_dataset: HybridDataset with real samples
            generated_data: List of (input, target) tensor tuples
            alpha: Expected proportion of generated samples
            validate_count: Whether to validate generated count
            device: Device to move tensors to ('cpu' or 'cuda')
        """
        self.real_dataset = real_dataset
        self.generated_data = generated_data
        self.alpha = alpha
        self.device = device
        
        self.num_real = len(real_dataset)
        self.num_generated = len(generated_data)
        
        # Validate count-based generation
        expected_count = int(self.num_real * alpha)
        if validate_count and abs(self.num_generated - expected_count) > 1:
            logger.warning(
                f"Generated count mismatch: expected ~{expected_count}, "
                f"got {self.num_generated}. "
                f"Count-based weighting requires proportional generation."
            )
        
        logger.info(f"AugmentedTensorDataset created:")
        logger.info(f"  Real samples: {self.num_real}")
        logger.info(f"  Generated samples: {self.num_generated}")
        logger.info(f"  Total samples: {len(self)}")
        logger.info(f"  Alpha: {alpha:.2f}")
        logger.info(f"  Expected generated: {expected_count}")
        
    def __len__(self) -> int:
        """Return total number of samples (real + generated)."""
        return self.num_real + self.num_generated
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get sample by index.
        
        Args:
            idx: Index in range [0, len(self))
            
        Returns:
            Tuple of (input_tensor, target_tensor)
            NO WEIGHT! All samples have implicit weight = 1.0
            
        Note:
            - First num_real indices return real samples
            - Remaining indices return generated samples
            - Shuffling handled by DataLoader, not here
            - All tensors moved to self.device
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        
        if idx < self.num_real:
            # Real sample from HybridDataset
            input_tensor, target_tensor = self.real_dataset[idx]
            # Move to device if needed
            if self.device != 'cpu':
                input_tensor = input_tensor.to(self.device)
                target_tensor = target_tensor.to(self.device)
            return input_tensor, target_tensor
        else:
            # Generated sample (already on correct device)
            gen_idx = idx - self.num_real
            return self.generated_data[gen_idx]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with statistics about the dataset
        """
        return {
            'num_real': self.num_real,
            'num_generated': self.num_generated,
            'total': len(self),
            'alpha': self.alpha,
            'real_ratio': self.num_real / len(self),
            'generated_ratio': self.num_generated / len(self),
        }


class AugmentedFieldDataset:
    """
    Combines real field data with generated predictions for physical training.
    
    This is NOT a PyTorch Dataset - it provides an iterator interface suitable
    for FieldTrainer which processes samples one at a time.
    
    Key Features:
    - Count-based: generates int(len(real) * alpha) samples
    - All samples have weight = 1.0 (no weight-based scaling)
    - Iterator interface (not indexable)
    - Returns 2-tuple (initial_fields, target_fields) - NO weights!
    
    Args:
        real_dataset: HybridDataset with return_fields=True (PhiFlow Fields)
        generated_data: List of (initial_fields, target_fields) tuples
        alpha: Proportion for generated data
        shuffle: If True, shuffle samples before iteration
        validate_count: If True, validates generated count matches expected
        
    Yields:
        Tuple of (initial_fields, target_fields) - no weights!
        
    Example:
        >>> real_dataset = HybridDataset(..., return_fields=True)
        >>> generated_data = generate_synthetic_predictions(...)
        >>> augmented = AugmentedFieldDataset(real_dataset, generated_data, alpha=0.1)
        >>> for initial_fields, target_fields in augmented:
        ...     # Train with equal weighting for all samples
        ...     loss = compute_loss(model, initial_fields, target_fields)
    """
    
    def __init__(
        self,
        real_dataset: HybridDataset,
        generated_data: List[Tuple[Dict[str, Field], Dict[str, List[Field]]]],
        alpha: float = 0.1,
        shuffle: bool = True,
        validate_count: bool = True,
    ):
        """Initialize augmented field dataset."""
        self.real_dataset = real_dataset
        self.generated_data = generated_data
        self.alpha = alpha
        self.shuffle = shuffle
        
        self.num_real = len(real_dataset)
        self.num_generated = len(generated_data)
        
        # Validate count-based generation
        expected_count = int(self.num_real * alpha)
        if validate_count and abs(self.num_generated - expected_count) > 1:
            logger.warning(
                f"Generated count mismatch: expected ~{expected_count}, "
                f"got {self.num_generated}. "
                f"Count-based weighting requires proportional generation."
            )
        
        logger.info(f"AugmentedFieldDataset created:")
        logger.info(f"  Real samples: {self.num_real}")
        logger.info(f"  Generated samples: {self.num_generated}")
        logger.info(f"  Total samples: {len(self)}")
        logger.info(f"  Alpha: {alpha:.2f}")
        logger.info(f"  Shuffle: {shuffle}")
        
    def __len__(self) -> int:
        """Return total number of samples (real + generated)."""
        return self.num_real + self.num_generated
    
    def __iter__(self):
        """
        Iterate over all samples (real + generated).
        
        Yields:
            Tuple of (initial_fields, target_fields)
            NO WEIGHT! All samples have implicit weight = 1.0
            
        Note:
            - Yields real samples first, then generated
            - If shuffle=True, samples are shuffled before iteration
        """
        # Create index list
        indices = list(range(len(self)))
        
        if self.shuffle:
            import random
            random.shuffle(indices)
        
        # Yield samples in order
        for idx in indices:
            if idx < self.num_real:
                # Real sample from HybridDataset
                yield self.real_dataset[idx]
            else:
                # Generated sample
                gen_idx = idx - self.num_real
                yield self.generated_data[gen_idx]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with statistics about the dataset
        """
        return {
            'num_real': self.num_real,
            'num_generated': self.num_generated,
            'total': len(self),
            'alpha': self.alpha,
            'real_ratio': self.num_real / len(self),
            'generated_ratio': self.num_generated / len(self),
        }
