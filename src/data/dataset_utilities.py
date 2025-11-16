"""
Dataset Utilities - Shared Components

Extracted from AbstractDataset to reduce complexity:
- DatasetBuilder: Setup and validation
- FilteringManager: Percentage filtering and resampling
"""

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import random
import torch
from phi.field import Field

from .data_manager import DataManager
from .augmentation_manager import AugmentationHandler
from src.utils.logger import get_logger

logger = get_logger(__name__)


from .abstract_dataset import AbstractDataset


# DatasetBuilder removed; use AbstractDataset.setup_cache and
# AbstractDataset.compute_sliding_window directly instead.


# AugmentationHandler moved to augmentation_manager.py


# FilteringManager moved to abstract_dataset.py



# FieldMetadata and helpers moved to field_dataset.py; import directly from there if needed

    

from phi.torch.flow import stack, batch
from typing import List, Dict, Tuple
from phi.field import Field

# In dataset_utilities.py or dataloader_factory.py

def tensor_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Collate per-sample tensors [V, T, H, W] into batches [B, V, T, H, W].
    
    Args:
        batch: List of (initial, targets) tuples, each [V, T, H, W]
    
    Returns:
        (batched_initial, batched_targets) each [B, V, T, H, W]
    """
    initials, targets = zip(*batch)

    logger.info(f"Device tensor_collate_fn: batching {len(batch)} samples on device {initials[0].device, targets[0].device}, ")
    
    # Stack along new batch dimension
    batched_initial = torch.stack(initials, dim=0)  # [B, V, T_init, H, W]
    batched_targets = torch.stack(targets, dim=0)   # [B, V, T_pred, H, W]
    
    return batched_initial, batched_targets


def field_collate_fn(field_batch: List[Tuple[Dict[str, Field], Dict[str, List[Field]]]]):
    """
    Collate per-sample Fields into batched Fields.
    
    Args:
        batch: List of (initial_fields, target_fields) tuples
    
    Returns:
        (batched_initial, batched_targets) with batch dimension in Fields
    """

    initial_fields, target_fields = zip(*field_batch)
    
    # Stack initial fields
    field_names = initial_fields[0].keys()
    stacked_initial = {}
    stacked_targets = {}
    for name in field_names:
        initial = [sample[name] for sample in initial_fields]
        targets = [sample[name] for sample in target_fields]
        stacked_initial[name] = stack(initial, batch('batch'))
        stacked_targets[name] = stack(targets, batch('batch'))
  
    return stacked_initial, stacked_targets
