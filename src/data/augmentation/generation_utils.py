"""
Prediction generation utilities for hybrid training augmentation.

This module provides functions to generate predictions using trained models
for data augmentation during hybrid training.
"""

import logging
import torch
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_synthetic_predictions(
    model: torch.nn.Module,
    real_dataset: Dataset,
    alpha: float,
    device: str = 'cpu',
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Generate predictions using a synthetic model for data augmentation.
    
    This function takes a trained synthetic model and generates predictions
    on real data samples. The number of predictions is proportional to alpha.
    
    Count-based generation: num_generated = int(len(real_dataset) * alpha)
    
    Args:
        model: Trained synthetic model (expects tensors as input/output)
        real_dataset: Dataset of real samples (returns (input, target) tuples)
        alpha: Proportion of generated samples (e.g., 0.1 = 10%)
        device: Device to run model on ('cpu' or 'cuda')
        batch_size: Batch size for generation
        num_workers: Number of DataLoader workers
        
    Returns:
        Tuple of (inputs_list, targets_list) where:
        - inputs_list: List of input tensors
        - targets_list: List of predicted target tensors
        
    Example:
        >>> model = load_trained_model("checkpoint.pt")
        >>> dataset = HybridDataset(...)
        >>> inputs, targets = generate_synthetic_predictions(
        ...     model=model,
        ...     real_dataset=dataset,
        ...     alpha=0.1,
        ...     device='cuda',
        ...     batch_size=32
        ... )
        >>> print(f"Generated {len(inputs)} predictions")
    """
    model.eval()
    model.to(device)
    
    # Calculate number of samples to generate
    num_real = len(real_dataset)
    num_generate = int(num_real * alpha)
    
    logger.info(
        f"Generating {num_generate} synthetic predictions "
        f"(alpha={alpha:.2f} * {num_real} real samples)"
    )
    
    if num_generate == 0:
        logger.warning("Alpha too small, no samples will be generated")
        return [], []
    
    # Create subset of real dataset for generation
    # Use proportional sampling to get diverse samples
    indices = _select_proportional_indices(num_real, num_generate)
    subset = torch.utils.data.Subset(real_dataset, indices)
    
    # Create DataLoader
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # Generate predictions
    inputs_list = []
    targets_list = []
    
    with torch.no_grad():
        for batch_inputs, _ in loader:
            # Move to device
            batch_inputs = batch_inputs.to(device)
            
            # Generate predictions
            batch_predictions = model(batch_inputs)
            
            # Store results (move back to CPU)
            for i in range(batch_inputs.shape[0]):
                inputs_list.append(batch_inputs[i].cpu())
                targets_list.append(batch_predictions[i].cpu())
    
    logger.info(f"Generated {len(inputs_list)} synthetic predictions")
    
    return inputs_list, targets_list


def generate_physical_predictions(
    model: torch.nn.Module,
    real_dataset: Dataset,
    alpha: float,
    device: str = 'cpu',
    num_rollout_steps: int = 10
) -> Tuple[List, List]:
    """
    Generate predictions using a physical model for data augmentation.
    
    This function takes a trained physical model and generates rollout
    predictions on real PhiFlow field samples. The number of predictions
    is proportional to alpha.
    
    Count-based generation: num_generated = int(len(real_dataset) * alpha)
    
    Args:
        model: Trained physical model (expects PhiFlow fields as input/output)
        real_dataset: Dataset of real field samples (returns (initial, targets) tuples)
        alpha: Proportion of generated samples (e.g., 0.1 = 10%)
        device: Device to run model on ('cpu' or 'cuda')
        num_rollout_steps: Number of rollout steps for prediction
        
    Returns:
        Tuple of (initial_fields_list, target_fields_list) where:
        - initial_fields_list: List of initial field states
        - target_fields_list: List of predicted rollout target fields
        
    Example:
        >>> model = load_trained_model("physical_checkpoint.pt")
        >>> dataset = create_field_dataset(...)
        >>> initial_list, targets_list = generate_physical_predictions(
        ...     model=model,
        ...     real_dataset=dataset,
        ...     alpha=0.1,
        ...     device='cuda',
        ...     num_rollout_steps=10
        ... )
        >>> print(f"Generated {len(initial_list)} physical predictions")
    """
    # Only call eval/to for PyTorch models (synthetic)
    # Physical models are not PyTorch modules
    if hasattr(model, 'eval'):
        model.eval()
    if hasattr(model, 'to'):
        model.to(device)
    
    # Calculate number of samples to generate
    num_real = len(real_dataset)
    num_generate = int(num_real * alpha)
    
    logger.info(
        f"Generating {num_generate} physical predictions "
        f"(alpha={alpha:.2f} * {num_real} real samples)"
    )
    
    if num_generate == 0:
        logger.warning("Alpha too small, no samples will be generated")
        return [], []
    
    # Select proportional indices
    indices = _select_proportional_indices(num_real, num_generate)
    
    # Generate predictions
    initial_fields_list = []
    target_fields_list = []
    
    with torch.no_grad():
        for idx in indices:
            # Get sample from dataset
            initial_fields, _ = real_dataset[idx]
            
            # Perform rollout prediction
            # Note: Physical model rollout logic depends on model architecture
            # This is a placeholder that will need to be adapted
            predictions = _perform_physical_rollout(
                model, initial_fields, num_rollout_steps, device
            )
            
            # Store results
            initial_fields_list.append(initial_fields)
            target_fields_list.append(predictions)
    
    logger.info(f"Generated {len(initial_fields_list)} physical predictions")
    
    return initial_fields_list, target_fields_list


def _select_proportional_indices(total_count: int, sample_count: int) -> List[int]:
    """
    Select indices proportionally across the dataset.
    
    This ensures diverse sampling rather than just taking the first N samples.
    
    Args:
        total_count: Total number of available samples
        sample_count: Number of samples to select
        
    Returns:
        List of selected indices
    """
    if sample_count >= total_count:
        return list(range(total_count))
    
    # Calculate step size for proportional sampling
    step = total_count / sample_count
    
    # Select indices evenly distributed
    indices = [int(i * step) for i in range(sample_count)]
    
    # Ensure no duplicates and within bounds
    indices = sorted(list(set(indices)))[:sample_count]
    
    logger.debug(f"Selected {len(indices)} proportional indices from {total_count}")
    
    return indices


def _perform_physical_rollout(
    model: torch.nn.Module,
    initial_fields,
    num_steps: int,
    device: str
):
    """
    Perform rollout prediction with physical model.
    
    This function performs multi-step rollout predictions using a physical model.
    The model's step() method is called iteratively to advance the simulation.
    
    Args:
        model: Physical model with step() method
        initial_fields: Initial field state (dict of PhiFlow Fields)
        num_steps: Number of rollout steps
        device: Device to run on
        
    Returns:
        List of predicted field states, one for each step (List[Dict[str, Field]])
        
    Note:
        This assumes the physical model follows the PhysicalModel interface
        with a step(current_state: Dict[str, Field]) -> Dict[str, Field] method.
    """
    logger.debug(f"Performing {num_steps}-step rollout with physical model")
    
    # Start from initial state
    current_state = initial_fields
    
    # Store all rollout states
    rollout_states = []
    
    # Perform rollout steps
    for step in range(num_steps):
        try:
            # Call model's step method
            current_state = model.step(current_state)
            # Store the state at this step
            rollout_states.append(current_state)
        except Exception as e:
            logger.warning(f"Rollout failed at step {step}/{num_steps}: {e}")
            # Return states collected so far
            return rollout_states if rollout_states else [current_state]
    
    return rollout_states


def generate_and_cache_predictions(
    model: torch.nn.Module,
    real_dataset: Dataset,
    cache_manager,
    alpha: float,
    model_type: str = 'synthetic',
    device: str = 'cpu',
    batch_size: int = 32,
    num_workers: int = 0,
    save_format: str = 'dict'
) -> int:
    """
    Generate predictions and save them to cache in one step.
    
    This is a convenience function that combines generation and caching.
    
    Args:
        model: Trained model
        real_dataset: Dataset of real samples
        cache_manager: CacheManager instance for saving
        alpha: Proportion of generated samples
        model_type: 'synthetic' or 'physical'
        device: Device to run model on
        batch_size: Batch size for synthetic generation
        num_workers: Number of DataLoader workers
        save_format: Format for saving ('dict' or 'tuple')
        
    Returns:
        Number of samples generated and cached
        
    Example:
        >>> from src.data.augmentation import CacheManager
        >>> 
        >>> manager = CacheManager("data/cache", "burgers_128_alpha0.1")
        >>> model = load_model("checkpoint.pt")
        >>> dataset = HybridDataset(...)
        >>> 
        >>> num_cached = generate_and_cache_predictions(
        ...     model=model,
        ...     real_dataset=dataset,
        ...     cache_manager=manager,
        ...     alpha=0.1,
        ...     model_type='synthetic',
        ...     device='cuda'
        ... )
        >>> print(f"Cached {num_cached} predictions")
    """
    logger.info(f"Generating and caching predictions (model_type={model_type})")
    
    # Generate predictions based on model type
    if model_type == 'synthetic':
        inputs_list, targets_list = generate_synthetic_predictions(
            model=model,
            real_dataset=real_dataset,
            alpha=alpha,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers
        )
    elif model_type == 'physical':
        inputs_list, targets_list = generate_physical_predictions(
            model=model,
            real_dataset=real_dataset,
            alpha=alpha,
            device=device
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Save to cache
    num_saved = 0
    for i, (input_data, target_data) in enumerate(zip(inputs_list, targets_list)):
        cache_manager.save_sample(
            index=i,
            input_data=input_data,
            target_data=target_data,
            format=save_format
        )
        num_saved += 1
    
    # Update metadata
    cache_manager.update_metadata({
        'model_type': model_type,
        'alpha': alpha,
        'num_real_samples': len(real_dataset),
        'num_generated_samples': num_saved,
        'device': device,
        'save_format': save_format
    })
    
    logger.info(f"Successfully cached {num_saved} predictions")
    
    return num_saved
