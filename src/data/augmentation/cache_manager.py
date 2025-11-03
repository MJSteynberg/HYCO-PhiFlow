"""
Cache Manager for hybrid training data augmentation.

This module provides CacheManager for managing the lifecycle of cached
generated predictions, including creation, validation, cleanup, and
disk space monitoring.
"""

import logging
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import torch

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages cache lifecycle for hybrid training augmentation.
    
    Responsibilities:
    - Create and organize cache directory structure
    - Save and load cached predictions
    - Track metadata (generation time, model version, etc.)
    - Monitor disk space usage
    - Clean up old/invalid caches
    - Validate cache integrity
    
    Directory Structure:
        cache_root/
            hybrid_generated/
                {experiment_name}/
                    metadata.json
                    sample_000000.pt
                    sample_000001.pt
                    ...
    
    Args:
        cache_root: Root directory for all caches (e.g., "data/cache")
        experiment_name: Name of experiment (e.g., "burgers_128_alpha0.1")
        
    Example:
        >>> manager = CacheManager(
        ...     cache_root="data/cache",
        ...     experiment_name="burgers_128_alpha0.1"
        ... )
        >>> 
        >>> # Save generated predictions
        >>> for i, (input_data, target_data) in enumerate(predictions):
        ...     manager.save_sample(i, input_data, target_data)
        >>> 
        >>> # Update metadata
        >>> manager.update_metadata({
        ...     'model_version': 'v1.0',
        ...     'generation_time': datetime.now().isoformat(),
        ...     'num_samples': 100
        ... })
        >>> 
        >>> # Check disk usage
        >>> usage = manager.get_disk_usage()
        >>> print(f"Cache size: {usage['size_mb']:.2f} MB")
        >>> 
        >>> # Clean up if needed
        >>> if usage['size_mb'] > 1000:
        ...     manager.clear_cache()
    """
    
    def __init__(
        self,
        cache_root: str,
        experiment_name: str,
        auto_create: bool = True
    ):
        self.cache_root = Path(cache_root)
        self.experiment_name = experiment_name
        
        # Cache directory structure
        self.cache_dir = self.cache_root / "hybrid_generated" / experiment_name
        self.metadata_path = self.cache_dir / "metadata.json"
        
        # Create directory if needed
        if auto_create:
            self._ensure_cache_dir()
        
        logger.info(
            f"CacheManager initialized for '{experiment_name}' "
            f"at {self.cache_dir}"
        )
    
    def _ensure_cache_dir(self) -> None:
        """Create cache directory structure if it doesn't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Cache directory ensured at {self.cache_dir}")
    
    def save_sample(
        self,
        index: int,
        input_data: torch.Tensor,
        target_data: torch.Tensor,
        format: str = 'dict'
    ) -> Path:
        """
        Save a single generated sample to cache.
        
        Args:
            index: Sample index (0-based)
            input_data: Input tensor
            target_data: Target/prediction tensor
            format: Save format ('dict' or 'tuple')
            
        Returns:
            Path to saved file
        """
        self._ensure_cache_dir()
        
        filename = f"sample_{index:06d}.pt"
        filepath = self.cache_dir / filename
        
        # Prepare data based on format
        if format == 'dict':
            data = {
                'input': input_data.cpu(),
                'target': target_data.cpu()
            }
        elif format == 'tuple':
            data = (input_data.cpu(), target_data.cpu())
        else:
            raise ValueError(f"Unknown format: {format}. Use 'dict' or 'tuple'")
        
        # Save to disk
        torch.save(data, filepath)
        logger.debug(f"Saved sample {index} to {filepath}")
        
        return filepath
    
    def save_batch(
        self,
        start_index: int,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        format: str = 'dict'
    ) -> List[Path]:
        """
        Save a batch of generated samples to cache.
        
        Args:
            start_index: Starting index for batch
            inputs: Batch of input tensors [batch_size, ...]
            targets: Batch of target tensors [batch_size, ...]
            format: Save format ('dict' or 'tuple')
            
        Returns:
            List of paths to saved files
        """
        batch_size = inputs.shape[0]
        saved_paths = []
        
        for i in range(batch_size):
            idx = start_index + i
            path = self.save_sample(
                idx,
                inputs[i],
                targets[i],
                format=format
            )
            saved_paths.append(path)
        
        logger.debug(f"Saved batch of {batch_size} samples starting at index {start_index}")
        return saved_paths
    
    def load_sample(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a single sample from cache.
        
        Args:
            index: Sample index to load
            
        Returns:
            Tuple of (input, target) tensors
        """
        filename = f"sample_{index:06d}.pt"
        filepath = self.cache_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Sample {index} not found at {filepath}")
        
        data = torch.load(filepath, map_location='cpu')
        
        # Handle both dict and tuple formats
        if isinstance(data, dict):
            return data['input'], data['target']
        elif isinstance(data, (tuple, list)):
            return data[0], data[1]
        else:
            raise ValueError(f"Unknown cache format in {filepath}")
    
    def count_samples(self) -> int:
        """
        Count the number of cached samples.
        
        Returns:
            Number of .pt files in cache directory
        """
        if not self.cache_dir.exists():
            return 0
        
        cache_files = list(self.cache_dir.glob("sample_*.pt"))
        return len(cache_files)
    
    def list_samples(self) -> List[Path]:
        """
        List all cached sample files.
        
        Returns:
            Sorted list of sample file paths
        """
        if not self.cache_dir.exists():
            return []
        
        cache_files = sorted(self.cache_dir.glob("sample_*.pt"))
        return cache_files
    
    def update_metadata(self, metadata: Dict) -> None:
        """
        Update cache metadata.
        
        Args:
            metadata: Dictionary of metadata to save/update
        """
        self._ensure_cache_dir()
        
        # Load existing metadata if available
        existing_metadata = {}
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                existing_metadata = json.load(f)
        
        # Merge with new metadata
        existing_metadata.update(metadata)
        
        # Add timestamp
        existing_metadata['last_updated'] = datetime.now().isoformat()
        
        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(existing_metadata, f, indent=2)
        
        logger.debug(f"Updated metadata at {self.metadata_path}")
    
    def load_metadata(self) -> Dict:
        """
        Load cache metadata.
        
        Returns:
            Dictionary of metadata, or empty dict if not found
        """
        if not self.metadata_path.exists():
            return {}
        
        with open(self.metadata_path, 'r') as f:
            return json.load(f)
    
    def get_disk_usage(self) -> Dict:
        """
        Calculate disk space usage for this cache.
        
        Returns:
            Dictionary with usage statistics:
            - size_bytes: Total size in bytes
            - size_mb: Total size in megabytes
            - size_gb: Total size in gigabytes
            - num_files: Number of cache files
        """
        if not self.cache_dir.exists():
            return {
                'size_bytes': 0,
                'size_mb': 0.0,
                'size_gb': 0.0,
                'num_files': 0
            }
        
        total_size = 0
        num_files = 0
        
        # Sum up all file sizes
        for file_path in self.cache_dir.rglob("*.pt"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                num_files += 1
        
        # Include metadata file if it exists
        if self.metadata_path.exists():
            total_size += self.metadata_path.stat().st_size
            num_files += 1
        
        return {
            'size_bytes': total_size,
            'size_mb': total_size / (1024 * 1024),
            'size_gb': total_size / (1024 * 1024 * 1024),
            'num_files': num_files
        }
    
    def clear_cache(self, confirm: bool = False) -> None:
        """
        Delete all cached samples and metadata.
        
        Args:
            confirm: Safety flag, must be True to execute
            
        Raises:
            ValueError: If confirm is False
        """
        if not confirm:
            raise ValueError(
                "Must set confirm=True to clear cache. "
                "This will delete all cached data."
            )
        
        if not self.cache_dir.exists():
            logger.warning(f"Cache directory does not exist: {self.cache_dir}")
            return
        
        # Get usage before deletion for logging
        usage = self.get_disk_usage()
        
        # Delete the entire cache directory
        shutil.rmtree(self.cache_dir)
        
        logger.info(
            f"Cleared cache for '{self.experiment_name}'. "
            f"Freed {usage['size_mb']:.2f} MB ({usage['num_files']} files)"
        )
    
    def validate_cache(
        self,
        expected_count: Optional[int] = None,
        check_loadable: bool = True
    ) -> Dict:
        """
        Validate cache integrity.
        
        Args:
            expected_count: Expected number of samples (optional)
            check_loadable: Whether to attempt loading each sample
            
        Returns:
            Dictionary with validation results:
            - valid: Overall validation status
            - num_samples: Number of samples found
            - expected_count: Expected count (if provided)
            - count_mismatch: Whether count doesn't match expected
            - corrupt_samples: List of indices that failed to load
            - errors: List of error messages
        """
        result = {
            'valid': True,
            'num_samples': 0,
            'expected_count': expected_count,
            'count_mismatch': False,
            'corrupt_samples': [],
            'errors': []
        }
        
        if not self.cache_dir.exists():
            result['valid'] = False
            result['errors'].append(f"Cache directory does not exist: {self.cache_dir}")
            return result
        
        # Count samples
        samples = self.list_samples()
        result['num_samples'] = len(samples)
        
        # Check count
        if expected_count is not None:
            if abs(result['num_samples'] - expected_count) > 1:
                result['count_mismatch'] = True
                result['errors'].append(
                    f"Sample count mismatch: expected {expected_count}, "
                    f"found {result['num_samples']}"
                )
        
        # Check loadability
        if check_loadable:
            for i, sample_path in enumerate(samples):
                try:
                    # Extract index from filename
                    index = int(sample_path.stem.split('_')[1])
                    _ = self.load_sample(index)
                except Exception as e:
                    result['valid'] = False
                    result['corrupt_samples'].append(index)
                    result['errors'].append(f"Failed to load sample {index}: {e}")
        
        if result['corrupt_samples']:
            result['valid'] = False
        
        return result
    
    def exists(self) -> bool:
        """
        Check if cache directory exists.
        
        Returns:
            True if cache directory exists
        """
        return self.cache_dir.exists()
    
    def is_empty(self) -> bool:
        """
        Check if cache is empty.
        
        Returns:
            True if cache has no samples
        """
        return self.count_samples() == 0
    
    def get_info(self) -> Dict:
        """
        Get comprehensive cache information.
        
        Returns:
            Dictionary with cache info:
            - experiment_name: Name of experiment
            - cache_dir: Path to cache directory
            - exists: Whether cache exists
            - num_samples: Number of cached samples
            - disk_usage: Disk usage statistics
            - metadata: Cache metadata (if available)
        """
        usage = self.get_disk_usage()
        metadata = self.load_metadata()
        
        return {
            'experiment_name': self.experiment_name,
            'cache_dir': str(self.cache_dir),
            'exists': self.exists(),
            'is_empty': self.is_empty(),
            'num_samples': self.count_samples(),
            'disk_usage': usage,
            'metadata': metadata
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CacheManager(experiment='{self.experiment_name}', "
            f"samples={self.count_samples()}, "
            f"size={self.get_disk_usage()['size_mb']:.2f}MB)"
        )
