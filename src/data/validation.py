"""
Cache Validation for Data Management

This module provides comprehensive validation for cached simulation data,
ensuring that cached data matches current configuration parameters.

Features:
- PDE parameter validation with checksums
- Resolution and domain matching
- Field name and frame count verification
- Version compatibility checking
- Detailed reporting of validation failures
"""

import hashlib
import json
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime


class CacheValidator:
    """
    Validates cached simulation data against current configuration.
    
    This class performs comprehensive validation to ensure cached data
    is still valid for the current configuration, checking:
    - PDE parameters (nu, buoyancy, etc.)
    - Domain size and resolution
    - Field names and types
    - Number of frames
    - Generation parameters (dt, save_interval)
    
    Attributes:
        config: Configuration dictionary with current parameters
        strict: If True, requires exact version match; if False, allows compatible versions
    """
    
    def __init__(self, config: Dict[str, Any], strict: bool = False):
        """
        Initialize the cache validator.
        
        Args:
            config: Configuration dictionary containing model, data, and generation params
            strict: Whether to use strict version matching
        """
        self.config = config
        self.strict = strict
    
    def validate_cache(
        self,
        cached_metadata: Dict[str, Any],
        field_names: List[str],
        num_frames: Optional[int] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate cached data against current configuration.
        
        Args:
            cached_metadata: Metadata dictionary from cached file
            field_names: List of field names that should be present
            num_frames: Minimum number of frames required (None = don't check)
        
        Returns:
            Tuple of (is_valid, reasons_if_invalid)
            - is_valid: True if cache is valid, False otherwise
            - reasons_if_invalid: List of strings explaining why cache is invalid
        """
        reasons = []
        
        # Check 1: Version compatibility
        cache_version = cached_metadata.get('version', '1.0')
        if not self._is_version_compatible(cache_version):
            reasons.append(
                f"Cache version '{cache_version}' is incompatible with current version '2.0'"
            )
        
        # Check 2: Field names
        if not self._validate_field_names(cached_metadata, field_names):
            cached_fields = set(cached_metadata.get('field_metadata', {}).keys())
            requested_fields = set(field_names)
            reasons.append(
                f"Field mismatch - cached: {cached_fields}, requested: {requested_fields}"
            )
        
        # Check 3: Number of frames
        if num_frames is not None:
            cached_frames = cached_metadata.get('num_frames', 0)
            if cached_frames < num_frames:
                reasons.append(
                    f"Insufficient frames - cached: {cached_frames}, requested: {num_frames}"
                )
        
        # Check 4: PDE parameters (only if metadata has new format)
        if cache_version >= '2.0' and 'generation_params' in cached_metadata:
            if not self._validate_pde_params(cached_metadata):
                reasons.append("PDE parameters have changed")
            
            # Check 5: Resolution
            if not self._validate_resolution(cached_metadata):
                cached_res = cached_metadata['generation_params'].get('resolution', 'unknown')
                current_res = self.config.get('model', {}).get('physical', {}).get('resolution', 'unknown')
                reasons.append(
                    f"Resolution mismatch - cached: {cached_res}, current: {current_res}"
                )
            
            # Check 6: Domain
            if not self._validate_domain(cached_metadata):
                cached_domain = cached_metadata['generation_params'].get('domain', 'unknown')
                current_domain = self.config.get('model', {}).get('physical', {}).get('domain', 'unknown')
                reasons.append(
                    f"Domain mismatch - cached: {cached_domain}, current: {current_domain}"
                )
            
            # Check 7: dt (timestep)
            if not self._validate_dt(cached_metadata):
                cached_dt = cached_metadata['generation_params'].get('dt', 'unknown')
                current_dt = self.config.get('model', {}).get('physical', {}).get('dt', 'unknown')
                reasons.append(
                    f"Timestep (dt) mismatch - cached: {cached_dt}, current: {current_dt}"
                )
        
        # Cache is valid if no reasons for invalidity were found
        is_valid = len(reasons) == 0
        return is_valid, reasons
    
    def _is_version_compatible(self, cache_version: str) -> bool:
        """
        Check if cache version is compatible with current version.
        
        Args:
            cache_version: Version string from cached metadata
        
        Returns:
            True if compatible, False otherwise
        """
        try:
            # Extract major version
            cache_major = int(cache_version.split('.')[0])
            current_major = 2  # Current cache format version
            
            if self.strict:
                return cache_major == current_major
            else:
                # Allow version 1.x and 2.x caches (backward compatible)
                return cache_major in [1, 2]
        except (ValueError, IndexError):
            # Invalid version format
            return False
    
    def _validate_field_names(
        self,
        cached_metadata: Dict[str, Any],
        field_names: List[str]
    ) -> bool:
        """
        Validate that cached data contains all requested fields.
        
        Args:
            cached_metadata: Cached metadata dictionary
            field_names: List of field names that should be present
        
        Returns:
            True if all fields present, False otherwise
        """
        cached_fields = set(cached_metadata.get('field_metadata', {}).keys())
        requested_fields = set(field_names)
        return cached_fields == requested_fields
    
    def _validate_pde_params(self, cached_metadata: Dict[str, Any]) -> bool:
        """
        Validate that PDE parameters match between cached and current config.
        
        Args:
            cached_metadata: Cached metadata dictionary
        
        Returns:
            True if PDE parameters match, False otherwise
        """
        # Get checksums
        cached_hash = cached_metadata.get('checksums', {}).get('pde_params_hash')
        if cached_hash is None:
            # Old cache format without checksums - consider invalid
            return False
        
        # Compute current hash
        current_params = self.config.get('model', {}).get('physical', {}).get('pde_params', {})
        current_hash = compute_hash(current_params)
        
        return cached_hash == current_hash
    
    def _validate_resolution(self, cached_metadata: Dict[str, Any]) -> bool:
        """
        Validate that resolution matches between cached and current config.
        
        Args:
            cached_metadata: Cached metadata dictionary
        
        Returns:
            True if resolution matches, False otherwise
        """
        cached_res = cached_metadata.get('generation_params', {}).get('resolution')
        current_res = self.config.get('model', {}).get('physical', {}).get('resolution')
        
        if cached_res is None or current_res is None:
            return False
        
        return cached_res == current_res
    
    def _validate_domain(self, cached_metadata: Dict[str, Any]) -> bool:
        """
        Validate that domain matches between cached and current config.
        
        Args:
            cached_metadata: Cached metadata dictionary
        
        Returns:
            True if domain matches, False otherwise
        """
        cached_domain = cached_metadata.get('generation_params', {}).get('domain')
        current_domain = self.config.get('model', {}).get('physical', {}).get('domain')
        
        if cached_domain is None or current_domain is None:
            return False
        
        return cached_domain == current_domain
    
    def _validate_dt(self, cached_metadata: Dict[str, Any]) -> bool:
        """
        Validate that timestep (dt) matches between cached and current config.
        
        Args:
            cached_metadata: Cached metadata dictionary
        
        Returns:
            True if dt matches, False otherwise
        """
        cached_dt = cached_metadata.get('generation_params', {}).get('dt')
        current_dt = self.config.get('model', {}).get('physical', {}).get('dt')
        
        if cached_dt is None or current_dt is None:
            return False
        
        # Use approximate equality for floating point comparison
        return abs(float(cached_dt) - float(current_dt)) < 1e-9


def compute_hash(obj: Any) -> str:
    """
    Compute a stable SHA256 hash of an object.
    
    This function converts the object to a JSON string with sorted keys
    to ensure consistent hashing across different runs.
    
    Args:
        obj: Object to hash (must be JSON-serializable)
    
    Returns:
        Hexadecimal hash string
    
    Example:
        >>> hash1 = compute_hash({'a': 1, 'b': 2})
        >>> hash2 = compute_hash({'b': 2, 'a': 1})
        >>> hash1 == hash2
        True
    """
    json_str = json.dumps(obj, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


def get_cache_version() -> str:
    """
    Get the current cache format version.
    
    Returns:
        Version string (e.g., '2.0')
    """
    return '2.0'


def get_phiflow_version() -> str:
    """
    Get the PhiFlow version being used.
    
    Returns:
        Version string, or 'unknown' if PhiFlow not available
    """
    try:
        import phi
        return phi.__version__
    except (ImportError, AttributeError):
        return 'unknown'
