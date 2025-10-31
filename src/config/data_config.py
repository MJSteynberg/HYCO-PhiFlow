from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import MISSING


@dataclass
class DataConfig:
    """Configuration for dataset."""
    
    data_dir: str = 'data/'
    dset_name: str = MISSING  # Required
    fields: List[str] = field(default_factory=list)  # Required
    fields_scheme: str = 'unknown'
    cache_dir: str = 'data/cache'
    
    # Validation options
    validate_cache: bool = True
    auto_clear_invalid: bool = False