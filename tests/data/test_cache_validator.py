"""
Unit tests for CacheValidator class and validation functions.

Tests the enhanced cache validation features from Phase 1:
- PDE parameter validation with checksums
- Resolution, domain, and dt validation
- Version compatibility checking
- Hash computation
"""

import pytest
from src.data.validation import CacheValidator, compute_hash, get_cache_version


class TestComputeHash:
    """Tests for the compute_hash function."""
    
    def test_compute_hash_consistent(self):
        """Test that compute_hash produces consistent results."""
        obj = {'nu': 0.1, 'batch_size': 1}
        hash1 = compute_hash(obj)
        hash2 = compute_hash(obj)
        assert hash1 == hash2
    
    def test_compute_hash_order_independent(self):
        """Test that compute_hash is independent of key order."""
        obj1 = {'a': 1, 'b': 2, 'c': 3}
        obj2 = {'c': 3, 'a': 1, 'b': 2}
        assert compute_hash(obj1) == compute_hash(obj2)
    
    def test_compute_hash_different_values(self):
        """Test that different values produce different hashes."""
        obj1 = {'nu': 0.1}
        obj2 = {'nu': 0.2}
        assert compute_hash(obj1) != compute_hash(obj2)
    
    def test_compute_hash_nested_dicts(self):
        """Test hashing nested dictionaries."""
        obj1 = {'params': {'nu': 0.1, 'dt': 0.5}}
        obj2 = {'params': {'dt': 0.5, 'nu': 0.1}}
        assert compute_hash(obj1) == compute_hash(obj2)
    
    def test_compute_hash_with_lists(self):
        """Test hashing objects with lists."""
        obj = {'fields': ['velocity', 'density'], 'value': 1}
        hash_val = compute_hash(obj)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64  # SHA256 produces 64 hex characters


class TestCacheValidatorInit:
    """Tests for CacheValidator initialization."""
    
    def test_init_basic(self):
        """Test basic initialization."""
        config = {
            'model': {
                'physical': {
                    'pde_params': {'nu': 0.1},
                    'resolution': {'x': 128, 'y': 128},
                    'domain': {'size_x': 100, 'size_y': 100},
                    'dt': 0.5
                }
            }
        }
        validator = CacheValidator(config)
        assert validator.config == config
    
    def test_init_sets_config(self):
        """Test that initialization sets config correctly."""
        config = {'model': {'physical': {}}}
        validator = CacheValidator(config)
        assert validator.config == config


class TestVersionCompatibility:
    """Tests for version compatibility checking."""
    
    def test_version_compatible_v2(self):
        """Test that v2.x is compatible."""
        config = {'model': {'physical': {}}}
        validator = CacheValidator(config)
        
        metadata = {'version': '2.0'}
        is_valid, _ = validator.validate_cache(metadata, [], None)
        assert is_valid  # Should pass version check (other checks may fail)
    
    def test_version_incompatible_v1(self):
        """Test that v1.x is incompatible (no backward compatibility)."""
        config = {'model': {'physical': {}}}
        validator = CacheValidator(config)
        
        # v1.0 should NOT be compatible (backward compatibility removed)
        assert not validator._is_version_compatible('1.0')
        assert not validator._is_version_compatible('1.5')
    
    def test_version_compatible_v2_all_variants(self):
        """Test that all v2.x variants are compatible."""
        config = {'model': {'physical': {}}}
        validator = CacheValidator(config)
        
        assert validator._is_version_compatible('2.0')
        assert validator._is_version_compatible('2.1')
        assert validator._is_version_compatible('2.9')


class TestFieldValidation:
    """Tests for field name validation."""
    
    def test_field_validation_matching(self):
        """Test field validation with matching fields."""
        config = {'model': {'physical': {}}}
        validator = CacheValidator(config)
        
        metadata = {
            'version': '2.0',
            'field_metadata': {'velocity': {}, 'density': {}}
        }
        
        assert validator._validate_field_names(metadata, ['velocity', 'density'])
    
    def test_field_validation_order_independent(self):
        """Test that field validation is order-independent."""
        config = {'model': {'physical': {}}}
        validator = CacheValidator(config)
        
        metadata = {
            'version': '2.0',
            'field_metadata': {'velocity': {}, 'density': {}}
        }
        
        # Both orders should match
        assert validator._validate_field_names(metadata, ['velocity', 'density'])
        assert validator._validate_field_names(metadata, ['density', 'velocity'])
    
    def test_field_validation_missing_field(self):
        """Test field validation when cached data is missing a field."""
        config = {'model': {'physical': {}}}
        validator = CacheValidator(config)
        
        metadata = {
            'version': '2.0',
            'field_metadata': {'velocity': {}}
        }
        
        # Requesting both velocity and density should fail
        assert not validator._validate_field_names(metadata, ['velocity', 'density'])
    
    def test_field_validation_extra_field(self):
        """Test field validation when cached data has extra fields."""
        config = {'model': {'physical': {}}}
        validator = CacheValidator(config)
        
        metadata = {
            'version': '2.0',
            'field_metadata': {'velocity': {}, 'density': {}, 'pressure': {}}
        }
        
        # Requesting only velocity and density should fail (extra field in cache)
        assert not validator._validate_field_names(metadata, ['velocity', 'density'])


class TestPDEParamsValidation:
    """Tests for PDE parameter validation."""
    
    def test_pde_params_validation_matching(self):
        """Test PDE params validation with matching parameters."""
        pde_params = {'nu': 0.1, 'batch_size': 1}
        config = {
            'model': {
                'physical': {
                    'pde_params': pde_params
                }
            }
        }
        validator = CacheValidator(config)
        
        metadata = {
            'version': '2.0',
            'generation_params': {},
            'checksums': {
                'pde_params_hash': compute_hash(pde_params)
            }
        }
        
        assert validator._validate_pde_params(metadata)
    
    def test_pde_params_validation_different(self):
        """Test PDE params validation with different parameters."""
        config = {
            'model': {
                'physical': {
                    'pde_params': {'nu': 0.1}
                }
            }
        }
        validator = CacheValidator(config)
        
        # Cached data has different nu value
        metadata = {
            'version': '2.0',
            'generation_params': {},
            'checksums': {
                'pde_params_hash': compute_hash({'nu': 0.2})
            }
        }
        
        assert not validator._validate_pde_params(metadata)
    
    def test_pde_params_validation_missing_hash(self):
        """Test PDE params validation when hash is missing."""
        config = {
            'model': {
                'physical': {
                    'pde_params': {'nu': 0.1}
                }
            }
        }
        validator = CacheValidator(config)
        
        # Old cache format without checksums
        metadata = {
            'version': '1.0',
            'generation_params': {}
        }
        
        assert not validator._validate_pde_params(metadata)


class TestResolutionValidation:
    """Tests for resolution validation."""
    
    def test_resolution_validation_matching(self):
        """Test resolution validation with matching resolution."""
        resolution = {'x': 128, 'y': 128}
        config = {
            'model': {
                'physical': {
                    'resolution': resolution
                }
            }
        }
        validator = CacheValidator(config)
        
        metadata = {
            'version': '2.0',
            'generation_params': {
                'resolution': resolution
            }
        }
        
        assert validator._validate_resolution(metadata)
    
    def test_resolution_validation_different(self):
        """Test resolution validation with different resolution."""
        config = {
            'model': {
                'physical': {
                    'resolution': {'x': 128, 'y': 128}
                }
            }
        }
        validator = CacheValidator(config)
        
        metadata = {
            'version': '2.0',
            'generation_params': {
                'resolution': {'x': 64, 'y': 64}  # Different resolution
            }
        }
        
        assert not validator._validate_resolution(metadata)
    
    def test_resolution_validation_missing(self):
        """Test resolution validation when resolution is missing."""
        config = {
            'model': {
                'physical': {
                    'resolution': {'x': 128, 'y': 128}
                }
            }
        }
        validator = CacheValidator(config)
        
        metadata = {
            'version': '2.0',
            'generation_params': {}  # No resolution
        }
        
        assert not validator._validate_resolution(metadata)


class TestDomainValidation:
    """Tests for domain validation."""
    
    def test_domain_validation_matching(self):
        """Test domain validation with matching domain."""
        domain = {'size_x': 100, 'size_y': 100}
        config = {
            'model': {
                'physical': {
                    'domain': domain
                }
            }
        }
        validator = CacheValidator(config)
        
        metadata = {
            'version': '2.0',
            'generation_params': {
                'domain': domain
            }
        }
        
        assert validator._validate_domain(metadata)
    
    def test_domain_validation_different(self):
        """Test domain validation with different domain."""
        config = {
            'model': {
                'physical': {
                    'domain': {'size_x': 100, 'size_y': 100}
                }
            }
        }
        validator = CacheValidator(config)
        
        metadata = {
            'version': '2.0',
            'generation_params': {
                'domain': {'size_x': 200, 'size_y': 200}  # Different domain
            }
        }
        
        assert not validator._validate_domain(metadata)


class TestDtValidation:
    """Tests for timestep (dt) validation."""
    
    def test_dt_validation_matching(self):
        """Test dt validation with matching timestep."""
        config = {
            'model': {
                'physical': {
                    'dt': 0.5
                }
            }
        }
        validator = CacheValidator(config)
        
        metadata = {
            'version': '2.0',
            'generation_params': {
                'dt': 0.5
            }
        }
        
        assert validator._validate_dt(metadata)
    
    def test_dt_validation_different(self):
        """Test dt validation with different timestep."""
        config = {
            'model': {
                'physical': {
                    'dt': 0.5
                }
            }
        }
        validator = CacheValidator(config)
        
        metadata = {
            'version': '2.0',
            'generation_params': {
                'dt': 1.0  # Different dt
            }
        }
        
        assert not validator._validate_dt(metadata)
    
    def test_dt_validation_floating_point_tolerance(self):
        """Test that dt validation handles floating point precision."""
        config = {
            'model': {
                'physical': {
                    'dt': 0.5
                }
            }
        }
        validator = CacheValidator(config)
        
        metadata = {
            'version': '2.0',
            'generation_params': {
                'dt': 0.5 + 1e-10  # Very small difference
            }
        }
        
        assert validator._validate_dt(metadata)


class TestCompleteValidation:
    """Tests for complete cache validation."""
    
    def test_validation_all_valid(self):
        """Test validation when all parameters match."""
        pde_params = {'nu': 0.1, 'batch_size': 1}
        resolution = {'x': 128, 'y': 128}
        domain = {'size_x': 100, 'size_y': 100}
        dt = 0.5
        
        config = {
            'model': {
                'physical': {
                    'pde_params': pde_params,
                    'resolution': resolution,
                    'domain': domain,
                    'dt': dt
                }
            }
        }
        validator = CacheValidator(config)
        
        metadata = {
            'version': '2.0',
            'field_metadata': {'velocity': {}},
            'num_frames': 10,
            'generation_params': {
                'resolution': resolution,
                'domain': domain,
                'dt': dt
            },
            'checksums': {
                'pde_params_hash': compute_hash(pde_params)
            }
        }
        
        is_valid, reasons = validator.validate_cache(metadata, ['velocity'], 5)
        assert is_valid
        assert len(reasons) == 0
    
    def test_validation_pde_params_mismatch(self):
        """Test validation when PDE params don't match."""
        config = {
            'model': {
                'physical': {
                    'pde_params': {'nu': 0.1},
                    'resolution': {'x': 128, 'y': 128},
                    'domain': {'size_x': 100, 'size_y': 100},
                    'dt': 0.5
                }
            }
        }
        validator = CacheValidator(config)
        
        metadata = {
            'version': '2.0',
            'field_metadata': {'velocity': {}},
            'num_frames': 10,
            'generation_params': {
                'resolution': {'x': 128, 'y': 128},
                'domain': {'size_x': 100, 'size_y': 100},
                'dt': 0.5
            },
            'checksums': {
                'pde_params_hash': compute_hash({'nu': 0.2})  # Different!
            }
        }
        
        is_valid, reasons = validator.validate_cache(metadata, ['velocity'], 5)
        assert not is_valid
        assert any('PDE parameters' in reason for reason in reasons)
    
    def test_validation_resolution_mismatch(self):
        """Test validation when resolution doesn't match."""
        pde_params = {'nu': 0.1}
        config = {
            'model': {
                'physical': {
                    'pde_params': pde_params,
                    'resolution': {'x': 128, 'y': 128},
                    'domain': {'size_x': 100, 'size_y': 100},
                    'dt': 0.5
                }
            }
        }
        validator = CacheValidator(config)
        
        metadata = {
            'version': '2.0',
            'field_metadata': {'velocity': {}},
            'num_frames': 10,
            'generation_params': {
                'resolution': {'x': 64, 'y': 64},  # Different!
                'domain': {'size_x': 100, 'size_y': 100},
                'dt': 0.5
            },
            'checksums': {
                'pde_params_hash': compute_hash(pde_params)
            }
        }
        
        is_valid, reasons = validator.validate_cache(metadata, ['velocity'], 5)
        assert not is_valid
        assert any('Resolution mismatch' in reason for reason in reasons)
    
    def test_validation_multiple_failures(self):
        """Test validation reports multiple failures."""
        config = {
            'model': {
                'physical': {
                    'pde_params': {'nu': 0.1},
                    'resolution': {'x': 128, 'y': 128},
                    'domain': {'size_x': 100, 'size_y': 100},
                    'dt': 0.5
                }
            }
        }
        validator = CacheValidator(config)
        
        metadata = {
            'version': '2.0',
            'field_metadata': {'density': {}},  # Wrong field
            'num_frames': 3,  # Too few frames
            'generation_params': {
                'resolution': {'x': 64, 'y': 64},  # Wrong resolution
                'domain': {'size_x': 100, 'size_y': 100},
                'dt': 1.0  # Wrong dt
            },
            'checksums': {
                'pde_params_hash': compute_hash({'nu': 0.2})  # Wrong params
            }
        }
        
        is_valid, reasons = validator.validate_cache(metadata, ['velocity'], 5)
        assert not is_valid
        assert len(reasons) >= 4  # At least 4 failures
        
        # Check that specific failures are reported
        reasons_str = ' '.join(reasons)
        assert 'Field mismatch' in reasons_str
        assert 'Insufficient frames' in reasons_str or 'frames' in reasons_str.lower()
        assert 'Resolution mismatch' in reasons_str
        assert 'dt' in reasons_str.lower() or 'Timestep' in reasons_str
        assert 'PDE' in reasons_str


class TestGetCacheVersion:
    """Test get_cache_version function."""
    
    def test_get_cache_version(self):
        """Test that get_cache_version returns current version."""
        version = get_cache_version()
        assert version == '2.0'
        assert isinstance(version, str)
