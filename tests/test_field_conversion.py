"""
Tests for field conversion utilities.

These tests verify that we can correctly convert between PyTorch tensors
and PhiFlow Fields while preserving all metadata.
"""

import pytest
import torch
from phi.torch.flow import *
from phi import math
from phi.field import CenteredGrid, StaggeredGrid
from phi.geom import Box

from src.utils.field_conversion import (
    FieldMetadata,
    tensor_to_field,
    field_to_tensor,
    tensors_to_fields,
    fields_to_tensors,
    create_field_metadata_from_model
)


class TestFieldMetadata:
    """Test FieldMetadata dataclass functionality."""
    
    def test_from_centered_field(self):
        """Test extracting metadata from a CenteredGrid."""
        domain = Box(x=1, y=1)
        resolution = spatial(x=32, y=32)
        
        # Create tensor with proper dimension naming: [x, y]
        values = math.random_normal(resolution)
        
        field = CenteredGrid(
            values,
            extrapolation.PERIODIC,
            bounds=domain
        )
        
        metadata = FieldMetadata.from_field(field)
        
        assert metadata.domain == domain
        assert metadata.resolution == resolution
        assert metadata.field_type == 'centered'
        assert 'x' in metadata.spatial_dims
        assert 'y' in metadata.spatial_dims
    
    def test_from_staggered_field(self):
        """Test extracting metadata from a StaggeredGrid."""
        domain = Box(x=1, y=1)
        resolution = spatial(x=32, y=32)
        
        # Create vector field with proper dimension naming: [x, y, vector]
        values = math.random_normal(resolution & channel(vector='x,y'))
        
        field = StaggeredGrid(
            values,
            extrapolation.ZERO,
            bounds=domain
        )
        
        metadata = FieldMetadata.from_field(field)
        
        assert metadata.domain == domain
        # StaggeredGrid may have slightly different resolution internally
        # Just check that it has spatial dimensions
        assert 'x' in metadata.resolution.names
        assert 'y' in metadata.resolution.names
        assert metadata.field_type == 'staggered'
    
    def test_from_cache_metadata(self):
        """Test reconstructing metadata from cached dictionary."""
        domain = Box(x=1, y=1)
        resolution = spatial(x=64, y=64)
        
        cached_meta = {
            'extrapolation': 'PERIODIC',
            'field_type': 'centered',
            'spatial_dims': ['x', 'y'],
            'channel_dims': []
        }
        
        metadata = FieldMetadata.from_cache_metadata(cached_meta, domain, resolution)
        
        from phi.math import extrapolation as extrap_module
        
        assert metadata.domain == domain
        assert metadata.resolution == resolution
        assert metadata.extrapolation == extrap_module.PERIODIC
        assert metadata.field_type == 'centered'
        assert metadata.spatial_dims == ('x', 'y')
    
    def test_extrapolation_parsing(self):
        """Test various extrapolation string formats."""
        domain = Box(x=1, y=1)
        resolution = spatial(x=32, y=32)
        
        from phi.math import extrapolation as extrap_module
        
        test_cases = [
            ('ZERO', extrap_module.ZERO),
            ('PERIODIC', extrap_module.PERIODIC),
            ('BOUNDARY', extrap_module.BOUNDARY),
            ('zero-gradient', extrap_module.ZERO_GRADIENT),
            ('<ZERO>', extrap_module.ZERO),  # PhiFlow repr format
        ]
        
        for extrap_str, expected in test_cases:
            cached_meta = {
                'extrapolation': extrap_str,
                'field_type': 'centered',
                'spatial_dims': ['x', 'y'],
                'channel_dims': []
            }
            
            metadata = FieldMetadata.from_cache_metadata(cached_meta, domain, resolution)
            assert metadata.extrapolation == expected, f"Failed for {extrap_str}"


class TestTensorToField:
    """Test tensor to Field conversions."""
    
    def test_centered_scalar_field(self):
        """Test converting a scalar tensor to CenteredGrid."""
        domain = Box(x=1, y=1)
        resolution = spatial(x=32, y=32)
        
        # Create tensor: [time, channels, x, y]
        tensor = torch.randn(10, 1, 32, 32)
        
        metadata = FieldMetadata(
            domain=domain,
            resolution=resolution,
            extrapolation=extrapolation.PERIODIC,
            field_type='centered',
            spatial_dims=('x', 'y'),
            channel_dims=()
        )
        
        # Convert single timestep
        field = tensor_to_field(tensor, metadata, time_slice=0)
        
        assert not field.is_staggered  # Check it's centered
        assert field.shape.spatial == resolution
        assert field.extrapolation == extrapolation.PERIODIC
    
    def test_staggered_vector_field(self):
        """Test converting a vector tensor to StaggeredGrid."""
        domain = Box(x=1, y=1)
        resolution = spatial(x=32, y=32)
        
        # Create tensor: [time, channels, x, y]
        tensor = torch.randn(10, 2, 32, 32)
        
        metadata = FieldMetadata(
            domain=domain,
            resolution=resolution,
            extrapolation=extrapolation.ZERO,
            field_type='staggered',
            spatial_dims=('x', 'y'),
            channel_dims=('vector',)
        )
        
        # Convert single timestep
        field = tensor_to_field(tensor, metadata, time_slice=5)
        
        assert field.is_staggered  # Check it's staggered
        assert field.shape.spatial == resolution
    
    def test_without_time_slice(self):
        """Test conversion with batch dimension preserved."""
        domain = Box(x=1, y=1)
        resolution = spatial(x=16, y=16)
        
        tensor = torch.randn(5, 1, 16, 16)
        
        metadata = FieldMetadata(
            domain=domain,
            resolution=resolution,
            extrapolation=extrapolation.PERIODIC,
            field_type='centered',
            spatial_dims=('x', 'y'),
            channel_dims=()
        )
        
        # Convert without time slice - should preserve batch dimension
        field = tensor_to_field(tensor, metadata, time_slice=None)

        assert not field.is_staggered
        # Field should have a batch or time dimension
        assert field.values.shape.batch.volume == 5
class TestFieldToTensor:
    """Test Field to tensor conversions."""
    
    def test_centered_field_to_tensor(self):
        """Test extracting tensor from CenteredGrid."""
        domain = Box(x=1, y=1)
        resolution = spatial(x=32, y=32)
        
        # Create field with proper dimensions
        values = math.random_normal(resolution)
        
        field = CenteredGrid(
            values,
            extrapolation.PERIODIC,
            bounds=domain
        )
        
        tensor = field_to_tensor(field)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[-2:] == (32, 32)  # Spatial dimensions preserved
    
    def test_staggered_field_to_tensor(self):
        """Test extracting tensor from StaggeredGrid."""
        domain = Box(x=1, y=1)
        resolution = spatial(x=32, y=32)
        
        # Create vector field
        values = math.random_normal(resolution & channel(vector='x,y'))
        
        field = StaggeredGrid(
            values,
            extrapolation.ZERO,
            bounds=domain
        )
        
        tensor = field_to_tensor(field)
        
        assert isinstance(tensor, torch.Tensor)
        # Should preserve vector dimension
        assert 2 in tensor.shape


class TestBatchConversions:
    """Test batch conversion utilities."""
    
    def test_tensors_to_fields(self):
        """Test converting multiple tensors to fields."""
        domain = Box(x=1, y=1)
        resolution = spatial(x=32, y=32)
        
        tensor_dict = {
            'velocity': torch.randn(10, 2, 32, 32),
            'density': torch.randn(10, 1, 32, 32)
        }
        
        metadata_dict = {
            'velocity': FieldMetadata(
                domain=domain,
                resolution=resolution,
                extrapolation=extrapolation.ZERO,
                field_type='staggered',
                spatial_dims=('x', 'y'),
                channel_dims=('vector',)
            ),
            'density': FieldMetadata(
                domain=domain,
                resolution=resolution,
                extrapolation=extrapolation.PERIODIC,
                field_type='centered',
                spatial_dims=('x', 'y'),
                channel_dims=()
            )
        }
        
        field_dict = tensors_to_fields(tensor_dict, metadata_dict, time_slice=3)
        
        assert 'velocity' in field_dict
        assert 'density' in field_dict
        assert field_dict['velocity'].is_staggered
        assert not field_dict['density'].is_staggered
    
    def test_fields_to_tensors(self):
        """Test converting multiple fields to tensors."""
        domain = Box(x=1, y=1)
        resolution = spatial(x=32, y=32)
        
        field_dict = {
            'velocity': StaggeredGrid(
                math.random_normal(resolution & channel(vector='x,y')),
                extrapolation.ZERO,
                bounds=domain
            ),
            'density': CenteredGrid(
                math.random_normal(resolution),
                extrapolation.PERIODIC,
                bounds=domain
            )
        }
        
        tensor_dict = fields_to_tensors(field_dict)
        
        assert 'velocity' in tensor_dict
        assert 'density' in tensor_dict
        assert isinstance(tensor_dict['velocity'], torch.Tensor)
        assert isinstance(tensor_dict['density'], torch.Tensor)
    
    def test_mismatched_keys_error(self):
        """Test that mismatched keys raise an error."""
        tensor_dict = {'velocity': torch.randn(10, 2, 32, 32)}
        metadata_dict = {'density': FieldMetadata(
            domain=Box(x=1, y=1),
            resolution=spatial(x=32, y=32),
            extrapolation=extrapolation.PERIODIC,
            field_type='centered',
            spatial_dims=('x', 'y'),
            channel_dims=()
        )}
        
        with pytest.raises(ValueError, match="Mismatched keys"):
            tensors_to_fields(tensor_dict, metadata_dict)


class TestRoundTripConversion:
    """Test round-trip conversions (Field -> Tensor -> Field)."""
    
    def test_centered_field_roundtrip(self):
        """Test that CenteredGrid survives a round trip."""
        domain = Box(x=1, y=1)
        resolution = spatial(x=32, y=32)
        
        # Create field with proper dimensions
        values = math.random_normal(resolution)
        
        original_field = CenteredGrid(
            values,
            extrapolation.PERIODIC,
            bounds=domain
        )
        
        # Extract metadata
        metadata = FieldMetadata.from_field(original_field)
        
        # Convert to tensor and back
        tensor = field_to_tensor(original_field)
        # Add time dimension for testing: [channels, x, y] -> [time, channels, x, y]
        tensor = tensor.unsqueeze(0)
        
        reconstructed_field = tensor_to_field(tensor, metadata, time_slice=0)
        
        # Check properties match
        assert reconstructed_field.is_staggered == original_field.is_staggered
        assert reconstructed_field.shape.spatial == original_field.shape.spatial
        assert reconstructed_field.extrapolation == original_field.extrapolation
        
        # Check values are close (may have minor numerical differences)
        original_values = original_field.values._native
        reconstructed_values = reconstructed_field.values._native
        # Ensure both tensors are on the same device
        if hasattr(original_values, 'device') and hasattr(reconstructed_values, 'device'):
            reconstructed_values = reconstructed_values.to(original_values.device)
        assert torch.allclose(original_values, reconstructed_values, atol=1e-6)
    
    def test_staggered_field_roundtrip(self):
        """Test that StaggeredGrid survives a round trip."""
        domain = Box(x=1, y=1)
        resolution = spatial(x=32, y=32)
        
        # Create vector field
        values = math.random_normal(resolution & channel(vector='x,y'))
        
        original_field = StaggeredGrid(
            values,
            extrapolation.ZERO,
            bounds=domain
        )
        
        metadata = FieldMetadata.from_field(original_field)
        
        tensor = field_to_tensor(original_field)
        tensor = tensor.unsqueeze(0)  # Add time dimension
        
        reconstructed_field = tensor_to_field(tensor, metadata, time_slice=0)
        
        assert reconstructed_field.is_staggered == original_field.is_staggered
        assert reconstructed_field.shape.spatial == original_field.shape.spatial
        assert reconstructed_field.extrapolation == original_field.extrapolation


class TestCreateFieldMetadataFromModel:
    """Test creating metadata from physical model instances."""
    
    def test_create_metadata_basic(self):
        """Test basic metadata creation from model-like object."""
        # Create a mock model object
        class MockModel:
            def __init__(self):
                self.domain = Box(x=1, y=1)
                self.resolution = spatial(x=64, y=64)
        
        model = MockModel()
        field_names = ['velocity', 'pressure']
        field_types = {'velocity': 'staggered', 'pressure': 'centered'}
        
        metadata_dict = create_field_metadata_from_model(model, field_names, field_types)
        
        assert 'velocity' in metadata_dict
        assert 'pressure' in metadata_dict
        assert metadata_dict['velocity'].field_type == 'staggered'
        assert metadata_dict['pressure'].field_type == 'centered'
        assert metadata_dict['velocity'].domain == model.domain
        assert metadata_dict['velocity'].resolution == model.resolution
    
    def test_default_field_types(self):
        """Test that fields default to 'centered' when types not specified."""
        class MockModel:
            def __init__(self):
                self.domain = Box(x=1, y=1)
                self.resolution = spatial(x=64, y=64)
        
        model = MockModel()
        field_names = ['temperature', 'concentration']
        
        metadata_dict = create_field_metadata_from_model(model, field_names)
        
        assert metadata_dict['temperature'].field_type == 'centered'
        assert metadata_dict['concentration'].field_type == 'centered'
    
    def test_velocity_gets_vector_channels(self):
        """Test that 'velocity' fields automatically get vector channel dims."""
        class MockModel:
            def __init__(self):
                self.domain = Box(x=1, y=1)
                self.resolution = spatial(x=64, y=64)
        
        model = MockModel()
        field_names = ['velocity', 'density']
        
        metadata_dict = create_field_metadata_from_model(model, field_names)
        
        assert metadata_dict['velocity'].channel_dims == ('vector',)
        assert metadata_dict['density'].channel_dims == ()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
