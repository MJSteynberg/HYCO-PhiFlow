"""
Unit tests for FieldTensorConverter class.

Tests the bidirectional conversion between PhiFlow Fields and PyTorch tensors,
including batch operations, channel concatenation, and error handling.
"""

import pytest
import torch
from phi.torch.flow import *
from phi import math
from phi.math import spatial, channel, batch as batch_dim
from phi.field import CenteredGrid, StaggeredGrid
from phi.geom import Box

from src.utils.field_conversion import (
    FieldTensorConverter,
    FieldMetadata,
    create_field_metadata_from_model
)


@pytest.fixture
def simple_domain():
    """Simple 2D domain for testing."""
    return Box(x=1, y=1)


@pytest.fixture
def simple_resolution():
    """Simple 2D resolution for testing."""
    return spatial(x=32, y=32)


@pytest.fixture
def scalar_metadata(simple_domain, simple_resolution):
    """Metadata for a scalar field (e.g., density)."""
    return FieldMetadata(
        domain=simple_domain,
        resolution=simple_resolution,
        extrapolation=extrapolation.PERIODIC,
        field_type='centered',
        spatial_dims=('x', 'y'),
        channel_dims=()
    )


@pytest.fixture
def vector_metadata(simple_domain, simple_resolution):
    """Metadata for a vector field (e.g., velocity)."""
    return FieldMetadata(
        domain=simple_domain,
        resolution=simple_resolution,
        extrapolation=extrapolation.PERIODIC,
        field_type='centered',
        spatial_dims=('x', 'y'),
        channel_dims=('vector',)
    )


@pytest.fixture
def staggered_metadata(simple_domain, simple_resolution):
    """Metadata for a staggered vector field."""
    return FieldMetadata(
        domain=simple_domain,
        resolution=simple_resolution,
        extrapolation=extrapolation.PERIODIC,
        field_type='staggered',
        spatial_dims=('x', 'y'),
        channel_dims=('vector',)
    )


class TestFieldTensorConverterInitialization:
    """Test converter initialization and setup."""
    
    def test_init_single_field(self, scalar_metadata):
        """Test initialization with single scalar field."""
        converter = FieldTensorConverter({'density': scalar_metadata})
        
        assert converter.field_names == ['density']
        assert converter.total_channels == 1
        assert converter.channel_counts['density'] == 1
        assert converter.channel_offsets['density'] == 0
    
    def test_init_multiple_fields(self, scalar_metadata, vector_metadata):
        """Test initialization with multiple fields."""
        converter = FieldTensorConverter({
            'density': scalar_metadata,
            'velocity': vector_metadata
        })
        
        assert set(converter.field_names) == {'density', 'velocity'}
        assert converter.total_channels == 3  # 1 for density + 2 for velocity
        
        # Check channel mapping
        channel_info = converter.get_channel_info()
        assert channel_info['density']['count'] == 1
        assert channel_info['velocity']['count'] == 2
    
    def test_channel_offset_calculation(self, scalar_metadata, vector_metadata):
        """Test that channel offsets are calculated correctly."""
        converter = FieldTensorConverter({
            'field1': scalar_metadata,
            'field2': vector_metadata,
            'field3': scalar_metadata
        })
        
        assert converter.channel_offsets['field1'] == 0
        assert converter.channel_offsets['field2'] == 1  # After 1 scalar
        assert converter.channel_offsets['field3'] == 3  # After 1 scalar + 1 vector


class TestScalarFieldConversion:
    """Test conversion of scalar fields."""
    
    def test_scalar_field_to_tensor_no_batch(self, simple_domain, simple_resolution, scalar_metadata):
        """Test converting single scalar field to tensor."""
        # Create a simple scalar field
        field = CenteredGrid(
            math.tensor(torch.ones(32, 32), simple_resolution),
            extrapolation.PERIODIC,
            bounds=simple_domain
        )
        
        converter = FieldTensorConverter({'density': scalar_metadata})
        tensor = converter.fields_to_tensors_batch({'density': field})
        
        # Should have shape [1, H, W] since no batch dimension
        assert tensor.shape == (1, 32, 32)
        assert torch.all(tensor == 1.0)
    
    def test_scalar_field_to_tensor_with_batch(self, simple_domain, simple_resolution, scalar_metadata):
        """Test converting batched scalar field to tensor."""
        # Create batched field
        batch_data = torch.ones(4, 32, 32)
        field = CenteredGrid(
            math.tensor(batch_data, batch_dim('batch') & simple_resolution),
            extrapolation.PERIODIC,
            bounds=simple_domain
        )
        
        converter = FieldTensorConverter({'density': scalar_metadata})
        tensor = converter.fields_to_tensors_batch({'density': field})
        
        # Should have shape [B, 1, H, W]
        assert tensor.shape == (4, 1, 32, 32)
        assert torch.all(tensor == 1.0)
    
    def test_tensor_to_scalar_field_no_batch(self, scalar_metadata):
        """Test converting tensor to scalar field without batch."""
        # Tensor in format [C, H, W]
        tensor = torch.ones(1, 32, 32) * 2.0
        
        converter = FieldTensorConverter({'density': scalar_metadata})
        fields = converter.tensors_to_fields_batch(tensor)
        
        assert 'density' in fields
        field = fields['density']
        assert field.shape.spatial.volume == 32 * 32
        # Use .all().all() for PhiML tensors
        assert (field.values == 2.0).all
    
    def test_tensor_to_scalar_field_with_batch(self, scalar_metadata):
        """Test converting batched tensor to scalar field."""
        # Tensor in format [B, C, H, W]
        tensor = torch.ones(4, 1, 32, 32) * 3.0
        
        converter = FieldTensorConverter({'density': scalar_metadata})
        fields = converter.tensors_to_fields_batch(tensor)
        
        assert 'density' in fields
        field = fields['density']
        assert 'batch' in field.shape or 'time' in field.shape  # Has batch dimension
        # Use .all().all() for PhiML tensors with batch dims
        assert (field.values == 3.0).all
    
    def test_roundtrip_scalar_field(self, simple_domain, simple_resolution, scalar_metadata):
        """Test field -> tensor -> field roundtrip for scalar."""
        # Create original field
        original_data = torch.randn(32, 32)
        original_field = CenteredGrid(
            math.tensor(original_data, simple_resolution),
            extrapolation.PERIODIC,
            bounds=simple_domain
        )
        
        # Convert to tensor and back
        converter = FieldTensorConverter({'density': scalar_metadata})
        tensor = converter.fields_to_tensors_batch({'density': original_field})
        reconstructed_fields = converter.tensors_to_fields_batch(tensor)
        reconstructed_field = reconstructed_fields['density']
        
        # Check values match (allowing for small numerical errors)
        # Use math.reshaped_native for proper comparison
        original_native = math.reshaped_native(original_field.values, ['x', 'y'])
        reconstructed_native = math.reshaped_native(reconstructed_field.values, ['x', 'y'])
        
        assert torch.allclose(
            original_native,
            reconstructed_native,
            atol=1e-5
        )


class TestVectorFieldConversion:
    """Test conversion of vector fields."""
    
    def test_vector_field_to_tensor_no_batch(self, simple_domain, simple_resolution, vector_metadata):
        """Test converting single vector field to tensor."""
        # Create vector field [x, y, vector=2]
        vector_data = torch.randn(32, 32, 2)
        field = CenteredGrid(
            math.tensor(vector_data, simple_resolution & channel(vector='x,y')),
            extrapolation.PERIODIC,
            bounds=simple_domain
        )
        
        converter = FieldTensorConverter({'velocity': vector_metadata})
        tensor = converter.fields_to_tensors_batch({'velocity': field})
        
        # Should have shape [2, H, W] (2 channels for vector components)
        assert tensor.shape == (2, 32, 32)
    
    def test_vector_field_to_tensor_with_batch(self, simple_domain, simple_resolution, vector_metadata):
        """Test converting batched vector field to tensor."""
        # Create batched vector field [batch, x, y, vector=2]
        vector_data = torch.randn(4, 32, 32, 2)
        field = CenteredGrid(
            math.tensor(vector_data, batch_dim('batch') & simple_resolution & channel(vector='x,y')),
            extrapolation.PERIODIC,
            bounds=simple_domain
        )
        
        converter = FieldTensorConverter({'velocity': vector_metadata})
        tensor = converter.fields_to_tensors_batch({'velocity': field})
        
        # Should have shape [B, 2, H, W]
        assert tensor.shape == (4, 2, 32, 32)
    
    def test_tensor_to_vector_field(self, vector_metadata):
        """Test converting tensor to vector field."""
        # Tensor in format [B, C, H, W] where C=2 for vector
        tensor = torch.randn(4, 2, 32, 32)
        
        converter = FieldTensorConverter({'velocity': vector_metadata})
        fields = converter.tensors_to_fields_batch(tensor)
        
        assert 'velocity' in fields
        field = fields['velocity']
        assert field.shape.channel.volume == 2  # Vector has 2 components
        assert field.shape.spatial.volume == 32 * 32
    
    def test_roundtrip_vector_field(self, simple_domain, simple_resolution, vector_metadata):
        """Test field -> tensor -> field roundtrip for vector."""
        # Create original vector field
        original_data = torch.randn(32, 32, 2)
        original_field = CenteredGrid(
            math.tensor(original_data, simple_resolution & channel(vector='x,y')),
            extrapolation.PERIODIC,
            bounds=simple_domain
        )
        
        # Convert to tensor and back
        converter = FieldTensorConverter({'velocity': vector_metadata})
        tensor = converter.fields_to_tensors_batch({'velocity': original_field})
        reconstructed_fields = converter.tensors_to_fields_batch(tensor)
        reconstructed_field = reconstructed_fields['velocity']
        
        # Check values match
        # Use math.reshaped_native with proper dimension order
        original_native = math.reshaped_native(original_field.values, ['x', 'y', 'vector'])
        reconstructed_native = math.reshaped_native(reconstructed_field.values, ['x', 'y', 'vector'])
        
        assert torch.allclose(
            original_native,
            reconstructed_native,
            atol=1e-5
        )


class TestMultiFieldConversion:
    """Test conversion with multiple fields simultaneously."""
    
    def test_multiple_fields_to_tensor(self, simple_domain, simple_resolution, 
                                       scalar_metadata, vector_metadata):
        """Test converting multiple fields to single concatenated tensor."""
        # Create fields
        density_field = CenteredGrid(
            math.tensor(torch.ones(32, 32), simple_resolution),
            extrapolation.PERIODIC,
            bounds=simple_domain
        )
        
        velocity_field = CenteredGrid(
            math.tensor(torch.ones(32, 32, 2) * 2, simple_resolution & channel(vector='x,y')),
            extrapolation.PERIODIC,
            bounds=simple_domain
        )
        
        converter = FieldTensorConverter({
            'density': scalar_metadata,
            'velocity': vector_metadata
        })
        
        tensor = converter.fields_to_tensors_batch({
            'density': density_field,
            'velocity': velocity_field
        })
        
        # Should have 3 channels total (1 + 2)
        assert tensor.shape == (3, 32, 32)
        
        # Check that density channel is 1.0 and velocity channels are 2.0
        assert torch.all(tensor[0] == 1.0)  # Density
        assert torch.all(tensor[1] == 2.0)  # Velocity x
        assert torch.all(tensor[2] == 2.0)  # Velocity y
    
    def test_tensor_to_multiple_fields(self, scalar_metadata, vector_metadata):
        """Test splitting concatenated tensor into multiple fields."""
        # Create concatenated tensor [3, H, W]
        tensor = torch.zeros(3, 32, 32)
        tensor[0] = 1.0  # Density
        tensor[1] = 2.0  # Velocity x
        tensor[2] = 3.0  # Velocity y
        
        converter = FieldTensorConverter({
            'density': scalar_metadata,
            'velocity': vector_metadata
        })
        
        fields = converter.tensors_to_fields_batch(tensor)
        
        # Check both fields are present
        assert 'density' in fields
        assert 'velocity' in fields
        
        # Check density values using reshaped_native
        density_values = math.reshaped_native(fields['density'].values, ['x', 'y'])
        assert torch.all(density_values == 1.0)
        
        # Check velocity values using reshaped_native
        velocity_values = math.reshaped_native(fields['velocity'].values, ['x', 'y', 'vector'])
        assert torch.all(velocity_values[..., 0] == 2.0)  # x component
        assert torch.all(velocity_values[..., 1] == 3.0)  # y component
    
    def test_roundtrip_multiple_fields(self, simple_domain, simple_resolution,
                                       scalar_metadata, vector_metadata):
        """Test roundtrip conversion with multiple fields."""
        # Create original fields
        density_data = torch.randn(32, 32)
        velocity_data = torch.randn(32, 32, 2)
        
        original_fields = {
            'density': CenteredGrid(
                math.tensor(density_data, simple_resolution),
                extrapolation.PERIODIC,
                bounds=simple_domain
            ),
            'velocity': CenteredGrid(
                math.tensor(velocity_data, simple_resolution & channel(vector='x,y')),
                extrapolation.PERIODIC,
                bounds=simple_domain
            )
        }
        
        converter = FieldTensorConverter({
            'density': scalar_metadata,
            'velocity': vector_metadata
        })
        
        # Roundtrip conversion
        tensor = converter.fields_to_tensors_batch(original_fields)
        reconstructed_fields = converter.tensors_to_fields_batch(tensor)
        
        # Check both fields match using reshaped_native
        density_orig = math.reshaped_native(original_fields['density'].values, ['x', 'y'])
        density_recon = math.reshaped_native(reconstructed_fields['density'].values, ['x', 'y'])
        
        velocity_orig = math.reshaped_native(original_fields['velocity'].values, ['x', 'y', 'vector'])
        velocity_recon = math.reshaped_native(reconstructed_fields['velocity'].values, ['x', 'y', 'vector'])
        
        assert torch.allclose(density_orig, density_recon, atol=1e-5)
        assert torch.allclose(velocity_orig, velocity_recon, atol=1e-5)


class TestBatchedConversion:
    """Test conversion with batch dimensions."""
    
    def test_batched_multiple_fields(self, simple_domain, simple_resolution,
                                     scalar_metadata, vector_metadata):
        """Test converting batched multiple fields."""
        batch_size = 4
        
        # Create batched fields
        density_data = torch.randn(batch_size, 32, 32)
        velocity_data = torch.randn(batch_size, 32, 32, 2)
        
        fields = {
            'density': CenteredGrid(
                math.tensor(density_data, batch_dim('batch') & simple_resolution),
                extrapolation.PERIODIC,
                bounds=simple_domain
            ),
            'velocity': CenteredGrid(
                math.tensor(velocity_data, batch_dim('batch') & simple_resolution & channel(vector='x,y')),
                extrapolation.PERIODIC,
                bounds=simple_domain
            )
        }
        
        converter = FieldTensorConverter({
            'density': scalar_metadata,
            'velocity': vector_metadata
        })
        
        tensor = converter.fields_to_tensors_batch(fields)
        
        # Should have shape [B, C, H, W]
        assert tensor.shape == (batch_size, 3, 32, 32)
    
    def test_batched_roundtrip(self, simple_domain, simple_resolution,
                               scalar_metadata, vector_metadata):
        """Test roundtrip with batched data."""
        batch_size = 8
        
        # Create original batched fields
        density_data = torch.randn(batch_size, 32, 32)
        velocity_data = torch.randn(batch_size, 32, 32, 2)
        
        original_fields = {
            'density': CenteredGrid(
                math.tensor(density_data, batch_dim('batch') & simple_resolution),
                extrapolation.PERIODIC,
                bounds=simple_domain
            ),
            'velocity': CenteredGrid(
                math.tensor(velocity_data, batch_dim('batch') & simple_resolution & channel(vector='x,y')),
                extrapolation.PERIODIC,
                bounds=simple_domain
            )
        }
        
        converter = FieldTensorConverter({
            'density': scalar_metadata,
            'velocity': vector_metadata
        })
        
        # Roundtrip
        tensor = converter.fields_to_tensors_batch(original_fields)
        reconstructed_fields = converter.tensors_to_fields_batch(tensor)
        
        # Check values match using reshaped_native with batch dimension
        density_orig = math.reshaped_native(original_fields['density'].values, ['batch', 'x', 'y'])
        density_recon = math.reshaped_native(reconstructed_fields['density'].values, ['time', 'x', 'y'])  # batch becomes 'time'
        
        velocity_orig = math.reshaped_native(original_fields['velocity'].values, ['batch', 'x', 'y', 'vector'])
        velocity_recon = math.reshaped_native(reconstructed_fields['velocity'].values, ['time', 'x', 'y', 'vector'])
        
        assert torch.allclose(density_orig, density_recon, atol=1e-5)
        assert torch.allclose(velocity_orig, velocity_recon, atol=1e-5)


class TestValidation:
    """Test validation methods."""
    
    def test_validate_fields_correct(self, simple_domain, simple_resolution,
                                     scalar_metadata, vector_metadata):
        """Test validation passes for correct fields."""
        fields = {
            'density': CenteredGrid(
                math.tensor(torch.ones(32, 32), simple_resolution),
                extrapolation.PERIODIC,
                bounds=simple_domain
            ),
            'velocity': CenteredGrid(
                math.tensor(torch.ones(32, 32, 2), simple_resolution & channel(vector='x,y')),
                extrapolation.PERIODIC,
                bounds=simple_domain
            )
        }
        
        converter = FieldTensorConverter({
            'density': scalar_metadata,
            'velocity': vector_metadata
        })
        
        assert converter.validate_fields(fields) is True
    
    def test_validate_fields_wrong_names(self, simple_domain, simple_resolution, scalar_metadata):
        """Test validation fails for wrong field names."""
        fields = {
            'wrong_name': CenteredGrid(
                math.tensor(torch.ones(32, 32), simple_resolution),
                extrapolation.PERIODIC,
                bounds=simple_domain
            )
        }
        
        converter = FieldTensorConverter({'density': scalar_metadata})
        
        with pytest.raises(ValueError, match="Field names mismatch"):
            converter.validate_fields(fields)
    
    def test_validate_tensor_correct(self, scalar_metadata, vector_metadata):
        """Test validation passes for correct tensor."""
        converter = FieldTensorConverter({
            'density': scalar_metadata,
            'velocity': vector_metadata
        })
        
        # Correct shape [B, C, H, W] with C=3
        tensor = torch.randn(4, 3, 32, 32)
        assert converter.validate_tensor(tensor) is True
        
        # Correct shape [C, H, W] with C=3
        tensor_no_batch = torch.randn(3, 32, 32)
        assert converter.validate_tensor(tensor_no_batch) is True
    
    def test_validate_tensor_wrong_channels(self, scalar_metadata, vector_metadata):
        """Test validation fails for wrong channel count."""
        converter = FieldTensorConverter({
            'density': scalar_metadata,
            'velocity': vector_metadata
        })
        
        # Wrong number of channels (should be 3, not 5)
        tensor = torch.randn(4, 5, 32, 32)
        
        with pytest.raises(ValueError, match="Expected 3 channels"):
            converter.validate_tensor(tensor)
    
    def test_validate_tensor_wrong_dimensions(self, scalar_metadata):
        """Test validation fails for wrong number of dimensions."""
        converter = FieldTensorConverter({'density': scalar_metadata})
        
        # Wrong dimensionality (2D instead of 3D or 4D)
        tensor = torch.randn(32, 32)
        
        with pytest.raises(ValueError, match="Expected tensor with 3 or 4 dimensions"):
            converter.validate_tensor(tensor)


class TestChannelInfo:
    """Test channel information methods."""
    
    def test_get_channel_info(self, scalar_metadata, vector_metadata):
        """Test getting channel layout information."""
        converter = FieldTensorConverter({
            'density': scalar_metadata,
            'velocity': vector_metadata,
            'pressure': scalar_metadata
        })
        
        info = converter.get_channel_info()
        
        assert info['density'] == {'count': 1, 'offset': 0}
        assert info['velocity'] == {'count': 2, 'offset': 1}
        assert info['pressure'] == {'count': 1, 'offset': 3}


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_mismatched_field_names_to_tensor(self, scalar_metadata, vector_metadata):
        """Test error when field names don't match metadata."""
        converter = FieldTensorConverter({
            'density': scalar_metadata,
            'velocity': vector_metadata
        })
        
        # Missing 'velocity' field
        fields = {
            'density': CenteredGrid(
                math.tensor(torch.ones(32, 32), spatial(x=32, y=32)),
                extrapolation.PERIODIC,
                bounds=Box(x=1, y=1)
            )
        }
        
        with pytest.raises(ValueError, match="Field names mismatch"):
            converter.fields_to_tensors_batch(fields)
    
    def test_wrong_channel_count_to_fields(self, scalar_metadata):
        """Test error when tensor has wrong number of channels."""
        converter = FieldTensorConverter({'density': scalar_metadata})
        
        # Tensor has 3 channels but should have 1
        tensor = torch.randn(3, 32, 32)
        
        with pytest.raises(ValueError, match="Expected 1 channels"):
            converter.tensors_to_fields_batch(tensor)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
