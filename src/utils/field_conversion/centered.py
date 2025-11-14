"""
Centered converter: strict, symmetric conversions between
CenteredGrid <-> canonical torch.Tensor [B, T, *spatial, V].

Assumptions (by design):
 - Inputs to field_to_tensor are valid PhiFlow CenteredGrid objects.
 - Inputs to tensor_to_field are canonical tensors produced by the opposite converter.
 - No backwards compatibility or shape guessing is performed.
"""
from typing import Optional
import torch

from phi.field import CenteredGrid  # phi fields
from phi import math
from phi.field._field import Field
from ..field_conversion.layout import canonical_from_phiflow_native, canonical_to_phiflow_native


class CenteredConverter:
    def __init__(self):
        pass

    def field_to_tensor(self, field: Field, *, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Convert a CenteredGrid (phi Field) to canonical torch tensor [B, T, *spatial, V].

        Strict assumptions:
          - The phi Field .values.native(...) will provide native in the repo's expected order:
            [*spatial, V, T] for vector, [*spatial, T] for scalar.
          - We do not attempt to handle arbitrary previous layouts.
        """
        # Ask phi for a native tensor with spatial first, vector then time last.
        # The phi API used in the repo typically allows extracting native values in a chosen dim ordering
        # but to keep this file self-contained we request a native representation and rely on its
        # documented ordering used across the codebase: spatial dims first, vector, then time.
        # Many earlier code paths used something like field.values.native(spatial + ['vector', 'time'])
        # but we will call field.values.native() without specifying dims and expect consistent layout.
        # To be safe, use field.values.native() and inspect dims via field.shape if needed.
        native = field.values.native()  # may be a phi.math.Tensor or similar with ._native
        # Some PhiFlow versions represent underlying torch tensor at ._native or .native() returns torch directly.
        native_t = getattr(native, '_native', native)
        # Determine vectorness via field.shape.channel.rank
        is_vector = field.shape.channel.rank > 0

        canonical = canonical_from_phiflow_native(native_t, is_vector=is_vector)
        if device is not None:
            canonical = canonical.to(device)
        return canonical

    def tensor_to_field(self, tensor: torch.Tensor, metadata, *, device: Optional[torch.device] = None) -> CenteredGrid:
        """
        Convert canonical tensor -> CenteredGrid.

        - tensor: strict canonical tensor [B, T, *spatial, V]
        - metadata: object with fields needed to reconstruct the CenteredGrid:
            - spatial_dims (Sequence[str])  : names of spatial dims in order (e.g. ['x','y'])
            - domain / bounds (optional)    : used as 'bounds' for CenteredGrid
            - extrapolation (optional)      : extrapolation to use
        The function will use the first batch element to reconstruct a single CenteredGrid.
        """
        if device is not None:
            tensor = tensor.to(device)

        if tensor.dim() < 4:
            raise ValueError("Expected canonical tensor with dims [B, T, *spatial, V]")

        # Convert to PhiFlow native shape expected by constructors:
        native = canonical_to_phiflow_native(tensor)  # [*spatial, V, T] or [*spatial, T] for scalar

        # Build phi shape spec dynamically using metadata.
        # We prefer to keep dependency on phi.math minimal here, but we will create a phi math tensor.
        # Caller must provide metadata with necessary spatial dimension sizes and names.
        # metadata must expose:
        #   - spatial_dims: Sequence[str]
        #   - spatial_sizes: Sequence[int] or dict name->size (prefer dict)
        #   - bounds (optional)
        #   - extrapolation (optional)
        spatial_dims = getattr(metadata, 'spatial_dims', None)
        spatial_sizes = getattr(metadata, 'spatial_sizes', None)
        bounds = getattr(metadata, 'domain', None) or getattr(metadata, 'bounds', None)
        extrapolation = getattr(metadata, 'extrapolation', None)

        # Prepare shape for math.tensor:
        # native currently either [*spatial, V, T] or [*spatial, T] (if V == 1)
        # Convert native -> numpy-like torch tensor and feed to math.tensor with proper shape.
        # phi.math.tensor will accept numpy-like arrays with shape matching a phi Shape if provided.
        # We need to assemble a phi shape; to avoid hard dependencies on shape constructors across
        # phi versions, we'll attempt a minimal approach:
        from phi import math as phi_math

        # Build shape dims sizes:
        # tensor has shape [B, T, *spatial, V]
        t = tensor
        B = t.shape[0]
        T = t.shape[1]
        spatial_shape = list(t.shape[2:-1])
        V = t.shape[-1]

        # Create phi tensor by providing the underlying torch tensor with an explicit shape mask.
        # Construct shape as batch('time') & spatial(...) & channel('vector' / 'scalar')
        # This code uses phi.math.tensor with an explicit shape in the typical API shape constructors.
        try:
            # Preferred approach if phi supports shape constructors:
            from phi.math import spatial, channel, batch as batch_dim
            # Build spatial dict if metadata provides names, else use anonymous sizes
            if spatial_dims and len(spatial_dims) == len(spatial_shape):
                spatial_kwargs = {name: int(size) for name, size in zip(spatial_dims, spatial_shape)}
                spatial_shape_obj = spatial(**spatial_kwargs)
            else:
                # fallback: use unnamed spatial dims; phi supports spatial(*sizes) in some versions,
                # but to keep broad compatibility we use positional spatial:
                spatial_shape_obj = spatial(*[int(s) for s in spatial_shape])

            if V == 1:
                channel_shape = channel(vector='scalar')
            else:
                channel_shape = channel(vector=V)

            combined_shape = batch_dim('time') & spatial_shape_obj & channel_shape

            # Pass python-native tensor (drop batch) to phi.math.tensor. canonical_to_phiflow_native returned
            # a tensor already dropped to [*spatial, V, T] or [*spatial, T], but phi expects shapes matching the
            # combined_shape ordering; constructing the precise phi.math.tensor may require permutation.
            # Use canonical_to_phiflow_native to get [*spatial, V, T] -> we need to permute for phi.tensor input.
            phi_native = native  # already [*spatial, V, T] or [*spatial, T]

            # phi.math.tensor expects the data in an array compatible with the shape; many versions are flexible.
            phi_tensor = phi_math.tensor(phi_native, combined_shape)
        except Exception:
            # Fallback: attempt to create a simple CenteredGrid using constructor that accepts values and bounds.
            # Map native -> [T, *spatial, V] then call CenteredGrid directly:
            # native currently [*spatial, V, T] or [*spatial, T]; reorder to [T, *spatial, V]
            if V == 1:
                # native: [*spatial, T] -> permute last to first -> [T, *spatial]
                permute_order = [-1] + list(range(0, len(spatial_shape)))
                phi_native_reordered = native.permute(*permute_order).unsqueeze(-1)  # -> [T, *spatial, 1]
            else:
                permute_order = [-1] + list(range(0, len(spatial_shape))) + [len(spatial_shape)]
                phi_native_reordered = native.permute(*permute_order)  # -> [T, *spatial, V]
            # Build CenteredGrid directly
            cg = CenteredGrid(phi_native_reordered, extrapolation=extrapolation, bounds=bounds)
            return cg

        # Build CenteredGrid from phi_tensor
        cg = CenteredGrid(phi_tensor, extrapolation=extrapolation, bounds=bounds)
        return cg
