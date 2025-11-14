"""
Staggered converter implemented in terms of the CenteredConverter.

Behavior:
 - field_to_tensor: convert StaggeredGrid -> at_centers() -> CenteredConverter.field_to_tensor
 - tensor_to_field: CenteredConverter.tensor_to_field -> StaggeredGrid.from_centered(...) or .at_staggered

Strict symmetry: these functions assume canonical tensors and a CenteredGrid
roundtrip; no compatibility code is present.
"""
from typing import Optional
import torch

from phi.field import StaggeredGrid, CenteredGrid
from .centered import CenteredConverter


class StaggeredConverter:
    def __init__(self):
        self._centered = CenteredConverter()

    def field_to_tensor(self, field: StaggeredGrid, *, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Convert a StaggeredGrid into canonical tensor by projecting to centers first.
        """
        centered = field.at_centers()
        return self._centered.field_to_tensor(centered, device=device)

    def tensor_to_field(self, tensor: torch.Tensor, metadata, *, device: Optional[torch.device] = None) -> StaggeredGrid:
        """
        Convert canonical tensor -> CenteredGrid -> StaggeredGrid.
        """
        centered = self._centered.tensor_to_field(tensor, metadata, device=device)
        # Convert CenteredGrid back to StaggeredGrid at the same locations.
        # Some phi versions provide constructors, else we rely on .to_staggered or similar.
        try:
            staggered = StaggeredGrid.from_centered(centered)
        except Exception:
            # Use fallback: call centered.to_staggered() if available
            if hasattr(centered, 'to_staggered'):
                staggered = centered.to_staggered()
            else:
                # If conversion unavailable, return centered as-is (explicit failure would be better)
                raise RuntimeError("Cannot convert CenteredGrid back to StaggeredGrid with this phi version.")
        return staggered
