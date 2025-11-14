"""BVTS (Batch, Vector, Time, Spatial...) helpers

Helpers to convert common input tensor layouts into the repository's
canonical BVTS layout: [B, V, T, *spatial]. These are used by the
data generation / caching pipeline and dataset loaders when normalizing
cached data.

Supported input layouts include per-field channel-first tensors
[C, T, H, W], single-frame [C, H, W], time-major [T, C, H, W], and already-BVTS
tensors. The functions raise clear errors for unexpected shapes to avoid
silent misinterpretation.
"""

from typing import Optional
import torch


def to_bvts(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a tensor into the canonical BVTS layout.

    Supported input shapes (common cases):
    - [T, C, H, W] -> [1, C, T, H, W]
    - [C, T, H, W] -> [1, C, T, H, W]
    - [C, H, W] -> [1, C, 1, H, W]
    - [B, V, T, H, W] -> returned unchanged

    Raises ValueError for unexpected shapes.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("to_bvts expects a torch.Tensor")

    # Already BVTS (common 2D spatial case)
    if tensor.dim() >= 5:
        # Assume [B, V, T, *spatial]
        return tensor

    # Time-major 4D: [T, C, H, W]
    if tensor.dim() == 4:
        # -> [1, C, T, H, W]
        return tensor.permute(1, 0, 2, 3)

    # Single-frame with channel: [C, H, W]
    if tensor.dim() == 3:
        # -> [1, C, 1, H, W]
        return tensor.unsqueeze(0).unsqueeze(2)

    # Single-frame scalar spatial [H, W]
    if tensor.dim() == 2:
        # -> [1, 1, 1, H, W]
        return tensor.unsqueeze(0).unsqueeze(0).unsqueeze(2)

    raise ValueError(f"Unsupported tensor shape for to_bvts(): {tuple(tensor.shape)}")


def from_bvts(tensor: torch.Tensor, squeeze_batch: bool = True) -> torch.Tensor:
    """Convert BVTS -> [T, C, *spatial].

    If `squeeze_batch` is True and batch dim == 1, the returned tensor will
    be shaped [T, C, *spatial]. If batch > 1 and squeezing is requested,
    a ValueError is raised since downstream consumers typically expect a
    single-simulation tensor when using non-BVTS utilities.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("from_bvts expects a torch.Tensor")

    if tensor.dim() >= 5:
        # [B, V, T, *spatial] -> if B==1 and squeeze_batch True then -> [V, T, *spatial]
        if squeeze_batch:
            if tensor.shape[0] != 1:
                raise ValueError(
                    "from_bvts currently only supports batch size 1 when converting to non-BVTS format"
                )
            t = tensor.squeeze(0)  # [V, T, *spatial]
        else:
            # Keeping a batch dimension is not supported for conversion to non-BVTS formats
            raise ValueError(
                "from_bvts: keeping batch dimension is not supported for non-BVTS conversion"
            )

        # Now convert [V, T, *spatial] -> [T, V, *spatial]
        perm = (1, 0) + tuple(range(2, t.dim()))
        return t.permute(*perm).contiguous()

    # If already [V, T, *spatial]
    if tensor.dim() >= 4 and tensor.shape[0] == tensor.shape[0]:
        # Heuristic: treat as [V, T, *spatial]
        perm = (1, 0) + tuple(range(2, tensor.dim()))
        return tensor.permute(*perm).contiguous()

    raise ValueError(f"Unsupported BVTS-like tensor shape for from_bvts(): {tuple(tensor.shape)}")


def assert_bvts(tensor: torch.Tensor) -> None:
    """Assert that a tensor follows the BVTS contract: [B, V, T, *spatial].

    This is a lightweight assertion to catch obvious mistakes during migration.
    """
    if not isinstance(tensor, torch.Tensor):
        raise AssertionError("assert_bvts: not a torch.Tensor")

    if tensor.dim() < 4:
        raise AssertionError(
            f"assert_bvts: BVTS tensor must have >=4 dims (batch, vector, time, spatial), got {tensor.dim()}"
        )

    # Basic shape sanity checks: batch and vector dims should be >=1
    if tensor.shape[0] < 1 or tensor.shape[1] < 1:
        raise AssertionError("assert_bvts: batch and vector dims must be >= 1")

