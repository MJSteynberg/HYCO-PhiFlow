"""
layout.py

Minimal helpers for the canonical tensor layout used across the codebase.

Canonical tensor layout (everywhere):
    [B, T, *spatial, V]

Where:
 - B: batch dimension (always present)
 - T: time dimension (always present)
 - *spatial: 1..N spatial dims (H, W, ...)
 - V: vector channel dimension (always present; V==1 for scalars)

These helpers are intentionally small — converters are strict and assume
the inputs were produced by the opposite converter.
"""
from typing import Sequence, Optional
import torch


def canonical_from_phiflow_native(native: torch.Tensor, *, is_vector: bool) -> torch.Tensor:
    """
    Convert a PhiFlow-style native tensor into canonical [B, T, *spatial, V].

    Expected *native* layout (strict):
      - Vector fields with time:  [*spatial, V, T]   (e.g. [H, W, V, T])
      - Scalar fields with time:  [*spatial, T]      (e.g. [H, W, T])

    This function performs the deterministic reorder:
      - Move time -> first axis of the returned time/spatial group
      - Move vector -> last axis of the returned time/spatial group
      - Add batch dim B=1 (converters always operate on single Field -> B=1)

    NOTE: We do NOT attempt to detect arbitrary native shapes. We strictly
    expect native shapes matching the above patterns because earlier
    converter logic guarantees the PhiFlow native order in our codebase.
    """
    t = native
    if is_vector:
        # Expect shape [*spatial, V, T]
        if t.dim() < 2:
            raise ValueError(f"Unexpected native vector tensor shape {tuple(t.shape)}")
        # Move T to first position of group: [T, *spatial, V]
        # Current: [..., V, T] -> permute to place T at index 0 of group
        # Easiest approach: bring T to front by moving last axis to front,
        # then move V to last axis (if V isn't already last due to permute order).
        t = t.permute(-1, *range(0, t.dim() - 2), t.dim() - 2)
        # Now t has shape [T, *spatial, V]
    else:
        # Expect shape [*spatial, T]
        if t.dim() < 1:
            raise ValueError(f"Unexpected native scalar tensor shape {tuple(t.shape)}")
        t = t.permute(-1, *range(0, t.dim() - 1))  # -> [T, *spatial]
        t = t.unsqueeze(-1)  # -> [T, *spatial, 1]

    # Add batch dim B=1 at axis 0 -> [1, T, *spatial, V]
    t = t.unsqueeze(0)
    return t


def canonical_to_phiflow_native(canonical: torch.Tensor) -> torch.Tensor:
    """
    Convert canonical [B, T, *spatial, V] -> PhiFlow native
    Format for PhiFlow native expected by constructors in this repo:
      - Vector fields: [*spatial, V, T]
      - Scalar fields: [*spatial, T] (we will return a trailing dimension only when V==1)
    This function will drop batch (takes first sample).
    """
    if canonical.dim() < 4:
        raise ValueError("canonical tensor must have at least 4 dims: [B, T, *spatial, V]")
    # Use first batch element
    tensor = canonical[0]  # [T, *spatial, V]
    T = tensor.shape[0]
    V = tensor.shape[-1]
    spatial_rank = tensor.dim() - 2  # excluding T and V
    # We want native: [*spatial, V, T] (vector) or [*spatial, T] (scalar)
    # Current: [T, *spatial, V]
    permute_order = list(range(1, 1 + spatial_rank)) + [tensor.dim() - 1, 0]
    native = tensor.permute(*permute_order)  # -> [*spatial, V, T]
    if V == 1:
        # For scalar fields, return [*spatial, T] (drop trailing V dim)
        native = native.squeeze(-2)  # remove V dimension (which was at -2 after permute)
    return native


def canonical_to_conv_input(canonical: torch.Tensor, merge_batch_time: bool = True) -> torch.Tensor:
    """
    Convert canonical -> PyTorch conv input.
    canonical: [B, T, *spatial, V]
    returns:
      - if merge_batch_time: [B*T, V, *spatial]  (channels-first, merged batch-time)
      - else: [B, T, V, *spatial]
    """
    if canonical.dim() < 4:
        raise ValueError("canonical tensor must have at least 4 dims: [B, T, *spatial, V]")

    b, t = canonical.shape[0], canonical.shape[1]
    spatial = canonical.shape[2:-1]
    v = canonical.shape[-1]

    # move vector to channel position: [B, T, V, *spatial]
    perm = [0, 1, -1] + list(range(2, 2 + len(spatial)))
    ct = canonical.permute(*perm)
    if merge_batch_time:
        ct = ct.reshape(b * t, v, *spatial)
    return ct
