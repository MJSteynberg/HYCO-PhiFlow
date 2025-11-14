"""BVTS validation utilities.

Provides:
 - BVTSValidationError: explicit exception type
 - assert_bvts_format: strict check for BVTS-shaped tensors
 - validate_bvts_dataset: quick dataset producer check
 - requires_bvts: decorator to validate tensor inputs/outputs

These helpers are small and explicit to make enforcement easy at
API boundaries (datasets, loaders, model forward calls, converters).
"""
from functools import wraps
from typing import Literal
import torch


class BVTSValidationError(ValueError):
    """Raised when a tensor does not conform to the BVTS contract."""


def assert_bvts_format(
    tensor: torch.Tensor,
    expected_spatial_dims: Literal[2, 3] = 2,
    min_time_steps: int = 1,
    context: str = "",
) -> None:
    """Assert that `tensor` is in BVTS format: [B, V, T, *spatial].

    Args:
        tensor: tensor to validate
        expected_spatial_dims: 2 for 2D (H, W), 3 for 3D (D, H, W)
        min_time_steps: minimum allowed length for the time axis
        context: optional context string for clearer errors

    Raises:
        BVTSValidationError on any violation
    """
    ctx = f" [{context}]" if context else ""

    if not isinstance(tensor, torch.Tensor):
        raise BVTSValidationError(f"Expected torch.Tensor{ctx}; got {type(tensor)}")

    expected_dims = 3 + expected_spatial_dims  # B, V, T + spatial
    if tensor.dim() != expected_dims:
        raise BVTSValidationError(
            f"BVTS tensor must have {expected_dims} dims [B, V, T, {'H, W' if expected_spatial_dims==2 else 'D, H, W'}], got {tensor.dim()}{ctx}"
        )

    if tensor.shape[0] < 1:
        raise BVTSValidationError(f"BVTS batch dimension must be >=1, got {tensor.shape[0]}{ctx}")

    if tensor.shape[1] < 1:
        raise BVTSValidationError(f"BVTS vector dimension must be >=1, got {tensor.shape[1]}{ctx}")

    if tensor.shape[2] < min_time_steps:
        raise BVTSValidationError(
            f"BVTS time dimension must be >= {min_time_steps}, got {tensor.shape[2]}{ctx}"
        )

    # spatial dims
    for i, dim_name in enumerate(['H', 'W'] if expected_spatial_dims == 2 else ['D', 'H', 'W']):
        dim_idx = 3 + i
        if tensor.shape[dim_idx] < 1:
            raise BVTSValidationError(
                f"BVTS spatial dimension {dim_name} must be >=1, got {tensor.shape[dim_idx]}{ctx}"
            )


def validate_bvts_dataset(dataset) -> None:
    """Validate that a dataset produces BVTS tensors for its first sample.

    This function is intentionally lightweight: it will attempt to index
    dataset[0] and inspect returned objects. If tensors are encountered
    they are validated; non-tensor results are ignored (for field datasets).

    Raises BVTSValidationError on failure.
    """
    try:
        sample = dataset[0]
    except Exception as e:
        raise BVTSValidationError(f"Failed to index dataset for validation: {e}")

    # Helper: strict validation - do not coerce; raise with diagnostics
    def _try_normalize_and_validate(item, ctx):
        if not isinstance(item, torch.Tensor):
            return  # non-tensor items ignored

        # Strict options accepted for dataset samples:
        # - Batched BVTS: 5D [B, V, T, *spatial] (B may be 1)
        # - Per-sample BVTS-like: 4D [V, T, *spatial] (common Dataset.__getitem__ form)
        # - Single-frame per-sample: 3D [V, *spatial] (treated as T==1)
        if isinstance(item, torch.Tensor):
            if item.dim() == 5:
                # Batched BVTS
                assert_bvts_format(item, context=ctx)
                return
            if item.dim() == 4:
                # Per-sample [V, T, *spatial]
                if item.shape[0] < 1 or item.shape[1] < 1:
                    raise BVTSValidationError(
                        f"Dataset {dataset.__class__.__name__} produced invalid per-sample tensor for {ctx}; "
                        f"shape={tuple(item.shape)}, expected [V,T,*spatial] with V>=1,T>=1"
                    )
                return
            if item.dim() == 3:
                # Per-sample single-frame [V, *spatial]
                if item.shape[0] < 1:
                    raise BVTSValidationError(
                        f"Dataset {dataset.__class__.__name__} produced invalid per-sample tensor for {ctx}; "
                        f"shape={tuple(item.shape)}, expected [V,*spatial] with V>=1"
                    )
                return

        sample_repr = repr(item)
        if isinstance(sample_repr, str) and len(sample_repr) > 200:
            sample_repr = sample_repr[:200] + "..."

        raise BVTSValidationError(
            f"Dataset {dataset.__class__.__name__} did not produce a BVTS-like tensor for {ctx}; got {getattr(item, 'shape', type(item))}, dtype={getattr(item, 'dtype', None)}. "
            f"Expected per-sample [V,T,H,W] (or [V,H,W]) or batched [B,V,T,H,W]. Sample repr: {sample_repr}"
        )

    # Tuple-based samples: (input, target) or (input, target, meta)
    if isinstance(sample, (tuple, list)):
        for i, item in enumerate(sample):
            _try_normalize_and_validate(item, f"{dataset.__class__.__name__}[0][{i}]")
    else:
        _try_normalize_and_validate(sample, f"{dataset.__class__.__name__}[0]")


def requires_bvts(func):
    """Decorator that asserts BVTS format for all tensor args/returns.

    It validates positional and keyword tensor arguments before calling
    the function. It will also validate any returned tensors (or tuple
    of tensors).
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Validate tensor args
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                assert_bvts_format(arg, context=f"{func.__name__} arg {i}")
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                assert_bvts_format(v, context=f"{func.__name__} kwarg {k}")

        result = func(*args, **kwargs)

        # Validate result if tensor or tuple/list of tensors
        if isinstance(result, torch.Tensor):
            assert_bvts_format(result, context=f"{func.__name__} output")
        elif isinstance(result, (tuple, list)):
            for i, item in enumerate(result):
                if isinstance(item, torch.Tensor):
                    assert_bvts_format(item, context=f"{func.__name__} output[{i}]")

        return result

    return wrapper
