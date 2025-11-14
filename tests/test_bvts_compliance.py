"""Lightweight BVTS compliance tests.

These tests avoid heavy project fixtures and instead validate the
validation helpers operate as expected.
"""
import pytest
import torch

from src.utils.field_conversion.validation import (
    assert_bvts_format,
    BVTSValidationError,
    validate_bvts_dataset,
    requires_bvts,
)


def test_assert_bvts_accepts_valid_tensor():
    t = torch.randn(1, 2, 1, 8, 8)
    # Should not raise
    assert_bvts_format(t)


def test_assert_bvts_rejects_invalid_shape():
    bad = torch.randn(3, 4, 5)
    with pytest.raises(BVTSValidationError):
        assert_bvts_format(bad)


def test_validate_bvts_dataset_accepts_good_dataset():
    class GoodDataset:
        def __getitem__(self, idx):
            return (torch.randn(1, 2, 1, 8, 8), torch.randn(1, 2, 1, 8, 8))

    ds = GoodDataset()
    # Should not raise
    validate_bvts_dataset(ds)


def test_validate_bvts_dataset_rejects_bad_dataset():
    class BadDataset:
        def __getitem__(self, idx):
            return torch.randn(4, 5, 6)

    ds = BadDataset()
    with pytest.raises(Exception):
        validate_bvts_dataset(ds)


def test_requires_bvts_decorator_validates_input_and_output():
    @requires_bvts
    def identity(x: torch.Tensor) -> torch.Tensor:
        return x

    good = torch.randn(1, 1, 1, 4, 4)
    # Should not raise
    out = identity(good)
    assert out.shape == good.shape

    bad = torch.randn(3, 4, 5)
    with pytest.raises(BVTSValidationError):
        identity(bad)
