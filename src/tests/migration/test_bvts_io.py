import torch
from src.utils.field_conversion.bvts import to_bvts, from_bvts, assert_bvts


def test_to_from_bvts_roundtrip_4d():
    # time-major [T, C, H, W]
    t = torch.randn(10, 3, 16, 16)
    bv = to_bvts(t)
    assert bv.shape == (1, 3, 10, 16, 16)
    # roundtrip
    back = from_bvts(bv)
    assert back.shape == t.shape


def test_to_bvts_single_frame_channel():
    t = torch.randn(3, 16, 16)
    bv = to_bvts(t)
    assert bv.shape == (1, 3, 1, 16, 16)


def test_assert_bvts_accepts_5d():
    bv = torch.randn(2, 3, 10, 8, 8)
    assert_bvts(bv)


def test_from_bvts_requires_batch1():
    bv = torch.randn(2, 3, 10, 8, 8)
    try:
        _ = from_bvts(bv)
        assert False, "Expected ValueError for batch>1"
    except ValueError:
        pass
