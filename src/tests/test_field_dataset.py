import torch

import src.utils.field_conversion as fc
import src.data.field_dataset as fd
from src.data.field_dataset import FieldDataset


class DummyConverter:
    def tensor_to_field(self, tensor_t, field_meta=None, time_slice=0):
        # Return a simple marker containing the incoming tensor shape
        return {"shape": tuple(tensor_t.shape)}


def make_fake_field_dataset(field_tensor: torch.Tensor, total_frames: int):
    # Create an uninitialized FieldDataset instance and set minimal attributes
    ds = FieldDataset.__new__(FieldDataset)
    ds.field_names = ["field"]
    ds.num_frames = total_frames
    return ds


def test_tensors_to_fields_bvts_only():
    # Prepare BVTS input: [B, C, T, H, W]
    B, C, T, H, W = 1, 3, 5, 8, 8
    bvts = torch.zeros((B, C, T, H, W))

    # Patch make_converter in the field_dataset module to return our DummyConverter
    old_make_converter = fd.make_converter
    fd.make_converter = lambda meta: DummyConverter()

    try:
    # BVTS case
        ds_bvts = make_fake_field_dataset(bvts, T)
        data = {"tensor_data": {"field": bvts}}
        fields_bvts = FieldDataset._tensors_to_fields(ds_bvts, data, {"field": None}, 1, 3)
        assert "field" in fields_bvts
        assert len(fields_bvts["field"]) == 2  # frames 1..2
        assert all(isinstance(x, dict) and "shape" in x for x in fields_bvts["field"])

    finally:
        # Restore
        fd.make_converter = old_make_converter
