"""Adapter utilities to convert cached simulation tensor_data to BVTS layout

Provides small helpers to convert per-field cached tensors into BVTS shape
and to concatenate multiple fields into a single BVTS tensor suitable for
feeding to BVTS-aware models.
"""
from typing import Dict, List
import torch

from src.utils.field_conversion.bvts import to_bvts


def sim_dict_to_bvts(sim_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert a simulation's tensor_data dict into BVTS tensors.

    Args:
        sim_data: mapping field_name -> tensor (common input shapes such as
                  [T, C, H, W], [C, T, H, W], or already BVTS)

    Returns:
        mapping field_name -> tensor in BVTS [B, V, T, *spatial]
    """
    converted = {}
    for name, tensor in sim_data.items():
        if isinstance(tensor, torch.Tensor):
            converted[name] = to_bvts(tensor)
        else:
            converted[name] = tensor
    return converted


def concat_fields_bvts(sim_data_bvts: Dict[str, torch.Tensor], field_names: List[str]) -> torch.Tensor:
    """Concatenate listed fields (already BVTS) into a single BVTS tensor.

    The concatenation is along the vector/channel dimension (dim=1).

    Returns:
        Tensor [B, V_total, T, *spatial]
    """
    tensors = [sim_data_bvts[name] for name in field_names]
    return torch.cat(tensors, dim=1)
