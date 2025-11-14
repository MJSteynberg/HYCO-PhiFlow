"""
DataManager: updated caching logic.

Key changes:
 - When loading a simulation, for each field we store a canonical tensor
   [B, T, *spatial, V] in the per-simulation cache dict.
 - No packing of multiple fields into a single tensor occurs here.
"""
from pathlib import Path
from typing import Dict, Any
import torch

# Import your phi utilities as used elsewhere in the project
from phi import math
from phi.field import CenteredGrid
from ..utils.field_conversion.layout import canonical_from_phiflow_native

class DataManager:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache: Dict[str, Any] = {}

    def load_and_cache_simulation(self, sim_path: Path, *, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """
        Load a single simulation and cache each field as a canonical tensor.

        Returns a dict mapping field_name -> canonical_tensor [B, T, *spatial, V]
        """
        sim_key = str(sim_path)
        if sim_key in self.cache:
            return self.cache[sim_key]

        # Example loading logic: adapt to your repo-specific loader
        # Assume `load_phi_simulation` returns a dict of Field objects keyed by field name
        sim_fields = self._load_simulation_fields(sim_path)

        tensor_data = {}
        for name, field in sim_fields.items():
            # field is a phi.Field (CenteredGrid or StaggeredGrid)
            # Project to centered for converters that expect centered native ordering
            if hasattr(field, "at_centers"):
                centered = field.at_centers()
            else:
                centered = field

            # Request native representation from phi; we assume native returns
            # a torch.Tensor or a phi-math object exposing ._native
            native = centered.values.native()
            native_t = getattr(native, "_native", native)

            # Determine if vector
            is_vector = centered.shape.channel.rank > 0

            canonical = canonical_from_phiflow_native(native_t, is_vector=is_vector)
            if device is not None:
                canonical = canonical.to(device)

            tensor_data[name] = canonical

        # cache and return
        self.cache[sim_key] = tensor_data
        return tensor_data

    def _load_simulation_fields(self, path: Path):
        """
        Placeholder loader. Replace with your real loader that returns a dict:
            { field_name: phi.Field, ... }
        """
        # THIS IS A STUB — replace with repository-specific loading code
        raise NotImplementedError("Replace _load_simulation_fields with project loader.")
