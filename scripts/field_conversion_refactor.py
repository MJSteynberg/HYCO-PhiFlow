"""
Quick smoke-test runner for the `src.utils.field_conversion` package.

This script imports the public factory functions and core classes and
performs light-weight instantiation calls so you can safely refactor the
conversion modules while ensuring the public API remains importable and
instantiable.

It avoids calling heavy conversion methods that require real PhiFlow
`Field` objects. Instead it:
- Builds FieldMetadata from a tiny DummyModel
- Creates converters via the factory helpers
- Calls small helper methods (e.g., get_channel_info, can_handle)

Run from the repo root:
    python3 scripts/field_conversion_refactor.py

The script prints results and exits with code 0 on success.
"""

import logging

import sys
import logging
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from phi.torch.flow import *

from src.utils.field_conversion import (
    FieldMetadata,
    create_field_metadata_from_model,
    make_converter,
    make_batch_converter,
    make_centered_converter,
    make_staggered_converter,
    BatchConcatenationConverter,
    CenteredConverter,
    StaggeredConverter,
)

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("field_conversion_refactor")


class DummyModel:
    """Minimal model-like object with domain and resolution attributes."""

    def __init__(self):
        # simple 2D unit box and a small resolution for tests
        self.domain = Box(x=(0.0, 1.0), y=(0.0, 1.0))
        self.resolution = spatial(x=8, y=8)


def main():
    logger.info("Starting field conversion smoke-test")

    # 1) Create FieldMetadata from a tiny DummyModel
    model = DummyModel()
    metadata_dict = create_field_metadata_from_model(model, ["density", "velocity"], {"velocity": "staggered"})
    logger.info(f"Created FieldMetadata for fields: {list(metadata_dict.keys())}")

    # 2) Create batch converter from metadata dict
    batch_conv = make_batch_converter(metadata_dict)
    logger.info(f"Batch converter created. Channel info: {batch_conv.get_channel_info()}")

    # 3) Create single-field converters via factory
    centered_meta = metadata_dict["density"]
    centered_conv = make_converter(centered_meta)
    logger.info(f"Centered converter created: {type(centered_conv)} - can_handle: {centered_conv.can_handle(centered_meta)}")

    staggered_meta = metadata_dict["velocity"]
    staggered_conv = make_converter(staggered_meta)
    logger.info(f"Staggered converter created: {type(staggered_conv)} - can_handle: {staggered_conv.can_handle(staggered_meta)}")

    # 4) Explicit factory helpers
    c1 = make_centered_converter()
    c2 = make_staggered_converter()
    logger.info(f"make_centered_converter returned {type(c1)}; make_staggered_converter returned {type(c2)}")

    # 5) Use make_converter with a dict to get a BatchConcatenationConverter
    bc = make_converter(metadata_dict)
    assert isinstance(bc, BatchConcatenationConverter)
    logger.info("make_converter(dict) -> BatchConcatenationConverter OK")

    logger.info("All smoke-test steps completed successfully.")

    # --- Now perform actual conversions using small PhiFlow Fields ---
    try:

        logger.info("Creating small CenteredGrid and StaggeredGrid for conversion tests...")

        # Create a simple centered field using a positional function
        centered_field = CenteredGrid(
            lambda x, y: math.sin(2 * math.pi * x) * math.cos(2 * math.pi * y),
            extrapolation=extrapolation.PERIODIC,
            x=8,
            y=8,
            bounds=model.domain,
        )

        # Create a staggered field by converting centered -> staggered
        staggered_field = StaggeredGrid(
            centered_field,
            extrapolation=extrapolation.PERIODIC,
            bounds=model.domain,
            x=8,
            y=8,
        )

        logger.info("Performing single-field conversions...")

        # Centered conversion
        t_center = centered_conv.field_to_tensor(centered_field)
        # Ensure we pass a torch.Tensor into tensor_to_field
        logger.info(f"centered field -> tensor shape: {getattr(t_center, 'shape', None)}")
        rec_center = centered_conv.tensor_to_field(t_center, centered_meta)
        logger.info(f"tensor -> reconstructed centered field type: {rec_center}")

        # Staggered conversion
        t_stag = staggered_conv.field_to_tensor(staggered_field)
        logger.info(f"staggered field -> tensor shape: {getattr(t_stag, 'shape', None)}")
        rec_stag = staggered_conv.tensor_to_field(t_stag, staggered_meta)
        logger.info(f"tensor -> reconstructed staggered field type: {rec_stag}")

        # Batch conversion (both fields)
        logger.info("Performing batch conversion (fields -> tensor -> fields)")
        fields = {"density": centered_field, "velocity": staggered_field}
        batch_tensor = batch_conv.fields_to_tensor_batch(fields)
        logger.info(f"batch tensor shape: {getattr(batch_tensor, 'shape', None)}")
        back_fields = batch_conv.tensor_to_fields_batch(batch_tensor)
        logger.info(f"reconstructed fields keys: {list(back_fields.keys())}")

        logger.info("Field conversion round-trips completed successfully.")
    except Exception as e:
        logger.exception("Field conversion round-trip failed")
        raise


if __name__ == "__main__":
    main()
