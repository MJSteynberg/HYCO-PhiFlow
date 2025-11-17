"""A minimal CLI for generating animations from saved Scene data.

This file provides a simple entry point to create GIF animations from the
fields in a saved simulation (Scene). It uses `DataManager` to load the
data and the internal `visualizer.animate_field` to write GIFs.
"""
from __future__ import annotations

import argparse
import yaml
import sys
from pathlib import Path
from typing import List, Optional
import torch

from src.data.data_manager import DataManager
from .visualizer import animate_field


def _load_config(cfg_path: Optional[str]):
    if cfg_path is None:
        cfg_path = "conf/config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _resolve_raw_data_dir(cfg: dict) -> Path:
    data_dir = Path(cfg["data"]["data_dir"])
    # Some datasets are nested as data/advection/advection/, try fallback
    if not data_dir.exists():
        nested = data_dir / cfg["data"].get("dset_name", "")
        if nested.exists():
            return nested
    return data_dir


def generate_animations(
    config_path: Optional[str],
    sim_index: int = 0,
    fields: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    fps: int = 10,
    num_frames: Optional[int] = None,
):
    cfg = _load_config(config_path)

    # Create DataManager with updated config paths if necessary
    cfg_copy = dict(cfg)
    cfg_copy["data"] = dict(cfg["data"])
    cfg_copy["data"]["data_dir"] = str(_resolve_raw_data_dir(cfg))

    dm = DataManager(cfg_copy)

    if fields is None:
        fields = cfg["data"].get("fields", [])

    # Load simulation tensors
    print(f"Loading simulation sim_{sim_index:06d} fields={fields}...")
    data = dm.load_simulation(sim_index, fields, num_frames)
    tensor_data = data["tensor_data"]

    output_dir = Path(output_dir) if output_dir else Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}
    for f in fields:
        if f not in tensor_data:
            print(f"Field '{f}' not found in simulation; skipping.")
            continue

        tensor = tensor_data[f]  # torch.Tensor
        # Normalize to [T, C, H, W]
        if tensor.ndim == 4 and tensor.shape[0] <= 4 and tensor.shape[1] > 4:
            # heuristic: [C, T, H, W]
            tensor = tensor.permute(1, 0, 2, 3)
        elif tensor.ndim == 4 and tensor.shape[1] <= 4 and tensor.shape[0] > 4:
            # [T, C, H, W] already
            pass
        elif tensor.ndim == 3:
            # [T, H, W]
            pass
        else:
            # For safety, fallback to DataManager's shape assumptions by converting
            # tensor to CPU ensures consistent dtype
            tensor = tensor

        out_path = output_dir / f"sim_{sim_index:06d}_{f}_animation.gif"
        print(f"Writing animation: {out_path}")
        animate_field(tensor, str(out_path), fps=fps)
        saved_paths[f] = out_path

    return saved_paths


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Generate animations from saved Scene data")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--sim-index", type=int, default=0, help="Simulation index to load")
    parser.add_argument(
        "--fields", type=str, default=None, help="Comma separated list of fields to animate"
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for animations")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for GIFs")
    parser.add_argument("--num-frames", type=int, default=None, help="How many frames to load (default: all)")

    args = parser.parse_args(argv)
    fields = None
    if args.fields is not None:
        fields = [f.strip() for f in args.fields.split(",") if f.strip()]

    generate_animations(
        args.config,
        args.sim_index,
        fields,
        args.output_dir,
        fps=args.fps,
        num_frames=args.num_frames,
    )


if __name__ == "__main__":
    main()
