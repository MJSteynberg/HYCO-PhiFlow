# src/visualization/plotter.py

import os
import torch
import yaml
import numpy as np
from pbdl.torch.loader import Dataloader
from phi.torch.flow import (
    Box,
    CenteredGrid,
    extrapolation,
    vis,
    math,
    channel,
    spatial,
    batch
)
import phi.field as field
import matplotlib.pyplot as plt
from typing import Dict, List

# --- Helper to load config ---
# (Used by run_evaluation.py)
def load_config(config_path: str) -> dict:
    """
    Loads a YAML config file.
    
    Args:
        config_path: Path to the .yaml configuration file.

    Returns:
        A dictionary containing the configuration.
    """
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# --- Config-driven Data Loader ---
# (Used by run_evaluation.py)
def load_data(data_dir: str, 
              dset_name: str, 
              fields_to_load: list,
              sim_index: int = 0,
              total_time_steps: int = 51):
    """
    Loads the full time sequence for a single simulation using pbdl.
    """
    print(f"Loading dataset '{dset_name}' from {data_dir} for sim {sim_index}...")
    print(f"pbdl will load fields: {fields_to_load}")
    
    hdf5_filepath = os.path.join(data_dir, f"{dset_name}.hdf5")
    if not os.path.exists(hdf5_filepath):
        print(f"Error: HDF5 file not found at {hdf5_filepath}")
        return None

    try:
        data_loader = Dataloader(
            dset_name,
            load_fields=fields_to_load,
            time_steps=1, 
            sel_sims=[sim_index], 
            batch_size=1,
            shuffle=False,
            local_datasets_dir=data_dir
        )
        
        print(f"Dataloader found {len(data_loader.dataset)} samples. Stacking all {total_time_steps} frames...")

        all_steps_list = []
        last_y_batch = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for x_batch, y_batch in data_loader:
            step_data = x_batch.squeeze(0).squeeze(0).to(device)
            all_steps_list.append(step_data)
            last_y_batch = y_batch
        
        if last_y_batch is not None:
            final_step_data = last_y_batch.squeeze(0).squeeze(0).to(device)
            all_steps_list.append(final_step_data)

        if not all_steps_list:
            print("Error: No data loaded. Dataloader was empty.")
            return None

        data_sequence = torch.stack(all_steps_list, dim=0) # Shape: (T, C, Y, X)
        
        print(f"Data stacked. Final sequence shape: {data_sequence.shape}")

        if data_sequence.shape[0] != total_time_steps:
             print(f"--- WARNING ---")
             print(f"Loader logic failed. Expected {total_time_steps} frames, but stacked {data_sequence.shape[0]}.")
             print(f"--- END WARNING ---")

        return data_sequence

    except Exception as e:
        print(f"Error loading data with pbdl: {e}")
        return None

# --- Generic, Spec-driven Field Reconstructor ---
# (Used by run_evaluation.py)
def reconstruct_fields_from_specs(data_sequence: torch.Tensor, 
                                  specs: Dict[str, int], 
                                  domain: Box) -> Dict[str, CenteredGrid]:
    """
    Converts a raw data tensor back into a dictionary of PhiFlow Fields
    based on a spec dictionary (e.g., {'density': 1, 'velocity': 2}).
    
    This is the preferred, generic way to reconstruct fields.
    """
    if data_sequence is None:
        return {}

    print(f"Reconstructing fields based on specs: {specs}")
    fields_dict = {}
    channel_index = 0
    
    for field_name, num_channels in specs.items():
        print(f"  Processing '{field_name}' ({num_channels} channels) at index {channel_index}...")
        
        data_slice_torch = data_sequence[:, channel_index:channel_index+num_channels, ...]
        
        if num_channels == 1:
            # --- Scalar Field (density, inflow, or 1D velocity) ---
            data_slice_torch = data_slice_torch.squeeze(1) # (T, Y, X)
            data_slice_torch = data_slice_torch.permute(0, 2, 1) # (T, Y, X) -> (T, X, Y)
            dims = batch('time') & spatial('x,y')
            
            values = math.tensor(data_slice_torch, dims)
            
            fields_dict[field_name] = CenteredGrid(
                values,
                extrapolation=extrapolation.BOUNDARY,
                bounds=domain
            )
            
            if field_name == 'velocity':
                fields_dict[field_name] = fields_dict[field_name].with_extrapolation(extrapolation.PERIODIC)
        
        elif num_channels == 2:
            # --- 2D Vector Field (velocity) ---
            data_slice_torch = data_slice_torch.permute(0, 3, 2, 1) # (T, 2, Y, X) -> (T, X, Y, 2)
            dims = batch('time') & spatial('x,y') & channel(vector='x,y')
            
            values = math.tensor(data_slice_torch, dims)
            
            fields_dict[field_name] = CenteredGrid(
                values,
                extrapolation=extrapolation.ZERO,
                bounds=domain
            )
            
        else:
            print(f"Warning: Skipping field '{field_name}'. Unsupported channel count: {num_channels}")

        channel_index += num_channels

    print(f"Reconstruction complete. Found fields: {list(fields_dict.keys())}")
    return fields_dict


# --- Comparison Plotter ---
# (Used by run_evaluation.py)
def plot_comparison(gt_fields: Dict[str, CenteredGrid], 
                    pred_fields: Dict[str, CenteredGrid],
                    sim_name: str = "",
                    sim_index: int = 0):
    """
    Shows a side-by-side animation for all common fields.
    """
    print("Plotting side-by-side comparisons...")
    
    common_fields = set(gt_fields.keys()) & set(pred_fields.keys())
    
    if not common_fields:
        print("No common fields found between Ground Truth and Prediction.")
        return

    for field_name in common_fields:
        gt_field = gt_fields[field_name]
        pred_field = pred_fields[field_name]
        
        if field_name == 'velocity':
            # This is a 2D vector, plot magnitude
            gt_field = field.vec_length(gt_field)
            pred_field = field.vec_length(pred_field)
            title = f"{field_name.capitalize()} Magnitude (Sim {sim_index})"
        else:
            # This is a scalar (density, inflow, or 1D velocity)
            title = f"{field_name.capitalize()} (Sim {sim_index})"
            
        print(f"  -> Plotting {title}")

        
        anim = vis.plot(
            {
                'Ground Truth': gt_field,
                'Prediction': pred_field,
            },
            animate='time', 
            title=title
        )
        save_dir = f"results/plots/{sim_name}"
        os.makedirs(save_dir, exist_ok=True)
        anim.save(f"{save_dir}/{field_name}_sim_{sim_index}.gif", fps=10)
        
