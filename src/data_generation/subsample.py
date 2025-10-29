# src/data_processing/subsampler.py

import os
import h5py
import yaml
import numpy as np
from tqdm import tqdm

# --- PhiFlow Imports ---
from phi.torch.flow import (
    Box,
    CenteredGrid,
    extrapolation,
    math,
    spatial,
    channel,
    field,
    batch
)

def load_config(config_path: str) -> dict:
    """Loads a YAML config file."""
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_subsampling(config_path: str, project_root: str):
    """
    Main function to load a dataset, subsample it,
    and save it to a new HDF5 file.
    """
    config = load_config(config_path)
    
    # --- Get Configs ---
    # Use original data-gen config to get domain info
    orig_config_path = os.path.join(project_root, config['original_config_path'])
    orig_config = load_config(orig_config_path)
    
    domain_cfg = orig_config['domain']
    orig_res_cfg = orig_config['resolution']
    orig_out_cfg = orig_config['output_data']

    # Subsampling config
    sub_cfg = config['subsampling']
    target_res_cfg = sub_cfg['target_resolution']
    
    # --- 1. GET THE NEW TIME STEP ---
    # Get 'time_step' from config, default to 1 (no subsampling)
    time_step = sub_cfg.get('time_step', 1)
    
    # --- Define Domain and Resolutions ---
    domain = Box(x=domain_cfg['size_x'], y=domain_cfg['size_y'])
    
    orig_resolution = spatial(
        x=orig_res_cfg['x'], 
        y=orig_res_cfg['y']
    )
    
    target_resolution = spatial(
        x=target_res_cfg['x'], 
        y=target_res_cfg['y']
    )
    
    # --- Define Target Grid Geometry ---
    # We create a dummy grid with the target shape to resample to.
    # Your generator.py saves all data as CenteredGrid values,
    # so we can use CenteredGrid here.
    target_geometry = CenteredGrid(
        0, 
        extrapolation=extrapolation.BOUNDARY, 
        bounds=domain, 
        resolution=target_resolution
    )

    # --- Setup I/O Paths ---
    input_dir = os.path.join(project_root, orig_out_cfg['output_dir'])
    input_path = os.path.join(input_dir, f"{orig_out_cfg['dataset_name']}.hdf5")
    
    output_dir = os.path.join(project_root, sub_cfg['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{sub_cfg['dataset_name']}.hdf5")

    print(f"Starting subsampling...")
    print(f"  Input file: {input_path}")
    print(f"  Output file: {output_path}")
    print(f"  Original Resolution: {orig_resolution}")
    print(f"  Target Resolution: {target_resolution}")
    print(f"  Time Subsampling Step: {time_step}")

    # --- Open Files ---
    with h5py.File(input_path, 'r') as f_in:
        with h5py.File(output_path, 'w') as f_out:
            
            in_sims_group = f_in['sims']
            out_sims_group = f_out.create_group('sims')
            
            # Get the number of simulations from the keys
            sim_keys = [k for k in in_sims_group.keys() if k.startswith('sim')]
            num_sims = len(sim_keys)
            
            # --- Loop over all simulations ---
            for i in tqdm(range(num_sims), desc="Subsampling Simulations"):
                sim_name = f'sim{i}'
                try:
                    # 1. Load original data (T, C, Y, X)
                    sim_data_np = in_sims_group[sim_name][::time_step, :, :, :]
                    
                    # 2. Convert to PhiFlow Tensor
                    # (T, C, Y, X) -> (time, channels, y, x)
                    values_tensor = math.tensor(
                        sim_data_np, 
                        batch('time'), 
                        channel('channels'), 
                        spatial('y, x')
                    )
                    
                    # 3. Create Original CenteredGrid
                    # We assume BOUNDARY extrapolation as a safe default.
                    original_grid = CenteredGrid(
                        values=values_tensor, 
                        bounds=domain, 
                        extrapolation=extrapolation.BOUNDARY
                    )
                    
                    # 4. Resample to target geometry
                    # field.resample uses linear interpolation by default
                    subsampled_grid = field.resample(original_grid, target_geometry)
                    
                    # 5. Convert back to NumPy
                    # (time, channels, y, x) -> (T, C, Y_new, X_new)
                    ordered_axes = ('time', 'channels', 'y', 'x')
                    subsampled_data_np = subsampled_grid.numpy(ordered_axes)
                    
                    # 6. Save to new HDF5
                    out_sims_group.create_dataset(sim_name, data=subsampled_data_np)

                except KeyError:
                    print(f"Warning: '{sim_name}' not found in input file. Skipping.")
                    continue
            
            # --- 7. Copy attributes ---
            print("Copying metadata attributes...")
            for key, val in in_sims_group.attrs.items():
                out_sims_group.attrs[key] = val
            
            # Add/update subsampling-specific metadata
            out_sims_group.attrs['Original Resolution'] = f"x: {orig_res_cfg['x']}, y: {orig_res_cfg['y']}"
            out_sims_group.attrs['Subsampled Resolution'] = f"x: {target_res_cfg['x']}, y: {target_res_cfg['y']}"

        print("\nSubsampling complete.")
        print(f"New dataset saved to: {output_path}")