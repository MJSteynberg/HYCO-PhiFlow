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
import matplotlib.pyplot as plt

# --- Helper to load config ---
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

# --- Logic moved from script ---

def load_data(data_dir: str, dset_name: str, sim_index: int = 0):
    """
    Loads the full time sequence for a single simulation using pbdl.
    
    Args:
        data_dir: Path to the directory containing the HDF5 file.
        dset_name: Name of the dataset (e.g., "smoke_v1").
        sim_index: Which simulation to load (default is 0).
        
    Returns:
        torch.Tensor: A tensor of shape (time, channels, y, x)
    """
    print(f"Loading dataset '{dset_name}' from {data_dir} for sim {sim_index}...")
    
    hdf5_filepath = os.path.join(data_dir, f"{dset_name}.hdf5")
    if not os.path.exists(hdf5_filepath):
        print(f"Error: HDF5 file not found at {hdf5_filepath}")
        return None

    try:
        # pbdl.Dataloader with time_steps=1 will iterate
        # over every saved time step for the selected sim.
        data_loader = Dataloader(
            dset_name,
            load_fields=['density', 'velocity'], # Matches config
            time_steps=1,
            sel_sims=[sim_index], # Load the specified simulation
            batch_size=1,
            shuffle=False,
            local_datasets_dir=data_dir
        )
        
        all_steps_list = []
        print(f"Stacking all time steps from dataloader for sim {sim_index}...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for data_batch, params in data_loader:
            # data_batch shape: (B, T, C, H, W) -> (1, 1, C, Y, X)
            # Squeeze both Batch and Time dimensions
            step_data = data_batch.squeeze(0).squeeze(0).to(device) # Shape: (C, Y, X)
            all_steps_list.append(step_data)
            
        if not all_steps_list:
            print("Error: No data loaded. Dataloader was empty.")
            return None

        # Stack all tensors along a new 'time' dimension (dim=0)
        data_sequence = torch.stack(all_steps_list, dim=0) # Shape: (T, C, Y, X)
        
        print(f"Data stacked. Final sequence shape: {data_sequence.shape}")
        return data_sequence

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    
def reconstruct_fields(data_sequence: torch.Tensor, domain: Box):
    """
    Converts the raw data tensor back into PhiFlow Field objects.

    Args:
        data_sequence (torch.Tensor): Data with shape (time, channels, y, x)
        domain (Box): The simulation domain object.
                                    
    Returns:
        (CenteredGrid, CenteredGrid): smoke_field, velocity_field
    """
    if data_sequence is None:
        return None, None

    print("Reconstructing PhiFlow fields...")
    
    if data_sequence.ndim == 3:
        print("Warning: Loaded data is 3D. Treating as a single time step.")
        data_sequence = data_sequence.unsqueeze(0) 
    
    # --- Separate Smoke and Velocity Data ---
    smoke_data_torch = data_sequence[:, 0, ...]    # Shape (T, Y, X)
    velo_data_torch = data_sequence[:, 1:3, ...] # Shape (T, 2, Y, X)
    
    # Permute from (T, C, H, W) to (T, H, W, C) -> (T, Y, X, C)
    velo_data_torch = velo_data_torch.permute(0, 2, 3, 1)

    # --- Create phiml.math Tensors with named dimensions ---
    smoke_values = math.tensor(
        smoke_data_torch,
        batch('time'),      
        spatial('y,x')    
    )
    
    velocity_values = math.tensor(
        velo_data_torch,
        batch('time'),      
        spatial('y,x'),   
        channel(vector='x,y')
    )

    # --- Create CenteredGrid Objects ---
    smoke_field = CenteredGrid(
        smoke_values,
        extrapolation=extrapolation.BOUNDARY,
        bounds=domain # Use the domain passed from config
    )
    
    velocity_field = CenteredGrid(
        velocity_values,
        extrapolation=extrapolation.ZERO,
        bounds=domain # Use the domain passed from config
    )
    
    print(f"Smoke field shape: {smoke_field.shape}")
    print(f"Velocity field shape: {velocity_field.shape}")
    
    return smoke_field, velocity_field

def visualize_animation(smoke_field: CenteredGrid, velocity_field: CenteredGrid, sim_index: int):
    """
    Shows an animation of smoke density.
    
    Args:
        smoke_field: The smoke density field.
        velocity_field: The velocity field (not currently used in plot).
        sim_index: The simulation index, for the title.
    """
    if smoke_field is None: return
    print("Showing animation (close window to continue)...")
    
    vis.plot(
        {'Smoke': smoke_field},
        animate='time',      
        title=f"Smoke Simulation (Sim {sim_index})",
    )
    plt.show()
    plt.close()

def run_visualization(config_path: str, project_root: str, sim_to_load: int = 0):
    """
    Main function to run visualization based on a config.

    Args:
        config_path: Path to the simulation config file.
        project_root: Absolute path to the project's root directory.
        sim_to_load: The index of the simulation to load from the HDF5 file.
    """
    # 1. Load the configuration
    config = load_config(config_path)

    # 2. Extract parameters from config
    # We now get all paths and domain info from the config
    domain_cfg = config['domain']
    out_cfg = config['output_data']
    
    # 3. Re-create the Domain object
    domain = Box(x=domain_cfg['size_x'], y=domain_cfg['size_y'])
    
    # 4. Define data path
    data_dir = os.path.join(project_root, out_cfg['output_dir'])
    dataset_name = out_cfg['dataset_name']
    
    print(f"Configuration loaded. Visualizing sim {sim_to_load}...")
    
    # 5. Load the data
    data = load_data(
        data_dir=data_dir,
        dset_name=dataset_name,
        sim_index=sim_to_load
    )
    
    # 6. Reconstruct fields
    smoke, velocity = reconstruct_fields(data, domain)
    
    # 7. Show animation
    visualize_animation(smoke, velocity, sim_to_load)