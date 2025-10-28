# src/visualization/plotter.py

import os
import torch
import yaml
import numpy as np
from pbdl.torch.loader import Dataloader  # <-- Make sure this is imported
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

# --- Config-driven Data Loader ---

def load_data(data_dir: str, 
              dset_name: str, 
              fields_to_load: list,
              sim_index: int = 0):
    """
    Loads the full time sequence for a single simulation using pbdl.
    
    Args:
        data_dir: Path to the directory containing the HDF5 file.
        dset_name: Name of the dataset (e.g., "smoke_v1").
        fields_to_load (list): List of field names to load (from config).
        sim_index: Which simulation to load (default is 0).
        
    Returns:
        torch.Tensor: A tensor of shape (time, channels, y, x)
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
            load_fields=fields_to_load, # Use the parameter
            time_steps=1,
            sel_sims=[sim_index], 
            batch_size=1,
            shuffle=False,
            local_datasets_dir=data_dir
        )
        
        all_steps_list = []
        print(f"Stacking all time steps from dataloader for sim {sim_index}...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for data_batch, params in data_loader:
            step_data = data_batch.squeeze(0).squeeze(0).to(device) # Shape: (C, Y, X)
            all_steps_list.append(step_data)
            
        if not all_steps_list:
            print("Error: No data loaded. Dataloader was empty.")
            return None

        data_sequence = torch.stack(all_steps_list, dim=0) # Shape: (T, C, Y, X)
        
        print(f"Data stacked. Final sequence shape: {data_sequence.shape}")
        return data_sequence

    except Exception as e:
        print(f"Error loading data with pbdl: {e}")
        print("Please ensure 'fields_to_save' in your config matches the pbdl schema.")
        return None

# --- Config-driven Field Reconstructor ---

def reconstruct_fields(data_sequence: torch.Tensor, 
                       fields_to_save: list,
                       domain: Box):
    """
    Converts the raw data tensor back into a dictionary of PhiFlow Fields.
    (This is the version from our last fix, which works)
    """
    if data_sequence is None:
        return {}

    print(f"Reconstructing fields based on config: {fields_to_save}")
    
    fields_dict = {}
    channel_index = 0
    
    for field_name in fields_to_save:
        field_name_lower = field_name.lower()
        print(f"  Processing field '{field_name}' at channel index {channel_index}...")

        if field_name_lower == 'density' or field_name_lower == 'inflow':
            # Case 1: Scalar Field (density, inflow)
            data_slice = data_sequence[:, channel_index, :, :]
            data_slice = data_slice.permute(0, 2, 1) # (T, Y, X) -> (T, X, Y)
            dims = batch('time') & spatial('x,y') 

            values = math.tensor(data_slice, dims)
            
            fields_dict[field_name_lower] = CenteredGrid(
                values,
                extrapolation=extrapolation.BOUNDARY,
                bounds=domain
            )
            channel_index += 1 

        elif field_name_lower == 'velocity':
            # Case 2: Vector Field (Velocity)
            data_slice_torch = data_sequence[:, channel_index:channel_index+2, :, :] # (T, 2, Y, X)
            data_slice_torch = data_slice_torch.permute(0, 3, 2, 1) # (T, X, Y, 2)
            dims = batch('time') & spatial('x,y') & channel(vector='x,y')
            
            values = math.tensor(data_slice_torch, dims)
            
            fields_dict['velocity'] = CenteredGrid(
                values,
                extrapolation=extrapolation.ZERO,
                bounds=domain
            )
            channel_index += 2
            
        else:
            print(f"Warning: Unknown field '{field_name}' in config. Skipping.")
            if 'velocity' in field_name_lower:
                channel_index += 2
            else:
                channel_index += 1

    print(f"Reconstruction complete. Found fields: {list(fields_dict.keys())}")
    return fields_dict

# --- Plotting Functions ---

def plot_density(density_field: CenteredGrid, title: str):
    """
    Shows an animation of a scalar field (like density or inflow).
    """
    if density_field is None: return
    print(f"Plotting '{title}' (close window to continue)...")
    
    vis.plot(
        density_field,
        animate='time',      
        title=title,
    )
    plt.show()
    plt.close()

def plot_velocity(velocity_field: CenteredGrid, title: str):
    """
    Shows an animation of a vector field (velocity streamlines).
    """
    if velocity_field is None: return
    print(f"Plotting '{title}' (close window to continue)...")
    
    vis.plot(
        velocity_field,
        animate='time',      
        title=title,
    )
    plt.show()
    plt.close()

# --- NEW FUNCTION ---
def plot_magnitude(velocity_field: CenteredGrid, title: str):
    """
    Calculates and shows an animation of the velocity magnitude.
    
    Args:
        velocity_field: The vector field to plot.
        title: The title for the plot.
    """
    if velocity_field is None: return
    print(f"Plotting '{title}' (close window to continue)...")
    
    # Calculate the magnitude
    magnitude = field.vec_length(velocity_field)
    
    # Plot the scalar magnitude field (will default to heatmap)
    vis.plot(
        magnitude,
        animate='time',      
        title=title,
    )
    plt.show()
    plt.close()

# --- Main Visualization Function (Updated) ---

def run_visualization(config_path: str, project_root: str, sim_to_load: int = 0):
    """
    Main function to run visualization based on a config.
    Will plot all fields found in the data.
    """
    # 1. Load the configuration
    config = load_config(config_path)

    # 2. Extract parameters from config
    domain_cfg = config['domain']
    out_cfg = config['output_data']
    
    # 3. Re-create the Domain object
    domain = Box(x=domain_cfg['size_x'], y=domain_cfg['size_y'])
    
    # 4. Define data path
    data_dir = os.path.join(project_root, out_cfg['output_dir'])
    dataset_name = out_cfg['dataset_name']
    
    # 5. Get the list of fields to load *from the config*
    fields_to_load = out_cfg['fields_to_save']
    
    print(f"Configuration loaded. Visualizing sim {sim_to_load}...")
    
    # 6. Load the data
    data = load_data(
        data_dir=data_dir,
        dset_name=dataset_name,
        fields_to_load=fields_to_load,
        sim_index=sim_to_load
    )
    
    if data is None:
        print("Failed to load data. Exiting.")
        return

    # 7. Reconstruct fields
    fields = reconstruct_fields(
        data_sequence=data, 
        fields_to_save=fields_to_load,
        domain=domain
    )
    
    # 8. Show animation for all reconstructed fields
    if not fields:
        print("No fields were reconstructed. Nothing to plot.")
        return
        
    if 'density' in fields:
        plot_density(
            fields['density'], 
            title=f"Density (Sim {sim_to_load})"
        )
        
    if 'velocity' in fields:
        # --- THIS IS THE CHANGE ---
        # Call plot_magnitude instead of plot_velocity
        plot_magnitude(
            fields['velocity'], 
            title=f"Velocity Magnitude (Sim {sim_to_load})"
        )
        # --- END CHANGE ---
        
    if 'inflow' in fields:
        plot_density(
            fields['inflow'], 
            title=f"Inflow (Sim {sim_to_load})"
        )