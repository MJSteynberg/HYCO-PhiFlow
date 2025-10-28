import os
import torch
import numpy as np
from pbdl.torch.loader import Dataloader
from phi.torch.flow import (
    Box,
    CenteredGrid,
    extrapolation,
    vis,
    math,
    channel,
    batch,
    spatial
)
import matplotlib.pyplot as plt

print("Imports complete. Ready to visualize data.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- Simulation Parameters (MUST match generation script) ---
# Domain Resolution
RES_X = 64
RES_Y = 80 
# Physical Domain Size
DOMAIN_SIZE_X = 80
DOMAIN_SIZE_Y = 100
# Domain object
DOMAIN = Box(x=DOMAIN_SIZE_X, y=DOMAIN_SIZE_Y)

# --- Data Generation Parameters ---
TOTAL_STEPS = 50
SAVE_INTERVAL = 1
# +1 for the initial state (step 0)
NUM_SAVED_STATES = (TOTAL_STEPS // SAVE_INTERVAL) + 1

# --- Data Loading Parameters ---
dataset_name = "smoke_v1"
local_data_directory = "./data"

def load_data(data_dir, dset_name, total_time_steps):
    """
    Loads the full simulation data using pbdl.Dataloader
    by requesting the entire time sequence as a single sample.
    
    Returns:
        torch.Tensor: A tensor of shape (time, channels, y, x)
    """
    print(f"Loading dataset '{dset_name}' from {data_dir}...")
    
    hdf5_filepath = os.path.join(data_dir, f"{dset_name}.hdf5")
    if not os.path.exists(hdf5_filepath):
        print(f"Error: HDF5 file not found at {hdf5_filepath}")
        print("Please run the data generation script first.")
        return None

    try:
        # --- MODIFIED ---
        # We set time_in to the total number of states we want to load.
        # This treats the entire sequence as our input.
        data_loader = Dataloader(
            dset_name,
            load_fields=['density', 'velocity'],
            time_steps=1,
            sel_sims=[0],
            batch_size=1,             # Get this one sequence
            shuffle=False,
            local_datasets_dir=data_dir
        )
        
        all_steps_list = []
        print("Stacking all time steps from dataloader...")
        
        # We loop over the data_loader, which is an iterable
        # (it will loop 10 times)
        for data_batch, params in data_loader:
            # data_batch shape is (B, T, C, H, W) -> (1, 1, 3, 40, 32)
            # Squeeze both Batch and Time dimensions
            step_data = data_batch.squeeze(0).squeeze(0).to(device) # Shape: (3, 40, 32)
            all_steps_list.append(step_data)
            
        if not all_steps_list:
            print("Error: No data loaded. Dataloader was empty.")
            return None

        # Stack all tensors along a new 'time' dimension (dim=0)
        data_sequence = torch.stack(all_steps_list, dim=0)
        
        # data_sequence final shape should be: (10, 3, 40, 32) (T, C, H, W)
        print(f"Data stacked. Final sequence shape: {data_sequence.shape}")
        return data_sequence

    except StopIteration:
         print("\nError: Dataloader was empty.")
         print(f"This might happen if 'time_in' ({total_time_steps}) is larger than the number of steps in the HDF5 file.")
         return None
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure 'pbdl' is installed and all arguments are correct.")
        return None

    
def reconstruct_fields(data_sequence):
    """
    Converts the raw data tensor back into PhiFlow Field objects.
    (Fix: Corrected vector channel order and handles 3D/4D)

    Args:
        data_sequence (torch.Tensor): Data with shape (time, channels, y, x)
                                    OR (channels, y, x)
    Returns:
        (CenteredGrid, CenteredGrid): smoke_field, velocity_field
    """
    if data_sequence is None:
        return None, None

    print("Reconstructing PhiFlow fields...")
    
    # --- Check dimensions and add batch dim if missing ---
    if data_sequence.ndim == 3:
        print("Warning: Loaded data is 3D. Treating as a single time step.")
        data_sequence = data_sequence.unsqueeze(0) 
    
    # data_sequence shape is now (T, C, Y, X)
    
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
        channel(vector='x,y') # --- THIS IS THE FIX ---
    )

    # --- Create CenteredGrid Objects ---
    smoke_field = CenteredGrid(
        smoke_values,
        extrapolation=extrapolation.BOUNDARY,
        bounds=DOMAIN
    )
    
    velocity_field = CenteredGrid(
        velocity_values,
        extrapolation=extrapolation.ZERO,
        bounds=DOMAIN
    )
    
    print(f"Smoke field shape: {smoke_field.shape}")
    print(f"Velocity field shape: {velocity_field.shape}")
    
    return smoke_field, velocity_field

def visualize_animation(smoke_field, velocity_field):
    """
    Shows an animation of smoke density and velocity vectors.
    """
    if smoke_field is None: return
    print("Showing animation (close window to continue)...")
    
    vis.plot(
        {'Smoke': smoke_field},
        animate='time',      
        title="Smoke Simulation",
    )
    plt.show()
    plt.close()

if __name__ == "__main__":
    # 1. Load the data
    data = load_data(
        data_dir=local_data_directory,
        dset_name=dataset_name,
        total_time_steps=NUM_SAVED_STATES # Pass in the total (e.g., 11)
    )
    
    # 2. Reconstruct fields
    # (This uses the 'reconstruct_fields' from the previous step,
    #  which correctly uses math.tensor() and named dimensions)
    smoke, velocity = reconstruct_fields(data)
    
    # 3. Show animation
    visualize_animation(smoke, velocity)

    