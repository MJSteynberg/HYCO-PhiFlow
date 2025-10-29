import os
import torch
import phiml.nn as pnn  # For re-building the U-Net
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

print("Imports complete. Ready to evaluate.")

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Simulation Parameters (MUST match generation script) ---
RES_X = 128
RES_Y = 128
DOMAIN_SIZE_X = 100
DOMAIN_SIZE_Y = 100
DOMAIN = Box(x=DOMAIN_SIZE_X, y=DOMAIN_SIZE_Y)

# --- Data Generation Parameters ---
TOTAL_STEPS = 50
SAVE_INTERVAL = 1
NUM_SAVED_STATES = (TOTAL_STEPS // SAVE_INTERVAL) + 1 # 51

# --- File Paths ---
dataset_name = "smoke_128"
local_data_directory = "./data"
# --- MODIFIED: Point to the new 4-step model ---
MODEL_PATH = os.path.join("./results/models", f"{dataset_name}_unet_autoregressive.pth")


def load_trained_model(model_path):
    """
    Loads the trained U-Net model from a .pth file.
    """
    print(f"Loading trained model from {model_path}...")
    
    # --- MODIFIED: Must match the 4-step model architecture ---
    model = pnn.u_net(
        in_channels=4,  # density, vx, vy, inflow
        out_channels=3, # 3 * 4 = 12
        levels=4,
        filters=64
    ).to(device)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run the training script first.")
        return None
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print("Model loaded successfully.")
    return model


def load_data(data_dir, dset_name, total_time_steps):
    """
    Loads the full simulation data (all frames) using a workaround
    for the pbdl.Dataloader.
    
    We set time_steps=1 to get 50 samples, which are pairs
    (frame 0, frame 1) ... (frame 49, frame 50).
    We stack all x_batches (0-49) and then add the last y_batch (50).
    
    Returns:
        torch.Tensor: A tensor of shape (time, channels, y, x)
    """
    print(f"Loading ground truth dataset '{dset_name}' from {data_dir}...")
    
    hdf5_filepath = os.path.join(data_dir, f"{dset_name}.hdf5")
    if not os.path.exists(hdf5_filepath):
        print(f"Error: HDF5 file not found at {hdf5_filepath}")
        return None

    try:
        data_loader = Dataloader(
            dset_name,
            load_fields=['density', 'velocity', 'inflow'],
            
            # --- THE WORKAROUND ---
            # 1. time_steps=1: This will find 50 samples.
            time_steps=1, 
            sel_sims=[10],
            batch_size=1,
            shuffle=False,
            local_datasets_dir=data_dir
        )
        
        # This will correctly report 50 samples
        print(f"Dataloader found {len(data_loader.dataset)} samples. Stacking all {total_time_steps} frames...")

        all_steps_list = []
        last_y_batch = None
        
        # --- 2. Fix the loop: Dataloader returns 2 items ---
        for x_batch, y_batch in data_loader:
            # x_batch is (B, T_in, C, H, W) -> (1, 1, 4, 80, 64)
            # y_batch is (B, T_out, C, H, W) -> (1, 1, 4, 80, 64)
            
            # 3. Append the input frame (x_batch)
            step_data = x_batch.squeeze(0).squeeze(0).to(device)
            all_steps_list.append(step_data)
            
            # 4. Store the target frame (y_batch)
            last_y_batch = y_batch
        
        # 5. After the loop (which ran 50 times), add the final frame
        if last_y_batch is not None:
            final_step_data = last_y_batch.squeeze(0).squeeze(0).to(device)
            all_steps_list.append(final_step_data)

        if not all_steps_list:
            print("Error: No data loaded. Dataloader was empty.")
            return None

        # Stack all tensors (50 + 1)
        data_sequence = torch.stack(all_steps_list, dim=0)
        
        # Final shape should be (51, 4, 80, 64)
        print(f"Data stacked. Final sequence shape: {data_sequence.shape}")
        
        if data_sequence.shape[0] != total_time_steps:
             print(f"--- WARNING ---")
             print(f"Loader logic failed. Expected {total_time_steps} frames, but stacked {data_sequence.shape[0]}.")
             print(f"--- END WARNING ---")

        return data_sequence
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


@torch.no_grad()
def run_autoregressive_rollout(model, initial_state_4channel, num_steps):
    """
    Generates a full simulation sequence using the multi-step model.
    At each step, we predict 4 steps but only use the first one.
    """
    print(f"Running autoregressive rollout for {num_steps} steps...")
    
    predicted_states_3channel = []
    
    # 1. Get the physics part of the initial state (channels 0, 1, 2)
    initial_physics = initial_state_4channel[0:3, ...]
    predicted_states_3channel.append(initial_physics)
    
    # 2. Get the static 1-channel inflow mask (channel 3)
    inflow_mask_1channel = initial_state_4channel[3, ...].unsqueeze(0)
    inflow_mask_batch = inflow_mask_1channel.unsqueeze(0).to(device)

    # 3. The first *input* to the model
    current_input_4channel = initial_state_4channel.unsqueeze(0).to(device)
    
    for step in range(num_steps):
        # 1. Predict 4 steps (12 channels)
        # Input: (1, 4, Y, X) -> Output: (1, 12, Y, X)
        predicted_3_channels = model(current_input_4channel)
        
        
        # 3. Store the 3-channel prediction (move to CPU)
        predicted_states_3channel.append(predicted_3_channels.squeeze(0))
        
        # 4. Create the *next* 4-channel input for the model
        current_input_4channel = torch.cat(
            (predicted_3_channels, inflow_mask_batch),
            dim=1 # Concatenate along the channel axis
        )
        
    # Stack all states (T=0 to T=50) along the time dimension
    full_sequence = torch.stack(predicted_states_3channel, dim=0)
    
    # Final shape will be (51, 3, 80, 64)
    print(f"Rollout complete. Predicted sequence shape: {full_sequence.shape}")
    return full_sequence


def reconstruct_fields(data_sequence):
    """
    Converts the raw data tensor back into PhiFlow Field objects.
    This function handles both 3-channel and 4-channel tensors
    because it only uses channels 0, 1, and 2.
    """
    if data_sequence is None: return None, None
    print("Reconstructing PhiFlow fields...")
    
    if data_sequence.ndim == 3:
        data_sequence = data_sequence.unsqueeze(0) 
    
    smoke_data_torch = data_sequence[:, 0, ...]    # Ch 0
    velo_data_torch = data_sequence[:, 1:3, ...] # Ch 1, 2
    velo_data_torch = velo_data_torch.permute(0, 2, 3, 1)

    smoke_values = math.tensor(smoke_data_torch, batch('time'), spatial('y,x'))
    velocity_values = math.tensor(velo_data_torch, batch('time'), spatial('y,x'), channel(vector='x,y'))

    smoke_field = CenteredGrid(smoke_values, extrapolation=extrapolation.BOUNDARY, bounds=DOMAIN)
    velocity_field = CenteredGrid(velocity_values, extrapolation=extrapolation.ZERO, bounds=DOMAIN)
    
    print(f"Smoke field shape: {smoke_field.shape}")
    print(f"Velocity field shape: {velocity_field.shape}")
    
    return smoke_field, velocity_field


def visualize_comparison(smoke_truth, vel_truth, smoke_pred, vel_pred):
    """
    Shows a side-by-side animation of ground truth and prediction.
    """
    if smoke_truth is None or smoke_pred is None:
        print("Cannot visualize. One of the fields is None.")
        return

    print("Showing side-by-side comparison (close window to exit)...")
    vis.plot(
        {
            'Ground Truth': smoke_truth,
            'Prediction': smoke_pred,
        },
        animate='time', 
        title="Ground Truth vs. Model Rollout"
    )
    plt.show()
    plt.close()


if __name__ == "__main__":
    # 1. Load the trained model
    model = load_trained_model(MODEL_PATH)
    
    if model is None:
        exit()

    # 2. Load the full ground truth (GT) sequence
    # This will have 4 channels: (density, vx, vy, inflow)
    gt_sequence_tensor = load_data(
        data_dir=local_data_directory,
        dset_name=dataset_name,
        total_time_steps=NUM_SAVED_STATES # 51
    )
    
    if gt_sequence_tensor is None:
        exit()
        
    # 3. Get the initial state (T=0) to start the rollout
    # Shape: (4, 80, 64)
    initial_state_tensor = gt_sequence_tensor[0]

    # 4. Run the autoregressive rollout
    # This will have 3 channels: (density, vx, vy)
    pred_sequence_tensor = run_autoregressive_rollout(
        model=model,
        initial_state_4channel=initial_state_tensor,
        num_steps=TOTAL_STEPS  # Predict 50 new steps
    )

    # 5. Reconstruct all fields for visualization
    # reconstruct_fields can handle the 4-channel GT and 3-channel pred
    print("Reconstructing ground truth fields...")
    smoke_gt, velocity_gt = reconstruct_fields(gt_sequence_tensor)
    
    print("Reconstructing predicted fields...")
    smoke_pred, velocity_pred = reconstruct_fields(pred_sequence_tensor)
    
    # 6. Show the side-by-side animation
    visualize_comparison(
        smoke_gt, velocity_gt,
        smoke_pred, velocity_pred
    )