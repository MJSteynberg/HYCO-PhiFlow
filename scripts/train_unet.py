# train_autoregressive.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from pbdl.torch.loader import Dataloader
import phiml.nn as pnn  # PhiFlow's unified NN API
import time

print("Imports complete. Ready for AUTOREGRESSIVE training.")

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Simulation Parameters (MUST match generation script) ---
RES_X = 512
RES_Y = 512
DOMAIN_SIZE_X = 100
DOMAIN_SIZE_Y = 100

# --- Data Loading Parameters ---
dataset_name = "smoke_v1"
local_data_directory = "./data"

# --- Training Parameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
EPOCHS = 500
# This now defines the *length of the autoregressive rollout* during training
NUM_PREDICT_STEPS = 4 

# --- Model & Checkpoint Paths ---
MODEL_PATH = "./smoke"
MODEL_NAME = f"{local_data_directory}_unet_autoregressive_{NUM_PREDICT_STEPS}step"
CHECKPOINT_PATH = os.path.join(MODEL_PATH, f"{MODEL_NAME}.pth")
MODEL_SCRIPT_PATH = os.path.join(MODEL_PATH, f"{MODEL_NAME}.pt")

def create_data_loader(data_dir, dset_name, batch_size):
    """
    Creates a pbdl.Dataloader configured for multi-step training.
    
    Args:
        data_dir (str): Path to the directory containing the HDF5 file.
        dset_name (str): Name of the dataset (HDF5 file name).
        batch_size (int): The batch size.
        
    Returns:
        pbdl.torch.loader.Dataloader: The configured data loader.
    """
    print(f"Setting up data loader for '{dset_name}'...")
    
    hdf5_filepath = os.path.join(data_dir, f"{dset_name}.hdf5")
    if not os.path.exists(hdf5_filepath):
        print(f"Error: Dataset not found at {hdf5_filepath}")
        return None

    loader = Dataloader(
        dset_name,
        load_fields=['density', 'velocity', 'inflow'],
        
        # --- THIS IS THE CORRECT CONFIGURATION ---
        time_steps=NUM_PREDICT_STEPS, 
        intermediate_time_steps=True,
        batch_size=batch_size,
        shuffle=True,
        sel_sims=[0],
        local_datasets_dir=data_dir
    )
    return loader


def create_model():
    """
    Creates the U-Net model.
    
    Note: The model architecture is unchanged. It still takes 4 input
    channels (d, vx, vy, inflow) and predicts 12 output channels
    (3 physics channels * 4 future steps).
    
    Returns:
        torch.nn.Module: The U-Net model.
    """
    print("Creating U-Net model...")
    
    # 4 input channels: (density, vx, vy, inflow_field)
    in_channels = 4
    
    # 12 output channels: (d, vx, vy) for t+1, (d, vx, vy) for t+2, ...
    out_channels = NUM_PREDICT_STEPS * 3
    
    model = pnn.u_net(
        in_channels=in_channels,
        out_channels=out_channels,
        levels=4,
        filters=64,
        batch_norm=True,
    )

    try:
        model.load_state_dict(torch.load(MODEL_SCRIPT_PATH, map_location=device))
        print(f"Loaded model weights from {MODEL_SCRIPT_PATH}")
    except FileNotFoundError:
        print("No pre-existing model weights found. Training from scratch.")
    
    model = model.to(device)
    print("Model created successfully and moved to device.")
    return model

def train_epoch(model, train_loader, optimizer, scheduler, loss_fn):
    """
    Runs one epoch of autoregressive training.

    Args:
        model (torch.nn.Module): The U-Net model.
        train_loader (Dataloader): The data loader for training data.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
        loss_fn (torch.nn.Module): The loss function.

    Returns:
        float: The average training loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    
    for x_batch, y_batch in train_loader:
        # x_batch shape: (B, 4, Y, X) -> State at t=0
        # y_batch shape: (B, T, 4, Y, X) -> States at t=1, t+2, ..., t+T
        
        # Extract the static inflow field from the input (channel 3)
        # This will be re-used for every step in the rollout
        inflow_field_4c = x_batch[:, 3:4, ...].to(device) # Shape: (B, 1, Y, X)
        
        # Initialize the 'current state' with the 3 physics channels from the input
        current_state_3c = x_batch[:, 0:3, ...].to(device) # Shape: (B, 3, Y, X)
        
        batch_rollout_loss = 0.0
        
        # We must zero the gradients *before* the rollout loop
        optimizer.zero_grad()

        # Autoregressive rollout loop
        for t_step in range(NUM_PREDICT_STEPS):
            # 1. Prepare the 4-channel input for the model
            model_input_4c = torch.cat([current_state_3c, inflow_field_4c], dim=1)
            
            # 2. Forward pass: (B, 4, Y, X) -> (B, 12, Y, X)
            # The model predicts all 4 future steps based on the *current* input
            pred_all_steps_12c = model(model_input_4c)
            
            # 3. Extract *only* the first predicted step (t+1)
            # We use this as our prediction for this step of the rollout
            pred_this_step_3c = pred_all_steps_12c[:, 0:3, :, :] # Shape: (B, 3, Y, X)
            
            # 4. Get the corresponding ground truth for this step
            # y_batch[:, t_step, ...] is the GT at time t+1, t+2, etc.
            gt_this_step_3c = y_batch[:, t_step, 0:3, ...].to(device)
            
            # 5. Calculate and accumulate the loss for this step
            step_loss = loss_fn(pred_this_step_3c, gt_this_step_3c)
            batch_rollout_loss = batch_rollout_loss + step_loss
            
            # 6. Set up for the next iteration:
            # The model's prediction becomes the input for the next step
            current_state_3c = pred_this_step_3c
            
        # --- End of Rollout Loop ---
        
        # We average the loss over the rollout steps
        avg_rollout_loss = batch_rollout_loss / NUM_PREDICT_STEPS
        
        # 7. Backpropagate the total accumulated loss from the entire rollout
        avg_rollout_loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        total_loss += avg_rollout_loss.item()
        
    # --- End of Epoch ---
    avg_loss = total_loss / len(train_loader)
    return avg_loss

if __name__ == "__main__":
    
    # 1. Create Data Loader
    train_loader = create_data_loader(
        data_dir=local_data_directory,
        dset_name=dataset_name,
        batch_size=BATCH_SIZE
    )
    
    if train_loader is None:
        exit()

    # 2. Create Model
    model = create_model()

    # 3. Set up Loss, Optimizer, and Scheduler
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader))

    # 4. Training Loop
    print(f"\nStarting autoregressive training for {EPOCHS} epochs...")
    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn
        )
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.6f} | Time: {epoch_time:.2f}s")

        # 5. Save Checkpoint
        if train_loss < best_loss and epoch % 10 == 0:
            best_loss = train_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> New best model saved to {CHECKPOINT_PATH}")

        # 6. Save the scripted model at the end of training
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print("Autoregressive training complete.")