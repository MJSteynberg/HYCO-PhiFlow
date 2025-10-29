# scripts/run_evaluation.py

import os
import sys
import torch
import yaml
from phi.torch.flow import Box, math
from typing import List

# --- Add project root to path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# --- NEW: Import our UNet model ---
from src.models.synthetic.unet import UNet

# --- NEW: Import our generic plotting/loading functions ---
from src.visualization.plotter import (
    load_config,
    load_data,
    reconstruct_fields_from_specs,
    plot_comparison
)

print("Imports complete. Ready to evaluate.")

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_trained_model(model_path, config):
    """
    (This function is unchanged from your last version)
    Loads the trained U-Net model using the new UNet class.
    """
    print(f"Loading trained model from {model_path}...")
    
    model = UNet(config=config).to(device)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run the training script first.")
        return None
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print("Model loaded successfully.")
    return model


@torch.no_grad()
def run_autoregressive_rollout(model: UNet, 
                               initial_state_tensor: torch.Tensor, 
                               data_loader_fields: List[str],
                               num_steps: int) -> torch.Tensor:
    """
    --- MODIFIED: This is now generic and input-agnostic ---
    Generates a full simulation sequence using the dictionary-based model.
    """
    print(f"Running autoregressive rollout for {num_steps} steps...")
    
    predicted_states_list = []
    
    # --- Generic Field Setup ---
    all_specs = {**model.INPUT_SPECS, **model.OUTPUT_SPECS}
    data_loader_channels = {f: all_specs[f] for f in data_loader_fields}
    
    # Get specs for dynamic (predicted) fields
    output_specs = model.OUTPUT_SPECS
    dynamic_fields = list(output_specs.keys())
    
    # Get names of static (input-only) fields
    static_fields = [f for f in model.INPUT_SPECS if f not in model.OUTPUT_SPECS]

    # --- Unpack Initial State (T=0) ---
    current_state_dict = {} # (e.g., {'density': ..., 'velocity': ...})
    static_field_dict = {}  # (e.g., {'inflow': ...})
    
    start_channel = 0
    for field_name in data_loader_fields:
        num_channels = data_loader_channels[field_name]
        end_channel = start_channel + num_channels
        
        field_tensor_batch = initial_state_tensor[start_channel:end_channel, ...].unsqueeze(0).to(device)
        
        if field_name in dynamic_fields:
            current_state_dict[field_name] = field_tensor_batch
        if field_name in static_fields:
            static_field_dict[field_name] = field_tensor_batch
            
        start_channel = end_channel

    # 1. Store the initial dynamic state (T=0)
    # We re-combine the dict values into a tensor for storage.
    initial_dynamic_tensor = torch.cat(
        [current_state_dict[key] for key in output_specs.keys()], dim=1
    ).squeeze(0)
    predicted_states_list.append(initial_dynamic_tensor)
    
    
    # 2. --- Generic Rollout Loop ---
    for step in range(num_steps):
        # 1. Prepare the input dictionary
        model_input_dict = {**current_state_dict, **static_field_dict}
        
        # 2. Predict: (Dict) -> (Dict)
        pred_dict = model(model_input_dict)
        
        # 3. Store the prediction tensor
        pred_tensor = torch.cat(
            [pred_dict[key] for key in output_specs.keys()], dim=1
        ).squeeze(0)
        
        predicted_states_list.append(pred_tensor)
        
        # 4. Set up for the next iteration
        current_state_dict = pred_dict
        
    # Stack all states (T=0 to T=50) along the time dimension
    full_sequence = torch.stack(predicted_states_list, dim=0)
    
    print(f"Rollout complete. Predicted sequence shape: {full_sequence.shape}")
    return full_sequence


if __name__ == "__main__":
    
    # --- NEW: Config-driven execution ---
    if len(sys.argv) < 2:
        print("Error: Please provide the path to a training config YAML.")
        print("Usage: python scripts/run_evaluation.py configs/smoke_128.yaml")
        exit()
        
    config_path = sys.argv[1]
    
    # 1. Load Configs
    train_config = load_config(config_path)
    
    # Get parameters from the *same config* used for training
    data_cfg = train_config['data_config']
    domain_cfg = train_config['domain']
    model_cfg = train_config['model']
    
    dataset_name = data_cfg['dset_name']
    data_dir = data_cfg['data_dir']
    data_loader_fields = data_cfg['data_loader_fields']
    
    sim_cfg = train_config['simulation']
    num_saved_states = (sim_cfg['total_steps'] // sim_cfg['save_interval']) + 1
    total_steps = sim_cfg['total_steps']

    # 2. Re-create the Domain object
    domain = Box(x=domain_cfg['size_x'], y=domain_cfg['size_y'])
    
    # 3. Load the trained model
    model_name = train_config['model_name']
    model_path = os.path.join(train_config['model_path'], f"{model_name}.pth")
    model = load_trained_model(model_path, config=model_cfg)
    
    if model is None:
        exit()

    # 4. Load the full ground truth (GT) sequence
    sim_to_load = 50 # Hard-code sim 30 for evaluation
    gt_sequence_tensor = load_data(
        data_dir=data_dir,
        dset_name=dataset_name,
        fields_to_load=data_loader_fields,
        sim_index=sim_to_load,
        total_time_steps=num_saved_states
    )
    
    if gt_sequence_tensor is None:
        exit()
        
    # 5. Get the initial state (T=0)
    initial_state_tensor = gt_sequence_tensor[0] # (C, Y, X)

    # 6. Run the autoregressive rollout
    pred_sequence_tensor = run_autoregressive_rollout(
        model=model,
        initial_state_tensor=initial_state_tensor,
        data_loader_fields=data_loader_fields,
        num_steps=total_steps
    )

    # 7. Reconstruct all fields for visualization
    # We use the *output_specs* for the prediction tensor
    # We use the *data_loader_fields* for the ground truth tensor
    
    print("Reconstructing ground truth fields...")
    gt_all_specs = {**model_cfg['input_specs'], **model_cfg['output_specs']}
    gt_specs_ordered = {f: gt_all_specs[f] for f in data_loader_fields}
    gt_fields = reconstruct_fields_from_specs(
        gt_sequence_tensor, gt_specs_ordered, domain
    )
    
    print("Reconstructing predicted fields...")
    pred_fields = reconstruct_fields_from_specs(
        pred_sequence_tensor, model_cfg['output_specs'], domain
    )
    
    # 8. Show the side-by-side animation
    plot_comparison(gt_fields, pred_fields, sim_name=dataset_name, sim_index=sim_to_load)