# scripts/run_plot_scene.py

import os
import sys
import yaml
import argparse
from typing import Dict

# --- Add project root to path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# --- PhiFlow Imports ---
from phi.torch.flow import *

# --- Helper import from our own code ---
from src.visualization.plotter import load_config


def plot_scene_data(config: Dict, sim_index: int):
    """
    Loads and animates all specified fields from a Scene for one simulation.
    """
    data_cfg = config['data']
    
    # 1. --- Construct the Scene Path ---
    dset_name = data_cfg['dset_name']
    scene_parent_dir = os.path.join(
        PROJECT_ROOT, 
        data_cfg['data_dir'], 
        dset_name
    )
    scene_name = f"sim_{sim_index:06d}"
    scene_path = os.path.join(scene_parent_dir, scene_name)

    if not os.path.exists(scene_path):
        print(f"Error: Scene directory not found at {scene_path}")
        print("Please run the 'generate_scene' task first.")
        return

    print(f"Loading data from Scene: {scene_path}")
    
    # 2. --- Open the Scene ---
    scene = Scene.at(scene_path)
    
    # 3. --- Get field names and check what's available ---
    available_fields = scene.fieldnames
    print(f"Available fields in scene: {available_fields}")
    
    # Get all frames (union of all fields)
    frames = scene.frames
    if not frames:
        print(f"Error: No frames found in scene.")
        print(f"Contents of {scene_path}:")
        print(os.listdir(scene_path))
        return
    
    print(f"Found {len(frames)} frames: {min(frames)} to {max(frames)}")
    
    # 4. --- Define save directory for plots ---
    save_dir = os.path.join(PROJECT_ROOT, "results", "plots_scene", dset_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Plots will be saved to: {save_dir}")

    # 5. --- Load and Plot Each Field ---
    fields_to_plot = data_cfg['fields']
    
    for field_name in fields_to_plot:
        # Skip fields that don't exist
        if field_name not in available_fields:
            print(f"  -> Warning: Field '{field_name}' not found in scene. Skipping.")
            continue
            
        print(f"  -> Loading and plotting '{field_name}'...")
        
        try:
            # Read all frames for this field
            field_frames = []
            for frame in frames:
                try:
                    field_data = scene.read_field(field_name, frame=frame, convert_to_backend=True)
                    field_frames.append(field_data)
                except FileNotFoundError:
                    print(f"     Warning: Frame {frame} not found for '{field_name}'. Skipping this frame.")
                    continue
            
            if not field_frames:
                print(f"     Error: No frames loaded for '{field_name}'. Skipping.")
                continue
            
            # Stack frames along time dimension
            field_sequence = stack(field_frames, batch('time'))
            
            title = f"{dset_name} - {field_name.capitalize()} (Sim {sim_index})"
            save_path = os.path.join(save_dir, f"{field_name}_sim_{sim_index:06d}.gif")
            
            # Handle velocity magnitude plot
            if field_name == 'velocity':
                plot_data = field.vec_length(field_sequence)
                title = f"{dset_name} - Velocity Magnitude (Sim {sim_index})"
                save_path = os.path.join(save_dir, f"velocity_magnitude_sim_{sim_index:06d}.gif")
            else:
                plot_data = field_sequence

            # 6. --- Plot and Save Animation ---
            vis.plot(plot_data, animate='time', show_color_bar=True, title=title)
            vis.savefig(save_path)
            
            print(f"     Saved animation to {save_path}")

        except Exception as e:
            print(f"     Error plotting field '{field_name}': {e}")
            import traceback
            traceback.print_exc()
            
    print("\nPlotting complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and plot data from a PhiFlow Scene.")
    parser.add_argument('--config', type=str, required=True, help='Path to the unified experiment YAML file.')
    parser.add_argument('--sim', type=int, default=0, help='The simulation index to plot (default: 0).')
    args = parser.parse_args()

    # 1. Load Config
    config = load_config(args.config)
    
    # 2. Get sim index
    sim_index = args.sim
    if sim_index == 0:
        # Try to get test_sim from config as a default
        test_sims = config.get('trainer_params', {}).get('test_sim', [0])
        if isinstance(test_sims, list) and test_sims:
            sim_index = test_sims[0]

    # 3. Run plotting
    plot_scene_data(config, sim_index=sim_index)