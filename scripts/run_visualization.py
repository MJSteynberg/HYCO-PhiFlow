# scripts/run_visualization.py

import os
import sys

# --- Add project root to path ---
# This allows us to import from 'src'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# --- Import core logic ---
from src.visualization.plotter import run_visualization


if __name__ == "__main__":
    # --- Define Configuration ---
    CONFIG_FILE_PATH = os.path.join(PROJECT_ROOT, 'configs', 'smoke.yaml')
    
    # --- Select simulation to visualize ---
    # You can change this to load any sim from your dataset
    SIM_INDEX_TO_LOAD = 5
    
    # --- Run ---
    run_visualization(
        CONFIG_FILE_PATH, 
        PROJECT_ROOT, 
        sim_to_load=SIM_INDEX_TO_LOAD
    )