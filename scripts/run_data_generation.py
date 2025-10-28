# scripts/run_data_generation.py

import os
import sys

# --- Add project root to path ---
# This allows us to import from 'src'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# --- Import core logic ---
from src.data_generation.generator import run_generation


if __name__ == "__main__":
    # --- Define Configuration ---
    CONFIG_FILE_PATH = os.path.join(PROJECT_ROOT, 'configs', 'smoke.yaml')
    
    # --- Run ---
    run_generation(CONFIG_FILE_PATH, PROJECT_ROOT)