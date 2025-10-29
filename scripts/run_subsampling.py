# scripts/run_subsampling.py

import os
import sys

# --- Add project root to path ---
# This allows us to import from 'src'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# --- Import core logic ---
# We assume the new module is in src/data_generation/subsample.py
try:
    from src.data_generation.subsample import run_subsampling
except ImportError:
    print("Error: Could not find 'src.data_generation.subsample'.")
    print("Please make sure 'src/data_generation/subsample.py' exists.")
    print("You may also need to create an empty 'src/data_generation/__init__.py' file.")
    sys.exit(1)


if __name__ == "__main__":
    # --- Define Configuration ---
    # This script will need its own config file
    # to specify *which* dataset to subsample and *what*
    # new resolution to use.
    CONFIG_FILE_PATH = os.path.join(
        PROJECT_ROOT, 
        'configs', 
        'smoke_subsample.yaml' # <-- Example config
    )
    
    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"Error: Config file not found at {CONFIG_FILE_PATH}")
        print("Please create a config file for subsampling.")
        print("An example 'subsample_burgers.yaml' is provided in the documentation.")
        sys.exit(1)

    # --- Run ---
    run_subsampling(
        CONFIG_FILE_PATH, 
        PROJECT_ROOT
    )