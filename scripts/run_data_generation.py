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
    experiments = ['burgers', 'smoke']

    CONFIG_FILE_PATHS = {
        'burgers': os.path.join(PROJECT_ROOT, 'configs', 'burgers.yaml'),
        'smoke': os.path.join(PROJECT_ROOT, 'configs', 'smoke.yaml')
    }

    # --- Run ---
    for exp in experiments:
        config_path = CONFIG_FILE_PATHS[exp]
        run_generation(config_path, PROJECT_ROOT)