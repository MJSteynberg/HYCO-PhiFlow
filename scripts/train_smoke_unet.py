# scripts/train_unet.py
import os
import sys
# --- Add project root to path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from src.training.synthetic.trainer import SyntheticTrainer

print("Starting synthetic model training...")

TRAINING_CONFIG = {
    # --- Data Loading ---
    "data_dir": "./data",
    "dset_name": "smoke_128",
    
    # --- NEW: Define the exact field order from the dataset ---
    "data_loader_fields": ['density', 'velocity', 'inflow'],
    
    # --- Training Parameters ---
    "learning_rate": 1e-4,
    "batch_size": 8,
    "epochs": 50,
    "num_predict_steps": 4, # Autoregressive rollout length

    # --- Model & Checkpoint Paths ---
    "model_path": "results/models",
    "model_name": "smoke_128_unet_autoregressive",

    # --- Model Configuration ---
    "model": {
        'input_specs': {
            'density': 1,
            'velocity': 2,
            'inflow': 1
        },
        'output_specs': {
            'density': 1,
            'velocity': 2
        },
        'levels': 4,
        'filters': 64,
        'batch_norm': True,
    }
}

if __name__ == "__main__":
    
    # 1. Initialize the Trainer
    trainer = SyntheticTrainer(config=TRAINING_CONFIG)
    
    # 2. Run Training
    trainer.train()

    print("Training script finished.")