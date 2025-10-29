# scripts/train_burgers_unet.py
import os
import sys
# --- Add project root to path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.training.synthetic.trainer import SyntheticTrainer

print("Starting synthetic model training for Burgers' equation...")

# --- This config is the *only* thing that needs to change ---
TRAINING_CONFIG = {
    # --- Data Loading ---
    "data_dir": "./data",
    "dset_name": "burgers_128", # Use the Burgers dataset
    
    # --- Data Loader Fields ---
    # The BurgersModel only saves 'velocity'
    "data_loader_fields": ['velocity'],
    
    # --- Training Parameters ---
    "learning_rate": 1e-4,
    "batch_size": 16,
    "epochs": 50,
    "num_predict_steps": 6, # Autoregressive rollout length

    # --- Model & Checkpoint Paths ---
    "model_path": "results/models",
    "model_name": "burgers_128_unet_autoregressive", # New model name

    # --- Model Configuration ---
    "model": {
        # Input: The model just needs the current velocity (1 channel)
        'input_specs': {
            'velocity': 1
        },
        # Output: The model predicts the next velocity (1 channel)
        'output_specs': {
            'velocity': 1
        },
        # U-Net params
        'levels': 4,
        'filters': 64,
        'batch_norm': True,
    }
}

if __name__ == "__main__":
    
    # 1. Initialize the Trainer
    trainer = SyntheticTrainer(config=TRAINING_CONFIG)
    print(trainer.model)
    # 2. Run Training
    trainer.train()

    print("Burgers training script finished.")