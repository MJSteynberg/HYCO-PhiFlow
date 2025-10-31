# run.py

import argparse
import yaml
import os
import sys
from typing import List, Dict, Any

# Add project root to path to allow imports from 'src'
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

# --- Import Task Runners ---
from src.data.generator import run_generation
from src.training.synthetic.trainer import SyntheticTrainer
from src.training.physical.trainer import PhysicalTrainer
from src.evaluation import Evaluator

def main():
    parser = argparse.ArgumentParser(description="Unified runner for the PDE modeling project.")
    parser.add_argument('--config', type=str, required=True, help='Path to the unified experiment YAML file.')
    
    # --- MODIFICATION: Removed --task argument ---
    # The tasks to run are now defined inside the config file.
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # --- Add project root and log directory to config ---
    config['project_root'] = PROJECT_ROOT
    
    run_config = config['run_params']

    print(f"--- Loaded Experiment: {run_config['experiment_name']} ---")
    
    # --- MODIFICATION: Get tasks from config ---
    tasks: List[str] = run_config['mode']
    if isinstance(tasks, str):
        tasks = [tasks]  # Ensure 'tasks' is a list

    print(f"--- Will Run Tasks: {tasks} ---")

    # --- MODIFICATION: Loop over tasks from config ---
    for task in tasks:
        print(f"\n--- Running Task: {task.upper()} ---")
        
        if task == 'generate':
            run_generation(config)

        elif task == 'generate_scene' :
            run_generation(config)
            
        elif task == 'train':
            model_type = run_config.get('model_type', 'synthetic')
            print(f"Model type specified: '{model_type}'")

            if model_type == 'synthetic':
                # SyntheticTrainer's __init__ takes the full config
                trainer = SyntheticTrainer(config)
                trainer.train()
                
            elif model_type == 'physical':
                # Use the new Scene-based trainer for inverse problems
                print("Using PhysicalTrainerScene for inverse problem.")
                trainer = PhysicalTrainer(config)
                trainer.train()
                
            else:
                raise ValueError(
                    f"Unknown model_type '{model_type}' in config. "
                    f"Must be 'synthetic' or 'physical'."
                )

        elif task == 'evaluate':
            model_type = run_config.get('model_type', 'synthetic')
            print(f"Model type specified: '{model_type}'")
            
            if model_type == 'synthetic':
                print("Running evaluation for synthetic model...")
                
                # Check if evaluation_params exist in config
                if 'evaluation_params' not in config:
                    print("Warning: No 'evaluation_params' found in config. Using defaults.")
                    config['evaluation_params'] = {
                        'test_sim': [0],
                        'num_frames': 51,
                        'metrics': ['mse', 'mae', 'rmse'],
                        'keyframe_count': 5,
                        'animation_fps': 10,
                        'save_animations': True,
                        'save_plots': True
                    }
                
                # Create evaluator and run evaluation
                evaluator = Evaluator(config)
                results = evaluator.evaluate()
                
                print(f"Evaluation complete! Results saved.")
                
            elif model_type == 'physical':
                print("Physical model evaluation not yet implemented.")
                print("Physical models are evaluated during training (inverse problem setup).")
                
            else:
                raise ValueError(
                    f"Unknown model_type '{model_type}' in config. "
                    f"Must be 'synthetic' or 'physical'."
                )
            
            
        else:
            print(f"--- Warning: Unknown task '{task}' in config. Skipping. ---")

    print(f"\n--- Experiment {run_config['experiment_name']} Finished ---")

if __name__ == "__main__":
    main()