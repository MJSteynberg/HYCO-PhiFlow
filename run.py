import argparse
import yaml
import os
import sys

# Add project root to path to allow imports from 'src'
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

# Import task runners/functions
# NOTE: These functions must be refactored to accept a single config dict
from src.data_generation.generator import run_generation
from src.training.synthetic.trainer import SyntheticTrainer
from src.data_generation.subsample import run_subsampling
# from src.evaluation.evaluator import run_evaluation # (Assuming this is refactored)

def main():
    parser = argparse.ArgumentParser(description="Unified runner for the PDE modeling project.")
    parser.add_argument('--config', type=str, required=True, help='Path to the unified experiment YAML file.')
    parser.add_argument('--task', type=str, required=True, choices=['generate', 'train', 'evaluate', 'subsample'], help='The task to run.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Add project root to config for consistent pathing
    config['project_root'] = PROJECT_ROOT

    print(f"Loaded configuration from {config['data']['dset_name']} for task '{args.task}'.")
    print(f"The loaded fields are: {config['data']['fields']}")

    if args.task == 'generate':
        print(f"--- Running Task: Data Generation ---")
        run_generation(config)
    elif args.task == 'train':
        print(f"--- Running Task: Training ---")
        trainer = SyntheticTrainer(config)
        trainer.train()
    elif args.task == 'evaluate':
        print(f"--- Running Task: Evaluation ---")
        # run_evaluation(config) # Placeholder for the refactored evaluation logic
        print("Evaluation task not yet fully refactored.")
    elif args.task == 'subsample':
        print(f"--- Running Task: Subsampling ---")
        run_subsampling(config)

if __name__ == "__main__":
    main()