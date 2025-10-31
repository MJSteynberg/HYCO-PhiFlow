"""
Hydra-based experiment runner.

Usage:
    python run_hydra.py experiment=burgers_experiment
    python run_hydra.py experiment=burgers_experiment trainer_params.epochs=200
    python run_hydra.py +experiment=smoke_experiment run_params.mode=[generate,train]
"""

import os
import sys
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generator import run_generation
from src.training.synthetic.trainer import SyntheticTrainer
from src.training.physical.trainer import PhysicalTrainer
from src.evaluation import Evaluator


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    
    # Print configuration (optional, for debugging)
    print("=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    # Convert to regular dict for compatibility with existing code
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Add project root
    config['project_root'] = str(PROJECT_ROOT)
    
    run_config = config['run_params']
    tasks = run_config['mode']
    
    if isinstance(tasks, str):
        tasks = [tasks]
    
    print(f"\n--- Experiment: {run_config['experiment_name']} ---")
    print(f"--- Tasks: {tasks} ---\n")
    
    # Execute tasks
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"RUNNING TASK: {task.upper()}")
        print(f"{'='*60}\n")
        
        if task == 'generate':
            run_generation(config)
        
        elif task == 'train':
            model_type = run_config.get('model_type', 'synthetic')
            
            if model_type == 'synthetic':
                trainer = SyntheticTrainer(config)
                trainer.train()
            elif model_type == 'physical':
                trainer = PhysicalTrainer(config)
                trainer.train()
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
        
        elif task == 'evaluate':
            model_type = run_config.get('model_type', 'synthetic')
            
            if model_type == 'synthetic':
                evaluator = Evaluator(config)
                evaluator.evaluate()
            else:
                print("Physical model evaluation not yet implemented.")
        
        else:
            print(f"Warning: Unknown task '{task}'")
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE: {run_config['experiment_name']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()