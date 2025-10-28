# src/data_generation/generator.py

import os
import yaml
import h5py
import numpy as np
import random

# --- PhiFlow Imports ---
from phi.torch.flow import (
    Box,
    CenteredGrid,
    extrapolation,
    math,
    batch,
    spatial,
    channel
)

# --- Our Model Imports ---
# We use a dynamic import to make it "plug-and-play"
# This import works because the calling script adds the project root to sys.path
import src.models.physical as physical_models


def load_config(config_path: str) -> dict:
    """Loads a YAML config file.
    
    Args:
        config_path: Path to the .yaml configuration file.

    Returns:
        A dictionary containing the configuration.
    """
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_physical_model(config: dict) -> physical_models.PhysicalModel:
    """
    Dynamically imports and instantiates a physical model from config.
    
    This function also generates a random inflow center based on
    config ranges.

    Args:
        config: Configuration dictionary.

    Returns:
        An instance of a PhysicalModel (e.g., SmokeModel).
    """
    model_name = config['model_name']
    
    # 1. Build Domain and Resolution
    domain_cfg = config['domain']
    res_cfg = config['resolution']
    domain = Box(x=domain_cfg['size_x'], y=domain_cfg['size_y'])
    resolution = spatial(x=res_cfg['x'], y=res_cfg['y'])
    
    # 2. Get PDE params and generate random inflow center
    pde_params = config.get('pde_params', {}).copy()

    # 3. Get the Model Class
    try:
        ModelClass = getattr(physical_models, model_name)
    except AttributeError:
        raise ImportError(f"Model '{model_name}' not found in src/models/physical/__init__.py")

    # 4. Instantiate the model
    model = ModelClass(
        domain=domain,
        resolution=resolution,
        dt=config['dt'],
        **pde_params
    )
    return model


def run_generation(config_path: str, project_root: str):
    """
    Main function to run data generation based on a config.

    Args:
        config_path: Path to the simulation config file.
        project_root: Absolute path to the project's root directory.
    """
    config = load_config(config_path)
    gen_cfg = config['data_generation']
    out_cfg = config['output_data']

    # --- Setup HDF5 File ---
    output_path = os.path.join(project_root, out_cfg['output_dir'])
    os.makedirs(output_path, exist_ok=True)
    dset_path = os.path.join(output_path, f"{out_cfg['dataset_name']}.hdf5")
    
    print(f"Starting {gen_cfg['num_simulations']} simulations.")
    print(f"Data will be saved to: {dset_path}")
    
    with h5py.File(dset_path, 'w') as f:
        
        sims_group = f.create_group('sims')
        
        for i in range(gen_cfg['num_simulations']):
            print(f"\n--- Running Simulation {i+1}/{gen_cfg['num_simulations']} ---")
            
            # --- 1. Get a new model with a new random inflow ---
            model = get_physical_model(config)
            
            # --- 2. Get initial state (with batch_size=1) ---
            state_list = [
                math.expand(s, batch(time=1))
                for s in model.get_initial_state()
            ]
            
            # --- 3. Run the simulation loop ---
            for t in range(1, gen_cfg['total_steps'] + 1):
                current_state = [s.time[-1] for s in state_list]
                next_state = model.step(*current_state)
                
                if t % gen_cfg['save_interval'] == 0:
                    for j in range(len(state_list)):
                        new_slice = math.expand(next_state[j], batch(time=1))
                        state_list[j] = math.concat([state_list[j], new_slice], 'time')

            # --- 4. Prepare data for saving ---
            vel, den = state_list[0], state_list[1] 
            vel_x_cen = vel.vector['x'].at(den)
            vel_y_cen = vel.vector['y'].at(den)
            
            sim_data = math.stack([
                den.values,
                vel_x_cen.values,
                vel_y_cen.values,
                model.inflow.values
            ], dim=channel('channels'))
            
            print(f"Simulation {i} data shape (with batch=1): {sim_data.shape}")
            sim_data_np = sim_data.batch[0].numpy('time, channels, y, x') 
            sims_group.create_dataset(f'sim{i}', data=sim_data_np)
        
        # --- 5. Save metadata ---
        print("\nAll simulations complete. Saving metadata...")
        
        sims_group.attrs['PDE'] = config.get('pde_name', 'Incompressible Navier-Stokes with Boussinesq')
        sims_group.attrs['Fields Scheme'] = out_cfg['fields_scheme']
        sims_group.attrs['Fields'] = out_cfg['fields_to_save']
        sims_group.attrs['Constants'] = [] 
        
        saved_dt = config['dt'] * gen_cfg['save_interval']
        sims_group.attrs['Dt'] = float(saved_dt)