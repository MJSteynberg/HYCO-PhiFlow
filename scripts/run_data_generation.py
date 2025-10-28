import os
import sys
import yaml
import h5py
import numpy as np
import random

# --- Add project root to path ---
# This allows us to import from 'src'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# --- PhiFlow Imports ---
from phi.torch.flow import (
    Box,
    CenteredGrid,
    extrapolation,
    math,
    batch,
    spatial
)

# --- Our Model Imports ---
# We use a dynamic import to make it "plug-and-play"
import src.models.physical as physical_models


def load_config(config_path: str) -> dict:
    """Loads a YAML config file."""
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_physical_model(config: dict) -> physical_models.PhysicalModel:
    """
    Dynamically imports and instantiates a physical model.
    """
    model_name = config['model_name']
    
    # 1. Build Domain and Resolution
    domain_cfg = config['domain']
    res_cfg = config['resolution']
    domain = Box(x=domain_cfg['size_x'], y=domain_cfg['size_y'])
    resolution = spatial(x=res_cfg['x'], y=res_cfg['y'])
    
    # 2. Get PDE params and generate random inflow center
    # Make a copy so we can pop items from it
    pde_params = config.get('pde_params', {}).copy()
    
    # --- FIX: Generate inflow_center from config ranges ---
    x_range = pde_params.pop('inflow_rand_x_range', [0.2, 0.6])
    y_range = pde_params.pop('inflow_rand_y_range', [0.1, 0.2])
    
    rand_x = domain.size[0] * (x_range[0] + x_range[1] * random.random())
    rand_y = domain.size[1] * (y_range[0] + y_range[1] * random.random())
    
    # Add the generated center to the params dict
    pde_params['inflow_center'] = (rand_x, rand_y)
    
    print(f"Generated new inflow position: ({rand_x:.1f}, {rand_y:.1f})")
    # 'inflow_radius' is already in pde_params
    # 'nu' and 'buoyancy' are also in there
    # --- END FIX ---

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
        **pde_params  # Unpacks nu, buoyancy, inflow_center, inflow_radius
    )
    return model


def run_generation(config_path: str):
    """
    Main function to run data generation based on a config.
    """
    config = load_config(config_path)
    gen_cfg = config['data_generation']
    out_cfg = config['output_data']

    # --- Setup HDF5 File ---
    output_path = os.path.join(PROJECT_ROOT, out_cfg['output_dir'])
    os.makedirs(output_path, exist_ok=True)
    dset_path = os.path.join(output_path, f"{out_cfg['dataset_name']}.hdf5")
    
    print(f"Starting {gen_cfg['num_simulations']} simulations.")
    print(f"Data will be saved to: {dset_path}")
    
    with h5py.File(dset_path, 'w') as f:
        
        # --- FIX: Match original group name ---
        sims_group = f.create_group('sims')
        # --- END FIX ---
        
        for i in range(gen_cfg['num_simulations']):
            print(f"\n--- Running Simulation {i+1}/{gen_cfg['num_simulations']} ---")
            
            # --- 1. Get a new model with a new random inflow ---
            model = get_physical_model(config)
            
            # --- 2. Get initial state (with batch_size=1) ---
            # We add a time dimension
            state_list = [
                s.expand(batch(time=1))
                for s in model.get_initial_state(batch_size=1)
            ]
            
            # --- 3. Run the simulation loop ---
            for t in range(1, gen_cfg['total_steps'] + 1):
                current_state = [s.time[-1] for s in state_list]
                next_state = model.step(*current_state)
                
                if t % gen_cfg['save_interval'] == 0:
                    for j in range(len(state_list)):
                        new_slice = next_state[j].expand(batch(time=1))
                        state_list[j] = math.concat([state_list[j], new_slice], 'time')

            # --- 4. Prepare data for saving (This part was correct) ---
            vel, den = state_list[0], state_list[1]
            vel_x_cen = vel.vector['x'].at(den)
            vel_y_cen = vel.vector['y'].at(den)
            
            sim_data = math.stack([
                den.values,
                vel_x_cen.values,
                vel_y_cen.values,
                model.inflow.values
            ], dim=channel('channels'))
            
            sim_data_np = sim_data.numpy('time, channels, y, x')
            sims_group.create_dataset(f'sim{i}', data=sim_data_np)
        
        # --- 5. Save metadata ---
        print("\nAll simulations complete. Saving metadata...")
        
        # --- FIX: Match original metadata keys and values ---
        f.attrs['PDE'] = config.get('pde_name', 'Incompressible Navier-Stokes with Boussinesq')
        f.attrs['Fields Scheme'] = out_cfg['fields_scheme']
        f.attrs['Fields'] = out_cfg['fields_to_save']
        f.attrs['Constants'] = []  # Match original
        
        # Match original Dt logic (sim_dt * save_interval)
        saved_dt = config['dt'] * gen_cfg['save_interval']
        f.attrs['Dt'] = float(saved_dt)
        # --- END FIX ---

if __name__ == "__main__":
    # We point to our new config file
    CONFIG_FILE_PATH = os.path.join(PROJECT_ROOT, 'configs', 'smoke.yaml')
    run_generation(CONFIG_FILE_PATH)