# src/data_generation/generator.py

import os
import yaml
import h5py
import numpy as np
import random
from tqdm import tqdm  # <-- 1. ADD THIS IMPORT

# --- PhiFlow Imports ---
from phi.torch.flow import (
    Box,
    CenteredGrid,
    extrapolation,
    math,
    batch,
    spatial,
    channel,
    StaggeredGrid # Import StaggeredGrid
)

# --- Our Model Imports ---
import src.models.physical as physical_models


def load_config(config_path: str) -> dict:
    """Loads a YAML config file.
    (This function is unchanged)
    """
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_physical_model(config: dict) -> physical_models.PhysicalModel:
    """
    Dynamically imports and instantiates a physical model from config.
    (This function is unchanged)
    """
    model_name = config['model_name']
    
    # 1. Build Domain and Resolution
    domain_cfg = config['domain']
    res_cfg = config['resolution']
    
    # --- MODIFICATION: Hardcode 2D resolution ---
    # This assumes 'x' and 'y' are always in the config
    domain = Box(x=domain_cfg['size_x'], y=domain_cfg['size_y'])
    resolution = spatial(x=res_cfg['x'], y=res_cfg['y'])
    # --- End Modification ---
    
    # 2. Get PDE params
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
        
        # --- 2. WRAP THE OUTER LOOP ---
        # This bar will show "Total Simulations [2/10]"
        for i in tqdm(range(gen_cfg['num_simulations']), desc="Total Simulations"):
            
            # (We keep this print, but remove the \n)
            print(f"--- Running Simulation {i+1}/{gen_cfg['num_simulations']} ---")
            
            # --- 1. Get a new model with a new random inflow ---
            model = get_physical_model(config)
            
            # --- 2. Get initial state (with batch_size=1) ---
            initial_state_dict = model.get_initial_state()
            state_lists = {
                name: math.expand(field, batch(time=1))
                for name, field in initial_state_dict.items()
            }
            
            # --- 3. WRAP THE INNER LOOP ---
            # This bar will show "Sim 1 Steps [|||| ] 50/100"
            # leave=False makes the inner bar disappear when it's done.
            for t in tqdm(range(1, gen_cfg['total_steps'] + 1), desc=f"Sim {i+1} Steps", leave=False):
                current_state_dict = {
                    name: field_list.time[-1] 
                    for name, field_list in state_lists.items()
                }
                
                next_state_dict = model.step(**current_state_dict)
                
                if t % gen_cfg['save_interval'] == 0:
                    for name, new_field in next_state_dict.items():
                        new_slice = math.expand(new_field, batch(time=1))
                        state_lists[name] = math.concat([state_lists[name], new_slice], 'time')

            # --- 4. Prepare data for saving (GENERALIZED) ---
            
            fields_to_save = out_cfg['fields_to_save']
            data_channels = [] 
            
            # (No need to print here, the progress bar shows it)
            # print(f"Preparing data for saving. Requested fields: {fields_to_save}")

            # --- MODIFICATION: Revert reference_grid logic ---
            # This logic is robust for both Smoke and 2D Burgers
            if 'density' in state_lists:
                reference_grid = state_lists['density'] # Smoke case
            elif 'velocity' in state_lists:
                # Burgers case: Create a CenteredGrid from the velocity
                vel_grid = state_lists['velocity']
                print("No 'density' field found. Creating reference grid from 'velocity' field.")
                centered_values = math.zeros(vel_grid.shape.without('vector'))
                reference_grid = CenteredGrid(
                    values=centered_values,
                    extrapolation=vel_grid.extrapolation,
                    bounds=vel_grid.bounds
                )
            else:
                reference_grid = next(iter(state_lists.values()))
            # --- End Modification ---

            # Loop over the fields requested in the config
            for field_name in fields_to_save:
                field_name_lower = field_name.lower() 

                if field_name_lower == 'density':
                    if 'density' not in state_lists:
                        raise ValueError("Config requests 'density' but it's not in the model state")
                    data_channels.append(state_lists['density'].values)
                
                # --- MODIFICATION: Hardcode 2D 'velocity' logic ---
                elif field_name_lower == 'velocity':
                    if 'velocity' not in state_lists:
                        raise ValueError("Config requests 'velocity' but it's not in the model state")
                    
                    vel = state_lists['velocity']
                    
                    # Assuming 2D, so we always add both x and y
                    # print("Adding 2D velocity (x, y) channels") # (Can be noisy with tqdm)
                    vel_x_cen = vel.vector['x'].at(reference_grid)
                    vel_y_cen = vel.vector['y'].at(reference_grid)
                    data_channels.append(vel_x_cen.values)
                    data_channels.append(vel_y_cen.values)
                # --- End Modification ---

                elif field_name_lower == 'velocity_x':
                    if 'velocity' not in state_lists:
                        raise ValueError("Config requests 'velocity_x' but model state does not have 'velocity'")
                    vel_x_cen = state_lists['velocity'].vector['x'].at(reference_grid)
                    data_channels.append(vel_x_cen.values)
                
                elif field_name_lower == 'velocity_y':
                    if 'velocity' not in state_lists:
                        raise ValueError("Config requests 'velocity_y' but model state does not have 'velocity'")
                    vel_y_cen = state_lists['velocity'].vector['y'].at(reference_grid)
                    data_channels.append(vel_y_cen.values)

                elif field_name_lower == 'inflow':
                    if not hasattr(model, 'inflow'):
                         raise ValueError("Config requests 'inflow' but model does not have 'inflow' attribute")
                    
                    inflow_values_tensor = model.inflow.values
                    time_size = reference_grid.shape.get_size('time')
                    inflow_expanded = math.expand(inflow_values_tensor, batch(time=time_size))
                    data_channels.append(inflow_expanded)
                    
                elif field_name_lower in state_lists:
                    # print(f"Adding field '{field_name_lower}' directly from state")
                    data_channels.append(state_lists[field_name_lower].values)
                    
                else:
                    raise ValueError(f"Unknown field '{field_name}' in config's 'fields_to_save'.")
            
            sim_data = math.stack(data_channels, dim=channel('channels'))
            
            # print(f"Simulation {i} data shape (with batch=1): {sim_data.shape}")
            
            # --- MODIFICATION: Hardcode 2D numpy ordering ---
            ordered_dims = ('time', 'channels', 'y', 'x')
            # --- End Modification ---

            sim_data_np = sim_data.batch[0].numpy(ordered_dims)
            sims_group.create_dataset(f'sim{i}', data=sim_data_np)
        
        # --- 5. Save metadata ---
        print("\nAll simulations complete. Saving metadata...")
        
        sims_group.attrs['PDE'] = config.get('pde_name', 'Unknown PDE')
        sims_group.attrs['Fields Scheme'] = out_cfg['fields_scheme']
        sims_group.attrs['Fields'] = out_cfg['fields_to_save'] 
        sims_group.attrs['Constants'] = [] 
        
        saved_dt = config['dt'] * gen_cfg['save_interval']
        sims_group.attrs['Dt'] = float(saved_dt)