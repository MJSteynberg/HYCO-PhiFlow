# src/data_generation/generator.py

import os
import yaml
import h5py
import numpy as np  # <-- Make sure numpy is imported
import random
from tqdm import tqdm

# --- PhiFlow Imports ---
from phi.torch.flow import (
    Box,
    CenteredGrid,
    extrapolation,
    math,
    batch,
    spatial,
    channel,
    StaggeredGrid
)

# --- Our Model Imports ---
import src.models.physical as physical_models


# (get_physical_model function remains the same)
def get_physical_model(config: dict) -> physical_models.PhysicalModel:
    phys_model_cfg = config['model']['physical']
    model_name = phys_model_cfg['name'] 
    
    try:
        ModelClass = getattr(physical_models, model_name)
    except AttributeError:
        raise ImportError(f"Model '{model_name}' not found in src/models/physical/__init__.py")
    
    # Pass the config dict directly - the base class handles parsing
    model = ModelClass(phys_model_cfg)
    return model


def run_generation(config: dict):
    """
    Main function to run data generation based on a config.
    """
    gen_cfg = config['generation_params']
    data_cfg = config['data']
    project_root = config['project_root']

    output_path = os.path.join(project_root, data_cfg['data_dir'])
    os.makedirs(output_path, exist_ok=True)
    dset_path = os.path.join(output_path, f"{data_cfg['dset_name']}.hdf5")
    
    print(f"Starting {gen_cfg['num_simulations']} simulations.")
    print(f"Data will be saved to: {dset_path}")
    
    with h5py.File(dset_path, 'w') as f:
        
        sims_group = f.create_group('sims')
    
        for i in tqdm(range(gen_cfg['num_simulations']), desc="Total Simulations"):
            
            model = get_physical_model(config)
            # --- Get model resolution explicitly ---
            model_resolution = model.resolution
            res_x_int = model_resolution.get_size('x')
            res_y_int = model_resolution.get_size('y')
            
            # --- 2. Get initial state (with batch_size=1) ---
            initial_state_dict = model.get_initial_state()
            state_lists = {
                name: math.expand(field, batch(time=1))
                for name, field in initial_state_dict.items()
            }
            
            # --- 3. Run Simulation ---
            for t in tqdm(range(1, gen_cfg['total_steps'] + 1), desc=f"Sim {i+1} Steps", leave=False):
                current_state_dict = {
                    name: field_list.time[-1] 
                    for name, field_list in state_lists.items()
                }
                next_state_dict = model.step(current_state_dict)
                if t % gen_cfg['save_interval'] == 0:
                    for name, new_field in next_state_dict.items():
                        new_slice = math.expand(new_field, batch(time=1))
                        state_lists[name] = math.concat([state_lists[name], new_slice], 'time')

            # --- 4. Prepare data for saving (GENERALIZED) ---
            
            fields_to_save = data_cfg['fields']
            
            # --- MODIFICATION: Store numpy arrays ---
            # We will store numpy arrays here, not phiml tensors
            data_channels_np = [] 
            # ---
            
            # Get a reference CenteredGrid (for resolution)
            reference_grid = next(iter(state_lists.values())).at_centers()
            
            # Define the final numpy dimension order (T, C, Y, X)
            # We'll order phiml tensors to (T, Y, X) before converting
            ordered_dims_spatial = ('time', 'y', 'x')

            for field_name in fields_to_save:
                field_name_lower = field_name.lower() 

                if field_name_lower == 'density':
                    if 'density' not in state_lists:
                        raise ValueError("Config requests 'density' but it's not in the model state")
                    
                    density_values = state_lists['density'].values.batch[0] # (T, Y, X)
                    print('density_values: ', density_values)
                    data_channels_np.append(density_values.numpy(ordered_dims_spatial))

                elif field_name_lower == 'velocity':
                    if 'velocity' not in state_lists:
                        raise ValueError("Config requests 'velocity' but it's not in the model state")
                    
                    vel = state_lists['velocity'] # (B, T, ...)
                    vx = math.pad(vel.vector['x'].values, {'x':(0,1)}, extrapolation.ZERO)
                    # DEPENDS ON THE EXTRAPOLATION USED IN THE MODEL
                    data_channels_np.append(vx.batch[0].numpy(ordered_dims_spatial))
                    vy = math.pad(vel.vector['y'].values, {'y':(0,1)}, extrapolation.ZERO)
                    data_channels_np.append(vy.batch[0].numpy(ordered_dims_spatial))
                    print(data_channels_np[-2].shape, data_channels_np[-1].shape)
                elif field_name_lower == 'inflow':
                    if not hasattr(model, 'inflow'):
                         raise ValueError("Config requests 'inflow' but model does not have 'inflow' attribute")
                    inflow_values_tensor = model.inflow.values
                    time_size = reference_grid.shape.get_size('time')
                    inflow_expanded = math.expand(inflow_values_tensor, batch(time=time_size))
                    
                    # .batch[0] is not needed if inflow has no batch dim
                    data_channels_np.append(inflow_expanded.numpy(ordered_dims_spatial)) 
                    
                elif field_name_lower in state_lists:
                    
                    field_values = state_lists[field_name_lower].values.batch[0] # (T, Y, X)
                    data_channels_np.append(field_values.numpy(ordered_dims_spatial))
                    
                else:
                    raise ValueError(f"Unknown field '{field_name}' in config's 'fields_to_save'.")
            
            # --- MODIFICATION: Stack in Numpy, not Phiml ---
            # stack on axis 1 to create (T, C, Y, X)
            print([x.shape for x in data_channels_np])
            sim_data_np = np.stack(data_channels_np, axis=1) 
            # ---
            
            sims_group.create_dataset(f'sim{i}', data=sim_data_np)
        
        # --- 5. Save metadata ---
        print("\nAll simulations complete. Saving metadata...")
        
        sims_group.attrs['PDE'] = config['model']['physical']['name']
        sims_group.attrs['Fields'] = data_cfg['fields'] 
        sims_group.attrs['Fields Scheme'] = data_cfg['fields_scheme']
        sims_group.attrs['Constants'] = [] 
        
        saved_dt = config['model']['physical']['dt'] * gen_cfg['save_interval']
        sims_group.attrs['Dt'] = float(saved_dt)