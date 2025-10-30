# src/data_generation/generator_scene.py

import os
import yaml
from tqdm import tqdm
import random

# --- PhiFlow Imports ---
from phi.torch.flow import *

# --- Our Model Imports ---
import src.models.physical as physical_models


def get_physical_model(config: dict) -> physical_models.PhysicalModel:
    """
    Dynamically imports and instantiates a physical model from config.
    """
    phys_model_cfg = config['model']['physical']
    model_name = phys_model_cfg['name']
    
    domain_cfg = phys_model_cfg['domain']
    res_cfg = phys_model_cfg['resolution']
    
    domain = Box(x=domain_cfg['size_x'], y=domain_cfg['size_y'])
    resolution = spatial(x=res_cfg['x'], y=res_cfg['y'])
    
    pde_params = phys_model_cfg.get('pde_params', {}).copy()

    try:
        ModelClass = getattr(physical_models, model_name)
    except AttributeError:
        raise ImportError(f"Model '{model_name}' not found in src/models/physical/__init__.py")

    model = ModelClass(
        domain=domain,
        resolution=resolution,
        dt=phys_model_cfg['dt'],
        **pde_params
    )
    return model


def run_generation_scene(config: dict):
    """
    Main function to run data generation based on a config,
    saving the output to a phi.vis.Scene directory.
    """
    # Get parameters from the UNIFIED config schema
    gen_cfg = config['generation_params']
    data_cfg = config['data']
    model_cfg = config['model']['physical']
    project_root = config['project_root']

    # --- Setup Output Directory ---
    output_dir = os.path.join(project_root, data_cfg['data_dir'], data_cfg['dset_name'])
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting {gen_cfg['num_simulations']} simulations.")
    print(f"Scene data will be saved in: {output_dir}")
    
    # --- Main Simulation Loop ---
    for i in tqdm(range(gen_cfg['num_simulations']), desc="Total Simulations"):
        
            
        scene = Scene.create(output_dir, copy_calling_script=False)
        
        # --- 2. Save metadata ---
        saved_dt = model_cfg['dt'] * gen_cfg['save_interval']
        
        metadata = {
            'PDE': model_cfg['name'],
            'Fields': data_cfg['fields'],
            'Fields_Scheme': data_cfg.get('fields_scheme', 'unknown'),
            'Dt': float(saved_dt),
            'Domain': model_cfg['domain'],
            'Resolution': model_cfg['resolution'],
            'PDE_Params': model_cfg.get('pde_params', {}),
            'Generation_Params': gen_cfg,
        }
        
        scene.put_properties(metadata)

        # --- 3. Get the physical model ---
        model = get_physical_model(config)
        
        # --- 4. Get initial state (t=0) ---
        current_state_dict = model.get_initial_state()
        
        # --- 5. Write initial frame (frame 0) ---
        state_to_save = {}
        for name in data_cfg['fields']:
            if name in current_state_dict:
                # Remove batch dimension for saving
                state_to_save[name] = current_state_dict[name].batch[0]
            elif name == 'inflow' and hasattr(model, 'inflow'):
                state_to_save[name] = model.inflow
        
        scene.write(state_to_save, frame=0)
        
        # --- 6. Run simulation steps ---
        save_interval = gen_cfg['save_interval']
        
        for t in tqdm(range(1, gen_cfg['total_steps'] + 1), desc=f"Sim {i} Steps", leave=False):
            
            # Step the model forward
            current_state_dict = model.step(current_state_dict)
            
            # Save at intervals
            if t % save_interval == 0:
                frame_index = t // save_interval
                
                state_to_save = {}
                for name in data_cfg['fields']:
                    if name in current_state_dict:
                        state_to_save[name] = current_state_dict[name].batch[0]
                    elif name == 'inflow' and hasattr(model, 'inflow'):
                        state_to_save[name] = model.inflow
                
                scene.write(state_to_save, frame=frame_index)

    print(f"\nScene generation complete. Data saved in {output_dir}")