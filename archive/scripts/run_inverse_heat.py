# scripts/run_inverse_heat.py

import os
import sys
import argparse
from typing import Dict, List

# --- Add project root to path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# --- PhiFlow Imports ---
from phi.torch.flow import *

# --- Our imports ---
from src.visualization.plotter import load_config
import src.models.physical as physical_models


def load_scene_data(config: Dict, sim_index: int):
    """
    Loads ground truth data from a Scene.
    Returns: dict with field sequences and metadata
    """
    data_cfg = config['data']
    
    scene_parent_dir = os.path.join(
        PROJECT_ROOT, 
        data_cfg['data_dir'], 
        data_cfg['dset_name']
    )
    scene_path = os.path.join(scene_parent_dir, f"sim_{sim_index:06d}")
    
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene not found at {scene_path}")
    
    print(f"Loading ground truth from: {scene_path}")
    scene = Scene.at(scene_path)
    
    # Get available field names
    available_fields = scene.fieldnames
    print(f"Available fields in scene: {available_fields}")
    
    # Load all frames for each field
    frames = scene.frames
    print(f"Found {len(frames)} frames")
    
    data = {}
    # Load whatever fields are available
    for field_name in available_fields:
        field_frames = []
        for frame in frames:
            field_data = scene.read_field(field_name, frame=frame, convert_to_backend=True)
            field_frames.append(field_data)
        data[field_name] = stack(field_frames, batch('time'))
        print(f"  Loaded {field_name}: {data[field_name].shape}")
    
    return data, scene.properties


def get_physical_model(config: dict):
    """
    Creates the physical model from config (same as generator_scene.py)
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


def run_inverse_problem(config: Dict):
    """
    Solves inverse problem: estimate parameters from observations.
    Uses trainer_params.learnable_parameters from config.
    """
    print("="*60)
    print("INVERSE PROBLEM: Heat Equation Parameter Estimation")
    print("="*60)
    
    trainer_cfg = config['trainer_params']
    model_cfg = config['model']['physical']
    gen_cfg = config['generation_params']
    
    # 1. --- Load ground truth data ---
    train_sims = trainer_cfg['train_sim']
    print(f"\nLoading {len(train_sims)} training simulations: {train_sims}")
    
    all_gt_data = []
    for sim_idx in train_sims:
        gt_data, metadata = load_scene_data(config, sim_idx)
        all_gt_data.append(gt_data)
    
    gt_data = all_gt_data[0]
    
    # Find the temperature field
    temp_field_name = None
    for possible_name in ['temp', 'temperature', 'T']:
        if possible_name in gt_data:
            temp_field_name = possible_name
            break
    
    if temp_field_name is None:
        temp_field_name = list(gt_data.keys())[0]
        print(f"Warning: Could not find 'temp' field. Using '{temp_field_name}' instead.")
    
    temperature_gt = gt_data[temp_field_name]
    num_frames = temperature_gt.shape.get_size('time')
    
    print(f"\nGround truth {temp_field_name} shape: {temperature_gt.shape}")
    print(f"Time steps: {num_frames}")
    
    # 2. --- Setup learnable parameters ---
    learnable_params = trainer_cfg['learnable_parameters']
    print(f"\nLearnable parameters:")
    for param in learnable_params:
        print(f"  - {param['name']}: initial_guess={param['initial_guess']}")
    
    true_values = metadata['PDE_Params']
    print(f"\nTrue parameter values: {true_values}")
    
    # 3. --- Initialize model ---
    model = get_physical_model(config)
    
    # 4. --- Setup training parameters ---
    num_epochs = trainer_cfg['epochs']
    num_predict_steps = trainer_cfg.get('num_predict_steps', num_frames - 1)
    
    print(f"\nTraining setup:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Prediction steps: {num_predict_steps}")
    
    # 5. --- Create initial guess tensor ---
    if len(learnable_params) == 1:
        # Single parameter - pass as scalar
        initial_guess = math.tensor(learnable_params[0]['initial_guess'])
        param_name = learnable_params[0]['name']
        
        print(f"\nInitial guess for {param_name}: {initial_guess}")
        
        # 6. --- Define loss function for single parameter ---
        def loss_function(diffusivity):
            """Loss function taking diffusivity directly as argument."""
            # Update model parameter
            setattr(model, param_name, diffusivity)
            
            # Get initial state (remove time dimension)
            initial_temp = temperature_gt.time[0]
            
            # Simulate forward and accumulate loss
            total_loss = math.tensor(0.0)
            current_state = {temp_field_name: initial_temp}
            
            for step in range(num_predict_steps):
                current_state = model.step(current_state)
                target = temperature_gt.time[step + 1]
                loss = math.l2_loss(current_state[temp_field_name] - target)
                total_loss += loss
            
            # Return average loss (scalar)
            return total_loss / num_predict_steps
        
    else:
        # Multiple parameters - would need tuple/list of initial guesses
        raise NotImplementedError("Multi-parameter optimization needs tuple of initial guesses")
    
    # 7. --- Run optimization ---
    print("\nStarting optimization...")
    
    # Initial loss
    initial_loss = loss_function(initial_guess)
    print(f"Initial loss: {initial_loss.numpy()}")
    
    # Create solve object with initial guess
    solve_params = math.Solve(
        method='L-BFGS-B',
        abs_tol=1e-6,
        x0=initial_guess,
        max_iterations=num_epochs
    )
    
    # Run minimization
    try:
        estimated_value = math.minimize(loss_function, solve_params)
        print(f"\nOptimization completed!")
    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        estimated_value = initial_guess
    
    # Final loss
    final_loss = loss_function(estimated_value)
    print(f"Final loss: {final_loss.numpy()}")
    
    # 8. --- Report results ---
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    estimated = estimated_value.numpy() if isinstance(estimated_value, math.Tensor) else estimated_value
    true_val = true_values[param_name]
    error = abs(estimated - true_val)
    
    print(f"\n{param_name}:")
    print(f"  True value:      {true_val}")
    print(f"  Estimated value: {estimated}")
    print(f"  Absolute error:  {error}")
    print(f"  Relative error:  {error/true_val*100:.2f}%")
    
    # 9. --- Save and visualize results ---
    save_dir = os.path.join(PROJECT_ROOT, "results", "inverse_heat", config['run_params']['experiment_name'])
    os.makedirs(save_dir, exist_ok=True)
    
    # Update model with optimized parameter
    setattr(model, param_name, estimated)
    
    # Simulate with estimated parameters
    print("\nSimulating with estimated parameters...")
    trajectory = [temperature_gt.time[0]]
    current_state = {temp_field_name: temperature_gt.time[0]}
    
    for _ in range(num_frames - 1):
        current_state = model.step(current_state)
        trajectory.append(current_state[temp_field_name])
    
    estimated_temperature = stack(trajectory, batch('time'))
    
    # Plot comparison
    print("Plotting results...")
    
    # Ground truth
    vis.plot(temperature_gt, title=f"Ground Truth", animate='time', show_color_bar=True)
    vis.savefig(os.path.join(save_dir, f"gt.gif"))
    
    # Estimated
    vis.plot(estimated_temperature, title=f"Estimated", animate='time', show_color_bar=True)
    vis.savefig(os.path.join(save_dir, f"estimated.gif"))
    
    # Error
    error_field = abs(estimated_temperature - temperature_gt)
    vis.plot(error_field, title=f"Absolute Error", play=False, show_color_bar=True)
    vis.savefig(os.path.join(save_dir, f"error.gif"))
    
    print(f"\nResults saved to {save_dir}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve inverse problem for heat equation")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_inverse_problem(config)