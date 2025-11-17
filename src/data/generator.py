# src/data_generation/generator_scene.py

import os
import yaml
from tqdm import tqdm
import random

# --- PhiFlow Imports ---
from phi.torch.flow import *

# --- Our Factory Imports ---
from src.factories.model_factory import ModelFactory
from src.models.physical.base import PhysicalModel
from matplotlib import pyplot as plt


def run_generation(config: dict):
    """
    Main function to run data generation based on a config,
    saving the output to a phi.vis.Scene directory.
    """
    # New config
    data_cfg = config["data"]
    model_cfg = config["model"]
    project_root = config["project_root"]

    # --- Setup Output Directory ---
    output_dir = os.path.join(project_root, data_cfg["data_dir"])
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting {data_cfg['num_simulations']} simulations.")
    print(f"Scene data will be saved in: {output_dir}")

    # --- Main Simulation Loop ---
    for i in tqdm(range(data_cfg["num_simulations"]), desc="Total Simulations"):

        scene = Scene.create(output_dir, copy_calling_script=False)

        # --- 2. Save metadata ---
        saved_dt = model_cfg["physical"]["dt"]

        metadata = {
            "PDE": model_cfg["physical"]["name"],
            "Fields": data_cfg["fields"],
            "Fields_Scheme": data_cfg["fields_scheme"],
            "Dt": float(saved_dt),
            "Domain": model_cfg["physical"]["domain"],
            "Resolution": model_cfg["physical"]["resolution"],
            "PDE_Params": model_cfg["physical"]["pde_params"],
        }

        scene.put_properties(metadata)

        # --- 3. Get the physical model ---
        model = ModelFactory.create_physical_model(config)

        # --- 4. Get initial state (t=0) ---
        current_state_dict = model.get_initial_state()

        # --- 5. Write initial frame (frame 0) ---
        state_to_save = {}
        for name in data_cfg["fields"]:
            # Remove batch dimension for saving
            state_to_save[name] = current_state_dict[name].batch[0]

        scene.write(state_to_save, frame=0)

        for t in range(1, data_cfg["trajectory_length"] + 1):

            # Step the model forward
            current_state_dict = model.forward(current_state_dict)

            # Save at intervals
            frame_index = t

            state_to_save = {}
            for name in data_cfg["fields"]:
                state_to_save[name] = current_state_dict[name].batch[0]

            scene.write(state_to_save, frame=frame_index)

    print(f"\nScene generation complete. Data saved in {output_dir}")
