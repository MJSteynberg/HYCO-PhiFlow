from phi.flow import *
from typing import Dict
from src.factories.model_factory import ModelFactory
import os


class DataGenerator:
    " Class that will generate data and save them as phiml tensors."
    def __init__(self, config: Dict):
        self.model = ModelFactory.create_physical_model(config)
        self.data_dir = config["data"]["data_dir"]
        self.dset_name = config["data"]["dset_name"]
        self.num_simulations = config["data"]["num_simulations"]
        self.trajectory_length = config["data"]["trajectory_length"]

        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        

    def generate_data(self):
        # Generate data in batched and save each batch separately

        initial_state = self.model.get_initial_state(self.num_simulations)

        # PhiFlow's iterate() INCLUDES the initial state in its output!
        # So iterate(f, time=n, x0) returns (n+1) timesteps: [x0, f(x0), ..., f^n(x0)]
        # To get exactly trajectory_length timesteps, we call rollout with (trajectory_length - 1)
        # This gives us: initial_state + (trajectory_length - 1) evolved states = trajectory_length total
        data = self.model.rollout(initial_state, self.trajectory_length - 1)

        # data already contains the full trajectory including initial state
        # No need to prepend anything!
        data_unstacked = unstack(data, 'batch')
        for i, sim_data in enumerate(data_unstacked):
            sim_path = os.path.join(self.data_dir, f"sim_{i:04d}.npz")
            # Extract tensor values from each Field in the state dict
            cache_data = {field_name: field.values for field_name, field in sim_data.items()}
            math.save(sim_path, cache_data)