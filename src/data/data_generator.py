from phi.flow import *
from typing import Dict
from src.factories.model_factory import ModelFactory
import os


class DataGenerator:
    """Generates physics simulation data and saves as unified PhiML tensors."""

    def __init__(self, config: Dict):
        self.model = ModelFactory.create_physical_model(config)
        self.data_dir = config["data"]["data_dir"]
        self.dset_name = config["data"]["dset_name"]
        self.num_simulations = config["data"]["num_simulations"]
        self.trajectory_length = config["data"]["trajectory_length"]
        os.makedirs(self.data_dir, exist_ok=True)

    def generate_data(self):
        """Generate simulation data and save as unified tensors."""
        initial_state = self.model.get_initial_state(self.num_simulations)

        # PhiFlow's iterate() includes initial state, so use (trajectory_length - 1)
        data = self.model.rollout(initial_state, self.trajectory_length - 1)

        # Unstack batches and save each simulation
        for i, sim_data in enumerate(unstack(data, 'batch')):
            sim_path = os.path.join(self.data_dir, f"sim_{i:04d}.npz")
            math.save(sim_path, sim_data)
