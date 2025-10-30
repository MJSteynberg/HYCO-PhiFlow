# src/training/synthetic/trainer_scene.py

import os
import time
from typing import Dict, Any, List
from tqdm import trange # Use trange for the epoch loop

# --- PhiFlow Imports ---
from phi.torch.flow import *
from phi.field import l2_loss
from phi.math import math, Tensor
import phiml.nn as nn # Import phiflow's nn module

# --- Repo Imports ---
# Import the synthetic models module
import src.models.synthetic as synthetic_models 


class SyntheticTrainer:
    """
    Trains a SyntheticModel (neural network) using data
    from a PhiFlow Scene.

    This trainer mirrors the structure of PhysicalTrainerScene but is
    adapted for training neural network weights via backpropagation
    using an Adam optimizer (phi.nn.adam).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the trainer from a unified configuration dictionary.

        Args:
            config (Dict[str, Any]): The experiment configuration.
        """
        self.config = config
        self.project_root = config.get('project_root', '.') # Get project root

        # --- Parse Configs ---
        self.data_config = config['data']
        # --- MODIFIED: Point to 'synthetic' model config ---
        self.model_config = config['model']['synthetic']
        self.trainer_config = config['trainer_params']
        
        # --- Get parameters ---
        self.train_sims: List[int] = self.trainer_config['train_sim']
        self.num_epochs: int = self.trainer_config['epochs']
        self.num_predict_steps: int = self.trainer_config['num_predict_steps']
        self.dt: float = 1.0 # Will be overwritten when data is loaded
        
        # --- Setup Model ---
        # The model is instantiated and holds its own learnable weights
        self.model = self._create_model()
        
        # --- MODIFIED: Setup Optimizer ---
        # Instead of 'initial_guesses', we create an Adam optimizer
        # that targets the parameters of self.model
        lr = self.trainer_config.get('learning_rate', 1e-3)
        self.optimizer = nn.adam(self.model, learning_rate=lr)
        
        # --- MODIFIED: Get Ground Truth field names ---
        # We need to load all fields that are either an INPUT
        # or an OUTPUT of the neural network.
        self.gt_fields: List[str] = list(
            set(self.model.INPUT_FIELDS + self.model.OUTPUT_FIELDS)
        )
        
        print(f"SyntheticTrainerScene initialized for model '{self.model_config['name']}'.")
        print(f"  Optimizing with {self.optimizer.__class__.__name__} (lr={lr}).")
        print(f"  Model Inputs:  {self.model.INPUT_FIELDS}")
        print(f"  Model Outputs: {self.model.OUTPUT_FIELDS}")
        print(f"  Data Fields to Load: {self.gt_fields}")


    def _create_model(self) -> synthetic_models.SyntheticModel:
        """
        Instantiates the synthetic model from the config.
        """
        model_name = self.model_config['name']
        
        try:
            # Get the class from the synthetic_models module
            ModelClass = getattr(synthetic_models, model_name)
        except AttributeError:
            raise ImportError(f"Model '{model_name}' not found in src/models/synthetic/__init__.py")

        # The SyntheticModel base class and its children (like UNet)
        # are designed to be initialized from their specific config dict.
        model = ModelClass(
            config=self.model_config 
        )
        return model
    
    def _load_ground_truth_data(self, sim_index: int) -> Dict[str, Tensor]:
        """
        Loads all ground truth data for one simulation from its Scene.
        Adds a batch dimension to match the model's expected format.
        
        (This is almost identical to the PhysicalTrainerScene method)
        """
        scene_parent_dir = os.path.join(
            self.project_root,
            self.data_config['data_dir'],
            self.data_config['dset_name']
        )
        scene_name = f"sim_{sim_index:06d}"
        scene_path = os.path.join(scene_parent_dir, scene_name)
        
        if not os.path.exists(scene_path):
            # Try to find the scene if the name is not zero-padded
            scene_path_alt = os.path.join(scene_parent_dir, f"sim_{sim_index}")
            if os.path.exists(scene_path_alt):
                scene_path = scene_path_alt
            else:
                raise FileNotFoundError(
                    f"Scene not found at {scene_path} or {scene_path_alt}"
                )
        
        print(f"Loading ground truth from: {scene_path}")
        scene = Scene.at(scene_path)
        
        # --- NEW: Read dt from scene metadata ---
        self.dt = scene.properties.get('Dt', 1.0)
        print(f"  Using dt = {self.dt} from Scene metadata.")
        
        frames = scene.frames
        # Limit frames based on num_predict_steps + 1 (for initial state)
        frames_to_load = frames[:self.num_predict_steps + 1]
        if len(frames_to_load) < self.num_predict_steps + 1:
            print(f"Warning: Scene only has {len(frames_to_load)} frames, "
                  f"but config requested {self.num_predict_steps + 1}.")
            # Update num_predict_steps to match available data
            self.num_predict_steps = len(frames_to_load) - 1
        
        print(f"Loading {len(frames_to_load)} frames (T=0 to T={self.num_predict_steps}).")
        
        data = {}
        # Load only the fields specified in self.gt_fields
        for field_name in self.gt_fields:
            if field_name not in scene.fieldnames:
                print(f"Warning: Field '{field_name}' not in Scene. Skipping.")
                continue
                
            field_frames = []
            for frame in frames_to_load:
                field_data = scene.read_field(
                    field_name, 
                    frame=frame, 
                    convert_to_backend=True
                )
                field_frames.append(field_data)
            
            # Stack along the 'time' dimension
            stacked_field = stack(field_frames, batch('time'))
            
            # --- ADD BATCH DIMENSION ---
            # Add batch dimension for consistency with model output
            stacked_field = math.expand(stacked_field, batch(batch=1))
            
            data[field_name] = stacked_field
            print(f"  Loaded '{field_name}' with shape {data[field_name].shape}")

        return data
    

    def train(self):
        """
        Runs the full training loop using Adam.
        """
        print(f"\n--- Starting Synthetic Model Training ---")
        
        # 1. --- Load Ground Truth Data ---
        sim_to_train = self.train_sims[0]
        if len(self.train_sims) > 1:
            print(f"Warning: Training on sim {sim_to_train}. "
                  f"Multi-sim training not yet implemented.")
                  
        gt_data_dict = self._load_ground_truth_data(sim_to_train)
        
        # Get the initial state (t=0) for all fields the model *needs as input*
        initial_state_dict = {
            name: field.time[0] 
            for name, field in gt_data_dict.items()
            if name in self.model.INPUT_FIELDS
        }

        # Get the ground truth *rollout* (t=1 to T) for all fields
        # the model *produces as output* (for loss calculation)
        gt_rollout_dict = {
            name: field.time[1:] 
            for name, field in gt_data_dict.items()
            if name in self.model.OUTPUT_FIELDS
        }
        
        # 2. --- Define the Loss Function ---
        # This function takes the data as input, which is the
        # standard for nn.update_weights.
        
        # @jit_compile # Can be added for performance
        def loss_function(initial_state_data, gt_rollout_data):
            """
            Calculates the L2 loss for a full autoregressive rollout.
            """
            total_loss = math.tensor(0.0)
            
            # Start with the initial state (t=0)
            # We must copy this dict!
            current_state = initial_state_data.copy()
            
            for step in range(self.num_predict_steps):
                # 1. Simulate one step forward using the neural net
                # The model's 'forward' method is its 'step'
                predicted_state = self.model(current_state)
                
                # 2. Calculate L2 loss for this step
                step_loss = 0.0
                for field_name, gt_rollout in gt_rollout_data.items():
                    # Get the ground truth for this time step (t=step+1)
                    target = gt_rollout.time[step] # Has batchᵇ=1
                    # Get the prediction for this field
                    pred = predicted_state[field_name] # Has batchᵇ=1
                    
                    # Compute L2 loss
                    field_loss = l2_loss(pred - target)
                    
                    # Sum over all dimensions (including batch)
                    step_loss += math.sum(field_loss)
                
                total_loss += step_loss
                
                # 3. Prepare state for the next step (autoregression)
                # Update the 'current_state' with the model's predictions.
                # Any fields that are inputs but not outputs (e.g., 'inflow')
                # will be preserved from the previous step.
                current_state.update(predicted_state)
            
            return total_loss / self.num_predict_steps

        # 3. --- Run Optimization ---
        print(f"\nStarting optimization with {self.optimizer.__class__.__name__}...")
        
        # Calculate initial loss
        initial_loss = loss_function(initial_state_dict, gt_rollout_dict)
        print(f"Initial loss: {initial_loss}")

        loss_history = []
        
        # Use trange for a nice progress bar
        for epoch in trange(self.num_epochs, desc="Training Epochs"):
            # nn.update_weights handles the forward pass, backward pass,
            # and optimizer step all in one.
            loss = nn.update_weights(
                self.model, 
                self.optimizer, 
                loss_function, 
                initial_state_dict, # Passed as first arg to loss_function
                gt_rollout_dict  # Passed as second arg to loss_function
            )
            loss_history.append(float(loss))
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}: Loss = {loss}")

        
        # 4. --- Report Results ---
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        
        final_loss = loss_history[-1] if loss_history else initial_loss
        
        print(f"  Model:       {self.model_config['name']}")
        print(f"  Epochs:      {self.num_epochs}")
        print(f"  Initial loss: {initial_loss}")
        print(f"  Final loss:   {final_loss}")
        print("="*60)