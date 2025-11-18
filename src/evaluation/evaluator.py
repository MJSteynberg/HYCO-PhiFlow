"""
Simple Evaluator for comparing real vs generated trajectories.

Uses PhiFlow's built-in visualization tools for easy plotting.
"""

from pathlib import Path
from typing import Dict, Any, List
import torch
from phi.vis import plot, show, close, smooth
from phi.math import nan_to_0

from src.models.synthetic.base import SyntheticModel
from src.data import DataManager
from src.utils.logger import get_logger
import matplotlib.pyplot as plt

logger = get_logger(__name__)


class Evaluator:
    """
    Simple evaluator that generates predictions and creates comparison animations.
    
    Workflow:
    1. Load model checkpoint
    2. Load test simulations
    3. Generate predictions (autoregressive rollout)
    4. Convert tensors back to Fields
    5. Create comparison animations using PhiFlow's plot()
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluator from config."""
        self.config = config
        self.eval_config = config['evaluation']
        
        # Setup paths
        self.output_dir = Path(self.eval_config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup data manager
        self.data_manager = DataManager(config)
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Model will be loaded in evaluate()
        self.model = None
        
    def evaluate(self):
        """
        Run evaluation on test simulations.
        
        Creates comparison animations for each simulation showing:
        - Real trajectory (from data)
        - Generated trajectory (from model)
        Side by side for each field.
        """
        logger.info("Starting evaluation...")
        
        # Load model
        self._load_model()
        
        # Evaluate each test simulation
        test_sims = self.eval_config['test_sim']
        for sim_idx in test_sims:
            logger.info(f"Evaluating simulation {sim_idx}...")
            self._evaluate_simulation(sim_idx)
            
        logger.info(f"Evaluation complete! Results saved to {self.output_dir}")
    
    def _load_model(self):
        """Load model from checkpoint."""
        from src.factories.model_factory import ModelFactory
        
        # Create model
        self.model = ModelFactory.create_synthetic_model(self.config)
        
        # Load checkpoint
        checkpoint_path = Path(self.eval_config['checkpoint_path'])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded model from {checkpoint_path}")
    
    def _evaluate_simulation(self, sim_idx: int):
        """Evaluate a single simulation."""
        # Load real data
        real_data = self._load_real_trajectory(sim_idx)
        
        # Generate predictions
        generated_data = self.generate_predictions([real_data])
        
        # Create visualizations
        self._create_comparison_animation(sim_idx, real_data, generated_data)
    
    def _load_real_trajectory(self, sim_idx: int) -> Dict[str, torch.Tensor]:
        """
        Load real trajectory from data.
        
        Returns:
            Dict mapping field names to tensors [T, C, H, W]
        """
        num_frames = self.eval_config['num_frames']
        field_names = self.config['data']['fields']
        
        # Load simulation
        sim_data = self.data_manager.load_simulation(
            sim_idx,
            field_names=field_names,
            num_frames=num_frames
        )
        
        return sim_data['tensor_data']
    
    @torch.no_grad()
    def generate_predictions(
        self,
        trajectories: List[Dict[str, torch.Tensor]],
    ):
        """
        Generate one-step predictions from physical model trajectories.
        
        NEW BEHAVIOR:
        - Takes physical trajectories directly (not through dataset)
        - For each trajectory, generates one-step predictions autoregressively
        - Returns as trajectory format: [initial_real, pred_from_real, pred_from_pred, ...]
        - This allows the same windowing logic via indexing
        
        Args:
            trajectories: List of trajectory dicts in cache format
                        Format: [{'tensor_data': {field_name: tensor[C, T, H, W]}}]
            device: Device to run predictions on
            batch_size: Batch size for inference (currently unused, could batch multiple trajectories)
            
        Returns:
            List of prediction trajectories in BVTS format [1, V, T, H, W]
        """
        self.model.eval()
        self.model.to(self.device)
        
        prediction_trajectories = []
        
        for tensor_data in trajectories:
            # Concatenate all fields along channel dimension
            # Each field: [C, T, H, W]
            field_tensors = [tensor_data[field_name] for field_name in sorted(tensor_data.keys())]
            full_trajectory = torch.cat(field_tensors, dim=0)  # [C_all, T, H, W]
            
            # Move to device
            full_trajectory = full_trajectory.to(self.device, non_blocking=True)
            
            # Get trajectory length
            num_steps = full_trajectory.shape[1] 
            trajectory_frames = [full_trajectory[:, 0:1, :, :]] 
            current_state = full_trajectory[:, 0:1, :, :]
            with torch.amp.autocast(enabled=True, device_type=self.device):
                # Generate predictions for remaining timesteps
                for t in range(num_steps-1):
                    # One-step prediction
                    next_state = self.model(current_state.unsqueeze(0)).squeeze(0)
                    # Store prediction (squeeze time dim)
                    trajectory_frames.append(next_state)  
                    current_state = next_state
            
            trajectory_tensor = torch.cat(trajectory_frames, dim=1)
            # Split into individual fields according to the trajectories given
            split_indices = []
            start_idx = 0
            for field_name in sorted(tensor_data.keys()):
                field_channels = tensor_data[field_name].shape[0]
                split_indices.append((start_idx, start_idx + field_channels))
                start_idx += field_channels
            
            split_trajectory = {
                field_name: trajectory_tensor[start:end, :, :, :]
                for (field_name, (start, end)) in zip(sorted(tensor_data.keys()), split_indices)
            }
            
            # Store as CPU tensor
            prediction_trajectories.append(split_trajectory)

        
        return prediction_trajectories

    
    def _create_comparison_animation(
        self,
        sim_idx: int,
        real_data: Dict[str, torch.Tensor],
        generated_data: Dict[str, torch.Tensor]
    ):
        """
        Create comparison animation using PhiFlow's visualization.
        
        This converts tensors to Fields and uses phi.vis.plot() to create
        side-by-side animations.
        """
        from phi.math import spatial, batch, channel
        from phi import math
        from phi.geom import Box
        
        # Get domain and resolution from config
        domain = Box(
            x=self.config['model']['physical']['domain']['size_x'],
            y=self.config['model']['physical']['domain']['size_y']
        )
        resolution = spatial(
            x=self.config['model']['physical']['resolution']['x'],
            y=self.config['model']['physical']['resolution']['y']
        )
        # Convert each field to PhiFlow Fields
        for field_name in real_data.keys():
            real_tensor = real_data[field_name]
            gen_tensor = generated_data[0][field_name]
            
            if real_tensor.shape[0] == 1:
                real_phiml = math.tensor(
                    real_tensor[0].cuda(),
                    batch('time'),
                    spatial('x', 'y')
                )
                gen_phiml = math.tensor(
                    gen_tensor[0].cuda(),
                    batch('time'),
                    spatial('x', 'y')
                )
            elif real_tensor.shape[0] == 2:
                real_phiml = math.norm(math.tensor(
                    real_tensor.cuda(),
                    channel('vector'),
                    batch('time'),
                    spatial('x', 'y')
                ))
                gen_phiml = math.norm(math.tensor(
                    gen_tensor.cuda(),
                    channel('vector'),
                    batch('time'),
                    spatial('x', 'y')
                ))
            
            
            # Make sure the synthetic prediction maximum is cut off at the real maximum for better visualization
           
            gen_phiml = nan_to_0(gen_phiml)
            real_phiml = nan_to_0(real_phiml)
            max_real = math.max(real_phiml)
            min_real = math.min(real_phiml)


            gen_phiml = math.clip(gen_phiml, min_real, max_real)

            
            ani = plot(
                {
                    'Real': real_phiml,
                    'Generated': gen_phiml
                },
                animate='time',

            )
            ani.save(
                self.output_dir / f'sim_{sim_idx}_{field_name}_comparison.gif',
                fps=10,
            )