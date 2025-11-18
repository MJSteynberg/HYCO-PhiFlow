"""
Simple Evaluator for comparing real vs generated trajectories.

Uses PhiFlow's built-in visualization tools for easy plotting.
"""

from pathlib import Path
from typing import Dict, Any, List
import torch
from phi.vis import plot, show, close, smooth
from phi.torch.flow import *
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
        self._load_synthetic_model()
        # self._load_physical_parameters()
        
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
        
        
        # Evaluate each test simulation
        test_sims = self.eval_config['test_sim']
        for sim_idx in test_sims:
            logger.info(f"Evaluating simulation {sim_idx}...")
            self._evaluate_simulation(sim_idx)
            
        logger.info(f"Evaluation complete! Results saved to {self.output_dir}")
    
    def _load_synthetic_model(self):
        """Load model from checkpoint."""
        from src.factories.model_factory import ModelFactory
        
        # Create model
        self.model = ModelFactory.create_synthetic_model(self.config)
        
        # Load checkpoint
        checkpoint_path = Path(self.eval_config['synthetic_checkpoint'])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded model from {checkpoint_path}")

    def _load_physical_parameters(self):
        """Load physical model for parameter visualization (optional)."""
        from src.factories.model_factory import ModelFactory
        
        self.physical_model = ModelFactory.create_physical_model(self.config)
        
        physical_checkpoint_path = self.eval_config.get('physical_checkpoint', None)
        checkpoint = torch.load(physical_checkpoint_path)
        self.learnable_parameters = [math.tensor(param, spatial("x,y")) for param in checkpoint["learnable_parameters"]]

        # Get the real parameters from the physical model
        learnable_params = self.config['trainer']['physical']['learnable_parameters']
        self.real_parameters = []
        self.param_names = [param['name'] for param in learnable_params]
        for param_name in self.param_names:
            self.real_parameters.append(getattr(self.physical_model, param_name))
    
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
                real_phiml = math.tensor(
                    real_tensor.cuda(),
                    channel(vector='x,y'),
                    batch('time'),
                    spatial('x', 'y')
                )
                real_field = CenteredGrid(
                    real_phiml,
                    extrapolation.PERIODIC,
                    x=resolution.get_size("x"),
                    y=resolution.get_size("y"),
                    bounds=domain,
                )
                gen_phiml = math.norm(math.tensor(
                    gen_tensor.cuda(),
                    channel('vector'),
                    batch('time'),
                    spatial('x', 'y')
                ))
                
                
            def normalized_vorticity(velocity: CenteredGrid) -> CenteredGrid:
                """
                Compute normalized vorticity: sign(ω) * sqrt(|ω| / quantile(ω, 0.8))
                
                This normalization:
                - Preserves sign (rotation direction)
                - Compresses dynamic range with sqrt
                - Normalizes by 80th percentile for consistent scaling
                """
                # Compute vorticity (curl)

                curl = field.curl(velocity)
                
                # Get the values as a tensor
                curl_values = curl.values
                
                # Compute 80th percentile (quantile)
                abs_curl = math.abs(curl_values)
                quantile_80 = math.quantile(abs_curl, 0.8)
                
                # Avoid division by zero
                quantile_80 = math.maximum(quantile_80, 1e-10)
                
                # Apply the transformation: sign(curl) * sqrt(|curl| / quantile)
                normalized = math.sign(curl_values) * math.sqrt(abs_curl / quantile_80)
                
                # Create new field with normalized values
                curl_normalized = field.CenteredGrid(
                    normalized,
                    extrapolation=curl.extrapolation,
                    bounds=curl.bounds,
                    resolution=curl.resolution
                )
                
                return curl_normalized
            # Make sure the synthetic prediction maximum is cut off at the real maximum for better visualization
            real_vorticity = normalized_vorticity(real_field)
            gen_phiml = nan_to_0(gen_phiml)
            real_phiml = nan_to_0(real_phiml)
            max_real = math.max(real_phiml)
            min_real = math.min(real_phiml)


            gen_phiml = math.clip(gen_phiml, min_real, max_real)

            
            ani = plot(
                {
                    'Real': real_vorticity,
                    'Generated': gen_phiml
                },
                animate='time',

            )
            ani.save(
                self.output_dir / f'sim_{sim_idx}_{field_name}_comparison.gif',
                fps=10,
            )
            # for name, real_param, learned_param in zip(self.param_names, self.real_parameters, self.learnable_parameters):
            #     plot(
            #         {
            #             f'Real {name}': real_param,
            #             f'Learned {name}': learned_param
            #         },
            #     )
            #     plt.savefig(self.output_dir / f'sim_{sim_idx}_{name}_comparison.png')
