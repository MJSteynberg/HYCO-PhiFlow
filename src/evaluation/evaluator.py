"""
Simple Evaluator for comparing real vs generated trajectories.

Uses PhiFlow's built-in visualization tools for easy plotting.
Fully compatible with PhiML-only codebase.
"""

from pathlib import Path
from typing import Dict, Any, List
from phi.vis import plot, show, close, smooth
from phi.torch.flow import *
from phi.math import nan_to_0, math

from src.models.synthetic.base import SyntheticModel
from src.utils.logger import get_logger
import matplotlib.pyplot as plt

logger = get_logger(__name__)


class Evaluator:
    """
    Simple evaluator that generates predictions and creates comparison animations.

    Workflow:
    1. Load model checkpoint (PhiML format)
    2. Load test simulations (PhiML format)
    3. Generate predictions (autoregressive rollout using PhiML tensors)
    4. Convert tensors to Fields for visualization
    5. Create comparison animations using PhiFlow's plot()
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluator from config."""
        self.config = config
        self.eval_config = config['evaluation']

        # Setup paths
        self.output_dir = Path(self.eval_config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data configuration
        self.data_dir = config['data']['data_dir']
        self.field_names = config['data']['fields']
        self.trajectory_length = config['data']['trajectory_length']


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

        # Evaluate each test simulation
        test_sims = self.eval_config['test_sim']
        for sim_idx in test_sims:
            logger.info(f"Evaluating simulation {sim_idx}...")
            self._evaluate_simulation(sim_idx)

        logger.info(f"Evaluation complete! Results saved to {self.output_dir}")

    def _load_synthetic_model(self):
        """Load model from checkpoint (PhiML format)."""
        from src.factories.model_factory import ModelFactory

        # Create model
        self.model = ModelFactory.create_synthetic_model(self.config)

        # Load checkpoint using PhiML's load method
        checkpoint_path = Path(self.eval_config['synthetic_checkpoint'])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.model.load(str(checkpoint_path))

        logger.info(f"Loaded PhiML model from {checkpoint_path}")

    def _load_physical_parameters(self):
        """Load physical model for parameter visualization (optional)."""
        import torch
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
        # Load real data (PhiML format)
        real_data = self._load_real_trajectory(sim_idx)

        # Generate predictions (PhiML tensors)
        generated_data = self.generate_predictions(real_data)

        # Create visualizations
        self._create_comparison_animation(sim_idx, real_data, generated_data)

    def _load_real_trajectory(self, sim_idx: int) -> Dict[str, Any]:
        """
        Load real trajectory from data using PhiML.

        Returns:
            Dict mapping field names to PhiML tensors/Fields with time dimension
        """
        import os

        # Load simulation using PhiML's math.load
        sim_path = os.path.join(self.data_dir, f"sim_{sim_idx:04d}.npz")
        if not os.path.exists(sim_path):
            raise FileNotFoundError(f"Simulation not found: {sim_path}")

        sim_data = math.load(sim_path)

        logger.debug(f"Loaded simulation {sim_idx} from {sim_path}")
        return sim_data
    
    def generate_predictions(self, sim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate autoregressive predictions from initial state using PhiML tensors.

        Args:
            sim_data: Dict mapping field names to PhiML tensors/Fields with time dimension
                     Format: {'velocity': Tensor/Field[time, x, y, vector], ...}

        Returns:
            Dict mapping field names to predicted trajectories (PhiML tensors)
            Format: {'velocity': Tensor[time, x, y, vector], ...}
        """
        logger.info("Generating predictions...")

        # Get the number of timesteps from the first field
        first_field = sim_data[self.field_names[0]]
        if isinstance(first_field, Field):
            num_steps = first_field.shape.get_size('time')
            # Extract initial state (t=0) from each field
            initial_state = {
                field_name: sim_data[field_name].time[0].values
                for field_name in self.field_names
            }
        else:
            # Already a tensor
            num_steps = first_field.shape.get_size('time')
            initial_state = {
                field_name: sim_data[field_name].time[0]
                for field_name in self.field_names
            }

        # Generate predictions autoregressively
        # Note: We predict num_steps - 1 because initial_state is already t=0
        current_state = initial_state
        predictions = [initial_state]  # Start with initial state at t=0

        for step in range(num_steps - 1):  # Predict remaining timesteps
            # Predict next state (dict of PhiML tensors)
            next_state = self.model(current_state)
            predictions.append(next_state)
            current_state = next_state

        # Stack predictions along time dimension
        prediction_trajectory = {}
        for field_name in self.field_names:
            # Stack all timesteps: [time, x, y, vector]
            prediction_trajectory[field_name] = math.stack(
                [pred[field_name] for pred in predictions],
                math.batch('time')
            )

        logger.info(f"Generated {num_steps} timesteps (including initial state)")
        return prediction_trajectory


    def _create_comparison_animation(
        self,
        sim_idx: int,
        real_data: Dict[str, Any],
        generated_data: Dict[str, Any]
    ):
        """
        Create comparison animation using PhiFlow's visualization.

        Args:
            sim_idx: Simulation index
            real_data: Dict of real trajectories (PhiML tensors/Fields)
            generated_data: Dict of generated trajectories (PhiML tensors)
        """
        from phi.math import spatial, batch, channel
        from phi.geom import Box

        # Get domain and resolution from config


        # Process each field
        for field_name in self.field_names:
            logger.info(f"Creating visualization for field: {field_name}")

            # Get real and generated data for this field
            real_field_data = real_data[field_name]
            gen_field_data = generated_data[field_name]

            # Check if this is a vector field (velocity)
            if 'vector' in real_field_data.shape.names:
                

                real_viz = real_field_data.vector[0]
                gen_viz = gen_field_data.vector[0]
            else:
                # For scalar fields, use directly
                real_viz = real_field_data
                gen_viz = gen_field_data

            # Limit the max values for better visualization
            max_real = math.max(real_viz)
            min_real = math.min(real_viz)
            gen_viz = math.clip(gen_viz, min_real, max_real)
            # Create animation
            plot({
                'Real': real_viz.time[0],
                'Generated': gen_viz.time[0]
            })

            plt.savefig(self.output_dir / f'sim_{sim_idx:04d}_{field_name}_comparison_0.png')
            plot({
                'Real': real_viz.time[1],
                'Generated': gen_viz.time[1]
            })

            plt.savefig(self.output_dir / f'sim_{sim_idx:04d}_{field_name}_comparison_1.png')

            ani = plot(
                {
                    'Real': real_viz,
                    'Generated': gen_viz
                },
                animate='time',
            )

            # Save animation
            output_path = self.output_dir / f'sim_{sim_idx:04d}_{field_name}_comparison.gif'
            ani.save(output_path, fps=10)
            logger.info(f"Saved animation to {output_path}")

        # Optional: Visualize learned physical parameters
        if hasattr(self, 'param_names'):
            for name, real_param, learned_param in zip(
                self.param_names, self.real_parameters, self.learnable_parameters
            ):
                plot({'Real': real_param, 'Learned': learned_param})
                plt.savefig(self.output_dir / f'sim_{sim_idx:04d}_{name}_comparison.png')
                logger.info(f"Saved parameter comparison for {name}")
