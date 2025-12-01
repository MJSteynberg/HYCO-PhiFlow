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
import numpy as np

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
        self.trajectory_length = config['data']['trajectory_length']
        # field_names will be inferred from loaded data tensor shape
        self.field_names = None

        # Model will be loaded in evaluate()
        self._load_synthetic_model()
        self._load_physical_parameters()

    def evaluate(self):
        """
        Run evaluation on test simulations.

        Creates comparison animations for each simulation showing:
        - Real trajectory (from data)
        - Generated trajectory (from model)
        Side by side for each field.
        """
        logger.info("Starting evaluation...")
        self._visualize_modulation_field()
        # Evaluate each test simulation
        test_sims = self.eval_config['test_sim']
        for sim_idx in test_sims:
            logger.info(f"Evaluating simulation {sim_idx}...")
            self._evaluate_simulation(sim_idx)
        
        # Visualize modulation field if available
        

        logger.info(f"Evaluation complete! Results saved to {self.output_dir}")

    def _load_synthetic_model(self):
        """Load model from checkpoint (PhiML format)."""
        from src.factories.model_factory import ModelFactory
        import os

        # Load a sample trajectory to get num_channels and field_names
        test_sims = self.eval_config['test_sim']
        sample_path = os.path.join(self.data_dir, f"sim_{test_sims[0]:04d}.npz")
        sample_data = math.load(sample_path)

        # Get num_channels and field_names from tensor shape
        num_channels = sample_data.shape['field'].size
        raw_names = sample_data.shape['field'].item_names
        # item_names can be nested like (('field1', 'field2'),) - extract properly
        if raw_names and isinstance(raw_names[0], tuple):
            self.field_names = list(raw_names[0])
        else:
            self.field_names = list(raw_names) if raw_names else None

        # Create model with num_channels
        self.model = ModelFactory.create_synthetic_model(self.config, num_channels=num_channels)
        

        # Get static fields from model to determine which are dynamic
        self.static_fields = self.model.static_fields if hasattr(self.model, 'static_fields') else []
        self.dynamic_fields = [f for f in self.field_names if f not in self.static_fields]
        logger.info(f"Loaded sample data: {num_channels} channels, fields={self.field_names}")
        logger.info(f"Dynamic fields (will be visualized): {self.dynamic_fields}")

        # Load checkpoint using PhiML's load method
        checkpoint_path = Path(self.eval_config['synthetic_checkpoint'])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        try:    
            self.model.load(str(checkpoint_path))
        except Exception as e:
            self.model.network = torch.compile(self.model.network)
            self.model.load(str(checkpoint_path))

        logger.info(f"Loaded PhiML model from {checkpoint_path}")

    def _load_physical_parameters(self):
        """Load physical model parameters for visualization."""
        from src.factories.model_factory import ModelFactory
        
        # Only load if physical checkpoint is specified
        physical_checkpoint_path = self.eval_config.get('physical_checkpoint', None)
        if not physical_checkpoint_path or not Path(physical_checkpoint_path).exists():
            logger.info("No physical checkpoint specified or found, skipping parameter visualization")
            return

        self.physical_model = ModelFactory.create_physical_model(self.config)

        # Load learned parameters (Tensor)
        try:
            self.learned_params = math.load(str(physical_checkpoint_path))
            logger.info(f"Loaded learned parameters from {physical_checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return

        # Get ground truth parameters
        self.ground_truth_params = self.physical_model.get_real_params()
        
    def _visualize_modulation_field(self):
        """Create visualization comparing ground truth vs learned parameters (both scalar and field)."""
        if not hasattr(self, 'learned_params') or not hasattr(self, 'ground_truth_params'):
            logger.info("No parameters to visualize")
            return

        from phi.vis import plot

        logger.info("Creating parameter visualizations...")

        real = self.ground_truth_params
        learned = self.learned_params

        # Visualize scalar parameters (e.g., diffusion coefficient)
        scalar_param_names = self.physical_model.scalar_param_names
        if scalar_param_names:
            logger.info(f"Visualizing scalar parameters: {scalar_param_names}")
            self._visualize_scalar_parameters(scalar_param_names, real, learned)

        # Get field parameter names from physical model
        field_param_names = self.physical_model.field_param_names
        if field_param_names:
            logger.info(f"Visualizing field parameters: {field_param_names}")
            self._visualize_field_parameters(field_param_names, real, learned)

    def _visualize_scalar_parameters(self, param_names, real, learned):
        """Visualize scalar parameters as bar chart."""
        # Extract scalar values
        real_values = []
        learned_values = []

        for param_name in param_names:
            real_val = float(real.field[param_name])
            learned_val = float(learned.field[param_name])
            real_values.append(real_val)
            learned_values.append(learned_val)

            logger.info(f"  {param_name}: Real={real_val:.6f}, Learned={learned_val:.6f}, Error={abs(learned_val - real_val):.6f}")

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(param_names))
        width = 0.35

        ax.bar(x - width/2, real_values, width, label='Ground Truth', alpha=0.8, color='#2E86AB')
        ax.bar(x + width/2, learned_values, width, label='Learned', alpha=0.8, color='#A23B72')

        ax.set_xlabel('Parameter', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Physical Parameter Recovery', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(param_names)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        # Add text labels showing exact values and error
        for i, (gt, learned_val) in enumerate(zip(real_values, learned_values)):
            error = abs(learned_val - gt)
            error_pct = (error / abs(gt) * 100) if gt != 0 else float('inf')
            y_pos = max(gt, learned_val) * 1.05
            ax.text(i, y_pos, f'Error: {error:.6f}\n({error_pct:.2f}%)',
                   ha='center', va='bottom', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        output_path = self.output_dir / 'scalar_parameter_recovery.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved scalar parameter comparison to {output_path}")

    def _visualize_field_parameters(self, param_names, real, learned):
        """Visualize field parameters as spatial plots."""
        # Visualize each field parameter component separately
        for param_name in param_names:
            logger.info(f"Creating visualization for parameter: {param_name}")

            # Extract individual field component
            real_param = real.field[param_name]
            learned_param = learned.field[param_name]

            # Create comparison plot for this parameter
            plot({
                f'Real {param_name}': real_param,
                f'Learned {param_name}': learned_param,
            })

            output_path = self.output_dir / f'param_{param_name}_comparison.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close('all')
            logger.info(f"Saved {param_name} comparison to {output_path}")

        # Also create vector field visualization if 2D
        try:
            if 'mod_field_x' in param_names and 'mod_field_y' in param_names:
                logger.info("Creating vector field visualization...")

                # Rename field channel to vector channel for vector visualization
                real_vec = math.stack(
                    [real.field['mod_field_x'], real.field['mod_field_y']],
                    channel(vector='x,y')
                )
                learned_vec = math.stack(
                    [learned.field['mod_field_x'], learned.field['mod_field_y']],
                    channel(vector='x,y')
                )

                # Create vector field comparison
                plot({
                    'Real Modulation (Vector)': math.norm(real_vec),
                    'Learned Modulation (Vector)': math.norm(learned_vec),
                })

                output_path = self.output_dir / 'modulation_vector_field_comparison.png'
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close('all')
                logger.info(f"Saved vector field comparison to {output_path}")

                # Calculate and log statistics
                l2_error = math.sqrt(math.sum((real_vec - learned_vec) ** 2))
                max_error = math.max(math.abs(real_vec - learned_vec))
                logger.info(f"Vector field L2 error: {l2_error:.6f}")
                logger.info(f"Vector field max error: {max_error:.6f}")

        except Exception as e:
            logger.warning(f"Could not create vector field visualization: {e}")

    def _evaluate_simulation(self, sim_idx: int):
        """Evaluate a single simulation."""
        # Load real data (PhiML format)
        real_data = self._load_real_trajectory(sim_idx)

        # Generate predictions (PhiML tensors)
        generated_data = self.generate_predictions(real_data)

        # Create visualizations
        self._create_comparison_animation(sim_idx, real_data, generated_data)

    def _load_real_trajectory(self, sim_idx: int) -> Tensor:
        """
        Load real trajectory from data using PhiML.

        Returns:
            Unified tensor: Tensor(time, x, y?, field)
        """
        import os
        from phi.math import Tensor

        # Load simulation using PhiML's math.load
        sim_path = os.path.join(self.data_dir, f"sim_{sim_idx:04d}.npz")
        if not os.path.exists(sim_path):
            raise FileNotFoundError(f"Simulation not found: {sim_path}")

        sim_data = math.load(sim_path)

        logger.debug(f"Loaded simulation {sim_idx} from {sim_path}, shape={sim_data.shape}")
        return sim_data
    
    def generate_predictions(self, sim_data: Tensor) -> Tensor:
        """
        Generate autoregressive predictions from initial state using unified tensors.

        Args:
            sim_data: Unified tensor with shape (time, x, y?, field)

        Returns:
            Predicted trajectory as unified tensor (time, x, y?, field)
        """
        from phi.math import Tensor

        logger.info("Generating predictions...")

        # Get the number of timesteps from tensor shape
        num_steps = sim_data.shape.get_size('time')

        # Extract initial state (t=0) - unified tensor
        initial_state = sim_data.time[0]

        # Generate predictions autoregressively
        current_state = initial_state
        predictions = [initial_state]  # Start with initial state at t=0

        for step in range(num_steps - 1):  # Predict remaining timesteps
            # Predict next state (unified tensor)
            next_state = self.model(current_state)
            predictions.append(next_state)
            current_state = next_state

        # Stack predictions along time dimension
        # expand_values=True handles dimension mismatches from native_call output
        prediction_trajectory = math.stack(predictions, math.batch('time'), expand_values=True)

        logger.info(f"Generated {num_steps} timesteps (including initial state)")
        return prediction_trajectory


    def _create_comparison_animation(
        self,
        sim_idx: int,
        real_data: Tensor,
        generated_data: Tensor
    ):
        """
        Create comparison animation using PhiFlow's visualization.

        Args:
            sim_idx: Simulation index
            real_data: Unified tensor (time, x, y?, field)
            generated_data: Unified tensor (time, x, y?, field)
        """
        from phi.math import spatial, batch, channel
        from phi.geom import Box

        # Only visualize dynamic fields (static fields don't change)
        for i, field_name in enumerate(self.dynamic_fields):
            logger.info(f"Creating visualization for dynamic field: {field_name}")

            # Get real and generated data for this field using field dimension
            real_field_data = real_data.field[field_name]
            gen_field_data = generated_data.field[field_name]

            # Use data directly (already scalar after field selection)
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

