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

        # Evaluate each test simulation
        test_sims = self.eval_config['test_sim']
        for sim_idx in test_sims:
            logger.info(f"Evaluating simulation {sim_idx}...")
            self._evaluate_simulation(sim_idx)
        
        # Visualize modulation field if available
        self._visualize_modulation_field()

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
        self.model.network = torch.compile(self.model.network)

        # Get static fields from model to determine which are dynamic
        self.static_fields = self.model.static_fields if hasattr(self.model, 'static_fields') else []
        self.dynamic_fields = [f for f in self.field_names if f not in self.static_fields]
        logger.info(f"Loaded sample data: {num_channels} channels, fields={self.field_names}")
        logger.info(f"Dynamic fields (will be visualized): {self.dynamic_fields}")

        # Load checkpoint using PhiML's load method
        checkpoint_path = Path(self.eval_config['synthetic_checkpoint'])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.model.load(str(checkpoint_path))

        logger.info(f"Loaded PhiML model from {checkpoint_path}")

    def _load_physical_parameters(self):
        """Load physical model parameters for visualization (optional)."""
        import numpy as np
        from src.factories.model_factory import ModelFactory

        # Only load if physical checkpoint is specified
        physical_checkpoint_path = self.eval_config.get('physical_checkpoint', None)
        if not physical_checkpoint_path or not Path(physical_checkpoint_path).exists():
            logger.info("No physical checkpoint specified or found, skipping parameter visualization")
            return

        self.physical_model = ModelFactory.create_physical_model(self.config)

        # Load learned parameters from numpy checkpoint
        checkpoint = np.load(physical_checkpoint_path)
        learnable_params = self.config['trainer']['physical']['learnable_parameters']
        self.param_names = [param['name'] for param in learnable_params]
        
        self.learnable_parameters = []
        for name in self.param_names:
            param_value = checkpoint[f'_{name}']
            self.learnable_parameters.append(float(param_value))  # Convert to scalar

        # Get the ground truth parameters from the physical model config
        self.ground_truth_parameters = []
        for param in learnable_params:
            if param['name'] == 'advection_coeff':
                # Get from pde_params
                self.ground_truth_parameters.append(float(eval(self.config['model']['physical']['pde_params']['value'])))
            elif param['name'] == 'modulation_amplitude':
                # Get from modulation config
                mod_config = self.config['model']['physical'].get('modulation', {})
                self.ground_truth_parameters.append(float(mod_config.get('amplitude', 0.0)))
        
        logger.info(f"Loaded physical parameters: {dict(zip(self.param_names, self.learnable_parameters))}")
        logger.info(f"Ground truth parameters: {dict(zip(self.param_names, self.ground_truth_parameters))}")

    def _visualize_modulation_field(self):
        """Create visualization comparing ground truth vs learned modulation field."""
        if not hasattr(self, 'param_names') or 'modulation_amplitude' not in self.param_names:
            return
            
        from phi.field import CenteredGrid
        from phi.math import math
        
        # Get amplitudes
        mod_idx = self.param_names.index('modulation_amplitude')
        gt_amplitude = self.ground_truth_parameters[mod_idx]
        learned_amplitude = self.learnable_parameters[mod_idx]
        
        # Get config parameters
        modulation_config = self.config['model']['physical'].get('modulation', {})
        num_lobes = int(modulation_config.get('lobes', 6))
        resolution = self.config['model']['physical']['domain']['dimensions']
        domain_size = 100  # From config
        
        # Create grids
        x_res = resolution['x']['resolution']
        y_res = resolution['y']['resolution']
        
        # Compute modulation fields on a grid
        x = np.linspace(0, domain_size, x_res)
        y = np.linspace(0, domain_size, y_res)
        xx, yy = np.meshgrid(x, y)
        
        # Center and ring parameters
        center_x, center_y = domain_size / 2, domain_size / 2
        ring_radius = domain_size * 0.3
        ring_width = domain_size * 0.15
        
        # Compute for both amplitudes
        dx = xx - center_x
        dy = yy - center_y
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        falloff = np.exp(-((r - ring_radius) / ring_width) ** 2)
        
        gt_modulation = falloff * gt_amplitude * np.sin(num_lobes * theta)
        learned_modulation = falloff * learned_amplitude * np.sin(num_lobes * theta)
        difference = gt_modulation - learned_modulation
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Ground truth
        im0 = axes[0, 0].imshow(gt_modulation, origin='lower', extent=[0, domain_size, 0, domain_size], cmap='RdBu_r')
        axes[0, 0].set_title(f'Ground Truth Modulation (amplitude={gt_amplitude:.3f})')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        plt.colorbar(im0, ax=axes[0, 0])
        
        # Learned
        im1 = axes[0, 1].imshow(learned_modulation, origin='lower', extent=[0, domain_size, 0, domain_size], cmap='RdBu_r')
        axes[0, 1].set_title(f'Learned Modulation (amplitude={learned_amplitude:.3f})')
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Difference
        im2 = axes[1, 0].imshow(difference, origin='lower', extent=[0, domain_size, 0, domain_size], cmap='RdBu_r')
        axes[1, 0].set_title('Difference (GT - Learned)')
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[1, 0])
        
        # Error statistics
        axes[1, 1].axis('off')
        error_text = f"""
Modulation Field Comparison

Ground Truth Amplitude: {gt_amplitude:.4f}
Learned Amplitude: {learned_amplitude:.4f}
Absolute Error: {abs(gt_amplitude - learned_amplitude):.4f}
Relative Error: {abs(gt_amplitude - learned_amplitude) / gt_amplitude * 100:.2f}%

Field Statistics:
GT Max: {np.max(gt_modulation):.4f}
GT Min: {np.min(gt_modulation):.4f}
Learned Max: {np.max(learned_modulation):.4f}
Learned Min: {np.min(learned_modulation):.4f}
Difference Max: {np.max(np.abs(difference)):.4f}
Difference RMS: {np.sqrt(np.mean(difference**2)):.4f}
"""
        axes[1, 1].text(0.1, 0.5, error_text, fontfamily='monospace', fontsize=10, verticalalignment='center')
        
        plt.tight_layout()
        output_path = self.output_dir / 'modulation_field_comparison.png'
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved modulation field comparison to {output_path}")

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

        # Visualize learned physical parameters if available
        if hasattr(self, 'param_names') and hasattr(self, 'learnable_parameters'):
            logger.info("Creating parameter recovery visualization...")
            
            # Create bar chart comparing ground truth vs learned
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(self.param_names))
            width = 0.35
            
            ax.bar(x - width/2, self.ground_truth_parameters, width, label='Ground Truth', alpha=0.8)
            ax.bar(x + width/2, self.learnable_parameters, width, label='Learned', alpha=0.8)
            
            ax.set_xlabel('Parameter')
            ax.set_ylabel('Value')
            ax.set_title('Physical Parameter Recovery')
            ax.set_xticks(x)
            ax.set_xticklabels(self.param_names)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add text labels showing exact values and error
            for i, (gt, learned) in enumerate(zip(self.ground_truth_parameters, self.learnable_parameters)):
                error = abs(learned - gt)
                error_pct = (error / abs(gt) * 100) if gt != 0 else float('inf')
                ax.text(i, max(gt, learned) * 1.05, f'Error: {error:.4f}\n({error_pct:.1f}%)', 
                       ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            output_path = self.output_dir / 'parameter_recovery.png'
            plt.savefig(output_path, dpi=150)
            plt.close(fig)
            logger.info(f"Saved parameter comparison to {output_path}")

