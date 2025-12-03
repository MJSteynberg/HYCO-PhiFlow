"""
Evaluator for comparing real vs generated trajectories with multiple model support.

Supports comparing:
- Two synthetic models (e.g., synthetic-only vs hybrid-synthetic)
- Two physical models (e.g., physical-only vs hybrid-physical)
- Ground truth parameters

Uses PhiFlow's built-in visualization tools for easy plotting.
Fully compatible with PhiML-only codebase.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
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
    Evaluator that generates predictions and creates comparison animations.

    Supports multiple models for comparison:
    - synthetic_checkpoint: Primary synthetic model (e.g., trained on real data only)
    - synthetic_checkpoint_hybrid: Secondary synthetic model (e.g., from hybrid training)
    - physical_checkpoint: Primary physical model (e.g., trained on real data only)
    - physical_checkpoint_hybrid: Secondary physical model (e.g., from hybrid training)

    Workflow:
    1. Load model checkpoints (PhiML format)
    2. Load test simulations (PhiML format)
    3. Generate predictions (autoregressive rollout using PhiML tensors)
    4. Create comparison animations/plots showing all models
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
        self.field_names = None

        # Models will be loaded in evaluate()
        self._load_synthetic_models()
        self._load_physical_parameters()

    def evaluate(self):
        """
        Run evaluation on test simulations.

        Creates comparison animations for each simulation showing:
        - Real trajectory (from data)
        - Synthetic model predictions
        - Hybrid synthetic model predictions (if available)
        """
        logger.info("Starting evaluation...")
        self._visualize_modulation_field()
        
        # Evaluate each test simulation
        test_sims = self.eval_config['test_sim']
        for sim_idx in test_sims:
            logger.info(f"Evaluating simulation {sim_idx}...")
            self._evaluate_simulation(sim_idx)

        logger.info(f"Evaluation complete! Results saved to {self.output_dir}")

    def _load_synthetic_models(self):
        """Load synthetic model checkpoints."""
        from src.factories.model_factory import ModelFactory
        import os

        # Load sample trajectory to get num_channels and field_names
        test_sims = self.eval_config['test_sim']
        sample_path = os.path.join(self.data_dir, f"sim_{test_sims[0]:04d}.npz")
        sample_data = math.load(sample_path)

        # Get num_channels and field_names from tensor shape
        num_channels = sample_data.shape['field'].size
        raw_names = sample_data.shape['field'].item_names
        if raw_names and isinstance(raw_names[0], tuple):
            self.field_names = list(raw_names[0])
        else:
            self.field_names = list(raw_names) if raw_names else None

        logger.info(f"Loaded sample data: {num_channels} channels, fields={self.field_names}")

        # Initialize model containers
        self.model = None  # Primary synthetic model
        self.model_hybrid = None  # Hybrid synthetic model

        # Load primary synthetic model
        checkpoint_path = self.eval_config.get('synthetic_checkpoint')
        if checkpoint_path and Path(checkpoint_path).exists():
            self.model = ModelFactory.create_synthetic_model(self.config, num_channels=num_channels)
            self._load_model_checkpoint(self.model, checkpoint_path, "synthetic")
            
            # Get static/dynamic fields from model
            self.static_fields = self.model.static_fields if hasattr(self.model, 'static_fields') else []
            self.dynamic_fields = [f for f in self.field_names if f not in self.static_fields]
            logger.info(f"Dynamic fields (will be visualized): {self.dynamic_fields}")
        else:
            logger.warning(f"Primary synthetic checkpoint not found: {checkpoint_path}")
            self.static_fields = []
            self.dynamic_fields = self.field_names or []

        # Load hybrid synthetic model (optional)
        hybrid_checkpoint_path = self.eval_config.get('synthetic_checkpoint_hybrid')
        if hybrid_checkpoint_path and Path(hybrid_checkpoint_path).exists():
            self.model_hybrid = ModelFactory.create_synthetic_model(self.config, num_channels=num_channels)
            self._load_model_checkpoint(self.model_hybrid, hybrid_checkpoint_path, "hybrid synthetic")
        else:
            logger.info("No hybrid synthetic checkpoint specified or found")

    def _load_model_checkpoint(self, model, checkpoint_path: str, name: str):
        """Load a model checkpoint with error handling."""
        try:
            model.load(str(checkpoint_path))
            logger.info(f"Loaded {name} model from {checkpoint_path}")
        except Exception as e:
            try:
                model.network = torch.compile(model.network)
                model.load(str(checkpoint_path))
                logger.info(f"Loaded {name} model (compiled) from {checkpoint_path}")
            except Exception as e2:
                logger.warning(f"Failed to load {name} model: {e2}")

    def _load_physical_parameters(self):
        """Load physical model parameters for visualization."""
        from src.factories.model_factory import ModelFactory

        # Initialize parameter containers
        self.physical_model = None
        self.ground_truth_params = None
        self.learned_params = None  # Primary physical model params
        self.learned_params_hybrid = None  # Hybrid physical model params

        # Check if any physical checkpoint is specified
        physical_checkpoint = self.eval_config.get('physical_checkpoint')
        physical_checkpoint_hybrid = self.eval_config.get('physical_checkpoint_hybrid')

        if not physical_checkpoint and not physical_checkpoint_hybrid:
            logger.info("No physical checkpoints specified, skipping parameter visualization")
            return

        # Create physical model for ground truth
        self.physical_model = ModelFactory.create_physical_model(self.config)
        self.ground_truth_params = self.physical_model.get_real_params()

        # Load primary physical parameters
        if physical_checkpoint and Path(physical_checkpoint).exists():
            try:
                self.learned_params = math.load(str(physical_checkpoint))
                logger.info(f"Loaded physical parameters from {physical_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to load physical checkpoint: {e}")

        # Load hybrid physical parameters
        if physical_checkpoint_hybrid and Path(physical_checkpoint_hybrid).exists():
            try:
                self.learned_params_hybrid = math.load(str(physical_checkpoint_hybrid))
                logger.info(f"Loaded hybrid physical parameters from {physical_checkpoint_hybrid}")
            except Exception as e:
                logger.warning(f"Failed to load hybrid physical checkpoint: {e}")

    def _visualize_modulation_field(self):
        """Create visualization comparing ground truth vs learned parameters."""
        if self.ground_truth_params is None:
            logger.info("No parameters to visualize")
            return

        if self.learned_params is None and self.learned_params_hybrid is None:
            logger.info("No learned parameters to visualize")
            return

        logger.info("Creating parameter visualizations...")

        # Visualize scalar parameters
        scalar_param_names = self.physical_model.scalar_param_names
        if scalar_param_names:
            logger.info(f"Visualizing scalar parameters: {scalar_param_names}")
            self._visualize_scalar_parameters(scalar_param_names)

        # Visualize field parameters
        field_param_names = self.physical_model.field_param_names
        if field_param_names:
            logger.info(f"Visualizing field parameters: {field_param_names}")
            self._visualize_field_parameters(field_param_names)

    def _visualize_scalar_parameters(self, param_names):
        """Visualize scalar parameters as bar chart with multiple models."""
        real = self.ground_truth_params

        # Collect values
        real_values = [float(real.field[name]) for name in param_names]

        labels = ['Ground Truth']
        all_values = [real_values]
        colors = ['#2E86AB']  # Blue

        if self.learned_params is not None:
            learned_values = [float(self.learned_params.field[name]) for name in param_names]
            labels.append('Physical Only')
            all_values.append(learned_values)
            colors.append('#E94F37')  # Red

        if self.learned_params_hybrid is not None:
            hybrid_values = [float(self.learned_params_hybrid.field[name]) for name in param_names]
            labels.append('Hybrid Physical')
            all_values.append(hybrid_values)
            colors.append('#44AF69')  # Green

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(param_names))
        n_bars = len(all_values)
        width = 0.8 / n_bars

        for i, (values, label, color) in enumerate(zip(all_values, labels, colors)):
            offset = (i - n_bars / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=label, alpha=0.8, color=color)

        ax.set_xlabel('Parameter', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Physical Parameter Recovery', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(param_names)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        # Log errors
        for i, name in enumerate(param_names):
            gt = real_values[i]
            if self.learned_params is not None:
                po_err = abs(all_values[1][i] - gt)
                logger.info(f"  {name}: Physical-Only Error = {po_err:.6f}")
            if self.learned_params_hybrid is not None:
                idx = 2 if self.learned_params is not None else 1
                hp_err = abs(all_values[idx][i] - gt)
                logger.info(f"  {name}: Hybrid-Physical Error = {hp_err:.6f}")

        plt.tight_layout()
        output_path = self.output_dir / 'scalar_parameter_recovery.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved scalar parameter comparison to {output_path}")

    def _visualize_field_parameters(self, param_names):
        """Visualize field parameters as spatial plots with multiple models."""
        real = self.ground_truth_params

        for param_name in param_names:
            logger.info(f"Creating visualization for parameter: {param_name}")

            real_param = real.field[param_name]

            # Build plot dictionary
            plot_dict = {f'Ground Truth': real_param}

            if self.learned_params is not None:
                plot_dict['Physical Only'] = self.learned_params.field[param_name]

            if self.learned_params_hybrid is not None:
                plot_dict['Hybrid Physical'] = self.learned_params_hybrid.field[param_name]

            # Create comparison plot
            plot(plot_dict)

            output_path = self.output_dir / f'param_{param_name}_comparison.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close('all')
            logger.info(f"Saved {param_name} comparison to {output_path}")

            # Visualize gradient (forcing) for potential fields
            if 'potential' in param_name.lower():
                self._visualize_forcing_field(param_name)

        # Vector field visualization for 2D modulation fields
        self._visualize_vector_field_if_applicable(param_names)

    def _visualize_forcing_field(self, param_name: str):
        """Visualize forcing field (gradient of potential) for all models."""
        logger.info(f"Creating gradient visualization for potential field: {param_name}")

        try:
            domain = self.physical_model.domain
            spatial_dims = self.physical_model.spatial_dims
            grid_kwargs = {name: self.physical_model.resolution.get_size(name) for name in spatial_dims}

            real_param = self.ground_truth_params.field[param_name]
            real_grid = CenteredGrid(real_param, PERIODIC, bounds=domain, **grid_kwargs)
            real_forcing = -real_grid.gradient(boundary=PERIODIC)

            # Build plot dictionary
            if self.physical_model.n_spatial_dims == 2:
                plot_dict = {'Ground Truth |f|': 0.1 * real_forcing}
            else:
                plot_dict = {'Ground Truth f': real_forcing.values}

            if self.learned_params is not None:
                learned_grid = CenteredGrid(
                    self.learned_params.field[param_name], PERIODIC, bounds=domain, **grid_kwargs
                )
                learned_forcing = -learned_grid.gradient(boundary=PERIODIC)
                if self.physical_model.n_spatial_dims == 2:
                    plot_dict['Physical Only |f|'] = 0.1 * learned_forcing
                else:
                    plot_dict['Physical Only f'] = learned_forcing.values

            if self.learned_params_hybrid is not None:
                hybrid_grid = CenteredGrid(
                    self.learned_params_hybrid.field[param_name], PERIODIC, bounds=domain, **grid_kwargs
                )
                hybrid_forcing = -hybrid_grid.gradient(boundary=PERIODIC)
                if self.physical_model.n_spatial_dims == 2:
                    plot_dict['Hybrid Physical |f|'] = 0.1 * hybrid_forcing
                else:
                    plot_dict['Hybrid Physical f'] = hybrid_forcing.values

            plot(plot_dict)

            output_path = self.output_dir / f'param_{param_name}_forcing_comparison.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close('all')
            logger.info(f"Saved {param_name} forcing comparison to {output_path}")

        except Exception as e:
            logger.warning(f"Failed to create gradient visualization for {param_name}: {e}")

    def _visualize_vector_field_if_applicable(self, param_names):
        """Create vector field visualization if 2D modulation fields exist."""
        if 'mod_field_x' not in param_names or 'mod_field_y' not in param_names:
            return

        try:
            logger.info("Creating vector field visualization...")

            real = self.ground_truth_params
            real_vec = math.stack(
                [real.field['mod_field_x'], real.field['mod_field_y']],
                channel(vector='x,y')
            )

            plot_dict = {'Ground Truth': math.norm(real_vec)}

            if self.learned_params is not None:
                learned_vec = math.stack(
                    [self.learned_params.field['mod_field_x'], self.learned_params.field['mod_field_y']],
                    channel(vector='x,y')
                )
                plot_dict['Physical Only'] = math.norm(learned_vec)

            if self.learned_params_hybrid is not None:
                hybrid_vec = math.stack(
                    [self.learned_params_hybrid.field['mod_field_x'], self.learned_params_hybrid.field['mod_field_y']],
                    channel(vector='x,y')
                )
                plot_dict['Hybrid Physical'] = math.norm(hybrid_vec)

            plot(plot_dict)

            output_path = self.output_dir / 'modulation_vector_field_comparison.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close('all')
            logger.info(f"Saved vector field comparison to {output_path}")

        except Exception as e:
            logger.warning(f"Could not create vector field visualization: {e}")

    def _evaluate_simulation(self, sim_idx: int):
        """Evaluate a single simulation with all available models."""
        # Load real data
        real_data = self._load_real_trajectory(sim_idx)

        # Generate predictions from all available models
        predictions = {}

        if self.model is not None:
            predictions['Synthetic'] = self._generate_predictions(self.model, real_data)

        if self.model_hybrid is not None:
            predictions['Hybrid Synthetic'] = self._generate_predictions(self.model_hybrid, real_data)

        # Create visualizations
        self._create_comparison_animation(sim_idx, real_data, predictions)

    def _load_real_trajectory(self, sim_idx: int) -> Tensor:
        """Load real trajectory from data using PhiML."""
        import os

        sim_path = os.path.join(self.data_dir, f"sim_{sim_idx:04d}.npz")
        if not os.path.exists(sim_path):
            raise FileNotFoundError(f"Simulation not found: {sim_path}")

        sim_data = math.load(sim_path)
        logger.debug(f"Loaded simulation {sim_idx} from {sim_path}, shape={sim_data.shape}")
        return sim_data

    def _generate_predictions(self, model, sim_data: Tensor) -> Tensor:
        """Generate autoregressive predictions from initial state."""
        logger.info("Generating predictions...")

        num_steps = sim_data.shape.get_size('time')
        initial_state = sim_data.time[0]

        current_state = initial_state
        predictions = [initial_state]

        for step in range(num_steps - 1):
            next_state = model(current_state)
            predictions.append(next_state)
            current_state = next_state

        prediction_trajectory = math.stack(predictions, math.batch('time'), expand_values=True)
        logger.info(f"Generated {num_steps} timesteps")
        return prediction_trajectory

    # Keep backward compatibility with old single-model generate_predictions
    def generate_predictions(self, sim_data: Tensor) -> Tensor:
        """Generate predictions using the primary synthetic model."""
        if self.model is None:
            raise RuntimeError("No synthetic model loaded")
        return self._generate_predictions(self.model, sim_data)

    def _create_comparison_animation(
        self,
        sim_idx: int,
        real_data: Tensor,
        predictions: Dict[str, Tensor]
    ):
        """Create comparison animation using PhiFlow's visualization."""

        for field_name in self.dynamic_fields:
            logger.info(f"Creating visualization for dynamic field: {field_name}")

            real_field = real_data.field[field_name]
            max_real = math.max(real_field) * 1.2
            min_real = math.min(real_field) * 1.2

            # Build plot dictionary
            plot_dict = {'Real': real_field}

            for model_name, pred_data in predictions.items():
                pred_field = pred_data.field[field_name]
                pred_field = math.clip(pred_field, min_real, max_real)
                plot_dict[model_name] = pred_field

            # Save static frame comparisons (t=0, t=1)
            plot({k: v.time[0] for k, v in plot_dict.items()})
            plt.savefig(self.output_dir / f'sim_{sim_idx:04d}_{field_name}_comparison_t0.png')
            plt.close('all')

            plot({k: v.time[1] for k, v in plot_dict.items()})
            plt.savefig(self.output_dir / f'sim_{sim_idx:04d}_{field_name}_comparison_t1.png')
            plt.close('all')

            # Create and save animation
            ani = plot(plot_dict, animate='time')

            output_path = self.output_dir / f'sim_{sim_idx:04d}_{field_name}_comparison.gif'
            ani.save(str(output_path), fps=10)
            logger.info(f"Saved animation to {output_path}")