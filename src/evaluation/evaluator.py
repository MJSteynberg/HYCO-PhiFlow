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
from src.evaluation.metrics import MetricsComputer, TrajectoryMetrics, ParameterMetrics
import matplotlib.pyplot as plt
import numpy as np
import json

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

        # Initialize metrics computer
        self.metrics_computer = MetricsComputer(field_names=self.field_names)

    def evaluate(self):
        """
        Run evaluation on test simulations.

        Creates comparison animations for each simulation showing:
        - Real trajectory (from data)
        - Synthetic model predictions
        - Hybrid synthetic model predictions (if available)

        Computes and saves quantitative metrics:
        - L2, L1, L∞ errors over time
        - Relative errors
        - Per-field metrics
        - Parameter recovery metrics
        """
        logger.info("Starting evaluation...")

        # Visualize and compute parameter metrics
        param_metrics = self._visualize_modulation_field()

        # Evaluate each test simulation
        test_sims = self.eval_config['test_sim']
        all_metrics = {}

        for sim_idx in test_sims:
            logger.info(f"Evaluating simulation {sim_idx}...")
            metrics = self._evaluate_simulation(sim_idx)
            all_metrics[f'sim_{sim_idx:04d}'] = metrics

        # Save aggregated metrics
        self._save_aggregated_metrics(all_metrics, param_metrics)

        # Create metrics visualizations
        self._visualize_aggregated_metrics(all_metrics)

        # Create consolidated spacetime heatmap for 1D problems
        train_sim_idx = self.eval_config.get('train_sim', [0])[0] if isinstance(self.eval_config.get('train_sim', 0), list) else self.eval_config.get('train_sim', 0)
        self._create_consolidated_spacetime_heatmap(train_sim_idx)

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

    def _visualize_modulation_field(self) -> Optional[Dict[str, ParameterMetrics]]:
        """
        Create visualization comparing ground truth vs learned parameters.

        Returns:
            Dictionary of parameter metrics for each model (or None if no parameters)
        """
        if self.ground_truth_params is None:
            logger.info("No parameters to visualize")
            return None

        if self.learned_params is None and self.learned_params_hybrid is None:
            logger.info("No learned parameters to visualize")
            return None

        logger.info("Creating parameter visualizations...")

        # Compute parameter metrics
        param_metrics = {}

        # Get parameter names
        scalar_param_names = self.physical_model.scalar_param_names
        field_param_names = self.physical_model.field_param_names

        # Compute metrics for each model
        if self.learned_params is not None:
            param_metrics['Physical Only'] = self.metrics_computer.compute_parameter_metrics(
                self.learned_params,
                self.ground_truth_params,
                scalar_param_names=scalar_param_names,
                field_param_names=field_param_names
            )
            logger.info("Parameter metrics (Physical Only):")
            self._log_parameter_metrics(param_metrics['Physical Only'])

        if self.learned_params_hybrid is not None:
            param_metrics['Hybrid Physical'] = self.metrics_computer.compute_parameter_metrics(
                self.learned_params_hybrid,
                self.ground_truth_params,
                scalar_param_names=scalar_param_names,
                field_param_names=field_param_names
            )
            logger.info("Parameter metrics (Hybrid Physical):")
            self._log_parameter_metrics(param_metrics['Hybrid Physical'])

        # Visualize scalar parameters
        if scalar_param_names:
            logger.info(f"Visualizing scalar parameters: {scalar_param_names}")
            self._visualize_scalar_parameters(scalar_param_names)

        # Visualize field parameters
        if field_param_names:
            logger.info(f"Visualizing field parameters: {field_param_names}")
            self._visualize_field_parameters(field_param_names)

        return param_metrics

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

    def _evaluate_simulation(self, sim_idx: int) -> Dict[str, TrajectoryMetrics]:
        """
        Evaluate a single simulation with all available models.

        Args:
            sim_idx: Simulation index

        Returns:
            Dictionary mapping model names to TrajectoryMetrics
        """
        # Load real data
        real_data = self._load_real_trajectory(sim_idx)

        # Generate predictions from all available models
        predictions = {}
        metrics = {}

        if self.model is not None:
            # Disable training mode for evaluation
            self.model.training = False

            predictions['Synthetic'] = self._generate_predictions(self.model, real_data)

            # Compute metrics
            metrics['Synthetic'] = self.metrics_computer.compute_trajectory_metrics(
                predictions['Synthetic'],
                real_data,
                compute_per_field=True
            )

            logger.info(f"Metrics for Synthetic model (sim {sim_idx}):")
            self._log_trajectory_metrics(metrics['Synthetic'])

        if self.model_hybrid is not None:
            # Disable training mode for evaluation
            self.model_hybrid.training = False

            predictions['Hybrid Synthetic'] = self._generate_predictions(self.model_hybrid, real_data)

            # Compute metrics
            metrics['Hybrid Synthetic'] = self.metrics_computer.compute_trajectory_metrics(
                predictions['Hybrid Synthetic'],
                real_data,
                compute_per_field=True
            )

            logger.info(f"Metrics for Hybrid Synthetic model (sim {sim_idx}):")
            self._log_trajectory_metrics(metrics['Hybrid Synthetic'])

        # Create visualizations
        self._create_comparison_animation(sim_idx, real_data, predictions)

        # Create heatmap visualizations for 1D problems
        self._create_spacetime_heatmaps(sim_idx, real_data, predictions)

        # Save metrics for this simulation
        self._save_simulation_metrics(sim_idx, metrics)

        return metrics

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

    def _create_spacetime_heatmaps(
        self,
        sim_idx: int,
        real_data: Tensor,
        predictions: Dict[str, Tensor]
    ):
        """
        Store data for consolidated spacetime heatmap (called per simulation).
        The actual heatmap is created in _create_consolidated_spacetime_heatmap.
        """
        # Check if this is a 1D problem
        spatial_dims = real_data.shape.spatial
        if spatial_dims.rank != 1:
            return

        # Initialize storage if not exists
        if not hasattr(self, '_heatmap_data'):
            self._heatmap_data = {}

        for field_name in self.dynamic_fields:
            if field_name not in self._heatmap_data:
                self._heatmap_data[field_name] = {}

            # Extract field data and convert to numpy arrays [time, space]
            real_field = real_data.field[field_name]
            spatial_dim = real_field.shape.spatial.names[0]

            # Get data as numpy arrays using reshaped_native
            data_arrays = {}
            data_arrays['Real'] = math.reshaped_native(real_field, ['time', spatial_dim])
            if hasattr(data_arrays['Real'], 'detach'):
                data_arrays['Real'] = data_arrays['Real'].detach().cpu().numpy()

            for model_name in ['Hybrid Synthetic', 'Synthetic']:
                if model_name in predictions:
                    pred_field = predictions[model_name].field[field_name]
                    data_arrays[model_name] = math.reshaped_native(pred_field, ['time', spatial_dim])
                    if hasattr(data_arrays[model_name], 'detach'):
                        data_arrays[model_name] = data_arrays[model_name].detach().cpu().numpy()

            self._heatmap_data[field_name][sim_idx] = data_arrays

    def _create_consolidated_spacetime_heatmap(self, train_sim_idx: int = 0):
        """
        Create consolidated spacetime heatmap showing multiple simulations side by side.

        Layout: 5 columns (Seen Data + 4 Unseen Data) x 3 rows (Real, HYCO Synthetic, Synthetic)
        """
        if not hasattr(self, '_heatmap_data') or not self._heatmap_data:
            logger.info("No heatmap data available for consolidated plot")
            return

        for field_name, sim_data in self._heatmap_data.items():
            logger.info(f"Creating consolidated spacetime heatmap for field: {field_name}")

            # Get all simulation indices and sort them
            sim_indices = sorted(sim_data.keys())
            if not sim_indices:
                continue

            # Separate seen (training) and unseen (test) data
            seen_idx = train_sim_idx if train_sim_idx in sim_indices else sim_indices[0]
            unseen_indices = [idx for idx in sim_indices if idx != seen_idx][:4]  # Take up to 4 unseen

            # Build column order: Seen first, then Unseen
            col_indices = [seen_idx] + unseen_indices
            num_cols = len(col_indices)

            # Determine model names from first simulation
            first_sim_data = sim_data[col_indices[0]]
            model_names = ['Real']
            if 'Hybrid Synthetic' in first_sim_data:
                model_names.append('Hybrid Synthetic')
            if 'Synthetic' in first_sim_data:
                model_names.append('Synthetic')
            num_rows = len(model_names)

            # Create figure
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), squeeze=False)

            # Determine global color scale across all data
            all_arrays = []
            for idx in col_indices:
                for arr in sim_data[idx].values():
                    all_arrays.append(arr)
            vmin = min(arr.min() for arr in all_arrays)
            vmax = max(arr.max() for arr in all_arrays)

            cmap = 'viridis_r'

            # Plot each column (simulation)
            for col, sim_idx in enumerate(col_indices):
                sim_arrays = sim_data[sim_idx]

                # Column title
                if col == 0:
                    col_title = 'Seen Data'
                else:
                    col_title = f'Unseen {col}'

                # Plot each row (model)
                for row, model_name in enumerate(model_names):
                    ax = axes[row, col]
                    arr = sim_arrays[model_name]

                    im = ax.imshow(
                        arr.T, aspect='auto', origin='lower',
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        extent=[0, arr.shape[0] - 1, 0, arr.shape[1] - 1]
                    )

                    # Title only on first row
                    if row == 0:
                        ax.set_title(col_title, fontsize=20, fontweight='bold')

                    # Y-label only on first column
                    if col == 0:
                        if model_name == 'Hybrid Synthetic':
                            label = 'HYCO\nSynthetic'
                        else:
                            label = model_name
                        ax.set_ylabel(label, fontsize=18, fontweight='bold', rotation=0, ha='right', va='center')
                    else:
                        ax.set_yticks([])

                    # X-label only on last row
                    if row == num_rows - 1:
                        ax.set_xlabel('Time Step', fontsize=18, fontweight='bold')
                    else:
                        ax.set_xticks([])

            # Add single colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.ax.tick_params(labelsize=14)

            plt.tight_layout(rect=[0, 0, 0.91, 1])
            output_path = self.output_dir / f'{field_name}_spacetime_heatmap_comparison.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved consolidated spacetime heatmap to {output_path}")

    # =========================================================================
    # Metrics logging, saving, and visualization
    # =========================================================================

    def _log_trajectory_metrics(self, metrics: TrajectoryMetrics):
        """Log trajectory metrics to console."""
        logger.info(f"  L2 error (mean): {metrics.l2_error_mean:.6e}")
        logger.info(f"  Relative L2 error (mean): {metrics.relative_l2_error_mean:.6f}")
        logger.info(f"  L1 error (mean): {metrics.l1_error_mean:.6e}")
        logger.info(f"  L∞ error (mean): {metrics.linf_error_mean:.6e}")
        logger.info(f"  L2 error (final): {metrics.l2_error_final:.6e}")

        if metrics.field_l2_errors:
            logger.info("  Per-field L2 errors:")
            for field_name, error in metrics.field_l2_errors.items():
                logger.info(f"    {field_name}: {error:.6e}")

    def _log_parameter_metrics(self, metrics: ParameterMetrics):
        """Log parameter metrics to console."""
        if metrics.scalar_l2_error:
            logger.info("  Scalar parameter errors:")
            for param_name, error in metrics.scalar_l2_error.items():
                rel_error = metrics.scalar_relative_error.get(param_name, 0.0)
                logger.info(f"    {param_name}: L2={error:.6e}, Relative={rel_error:.6f}")

        if metrics.field_l2_error:
            logger.info("  Field parameter errors:")
            for param_name, error in metrics.field_l2_error.items():
                rel_error = metrics.field_relative_l2_error.get(param_name, 0.0)
                logger.info(f"    {param_name}: L2={error:.6e}, Relative={rel_error:.6f}")

    def _save_simulation_metrics(self, sim_idx: int, metrics: Dict[str, TrajectoryMetrics]):
        """Save metrics for a single simulation to JSON."""
        metrics_dict = {}
        for model_name, model_metrics in metrics.items():
            metrics_dict[model_name] = model_metrics.to_dict()

        output_path = self.output_dir / f'sim_{sim_idx:04d}_metrics.json'
        with open(output_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        logger.info(f"Saved metrics to {output_path}")

    def _save_aggregated_metrics(
        self,
        all_metrics: Dict[str, Dict[str, TrajectoryMetrics]],
        param_metrics: Optional[Dict[str, ParameterMetrics]]
    ):
        """Save aggregated metrics across all simulations."""
        # Aggregate trajectory metrics
        aggregated = {}

        for sim_name, sim_metrics in all_metrics.items():
            aggregated[sim_name] = {}
            for model_name, metrics in sim_metrics.items():
                aggregated[sim_name][model_name] = metrics.to_dict()

        # Add parameter metrics if available
        if param_metrics:
            aggregated['parameter_recovery'] = {}
            for model_name, metrics in param_metrics.items():
                aggregated['parameter_recovery'][model_name] = metrics.to_dict()

        # Compute average metrics across simulations
        if all_metrics:
            model_names = list(next(iter(all_metrics.values())).keys())
            aggregated['average_across_simulations'] = {}

            for model_name in model_names:
                # Collect metrics for this model across all sims
                model_metrics_list = [
                    sim_metrics[model_name]
                    for sim_metrics in all_metrics.values()
                    if model_name in sim_metrics
                ]

                if model_metrics_list:
                    avg_metrics = {
                        'l2_error_mean': np.mean([m.l2_error_mean for m in model_metrics_list]),
                        'relative_l2_error_mean': np.mean([m.relative_l2_error_mean for m in model_metrics_list]),
                        'l1_error_mean': np.mean([m.l1_error_mean for m in model_metrics_list]),
                        'linf_error_mean': np.mean([m.linf_error_mean for m in model_metrics_list]),
                        'l2_error_final': np.mean([m.l2_error_final for m in model_metrics_list]),
                    }

                    aggregated['average_across_simulations'][model_name] = avg_metrics

                    logger.info(f"Average metrics for {model_name}:")
                    logger.info(f"  L2 error (mean): {avg_metrics['l2_error_mean']:.6e}")
                    logger.info(f"  Relative L2 error (mean): {avg_metrics['relative_l2_error_mean']:.6f}")
                    logger.info(f"  L1 error (mean): {avg_metrics['l1_error_mean']:.6e}")

        # Save to JSON
        output_path = self.output_dir / 'all_metrics.json'
        with open(output_path, 'w') as f:
            json.dump(aggregated, f, indent=2)

        logger.info(f"Saved aggregated metrics to {output_path}")

    def _visualize_aggregated_metrics(self, all_metrics: Dict[str, Dict[str, TrajectoryMetrics]]):
        """Create visualizations for aggregated metrics."""
        if not all_metrics:
            return

        # Get model names
        model_names = list(next(iter(all_metrics.values())).keys())

        # Create temporal evolution plots for each model
        for model_name in model_names:
            self._plot_temporal_evolution(all_metrics, model_name)

        # Create comparison plots across models
        self._plot_model_comparison(all_metrics, model_names)

        # Create spatial error comparison plot (training vs test)
        train_sim_idx = self.eval_config.get('train_sim', [0])[0] if isinstance(self.eval_config.get('train_sim', 0), list) else self.eval_config.get('train_sim', 0)
        self._plot_spatial_error_comparison(all_metrics, train_sim_idx)

    def _plot_temporal_evolution(self, all_metrics: Dict[str, Dict[str, TrajectoryMetrics]], model_name: str):
        """Plot error evolution over time for a specific model."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Error Evolution Over Time - {model_name}', fontsize=16, fontweight='bold')

        # Collect metrics across all simulations
        for sim_name, sim_metrics in all_metrics.items():
            if model_name not in sim_metrics:
                continue

            metrics = sim_metrics[model_name]
            timesteps = np.arange(len(metrics.l2_error_spatial))

            # L2 error
            axes[0, 0].plot(timesteps, metrics.l2_error_spatial, label=sim_name, alpha=0.7)
            axes[0, 0].set_xlabel('Timestep')
            axes[0, 0].set_ylabel('L2 Error')
            axes[0, 0].set_title('Spatial L2 Error')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend(fontsize=8)

            # Relative L2 error
            axes[0, 1].plot(timesteps, metrics.relative_l2_error, label=sim_name, alpha=0.7)
            axes[0, 1].set_xlabel('Timestep')
            axes[0, 1].set_ylabel('Relative L2 Error')
            axes[0, 1].set_title('Relative L2 Error')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend(fontsize=8)

            # L1 error
            axes[1, 0].plot(timesteps, metrics.l1_error_spatial, label=sim_name, alpha=0.7)
            axes[1, 0].set_xlabel('Timestep')
            axes[1, 0].set_ylabel('L1 Error (MAE)')
            axes[1, 0].set_title('Spatial L1 Error')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend(fontsize=8)

            # L∞ error
            axes[1, 1].plot(timesteps, metrics.linf_error_spatial, label=sim_name, alpha=0.7)
            axes[1, 1].set_xlabel('Timestep')
            axes[1, 1].set_ylabel('L∞ Error (Max)')
            axes[1, 1].set_title('Spatial L∞ Error')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend(fontsize=8)

        plt.tight_layout()
        output_path = self.output_dir / f'temporal_evolution_{model_name.replace(" ", "_")}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved temporal evolution plot to {output_path}")

    def _plot_model_comparison(self, all_metrics: Dict[str, Dict[str, TrajectoryMetrics]], model_names: List[str]):
        """Create bar plots comparing different models."""
        if len(model_names) < 2:
            return  # Need at least 2 models to compare

        # Collect average metrics for each model
        avg_l2_errors = []
        avg_rel_l2_errors = []
        avg_l1_errors = []

        for model_name in model_names:
            model_metrics_list = [
                sim_metrics[model_name]
                for sim_metrics in all_metrics.values()
                if model_name in sim_metrics
            ]

            if model_metrics_list:
                avg_l2_errors.append(np.mean([m.l2_error_mean for m in model_metrics_list]))
                avg_rel_l2_errors.append(np.mean([m.relative_l2_error_mean for m in model_metrics_list]))
                avg_l1_errors.append(np.mean([m.l1_error_mean for m in model_metrics_list]))
            else:
                avg_l2_errors.append(0)
                avg_rel_l2_errors.append(0)
                avg_l1_errors.append(0)

        # Create bar plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Model Comparison (Averaged Across Simulations)', fontsize=16, fontweight='bold')

        x = np.arange(len(model_names))
        width = 0.6

        # L2 error
        axes[0].bar(x, avg_l2_errors, width, color='#2E86AB', alpha=0.8)
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('L2 Error')
        axes[0].set_title('Average L2 Error')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names, rotation=15, ha='right')
        axes[0].grid(axis='y', alpha=0.3)

        # Relative L2 error
        axes[1].bar(x, avg_rel_l2_errors, width, color='#E94F37', alpha=0.8)
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Relative L2 Error')
        axes[1].set_title('Average Relative L2 Error')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(model_names, rotation=15, ha='right')
        axes[1].grid(axis='y', alpha=0.3)

        # L1 error
        axes[2].bar(x, avg_l1_errors, width, color='#44AF69', alpha=0.8)
        axes[2].set_xlabel('Model')
        axes[2].set_ylabel('L1 Error (MAE)')
        axes[2].set_title('Average L1 Error')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(model_names, rotation=15, ha='right')
        axes[2].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / 'model_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved model comparison plot to {output_path}")

    def _plot_spatial_error_comparison(self, all_metrics: Dict[str, Dict[str, TrajectoryMetrics]], train_sim_idx: int = 0):
        """
        Create comparison plot of spatial errors for training and test data.

        Plot 1 (top): Relative L2 error over time for training data
        Plot 2 (bottom): Relative L2 error over time for test data (mean ± 1 std)

        Args:
            all_metrics: Dictionary mapping sim names to model metrics
            train_sim_idx: Simulation index to use as training data
        """
        # Get model names
        model_names = list(next(iter(all_metrics.values())).keys())

        if len(model_names) < 1:
            logger.warning("Need at least 1 model for spatial error comparison")
            return

        # Define colors for models (matching the style from the provided code)
        color_map = {
            'Synthetic': '#E94F37',  # Red/Orange
            'Hybrid Synthetic': '#2E86AB',  # Blue
            'NN': '#44AF69',  # Green
            'HYCO Synthetic': '#2E86AB',  # Blue
            'HYCO Physical': '#44AF69',  # Green
        }

        # Create figure with 2 subplots stacked vertically
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # =====================================================================
        # Plot 1: Training data
        # =====================================================================
        train_sim_name = f'sim_{train_sim_idx:04d}'
        if train_sim_name in all_metrics:
            ax = axes[0]

            for model_name in model_names:
                if model_name not in all_metrics[train_sim_name]:
                    continue

                metrics = all_metrics[train_sim_name][model_name]
                timesteps = np.arange(len(metrics.relative_l2_error))

                color = color_map.get(model_name, '#44AF69')
                ax.plot(timesteps, metrics.relative_l2_error,
                       label=model_name, linewidth=2.5, color=color)

            ax.set_xlabel('Time Step', fontsize=18, fontweight='bold')
            ax.set_ylabel('Relative L2 Error', fontsize=18, fontweight='bold')
            ax.set_title('Train Data', fontsize=20, fontweight='bold')
            ax.set_xlim(0, len(timesteps) - 1)
            ax.legend(fontsize=14, loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=14)
        else:
            logger.warning(f"Training simulation {train_sim_name} not found in metrics")

        # =====================================================================
        # Plot 2: Test data (average with std dev)
        # =====================================================================
        ax = axes[1]

        # Collect metrics for each model across test simulations
        for model_name in model_names:
            # Collect relative L2 errors for all test sims (excluding training sim)
            rel_l2_errors_list = []

            for sim_name, sim_metrics in all_metrics.items():
                if sim_name == train_sim_name:
                    continue  # Skip training sim

                if model_name not in sim_metrics:
                    continue

                rel_l2_errors_list.append(sim_metrics[model_name].relative_l2_error)

            if not rel_l2_errors_list:
                logger.warning(f"No test data found for model {model_name}")
                continue

            # Convert to numpy array (shape: [n_sims, n_timesteps])
            rel_l2_array = np.array(rel_l2_errors_list)

            # Compute mean and std dev across simulations
            mean_error = np.mean(rel_l2_array, axis=0)
            std_error = np.std(rel_l2_array, axis=0)

            timesteps = np.arange(len(mean_error))

            color = color_map.get(model_name, '#44AF69')

            # Plot mean line
            ax.plot(timesteps, mean_error, label=model_name,
                   linewidth=2.5, color=color)

            # Add shaded area for ±1 std dev
            ax.fill_between(timesteps,
                           mean_error - std_error,
                           mean_error + std_error,
                           alpha=0.2, color=color)

        ax.set_xlabel('Time Step', fontsize=18, fontweight='bold')
        ax.set_ylabel('Relative L2 Error', fontsize=18, fontweight='bold')
        ax.set_title('Test Data', fontsize=20, fontweight='bold')
        ax.set_xlim(0, len(timesteps) - 1)
        ax.legend(fontsize=14, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=14)

        plt.tight_layout()
        output_path = self.output_dir / 'spatial_error_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved spatial error comparison plot to {output_path}")