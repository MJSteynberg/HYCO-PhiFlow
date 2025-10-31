"""
Main Evaluator class for comprehensive model evaluation.

This module provides the Evaluator class that orchestrates the complete
evaluation workflow: loading models, running inference, computing metrics,
and generating all visualizations.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import DataManager, HybridDataset
from src.models.synthetic.unet import UNet
from .metrics import compute_all_metrics, compute_metrics_per_field, aggregate_metrics
from .visualizations import (
    create_comparison_gif,
    create_comparison_gif_from_specs,
    plot_error_vs_time,
    plot_error_vs_time_multi_field,
    plot_keyframe_comparison,
    plot_keyframe_comparison_multi_field,
    plot_error_heatmap,
    create_evaluation_summary,
)


class Evaluator:
    """
    Main evaluator class for comprehensive model evaluation.
    
    This class orchestrates the complete evaluation workflow:
    1. Load trained synthetic model
    2. Run inference on test simulations
    3. Compute error metrics
    4. Generate all visualizations (animations, plots, keyframes)
    5. Save organized results with JSON summaries
    
    Supports both single and multi-field evaluation with automatic
    organization of results by simulation.
    
    Attributes:
        config: Configuration dictionary
        device: PyTorch device (CPU or CUDA)
        model: Loaded synthetic model
        data_manager: DataManager for loading test data
        field_specs: Dictionary mapping field names to channel counts
        output_specs: Model output specifications
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Evaluator from configuration.
        
        Args:
            config: Configuration dictionary containing:
                - data: Data configuration (fields, dataset name, etc.)
                - model/synthetic: Model configuration (architecture, paths, specs)
                - evaluation_params: Evaluation parameters (test_sim, metrics, etc.)
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"\n{'='*60}")
        print("INITIALIZING EVALUATOR")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        
        # Extract configuration
        self.data_config = config['data']
        self.model_config = config['model']['synthetic']
        self.eval_config = config.get('evaluation_params', {})
        
        # Field specifications
        self.input_specs = self.model_config['input_specs']
        self.output_specs = self.model_config['output_specs']
        self.field_names = self.data_config['fields']
        
        # Evaluation parameters
        self.test_sim = self.eval_config.get('test_sim', [0])
        self.num_frames = self.eval_config.get('num_frames', 51)
        self.metrics_to_compute = self.eval_config.get('metrics', ['mse', 'mae', 'rmse'])
        self.num_keyframes = self.eval_config.get('keyframe_count', 5)
        self.animation_fps = self.eval_config.get('animation_fps', 10)
        self.save_animations = self.eval_config.get('save_animations', True)
        self.save_plots = self.eval_config.get('save_plots', True)
        
        # Initialize components
        self.model = None
        self.data_manager = None
        
        print(f"Test simulations: {self.test_sim}")
        print(f"Evaluation frames: {self.num_frames}")
        print(f"Metrics to compute: {self.metrics_to_compute}")
        print(f"{'='*60}\n")
    
    def load_model(self) -> nn.Module:
        """
        Load the trained synthetic model.
        
        Returns:
            Loaded model in evaluation mode
            
        Raises:
            FileNotFoundError: If model checkpoint doesn't exist
        """
        print("Loading trained model...")
        
        # Get checkpoint path
        model_path_dir = self.model_config['model_path']
        model_save_name = self.model_config['model_save_name']
        checkpoint_path = os.path.join(model_path_dir, f"{model_save_name}.pth")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at {checkpoint_path}. "
                f"Please train the model first."
            )
        
        # Create model architecture
        model_name = self.model_config['name']
        
        if model_name == 'UNet':
            # UNet expects full config dictionary
            model = UNet(self.model_config)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Checkpoint is the state dict directly
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        print(f"  [OK] Model loaded from {checkpoint_path}")
        print(f"  [OK] Model architecture: {model_name}")
        print(f"  [OK] Input channels: {model.in_channels}")
        print(f"  [OK] Output channels: {model.out_channels}")
        
        self.model = model
        return model
    
    def setup_data_manager(self) -> DataManager:
        """
        Set up DataManager for loading test data.
        
        Returns:
            Configured DataManager instance
        """
        print("\nSetting up data manager...")
        
        data_dir = self.data_config['data_dir']
        dset_name = self.data_config['dset_name']
        
        raw_data_dir = os.path.join(data_dir, dset_name)
        cache_dir = os.path.join(data_dir, 'cache', f'eval_{dset_name}')
        
        self.data_manager = DataManager(
            raw_data_dir=raw_data_dir,
            cache_dir=cache_dir,
            config=self.data_config
        )
        
        print(f"  [OK] Data manager ready")
        print(f"  [OK] Dataset: {dset_name}")
        print(f"  [OK] Fields: {self.field_names}")
        
        return self.data_manager
    
    def run_inference(
        self,
        sim_index: int,
        num_rollout_steps: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Run model inference on a simulation.
        
        Performs autoregressive rollout: use model output as next input.
        
        Args:
            sim_index: Index of simulation to evaluate
            num_rollout_steps: Number of autoregressive steps (default: num_frames - 1)
            
        Returns:
            Dictionary with:
                - 'prediction': Model predictions [T, C, H, W]
                - 'ground_truth': True data [T, C, H, W]
                - 'initial_state': Initial condition [C, H, W]
        """
        print(f"\n  Running inference on simulation {sim_index}...")
        
        if num_rollout_steps is None:
            num_rollout_steps = self.num_frames - 1
        
        # Load data
        data = self.data_manager.get_or_load_simulation(
            sim_index=sim_index,
            field_names=self.field_names,
            num_frames=self.num_frames
        )
        
        ground_truth = data['tensor_data']
        
        # Concatenate all fields into single tensor along channel dimension
        # Handle fields with different channel counts (e.g., density=1, velocity=2)
        field_tensors = []
        for field_name in self.field_names:
            field_tensor = ground_truth[field_name]  # [T, C, H, W]
            field_tensors.append(field_tensor)
        
        gt_tensor = torch.cat(field_tensors, dim=1)  # Concatenate along channel dim [T, C_total, H, W]
        
        # Slice ground truth to match number of frames we're predicting
        # We predict num_frames total (including initial state)
        gt_tensor = gt_tensor[:self.num_frames]
        
        # Initial state (t=0)
        initial_state = gt_tensor[0:1].to(self.device)  # [1, C, H, W]
        
        # Autoregressive rollout
        predictions = [initial_state]
        current_state = initial_state
        
        with torch.no_grad():
            for step in range(num_rollout_steps):
                # Model predicts next state
                next_state = self.model(current_state)
                predictions.append(next_state)
                
                # Use prediction as next input
                current_state = next_state
        
        # Stack predictions
        prediction_tensor = torch.cat(predictions, dim=0)  # [T, C, H, W]
        
        print(f"    [OK] Rollout complete: {prediction_tensor.shape[0]} frames")
        print(f"    [OK] Prediction shape: {prediction_tensor.shape}")
        print(f"    [OK] Ground truth shape: {gt_tensor.shape}")
        
        return {
            'prediction': prediction_tensor.cpu(),
            'ground_truth': gt_tensor.cpu(),
            'initial_state': initial_state[0].cpu()
        }
    
    def compute_metrics(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Compute all error metrics.
        
        Args:
            prediction: Model predictions [T, C, H, W]
            ground_truth: True data [T, C, H, W]
            
        Returns:
            Dictionary with:
                - Per-field metrics
                - Aggregate statistics
                - Timestep-wise errors
        """
        print(f"  Computing metrics...")
        
        # Compute per-field metrics
        # Use input_specs to get ALL fields (including static ones)
        # because both prediction and ground_truth contain all fields
        field_specs = {}
        for field_name in self.field_names:
            if field_name in self.input_specs:
                field_specs[field_name] = self.input_specs[field_name]
        
        field_metrics = compute_metrics_per_field(
            prediction, ground_truth, field_specs, self.metrics_to_compute
        )
        
        # Compute aggregate statistics for each field/metric
        aggregates = {}
        for field_name, metrics in field_metrics.items():
            aggregates[field_name] = {}
            for metric_name, values in metrics.items():
                aggregates[field_name][metric_name] = aggregate_metrics(values)
        
        print(f"    [OK] Metrics computed for {len(field_specs)} fields")
        
        return {
            'field_metrics': field_metrics,
            'aggregates': aggregates,
            'field_specs': field_specs
        }
    
    def generate_visualizations(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        sim_index: int,
        save_dir: Union[str, Path]
    ) -> Dict[str, Dict[str, Path]]:
        """
        Generate all visualizations for a simulation.
        
        Args:
            prediction: Model predictions [T, C, H, W]
            ground_truth: True data [T, C, H, W]
            sim_index: Simulation index
            save_dir: Base directory for saving outputs
            
        Returns:
            Dictionary mapping visualization types to file paths
        """
        save_dir = Path(save_dir)
        print(f"  Generating visualizations...")
        
        saved_paths = {
            'animations': {},
            'error_plots': {},
            'keyframes': {},
            'heatmaps': {}
        }
        
        # Field specifications
        # Use input_specs to get ALL fields (including static ones)
        # because both prediction and ground_truth contain all fields
        field_specs = {}
        for field_name in self.field_names:
            if field_name in self.input_specs:
                field_specs[field_name] = self.input_specs[field_name]
        
        # 1. Animations
        if self.save_animations:
            print(f"    Creating animations...")
            anim_dir = save_dir / 'animations'
            paths = create_comparison_gif_from_specs(
                prediction, ground_truth, field_specs, anim_dir,
                fps=self.animation_fps, show_difference=True
            )
            saved_paths['animations'] = paths
        
        # 2. Error plots
        if self.save_plots:
            print(f"    Creating error plots...")
            plot_dir = save_dir / 'plots'
            paths = plot_error_vs_time_multi_field(
                prediction, ground_truth, field_specs, plot_dir,
                metrics=self.metrics_to_compute
            )
            saved_paths['error_plots'] = paths
        
        # 3. Keyframe comparisons
        if self.save_plots:
            print(f"    Creating keyframe comparisons...")
            keyframe_dir = save_dir / 'plots'
            paths = plot_keyframe_comparison_multi_field(
                prediction, ground_truth, field_specs, keyframe_dir,
                num_keyframes=self.num_keyframes,
                show_difference=True,
                show_metrics=True
            )
            saved_paths['keyframes'] = paths
        
        # 4. Error heatmaps (for multi-channel fields)
        if self.save_plots:
            heatmap_dir = save_dir / 'plots'
            channel_idx = 0
            for field_name, num_channels in field_specs.items():
                if num_channels > 1:
                    pred_field = prediction[:, channel_idx:channel_idx+num_channels, :, :]
                    gt_field = ground_truth[:, channel_idx:channel_idx+num_channels, :, :]
                    
                    heatmap_path = heatmap_dir / f"{field_name}_error_heatmap.png"
                    plot_error_heatmap(pred_field, gt_field, field_name, heatmap_path)
                    saved_paths['heatmaps'][field_name] = heatmap_path
                
                channel_idx += num_channels
        
        print(f"    [OK] All visualizations created")
        
        return saved_paths
    
    def save_metrics_to_json(
        self,
        metrics: Dict[str, Any],
        save_path: Union[str, Path]
    ) -> None:
        """
        Save metrics to JSON file.
        
        Args:
            metrics: Metrics dictionary
            save_path: Path to JSON file
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert torch tensors to lists for JSON serialization
        json_metrics = {}
        
        if 'aggregates' in metrics:
            json_metrics['aggregates'] = metrics['aggregates']
        
        if 'field_metrics' in metrics:
            json_metrics['per_timestep'] = {}
            for field_name, field_metrics in metrics['field_metrics'].items():
                json_metrics['per_timestep'][field_name] = {}
                for metric_name, values in field_metrics.items():
                    json_metrics['per_timestep'][field_name][metric_name] = \
                        values.detach().cpu().numpy().tolist()
        
        with open(save_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"    [OK] Metrics saved to {save_path}")
    
    def evaluate_simulation(
        self,
        sim_index: int,
        save_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Run complete evaluation on a single simulation.
        
        Args:
            sim_index: Index of simulation to evaluate
            save_dir: Directory to save results (auto-generated if None)
            
        Returns:
            Dictionary with results:
                - metrics: Computed error metrics
                - visualizations: Paths to generated visualizations
                - inference_results: Prediction and ground truth tensors
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING SIMULATION {sim_index}")
        print(f"{'='*60}")
        
        # Set up save directory
        if save_dir is None:
            dset_name = self.data_config['dset_name']
            model_name = self.model_config['model_save_name']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path('results') / 'evaluation' / f"{dset_name}_{model_name}" / f"sim_{sim_index:06d}"
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Run inference
        inference_results = self.run_inference(sim_index)
        prediction = inference_results['prediction']
        ground_truth = inference_results['ground_truth']
        
        # 2. Compute metrics
        metrics = self.compute_metrics(prediction, ground_truth)
        
        # 3. Generate visualizations
        visualizations = self.generate_visualizations(
            prediction, ground_truth, sim_index, save_dir
        )
        
        # 4. Save metrics to JSON
        metrics_dir = save_dir / 'metrics'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        self.save_metrics_to_json(metrics, metrics_dir / 'metrics_summary.json')
        
        # 5. Print summary
        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {save_dir}")
        print(f"\nMetric Summary:")
        for field_name, field_agg in metrics['aggregates'].items():
            print(f"\n  {field_name.upper()}:")
            for metric_name, stats in field_agg.items():
                print(f"    {metric_name.upper()}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
        print(f"{'='*60}\n")
        
        return {
            'metrics': metrics,
            'visualizations': visualizations,
            'inference_results': inference_results,
            'save_dir': save_dir
        }
    
    def evaluate(
        self,
        sim_indices: Optional[List[int]] = None,
        base_save_dir: Optional[Union[str, Path]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Run evaluation on multiple simulations.
        
        This is the main entry point for evaluation.
        
        Args:
            sim_indices: List of simulation indices (uses test_sim from config if None)
            base_save_dir: Base directory for all results
            
        Returns:
            Dictionary mapping sim_index to evaluation results
        """
        if sim_indices is None:
            sim_indices = self.test_sim
        
        print(f"\n{'='*60}")
        print(f"STARTING EVALUATION")
        print(f"{'='*60}")
        print(f"Simulations to evaluate: {sim_indices}")
        print(f"{'='*60}\n")
        
        # Load model
        if self.model is None:
            self.load_model()
        
        # Setup data manager
        if self.data_manager is None:
            self.setup_data_manager()
        
        # Evaluate each simulation
        all_results = {}
        
        for sim_idx in sim_indices:
            if base_save_dir is not None:
                sim_save_dir = Path(base_save_dir) / f"sim_{sim_idx:06d}"
            else:
                sim_save_dir = None
            
            results = self.evaluate_simulation(sim_idx, sim_save_dir)
            all_results[sim_idx] = results
        
        # Create summary across all simulations
        if len(sim_indices) > 1 and base_save_dir is not None:
            self._create_aggregate_summary(all_results, base_save_dir)
        
        print(f"\n{'='*60}")
        print(f"ALL EVALUATIONS COMPLETE")
        print(f"{'='*60}")
        print(f"Evaluated {len(sim_indices)} simulations")
        if base_save_dir is not None:
            print(f"Results saved to: {base_save_dir}")
        print(f"{'='*60}\n")
        
        return all_results
    
    def _create_aggregate_summary(
        self,
        all_results: Dict[int, Dict[str, Any]],
        base_save_dir: Union[str, Path]
    ) -> None:
        """
        Create aggregate summary across all evaluated simulations.
        
        Args:
            all_results: Results from all simulations
            base_save_dir: Base directory for saving summary
        """
        print(f"\nCreating aggregate summary...")
        
        base_save_dir = Path(base_save_dir)
        summary_dir = base_save_dir / 'summary'
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        # Aggregate metrics across simulations
        aggregate_metrics = {}
        
        for sim_idx, results in all_results.items():
            metrics = results['metrics']['aggregates']
            
            for field_name, field_metrics in metrics.items():
                if field_name not in aggregate_metrics:
                    aggregate_metrics[field_name] = {
                        metric: [] for metric in field_metrics.keys()
                    }
                
                for metric_name, stats in field_metrics.items():
                    aggregate_metrics[field_name][metric_name].append(stats['mean'])
        
        # Compute statistics across simulations
        summary = {}
        for field_name, field_data in aggregate_metrics.items():
            summary[field_name] = {}
            for metric_name, values in field_data.items():
                import numpy as np
                summary[field_name][metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        # Save aggregate summary
        with open(summary_dir / 'aggregate_metrics.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  [OK] Aggregate summary saved to {summary_dir}")
        print(f"\n  Aggregate Statistics Across {len(all_results)} Simulations:")
        for field_name, field_stats in summary.items():
            print(f"\n    {field_name.upper()}:")
            for metric_name, stats in field_stats.items():
                print(f"      {metric_name.upper()}: mean={stats['mean']:.6f} +/- {stats['std']:.6f}")
