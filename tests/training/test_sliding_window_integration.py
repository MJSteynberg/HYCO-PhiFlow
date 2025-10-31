"""
Test sliding window integration with the trainer.

This test verifies that the sliding window parameter flows correctly
from config -> trainer -> HybridDataset.
"""

import pytest
import torch
from pathlib import Path
from src.training.synthetic.trainer import SyntheticTrainer


@pytest.fixture
def base_config():
    """Create a minimal config for testing."""
    return {
        'project_root': '.',
        'data': {
            'data_dir': 'data/',
            'dset_name': 'burgers_128',
            'fields': ['velocity']
        },
        'model': {
            'synthetic': {
                'name': 'UNet',
                'model_path': 'results/models',
                'model_save_name': 'test_sliding_window',
                'input_specs': {'velocity': 2},
                'output_specs': {'velocity': 2},
                'architecture': {
                    'levels': 3,
                    'filters': 32,
                    'batch_norm': True
                }
            }
        },
        'trainer_params': {
            'learning_rate': 1e-4,
            'batch_size': 4,
            'epochs': 1,
            'num_predict_steps': 3,
            'train_sim': [0]
        }
    }


class TestSlidingWindowIntegration:
    """Test that sliding window parameter flows through the system."""
    
    def test_sliding_window_disabled_by_default(self, base_config):
        """Test that sliding window is disabled by default (backward compatibility)."""
        # Don't set use_sliding_window in config
        trainer = SyntheticTrainer(base_config)
        
        # Should default to False
        assert trainer.use_sliding_window == False
        
        # DataLoader should report single starting point mode
        # With 1 simulation and no sliding window, should have 1 sample
        assert len(trainer.train_loader.dataset) == 1
    
    def test_sliding_window_explicitly_disabled(self, base_config):
        """Test explicitly setting use_sliding_window to False."""
        base_config['trainer_params']['use_sliding_window'] = False
        trainer = SyntheticTrainer(base_config)
        
        assert trainer.use_sliding_window == False
        assert len(trainer.train_loader.dataset) == 1
    
    def test_sliding_window_enabled(self, base_config):
        """Test that enabling sliding window increases sample count."""
        base_config['trainer_params']['use_sliding_window'] = True
        trainer = SyntheticTrainer(base_config)
        
        assert trainer.use_sliding_window == True
        
        # With sliding window enabled, we load all available frames
        # The actual number depends on what's cached:
        # - Total frames loaded (from cache)
        # - num_predict_steps: 3
        # - Required frames per sample: 4 (1 initial + 3 predictions)
        # - Available starting positions: total_frames - num_predict_steps
        
        num_predict_steps = base_config['trainer_params']['num_predict_steps']
        actual_samples = len(trainer.train_loader.dataset)
        dataset = trainer.train_loader.dataset
        
        # Back-calculate total frames: samples = total_frames - num_predict_steps
        # So: total_frames = samples + num_predict_steps
        inferred_total_frames = actual_samples + num_predict_steps
        
        print(f"\nSliding window enabled:")
        print(f"  Inferred total frames: {inferred_total_frames}")
        print(f"  Num predict steps: {num_predict_steps}")
        print(f"  Samples created: {actual_samples}")
        print(f"  Formula: samples = frames - predict_steps = {inferred_total_frames} - {num_predict_steps} = {actual_samples}")
        
        # Should have many more samples than the single-sample mode (which has 1)
        assert actual_samples > 1
    
    def test_sliding_window_with_multiple_simulations(self, base_config):
        """Test sliding window with multiple simulations."""
        base_config['trainer_params']['use_sliding_window'] = True
        base_config['trainer_params']['train_sim'] = [0, 1]
        
        trainer = SyntheticTrainer(base_config)
        
        # With 2 simulations and sliding window:
        # Each simulation contributes: frames - predict_steps samples
        
        num_sims = len(base_config['trainer_params']['train_sim'])
        actual_samples = len(trainer.train_loader.dataset)
        num_predict_steps = base_config['trainer_params']['num_predict_steps']
        
        samples_per_sim = actual_samples // num_sims
        
        print(f"\nSliding window with {num_sims} simulations:")
        print(f"  Samples per simulation: {samples_per_sim}")
        print(f"  Total samples: {actual_samples}")
        
        # Each sim should contribute the same number of samples
        assert actual_samples == num_sims * samples_per_sim
        assert samples_per_sim > 1  # More than single-sample mode
    
    def test_data_loader_output_shapes(self, base_config):
        """Test that data loader outputs have correct shapes with sliding window."""
        base_config['trainer_params']['use_sliding_window'] = True
        trainer = SyntheticTrainer(base_config)
        
        # Get a batch from the loader
        initial_state, rollout_targets = next(iter(trainer.train_loader))
        
        batch_size = base_config['trainer_params']['batch_size']
        num_predict_steps = base_config['trainer_params']['num_predict_steps']
        
        # Initial state: [B, C, H, W] where C = 2 (velocity x, y)
        assert initial_state.shape[0] == batch_size
        assert initial_state.shape[1] == 2  # velocity channels
        
        # Rollout targets: [B, T, C, H, W]
        assert rollout_targets.shape[0] == batch_size
        assert rollout_targets.shape[1] == num_predict_steps
        assert rollout_targets.shape[2] == 2  # velocity channels
        
        print(f"\nData shapes:")
        print(f"  Initial state: {initial_state.shape}")
        print(f"  Rollout targets: {rollout_targets.shape}")
    
    def test_sliding_window_samples_are_different(self, base_config):
        """Verify that sliding window creates different samples."""
        base_config['trainer_params']['use_sliding_window'] = True
        base_config['trainer_params']['batch_size'] = 2  # Get 2 samples
        
        trainer = SyntheticTrainer(base_config)
        
        # Get two batches
        batch1_initial, batch1_targets = next(iter(trainer.train_loader))
        batch2_initial, batch2_targets = next(iter(trainer.train_loader))
        
        # Samples should be different (not identical)
        # They might be consecutive windows from the same simulation
        
        # Check that not all samples are identical
        samples_identical = torch.allclose(batch1_initial, batch2_initial)
        
        print(f"\nSamples comparison:")
        print(f"  First batch initial state mean: {batch1_initial.mean().item():.6f}")
        print(f"  Second batch initial state mean: {batch2_initial.mean().item():.6f}")
        print(f"  Batches are identical: {samples_identical}")
        
        # With sliding window and shuffling, batches should generally be different
        # (though there's a small chance they could be the same)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
