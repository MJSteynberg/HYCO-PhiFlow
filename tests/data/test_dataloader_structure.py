"""
Test to visualize what training samples the dataloader creates.

This answers the question: Do we create all pairs of input/output,
or just input at t=0 and the next few for output?
"""

import pytest
import torch
from pathlib import Path

from src.data import DataManager, HybridDataset


class TestDataLoaderStructure:
    """Visualize the structure of training samples."""

    def test_what_samples_are_created(self):
        """Show exactly what samples the dataloader creates."""
        print("\n" + "=" * 70)
        print("DATALOADER SAMPLE STRUCTURE")
        print("=" * 70)

        # Setup
        raw_data_dir = Path("data/smoke_128")
        cache_dir = Path("data/cache")

        data_manager = DataManager(
            raw_data_dir=str(raw_data_dir),
            cache_dir=str(cache_dir),
            config={"dset_name": "smoke_128"},
        )

        # Create dataset with multiple simulations
        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=[0, 1],  # Use 2 simulations
            field_names=["density", "velocity", "inflow"],
            num_frames=10,  # Load 10 frames total
            num_predict_steps=3,  # Predict 3 steps ahead
            dynamic_fields=["density", "velocity"],
            static_fields=["inflow"],
        )

        print(f"\nDataset Configuration:")
        print(f"  Simulations: {dataset.sim_indices}")
        print(f"  Total frames loaded per sim: {dataset.num_frames}")
        print(f"  Prediction steps: {dataset.num_predict_steps}")
        print(f"  Dataset length: {len(dataset)} samples")

        print(f"\n" + "-" * 70)
        print("CURRENT APPROACH: ONE SAMPLE PER SIMULATION")
        print("-" * 70)

        # Show what each sample looks like
        for idx in range(len(dataset)):
            initial_state, rollout_targets = dataset[idx]

            print(f"\nSample {idx} (Simulation {dataset.sim_indices[idx]}):")
            print(f"  Initial state (t=0): shape {initial_state.shape}")
            print(f"    - Contains: ALL fields (density + velocity + inflow)")
            print(f"    - Channels: 4 (1 + 2 + 1)")
            print(f"  ")
            print(f"  Rollout targets (t=1,2,3): shape {rollout_targets.shape}")
            print(f"    - Contains: DYNAMIC fields only (density + velocity)")
            print(f"    - Channels: 3 (1 + 2)")
            print(f"    - Timesteps: {rollout_targets.shape[0]}")

        print(f"\n" + "-" * 70)
        print("TRAINING PROCESS:")
        print("-" * 70)
        print(
            """
For each sample in a batch:
  1. Start with initial_state at t=0 [B, 4, H, W]
  2. Loop for num_predict_steps (3 steps):
     - Predict next state: model(current_state) -> [B, 4, H, W]
     - Extract dynamic fields for loss: [B, 3, H, W]
     - Compare with ground truth at this timestep
     - Update current_state = prediction (feed prediction back)
  3. Average loss over the 3 steps
        """
        )

        print(f"\n" + "-" * 70)
        print("KEY INSIGHT:")
        print("-" * 70)
        print(
            """
✗ We do NOT create multiple (input, output) pairs per simulation
✓ We create ONE sample per simulation:
  - Input: state at t=0
  - Output: states at t=1, t=2, t=3
  - Training uses AUTOREGRESSIVE rollout: prediction -> next input
        """
        )

    def test_alternative_approach_discussion(self):
        """Discuss alternative approaches for creating training samples."""
        print("\n" + "=" * 70)
        print("ALTERNATIVE APPROACHES (NOT CURRENTLY IMPLEMENTED)")
        print("=" * 70)

        print(
            """
APPROACH 1 (Current): Single Starting Point
───────────────────────────────────────────
- One sample per simulation
- Start at t=0, predict t=1,2,3
- Pros: Simple, true autoregressive training
- Cons: Limited training data, no variation in starting points

Simulation with 10 frames:
  [t0] -> [t1, t2, t3]
  ↑ ONE sample

APPROACH 2: Sliding Window
──────────────────────────
- Multiple samples per simulation
- Each timestep becomes a starting point
- Pros: More training data, robust to different states
- Cons: More complex, slower training

Simulation with 10 frames:
  [t0] -> [t1, t2, t3]
  [t1] -> [t2, t3, t4]
  [t2] -> [t3, t4, t5]
  ...
  [t6] -> [t7, t8, t9]
  ↑ 7 samples from one simulation!

APPROACH 3: Single-Step Prediction
───────────────────────────────────
- One sample per consecutive pair
- Pros: Maximum data, easier optimization
- Cons: Not autoregressive, may not generalize to multi-step

Simulation with 10 frames:
  [t0] -> [t1]
  [t1] -> [t2]
  [t2] -> [t3]
  ...
  [t8] -> [t9]
  ↑ 9 samples, but only single-step predictions
        """
        )

        print("\n" + "-" * 70)
        print("RECOMMENDATION:")
        print("-" * 70)
        print(
            """
For smoke simulation (where we want long rollouts):
  Current approach (Approach 1) is CORRECT for:
  - Learning autoregressive dynamics
  - Testing long-term stability
  - Matching evaluation conditions

To get more training data, you should:
  ✓ Generate MORE simulations (increase num_simulations)
  ✗ Don't use sliding window (changes training dynamics)
  
Current config: train_sim: [0,1,2,3,4,5,6,7,8,9] = 10 samples
Could expand to: train_sim: [0-99] = 100 samples
        """
        )

    def test_visualize_data_flow_in_training(self):
        """Show the exact data flow during one training iteration."""
        print("\n" + "=" * 70)
        print("DATA FLOW IN ONE TRAINING ITERATION")
        print("=" * 70)

        # Setup
        raw_data_dir = Path("data/smoke_128")
        cache_dir = Path("data/cache")

        data_manager = DataManager(
            raw_data_dir=str(raw_data_dir),
            cache_dir=str(cache_dir),
            config={"dset_name": "smoke_128"},
        )

        dataset = HybridDataset(
            data_manager=data_manager,
            sim_indices=[0],
            field_names=["density", "velocity", "inflow"],
            num_frames=10,
            num_predict_steps=3,
            dynamic_fields=["density", "velocity"],
            static_fields=["inflow"],
        )

        # Get one sample
        initial_state, rollout_targets = dataset[0]

        print("\n1. DATALOADER PROVIDES:")
        print(f"   initial_state: {initial_state.shape}")
        print(f"   rollout_targets: {rollout_targets.shape}")

        print("\n2. TRAINING LOOP (simplified):")
        print(f"   current_state = initial_state  # {initial_state.shape}")
        print(f"   ")
        print(f"   for t_step in range(3):  # num_predict_steps")
        print(f"       prediction = model(current_state)  # {initial_state.shape}")
        print(f"       pred_dynamic = extract_dynamic(prediction)  # [B, 3, H, W]")
        print(f"       gt_this_step = rollout_targets[:, t_step]  # [B, 3, H, W]")
        print(f"       ")
        print(f"       loss += MSE(pred_dynamic, gt_this_step)")
        print(f"       ")
        print(f"       current_state = prediction  # Feed prediction back!")

        print("\n3. KEY POINT: AUTOREGRESSIVE ROLLOUT")
        print("   ─────────────────────────────────────")
        print("   t=0: model(state_0) -> predict_1")
        print("   t=1: model(predict_1) -> predict_2  ← Uses previous prediction!")
        print("   t=2: model(predict_2) -> predict_3  ← Uses previous prediction!")
        print("")
        print("   This is why errors ACCUMULATE:")
        print("   - Any error in predict_1 affects predict_2")
        print("   - Errors compound exponentially")
        print("   - Hence the explosion we saw in smoke evaluation!")

        print("\n4. GROUND TRUTH TARGETS:")
        print(f"   rollout_targets[0] = true state at t=1")
        print(f"   rollout_targets[1] = true state at t=2")
        print(f"   rollout_targets[2] = true state at t=3")
        print(f"   ")
        print(f"   These are the TRUE states from simulation,")
        print(f"   NOT from previous predictions.")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
