#!/bin/bash
# =============================================================================
# Inviscid Burgers 2d Full Experiment
# =============================================================================
# This script runs a complete experiment comparing:
# 1. Synthetic model trained on real data only
# 2. Physical model trained on real data only  
# 3. Hybrid training (alternating physical and synthetic)
#
# After training, runs evaluation which generates comparison plots for:
# - Physical parameters (physical-only vs hybrid-physical vs ground truth)
# - Solution trajectories (synthetic-only vs hybrid-synthetic vs real)
# =============================================================================

set -e  # Exit on error

# Configuration
CONFIG_NAME="navier_stokes_2d"
RESULTS_DIR="results/experiments/navier_stokes_2d_comparison"

# Model checkpoint names
SYNTHETIC_ONLY="navier_stokes_synthetic_only_2d"
PHYSICAL_ONLY="navier_stokes_physical_only_2d"
HYBRID_SYNTHETIC="navier_stokes_hybrid_synthetic_2d"
HYBRID_PHYSICAL="navier_stokes_hybrid_physical_2d"

echo "=============================================="
echo "  Navier-Stokes 2d Comparison Experiment"
echo "=============================================="
echo ""
echo "Note: Synthetic-only training uses 'standalone_epochs' from config"
echo "      which equals: epochs * (cycles + warmup) for fair comparison"
echo ""

# -----------------------------------------------------------------------------
# Step 0: Generate data (if not already present)
# -----------------------------------------------------------------------------
echo "[Step 0/4] Checking/generating data..."
if [ ! -d "data/navier_stokes_2d" ] || [ -z "$(ls -A data/navier_stokes_2d 2>/dev/null)" ]; then
    echo "  Data not found, generating..."
    python run.py --config-name=${CONFIG_NAME} \
        general.tasks='[generate]'
else
    echo "  Data already exists, skipping generation."
fi
echo ""

# -----------------------------------------------------------------------------
# Step 1: Train synthetic model on real data only
# Uses standalone_epochs (= epochs * (cycles + warmup)) for fair comparison
# -----------------------------------------------------------------------------
echo "[Step 1/4] Training SYNTHETIC model on real data..."
python run.py --config-name=${CONFIG_NAME} \
    general.mode='synthetic' \
    general.tasks='[train]' \
    general.experiment_name='navier_stokes_2d_synthetic_only' \
    model.synthetic.model_save_name=${SYNTHETIC_ONLY} \
    trainer.synthetic.epochs='${trainer.synthetic.standalone_epochs}'

echo "  Synthetic-only model saved to results/models/${SYNTHETIC_ONLY}.pth"
echo ""

# -----------------------------------------------------------------------------
# Step 2: Train physical model on real data only
# Uses standalone_epochs (= epochs * cycles) for fair comparison
# -----------------------------------------------------------------------------
echo "[Step 2/4] Training PHYSICAL model on real data..."
python run.py --config-name=${CONFIG_NAME} \
    general.mode='physical' \
    general.tasks='[train]' \
    general.experiment_name='navier_stokes_2d_physical_only' \
    model.physical.model_save_name=${PHYSICAL_ONLY} \
    trainer.physical.epochs='${trainer.physical.standalone_epochs}'

echo "  Physical-only model saved to results/models/${PHYSICAL_ONLY}.npz"
echo ""

# -----------------------------------------------------------------------------
# Step 3: Hybrid training
# -----------------------------------------------------------------------------
echo "[Step 3/4] Running HYBRID training..."
python run.py --config-name=${CONFIG_NAME} \
    general.mode='hybrid' \
    general.tasks='[train]' \
    general.experiment_name='navier_stokes_2d_hybrid' \
    model.physical.model_save_name=${HYBRID_PHYSICAL} \
    model.synthetic.model_save_name=${HYBRID_SYNTHETIC}

echo "  Hybrid models saved:"
echo "    - results/models/${HYBRID_SYNTHETIC}.pth"
echo "    - results/models/${HYBRID_PHYSICAL}.npz"
echo ""

# -----------------------------------------------------------------------------
# Step 4: Run evaluation with all models
# -----------------------------------------------------------------------------
echo "[Step 4/4] Running evaluation with all models..."
python run.py --config-name=${CONFIG_NAME} \
    general.tasks='[evaluate]' \
    evaluation.output_dir=${RESULTS_DIR} \
    evaluation.synthetic_checkpoint="results/models/${SYNTHETIC_ONLY}.pth" \
    evaluation.synthetic_checkpoint_hybrid="results/models/${HYBRID_SYNTHETIC}.pth" \
    evaluation.physical_checkpoint="results/models/${PHYSICAL_ONLY}.npz" \
    evaluation.physical_checkpoint_hybrid="results/models/${HYBRID_PHYSICAL}.npz"

echo ""
echo "=============================================="
echo "  Experiment Complete!"
echo "=============================================="
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo ""
echo "Generated files:"
echo "  - param_*_comparison.png         : Physical params (physical-only vs hybrid vs true)"
echo "  - param_*_forcing_comparison.png : Forcing field comparison (-∇φ)"
echo "  - sim_*_*_comparison.gif         : Solution animations (synthetic vs hybrid vs real)"
echo "  - sim_*_*_comparison_t0.png      : Initial state comparison"
echo "  - sim_*_*_comparison_t1.png      : First timestep comparison"
echo ""