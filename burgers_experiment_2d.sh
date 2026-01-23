#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1 -p a100
#SBATCH --time=24:00:00
#SBATCH --job-name="inviscid_burgers_2d"
#SBATCH --output=inviscid_burgers_2d.out
#SBATCH --error=inviscid_burgers_2d.err

# Print job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"

# Load required modules
echo "Loading modules..."
module purge
module load python 
conda activate phi-env

# Display loaded modules and Python environment info
echo "Loaded modules:"
module list

echo "Python version:"
python --version

echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Check GPU availability
echo "GPU information:"
nvidia-smi

# Set up environment variables
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0

# Change to the job submission directory
cd $SLURM_SUBMIT_DIR


# Print system information
echo "System information:"
echo "Available memory: $(free -h | grep Mem | awk '{print $2}')"
echo "Available disk space: $(df -h . | tail -1 | awk '{print $4}')"

set -e  # Exit on error

# Configuration
CONFIG_NAME="inviscid_burgers_2d"
RESULTS_DIR="results/experiments/inviscid_burgers_2d_comparison"

# Model checkpoint names
SYNTHETIC_ONLY="inviscid_burgers_synthetic_only_2d"
PHYSICAL_ONLY="inviscid_burgers_physical_only_2d"
HYBRID_SYNTHETIC="inviscid_burgers_hybrid_synthetic_2d"
HYBRID_PHYSICAL="inviscid_burgers_hybrid_physical_2d"

# echo "=============================================="
# echo "  Inviscid Burgers 2d Comparison Experiment"
# echo "=============================================="
# echo ""
# echo "Note: Synthetic-only training uses 'standalone_epochs' from config"
# echo "      which equals: epochs * (cycles + warmup) for fair comparison"
# echo ""

# # -----------------------------------------------------------------------------
# # Step 0: Generate data (if not already present)
# # -----------------------------------------------------------------------------
# echo "[Step 0/4] Checking/generating data..."
# if [ ! -d "data/inviscid_burgers_2d" ] || [ -z "$(ls -A data/inviscid_burgers_2d 2>/dev/null)" ]; then
#     echo "  Data not found, generating..."
#     python run.py --config-name=${CONFIG_NAME} \
#         general.tasks='[generate]'
# else
#     echo "  Data already exists, skipping generation."
# fi
# echo ""

# # -----------------------------------------------------------------------------
# # Step 1: Train synthetic model on real data only
# # Uses standalone_epochs (= epochs * (cycles + warmup)) for fair comparison
# # -----------------------------------------------------------------------------
# echo "[Step 1/4] Training SYNTHETIC model on real data..."
# python run.py --config-name=${CONFIG_NAME} \
#     general.mode='synthetic' \
#     general.tasks='[train]' \
#     general.experiment_name='inviscid_burgers_2d_synthetic_only' \
#     model.synthetic.model_save_name=${SYNTHETIC_ONLY} \
#     trainer.synthetic.epochs='${trainer.synthetic.standalone_epochs}'

# echo "  Synthetic-only model saved to results/models/${SYNTHETIC_ONLY}.pth"
# echo ""

# # -----------------------------------------------------------------------------
# # Step 2: Train physical model on real data only
# # Uses standalone_epochs (= epochs * cycles) for fair comparison
# # -----------------------------------------------------------------------------
# echo "[Step 2/4] Training PHYSICAL model on real data..."
# python run.py --config-name=${CONFIG_NAME} \
#     general.mode='physical' \
#     general.tasks='[train]' \
#     general.experiment_name='inviscid_burgers_2d_physical_only' \
#     model.physical.model_save_name=${PHYSICAL_ONLY} \
#     trainer.physical.epochs='${trainer.physical.standalone_epochs}'

# echo "  Physical-only model saved to results/models/${PHYSICAL_ONLY}.npz"
# echo ""

# -----------------------------------------------------------------------------
# Step 3: Hybrid training
# -----------------------------------------------------------------------------
echo "[Step 3/4] Running HYBRID training..."
python run.py --config-name=${CONFIG_NAME} \
    general.mode='hybrid' \
    general.tasks='[train]' \
    general.experiment_name='inviscid_burgers_2d_hybrid' \
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