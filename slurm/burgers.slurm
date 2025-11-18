#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1 -p a100
#SBATCH --time=12:00:00
#SBATCH --job-name="burgers"
#SBATCH --output=burgers.out
#SBATCH --error=burgers.err


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

# Execute the main training script
python run.py --config-name=burgers.yaml

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code $?"
    exit 1
fi

echo "Job completed on $(date)"