"""Test evaluation script for Burgers 1D with parameter visualization."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.evaluation import Evaluator
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger("test_burgers_eval", level="INFO")

def test_burgers_evaluation():
    """Test evaluation with parameter recovery visualization."""
    logger.info("="*60)
    logger.info("Testing Burgers 1D Evaluation with Parameter Visualization")
    logger.info("="*60)

    # Load config
    config_path = PROJECT_ROOT / "conf" / "burgers_1d.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Ensure we have necessary checkpoints
    synthetic_checkpoint = Path(config['evaluation']['synthetic_checkpoint'])
    physical_checkpoint = Path(config['evaluation']['physical_checkpoint'])

    if not synthetic_checkpoint.exists():
        logger.warning(f"Synthetic checkpoint not found: {synthetic_checkpoint}")
        logger.info("Please run training first: python run.py --config-name=burgers_1d general.mode=synthetic general.tasks=\"['train']\"")
        return

    if not physical_checkpoint.exists():
        logger.warning(f"Physical checkpoint not found: {physical_checkpoint}")
        logger.info("Please run physical training first: python run.py --config-name=burgers_1d general.mode=physical general.tasks=\"['train']\"")
        return

    # Create evaluator
    logger.info("Creating evaluator...")
    evaluator = Evaluator(config)

    # Run evaluation
    logger.info("Running evaluation...")
    evaluator.evaluate()

    # Check output files
    output_dir = Path(config['evaluation']['output_dir'])
    param_plot = output_dir / 'scalar_parameter_recovery.png'

    if param_plot.exists():
        logger.info(f"✓ Scalar parameter recovery plot created: {param_plot}")
    else:
        logger.warning(f"✗ Scalar parameter recovery plot not found")

    # List all generated files
    logger.info("\nGenerated files:")
    for file in sorted(output_dir.glob('*')):
        logger.info(f"  - {file.name}")

    logger.info("\n" + "="*60)
    logger.info("✓ Evaluation test completed!")
    logger.info("="*60)

if __name__ == "__main__":
    test_burgers_evaluation()
