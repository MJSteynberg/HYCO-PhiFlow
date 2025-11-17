import tempfile
from pathlib import Path

from src.evaluation.cli import generate_animations


def test_generate_simple_animation():
    # Use the repository's default config which points to data/advection.
    # Use the first simulation available (sim_000000) and request a handful of frames.
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "eval_out"
        saved = generate_animations(None, sim_index=0, fields=["density"], output_dir=str(out_dir), fps=5, num_frames=10)
        # Check that the density animation was created
        expected = out_dir / "sim_000000_density_animation.gif"
        assert expected.exists(), f"Expected animation at {expected} but does not exist"
