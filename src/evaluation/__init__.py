"""Minimal evaluation package for generating animations.

This package provides a small CLI and a utility to generate simple
animations (GIF/MP4) from existing Scene data. It purposefully avoids
changing the rest of the codebase and reuses the DataManager to read
simulations from disk.
"""

from .cli import main  # pragma: no cover
