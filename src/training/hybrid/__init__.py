"""
Hybrid Trainer for HYCO

This module implements the hybrid training approach that alternates between
synthetic and physical model training with cross-model data augmentation.
"""

from .trainer import HybridTrainer

__all__ = ['HybridTrainer']
