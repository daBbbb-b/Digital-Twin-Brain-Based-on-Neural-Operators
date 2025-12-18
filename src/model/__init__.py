"""
Neural operator models for stimulus inversion.

This module contains neural operator implementations (FNO) and training utilities.
"""

from .fno import FNO
from .trainer import Trainer
from .inference import run_inference

__all__ = ['FNO', 'Trainer', 'run_inference']
