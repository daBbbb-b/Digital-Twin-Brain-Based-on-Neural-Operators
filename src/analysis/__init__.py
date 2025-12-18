"""
Analysis and visualization module.

This module provides tools for visualizing stimulus maps and evaluating
consistency with functional brain maps.
"""

from .viz import visualize_stimulus_map, plot_training_curves
from .stats import compute_consistency_metrics

__all__ = ['visualize_stimulus_map', 'plot_training_curves', 'compute_consistency_metrics']
