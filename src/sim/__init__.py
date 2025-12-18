"""
Simulation module for generating synthetic brain dynamics data.

This module provides ODE and PDE-based models to simulate brain activity
patterns and task stimuli.
"""

from .generate_ode import generate_ode_data
from .generate_pde import generate_pde_data

__all__ = ['generate_ode_data', 'generate_pde_data']
