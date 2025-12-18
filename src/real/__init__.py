"""
Real fMRI data processing module.

This module handles preprocessing of real fMRI task data and applies trained
models for stimulus inference.
"""

from .preprocess import preprocess_fmri
from .run_inference import infer_from_real_data

__all__ = ['preprocess_fmri', 'infer_from_real_data']
