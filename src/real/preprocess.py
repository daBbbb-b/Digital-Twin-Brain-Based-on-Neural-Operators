"""
Real fMRI data preprocessing module.

This module provides functions to preprocess real fMRI task data for input
to trained neural operator models. Includes time series extraction, normalization,
connectivity estimation, and data formatting.

Scientific background:
- fMRI data is 4D: [x, y, z, time]
- Must extract time series from regions of interest (ROIs)
- Normalize and detrend time series
- Estimate or load connectivity matrix
- Format as (X, W) for model input
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from pathlib import Path


def extract_roi_timeseries(fmri_data: np.ndarray, 
                           roi_mask: np.ndarray) -> np.ndarray:
    """
    Extract ROI time series from 4D fMRI data.
    
    Parameters
    ----------
    fmri_data : np.ndarray
        4D fMRI data [x, y, z, time]
    roi_mask : np.ndarray
        3D ROI mask [x, y, z] with integer labels for each ROI
        
    Returns
    -------
    np.ndarray
        ROI time series [N, T] where N is number of ROIs
    """
    n_rois = int(roi_mask.max())
    n_timepoints = fmri_data.shape[-1]
    
    timeseries = np.zeros((n_rois, n_timepoints))
    
    for roi_id in range(1, n_rois + 1):
        roi_voxels = roi_mask == roi_id
        # Average signal across all voxels in this ROI
        timeseries[roi_id - 1, :] = fmri_data[roi_voxels, :].mean(axis=0)
    
    return timeseries


def detrend_timeseries(X: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    Remove linear or polynomial trends from time series.
    
    Parameters
    ----------
    X : np.ndarray
        Time series data [N, T]
    method : str
        Detrending method ('linear' or 'constant')
        
    Returns
    -------
    np.ndarray
        Detrended time series [N, T]
    """
    N, T = X.shape
    X_detrended = np.zeros_like(X)
    
    for i in range(N):
        X_detrended[i, :] = signal.detrend(X[i, :], type=method)
    
    return X_detrended


def bandpass_filter(X: np.ndarray, 
                    TR: float,
                    lowcut: float = 0.01,
                    highcut: float = 0.1,
                    order: int = 4) -> np.ndarray:
    """
    Apply bandpass filter to time series.
    
    Parameters
    ----------
    X : np.ndarray
        Time series data [N, T]
    TR : float
        Repetition time (sampling interval) in seconds
    lowcut : float
        Low cutoff frequency in Hz
    highcut : float
        High cutoff frequency in Hz
    order : int
        Filter order
        
    Returns
    -------
    np.ndarray
        Filtered time series [N, T]
    """
    N, T = X.shape
    fs = 1.0 / TR  # Sampling frequency
    nyquist = fs / 2.0
    
    # Design Butterworth bandpass filter
    low = lowcut / nyquist
    high = highcut / nyquist
    
    if high >= 1.0:
        high = 0.99
    
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter to each ROI
    X_filtered = np.zeros_like(X)
    for i in range(N):
        X_filtered[i, :] = signal.filtfilt(b, a, X[i, :])
    
    return X_filtered


def normalize_timeseries(X: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize time series.
    
    Parameters
    ----------
    X : np.ndarray
        Time series data [N, T]
    method : str
        Normalization method ('zscore', 'minmax', or 'percent')
        
    Returns
    -------
    np.ndarray
        Normalized time series [N, T]
    """
    N, T = X.shape
    X_normalized = np.zeros_like(X)
    
    for i in range(N):
        if method == 'zscore':
            mean = np.mean(X[i, :])
            std = np.std(X[i, :])
            if std > 0:
                X_normalized[i, :] = (X[i, :] - mean) / std
            else:
                X_normalized[i, :] = X[i, :] - mean
        elif method == 'minmax':
            min_val = np.min(X[i, :])
            max_val = np.max(X[i, :])
            if max_val > min_val:
                X_normalized[i, :] = (X[i, :] - min_val) / (max_val - min_val)
            else:
                X_normalized[i, :] = X[i, :]
        elif method == 'percent':
            mean = np.mean(X[i, :])
            if mean != 0:
                X_normalized[i, :] = (X[i, :] - mean) / mean * 100
            else:
                X_normalized[i, :] = X[i, :]
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    return X_normalized


def preprocess_fmri(X: np.ndarray,
                    TR: float = 2.0,
                    detrend: bool = True,
                    bandpass: bool = True,
                    normalize: str = 'zscore',
                    smooth_sigma: Optional[float] = None) -> np.ndarray:
    """
    Full preprocessing pipeline for fMRI time series.
    
    Parameters
    ----------
    X : np.ndarray
        Raw time series data [N, T]
    TR : float
        Repetition time in seconds
    detrend : bool
        Whether to detrend time series
    bandpass : bool
        Whether to apply bandpass filter
    normalize : str
        Normalization method ('zscore', 'minmax', 'percent', or None)
    smooth_sigma : float, optional
        Temporal smoothing sigma (Gaussian filter)
        
    Returns
    -------
    np.ndarray
        Preprocessed time series [N, T]
    """
    X_proc = X.copy()
    
    # Detrend
    if detrend:
        X_proc = detrend_timeseries(X_proc, method='linear')
    
    # Bandpass filter
    if bandpass:
        X_proc = bandpass_filter(X_proc, TR=TR, lowcut=0.01, highcut=0.1)
    
    # Temporal smoothing
    if smooth_sigma is not None:
        for i in range(X_proc.shape[0]):
            X_proc[i, :] = gaussian_filter1d(X_proc[i, :], sigma=smooth_sigma)
    
    # Normalize
    if normalize is not None and normalize != 'none':
        X_proc = normalize_timeseries(X_proc, method=normalize)
    
    return X_proc


def prepare_for_inference(X: np.ndarray,
                         W: Optional[np.ndarray] = None,
                         TR: float = 2.0,
                         preprocess: bool = True) -> Dict[str, np.ndarray]:
    """
    Prepare real fMRI data for neural operator inference.
    
    Parameters
    ----------
    X : np.ndarray
        fMRI time series [N, T] or [n_samples, N, T]
    W : np.ndarray, optional
        Connectivity matrix [N, N]. If None, will be estimated.
    TR : float
        Repetition time in seconds
    preprocess : bool
        Whether to apply preprocessing
        
    Returns
    -------
    dict
        Dictionary with keys 'X' and 'W' ready for model input
    """
    # Handle batch dimension
    single_sample = False
    if X.ndim == 2:
        X = X[np.newaxis, :, :]
        single_sample = True
    
    n_samples, N, T = X.shape
    
    # Preprocess each sample
    if preprocess:
        X_proc = np.zeros_like(X)
        for i in range(n_samples):
            X_proc[i] = preprocess_fmri(X[i], TR=TR)
    else:
        X_proc = X
    
    # Estimate connectivity if not provided
    if W is None:
        from src.ec.surrogate import SurrogateEC
        surrogate = SurrogateEC(method='correlation')
        W = surrogate.fit(X_proc)
    
    # Remove batch dimension if input was single sample
    if single_sample:
        X_proc = X_proc[0]
    
    return {
        'X': X_proc.astype(np.float32),
        'W': W.astype(np.float32)
    }


if __name__ == '__main__':
    # Example usage
    print("Testing fMRI preprocessing...")
    
    # Create synthetic fMRI-like data
    N, T = 50, 200
    TR = 2.0
    
    # Simulate BOLD signal with trend and noise
    t = np.arange(T)
    X = np.zeros((N, T))
    for i in range(N):
        # Low-frequency oscillation + linear trend + noise
        X[i, :] = np.sin(2 * np.pi * 0.05 * t) + 0.01 * t + np.random.randn(T) * 0.5
    
    print(f"Input shape: {X.shape}")
    print(f"Input range: [{X.min():.3f}, {X.max():.3f}]")
    
    # Preprocess
    X_proc = preprocess_fmri(X, TR=TR, detrend=True, bandpass=True, normalize='zscore')
    
    print(f"Preprocessed shape: {X_proc.shape}")
    print(f"Preprocessed range: [{X_proc.min():.3f}, {X_proc.max():.3f}]")
    print(f"Preprocessed mean: {X_proc.mean():.6f}")
    print(f"Preprocessed std: {X_proc.std():.6f}")
    
    # Prepare for inference
    print("\nPreparing for inference...")
    data = prepare_for_inference(X, TR=TR, preprocess=True)
    print(f"X shape: {data['X'].shape}")
    print(f"W shape: {data['W'].shape}")
