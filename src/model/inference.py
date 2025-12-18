"""
Inference module for applying trained models to new data.

This module provides functions to load trained neural operator models and apply
them to new brain dynamics data for stimulus inversion.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union


def run_inference(model: nn.Module, 
                  X: np.ndarray, 
                  W: np.ndarray,
                  device: str = 'cpu',
                  batch_size: int = 16) -> np.ndarray:
    """
    Run inference with trained model on new data.
    
    Parameters
    ----------
    model : nn.Module
        Trained neural operator model
    X : np.ndarray
        Observed brain activity [n_samples, N, T] or [N, T]
    W : np.ndarray
        Connectivity matrix [N, N]
    device : str
        Device to run inference on ('cpu' or 'cuda')
    batch_size : int
        Batch size for inference
        
    Returns
    -------
    np.ndarray
        Estimated stimulus [n_samples, N, T] or [N, T]
    """
    model.eval()
    model.to(device)
    
    # Handle single sample case
    single_sample = False
    if X.ndim == 2:
        X = X[np.newaxis, :, :]
        single_sample = True
    
    n_samples, N, T = X.shape
    u_hat_list = []
    
    # Convert to tensors
    X_tensor = torch.from_numpy(X).float()
    W_tensor = torch.from_numpy(W).float().to(device)
    
    # Run inference in batches
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_X = X_tensor[i:i+batch_size].to(device)
            batch_u_hat = model(batch_X, W_tensor)
            u_hat_list.append(batch_u_hat.cpu().numpy())
    
    # Concatenate results
    u_hat = np.concatenate(u_hat_list, axis=0)
    
    # Return single sample if input was single sample
    if single_sample:
        u_hat = u_hat[0]
    
    return u_hat


def load_model_and_infer(checkpoint_path: str,
                         model_class: type,
                         model_kwargs: Dict,
                         X: np.ndarray,
                         W: np.ndarray,
                         device: str = 'cpu') -> np.ndarray:
    """
    Load trained model from checkpoint and run inference.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to model checkpoint
    model_class : type
        Model class (e.g., FNO)
    model_kwargs : dict
        Arguments for model initialization
    X : np.ndarray
        Observed brain activity [n_samples, N, T] or [N, T]
    W : np.ndarray
        Connectivity matrix [N, N]
    device : str
        Device to run inference on
        
    Returns
    -------
    np.ndarray
        Estimated stimulus [n_samples, N, T] or [N, T]
    """
    # Create model
    model = model_class(**model_kwargs)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Training epoch: {checkpoint['epoch']}, Val loss: {checkpoint['val_loss']:.6f}")
    
    # Run inference
    u_hat = run_inference(model, X, W, device=device)
    
    return u_hat


def evaluate_reconstruction(u_true: np.ndarray, 
                           u_hat: np.ndarray) -> Dict[str, float]:
    """
    Evaluate stimulus reconstruction quality.
    
    Parameters
    ----------
    u_true : np.ndarray
        True stimulus [n_samples, N, T] or [N, T]
    u_hat : np.ndarray
        Estimated stimulus [n_samples, N, T] or [N, T]
        
    Returns
    -------
    dict
        Dictionary of evaluation metrics
    """
    # Ensure same shape
    if u_true.ndim == 2:
        u_true = u_true[np.newaxis, :, :]
        u_hat = u_hat[np.newaxis, :, :]
    
    # Compute metrics
    mse = np.mean((u_true - u_hat) ** 2)
    mae = np.mean(np.abs(u_true - u_hat))
    
    # Correlation (flatten all dimensions)
    u_true_flat = u_true.flatten()
    u_hat_flat = u_hat.flatten()
    correlation = np.corrcoef(u_true_flat, u_hat_flat)[0, 1]
    
    # R-squared
    ss_res = np.sum((u_true - u_hat) ** 2)
    ss_tot = np.sum((u_true - np.mean(u_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Per-sample correlation
    per_sample_corr = []
    for i in range(u_true.shape[0]):
        corr = np.corrcoef(u_true[i].flatten(), u_hat[i].flatten())[0, 1]
        per_sample_corr.append(corr)
    mean_per_sample_corr = np.mean(per_sample_corr)
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'correlation': float(correlation),
        'r2': float(r2),
        'mean_per_sample_correlation': float(mean_per_sample_corr)
    }


def save_inference_results(output_path: str,
                           X: np.ndarray,
                           u_hat: np.ndarray,
                           W: np.ndarray,
                           u_true: Optional[np.ndarray] = None,
                           meta: Optional[Dict] = None):
    """
    Save inference results to file.
    
    Parameters
    ----------
    output_path : str
        Path to save results (.npz file)
    X : np.ndarray
        Input brain activity
    u_hat : np.ndarray
        Estimated stimulus
    W : np.ndarray
        Connectivity matrix
    u_true : np.ndarray, optional
        True stimulus (if available)
    meta : dict, optional
        Additional metadata
    """
    save_dict = {
        'X': X,
        'u_hat': u_hat,
        'W': W
    }
    
    if u_true is not None:
        save_dict['u_true'] = u_true
        # Compute and save metrics
        metrics = evaluate_reconstruction(u_true, u_hat)
        save_dict['metrics'] = metrics
    
    if meta is not None:
        save_dict['meta'] = meta
    
    np.savez(output_path, **save_dict)
    print(f"Saved inference results to {output_path}")


if __name__ == '__main__':
    # Example usage
    print("Testing inference module...")
    
    from src.model.fno import FNO
    
    # Create model
    N, T = 50, 100
    model = FNO(N=N, T=T, hidden_channels=16, n_layers=2, modes=8)
    
    # Create dummy data
    X = np.random.randn(10, N, T).astype(np.float32)
    W = np.random.randn(N, N).astype(np.float32)
    u_true = np.random.randn(10, N, T).astype(np.float32)
    
    # Run inference
    print("\nRunning inference...")
    u_hat = run_inference(model, X, W, device='cpu', batch_size=4)
    
    print(f"Input X shape: {X.shape}")
    print(f"Output u_hat shape: {u_hat.shape}")
    
    # Evaluate
    print("\nEvaluating reconstruction...")
    metrics = evaluate_reconstruction(u_true, u_hat)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test single sample
    print("\nTesting single sample inference...")
    X_single = X[0]
    u_hat_single = run_inference(model, X_single, W, device='cpu')
    print(f"Single input shape: {X_single.shape}")
    print(f"Single output shape: {u_hat_single.shape}")
