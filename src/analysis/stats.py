"""
Statistical analysis for stimulus reconstruction and consistency evaluation.

This module provides functions to compute metrics for evaluating stimulus
reconstruction quality and consistency with functional brain maps.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats
from scipy.spatial.distance import cosine


def compute_consistency_metrics(u_true: np.ndarray,
                                u_hat: np.ndarray,
                                detailed: bool = False) -> Dict[str, float]:
    """
    Compute comprehensive consistency metrics between true and estimated stimuli.
    
    Parameters
    ----------
    u_true : np.ndarray
        True stimulus [N, T] or [n_samples, N, T]
    u_hat : np.ndarray
        Estimated stimulus [N, T] or [n_samples, N, T]
    detailed : bool
        If True, compute per-region and per-time metrics
        
    Returns
    -------
    dict
        Dictionary of consistency metrics
    """
    # Handle batch dimension
    if u_true.ndim == 2:
        u_true = u_true[np.newaxis, :, :]
        u_hat = u_hat[np.newaxis, :, :]
    
    n_samples, N, T = u_true.shape
    
    metrics = {}
    
    # Global metrics
    metrics['mse'] = float(np.mean((u_true - u_hat) ** 2))
    metrics['rmse'] = float(np.sqrt(metrics['mse']))
    metrics['mae'] = float(np.mean(np.abs(u_true - u_hat)))
    
    # Normalized metrics
    u_true_std = np.std(u_true)
    if u_true_std > 0:
        metrics['nrmse'] = float(metrics['rmse'] / u_true_std)
    else:
        metrics['nrmse'] = float('nan')
    
    # Correlation
    u_true_flat = u_true.flatten()
    u_hat_flat = u_hat.flatten()
    metrics['pearson_r'] = float(np.corrcoef(u_true_flat, u_hat_flat)[0, 1])
    
    # Spearman correlation (rank-based, robust to outliers)
    spearman_r, spearman_p = stats.spearmanr(u_true_flat, u_hat_flat)
    metrics['spearman_r'] = float(spearman_r)
    metrics['spearman_p'] = float(spearman_p)
    
    # R-squared
    ss_res = np.sum((u_true - u_hat) ** 2)
    ss_tot = np.sum((u_true - np.mean(u_true)) ** 2)
    metrics['r2'] = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else float('nan')
    
    # Cosine similarity
    cosine_sim = 1 - cosine(u_true_flat, u_hat_flat)
    metrics['cosine_similarity'] = float(cosine_sim)
    
    # Per-sample metrics
    per_sample_corr = []
    per_sample_r2 = []
    for i in range(n_samples):
        # Correlation
        corr = np.corrcoef(u_true[i].flatten(), u_hat[i].flatten())[0, 1]
        per_sample_corr.append(corr)
        
        # R2
        ss_res_i = np.sum((u_true[i] - u_hat[i]) ** 2)
        ss_tot_i = np.sum((u_true[i] - np.mean(u_true[i])) ** 2)
        r2_i = 1 - (ss_res_i / ss_tot_i) if ss_tot_i > 0 else 0
        per_sample_r2.append(r2_i)
    
    metrics['mean_per_sample_correlation'] = float(np.mean(per_sample_corr))
    metrics['std_per_sample_correlation'] = float(np.std(per_sample_corr))
    metrics['mean_per_sample_r2'] = float(np.mean(per_sample_r2))
    
    if detailed:
        # Per-region metrics
        per_region_corr = []
        for i in range(N):
            region_true = u_true[:, i, :].flatten()
            region_hat = u_hat[:, i, :].flatten()
            corr = np.corrcoef(region_true, region_hat)[0, 1]
            per_region_corr.append(corr)
        
        metrics['mean_per_region_correlation'] = float(np.mean(per_region_corr))
        metrics['std_per_region_correlation'] = float(np.std(per_region_corr))
        metrics['per_region_correlation'] = per_region_corr
        
        # Per-time metrics
        per_time_corr = []
        for t in range(T):
            time_true = u_true[:, :, t].flatten()
            time_hat = u_hat[:, :, t].flatten()
            corr = np.corrcoef(time_true, time_hat)[0, 1]
            per_time_corr.append(corr)
        
        metrics['mean_per_time_correlation'] = float(np.mean(per_time_corr))
        metrics['std_per_time_correlation'] = float(np.std(per_time_corr))
        metrics['per_time_correlation'] = per_time_corr
    
    return metrics


def spatial_consistency(u: np.ndarray, W: np.ndarray) -> float:
    """
    Compute spatial consistency between stimulus and connectivity structure.
    
    Measures whether stimulus patterns respect connectivity structure.
    
    Parameters
    ----------
    u : np.ndarray
        Stimulus [N, T]
    W : np.ndarray
        Connectivity matrix [N, N]
        
    Returns
    -------
    float
        Spatial consistency score
    """
    N, T = u.shape
    
    # Compute spatial correlation of stimulus
    u_spatial_corr = np.corrcoef(u)
    
    # Compute correlation between connectivity and stimulus correlation
    # Flatten upper triangular parts
    triu_indices = np.triu_indices(N, k=1)
    W_flat = W[triu_indices]
    u_corr_flat = u_spatial_corr[triu_indices]
    
    consistency = np.corrcoef(W_flat, u_corr_flat)[0, 1]
    
    return float(consistency)


def temporal_consistency(u: np.ndarray, smoothness_weight: float = 0.5) -> float:
    """
    Compute temporal consistency of stimulus.
    
    Measures smoothness and autocorrelation of stimulus time series.
    
    Parameters
    ----------
    u : np.ndarray
        Stimulus [N, T]
    smoothness_weight : float
        Weight for smoothness vs autocorrelation
        
    Returns
    -------
    float
        Temporal consistency score
    """
    N, T = u.shape
    
    # Compute temporal derivative (smoothness)
    du_dt = np.diff(u, axis=1)
    smoothness = 1.0 / (1.0 + np.mean(du_dt ** 2))
    
    # Compute lag-1 autocorrelation
    autocorr_list = []
    for i in range(N):
        if np.std(u[i, :]) > 0:
            autocorr = np.corrcoef(u[i, :-1], u[i, 1:])[0, 1]
            autocorr_list.append(autocorr)
    
    mean_autocorr = np.mean(autocorr_list) if autocorr_list else 0
    
    # Combine metrics
    consistency = smoothness_weight * smoothness + (1 - smoothness_weight) * mean_autocorr
    
    return float(consistency)


def stimulus_snr(u_true: np.ndarray, u_hat: np.ndarray) -> float:
    """
    Compute signal-to-noise ratio of stimulus estimation.
    
    Parameters
    ----------
    u_true : np.ndarray
        True stimulus
    u_hat : np.ndarray
        Estimated stimulus
        
    Returns
    -------
    float
        SNR in dB
    """
    signal_power = np.mean(u_true ** 2)
    noise_power = np.mean((u_true - u_hat) ** 2)
    
    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = float('inf')
    
    return float(snr_db)


def functional_map_consistency(u: np.ndarray, 
                               functional_map: np.ndarray,
                               top_k: int = 10) -> Dict[str, float]:
    """
    Evaluate consistency between stimulus map and known functional organization.
    
    Parameters
    ----------
    u : np.ndarray
        Stimulus [N, T]
    functional_map : np.ndarray
        Known functional map [N] with functional labels or activation strength
    top_k : int
        Number of top regions to consider
        
    Returns
    -------
    dict
        Consistency metrics
    """
    N, T = u.shape
    
    # Compute mean stimulus per region
    u_mean = np.mean(u, axis=1)
    
    # Identify top-k active regions in stimulus
    top_stim_regions = np.argsort(np.abs(u_mean))[-top_k:]
    
    # Identify top-k regions in functional map
    top_func_regions = np.argsort(np.abs(functional_map))[-top_k:]
    
    # Compute overlap
    overlap = len(set(top_stim_regions) & set(top_func_regions)) / top_k
    
    # Compute correlation between stimulus strength and functional map
    correlation = np.corrcoef(u_mean, functional_map)[0, 1]
    
    return {
        'overlap': float(overlap),
        'correlation': float(correlation),
        'top_k': top_k
    }


def compute_all_metrics(u_true: np.ndarray,
                       u_hat: np.ndarray,
                       W: np.ndarray,
                       functional_map: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute all available metrics in one call.
    
    Parameters
    ----------
    u_true : np.ndarray
        True stimulus
    u_hat : np.ndarray
        Estimated stimulus
    W : np.ndarray
        Connectivity matrix
    functional_map : np.ndarray, optional
        Known functional map
        
    Returns
    -------
    dict
        All metrics
    """
    metrics = {}
    
    # Consistency metrics
    metrics.update(compute_consistency_metrics(u_true, u_hat, detailed=False))
    
    # SNR
    metrics['snr_db'] = stimulus_snr(u_true, u_hat)
    
    # Spatial consistency
    if u_hat.ndim == 2:
        metrics['spatial_consistency'] = spatial_consistency(u_hat, W)
        metrics['temporal_consistency'] = temporal_consistency(u_hat)
    else:
        # Average over samples
        spatial_list = []
        temporal_list = []
        for i in range(u_hat.shape[0]):
            spatial_list.append(spatial_consistency(u_hat[i], W))
            temporal_list.append(temporal_consistency(u_hat[i]))
        metrics['spatial_consistency'] = float(np.mean(spatial_list))
        metrics['temporal_consistency'] = float(np.mean(temporal_list))
    
    # Functional map consistency
    if functional_map is not None:
        if u_hat.ndim == 2:
            func_metrics = functional_map_consistency(u_hat, functional_map)
        else:
            # Average over samples
            func_metrics = functional_map_consistency(u_hat[0], functional_map)
        metrics.update({f'functional_{k}': v for k, v in func_metrics.items()})
    
    return metrics


if __name__ == '__main__':
    # Example usage
    print("Testing statistical analysis functions...")
    
    # Create synthetic data
    N, T = 50, 100
    n_samples = 10
    
    u_true = np.random.randn(n_samples, N, T)
    u_hat = u_true + np.random.randn(n_samples, N, T) * 0.3
    W = np.random.randn(N, N) * 0.2
    np.fill_diagonal(W, 0)
    functional_map = np.random.randn(N)
    
    print("\n1. Consistency metrics...")
    metrics = compute_consistency_metrics(u_true, u_hat, detailed=True)
    print("Basic metrics:")
    for key in ['mse', 'rmse', 'mae', 'pearson_r', 'r2']:
        print(f"  {key}: {metrics[key]:.4f}")
    
    print("\n2. Spatial consistency...")
    spatial_cons = spatial_consistency(u_hat[0], W)
    print(f"  Spatial consistency: {spatial_cons:.4f}")
    
    print("\n3. Temporal consistency...")
    temporal_cons = temporal_consistency(u_hat[0])
    print(f"  Temporal consistency: {temporal_cons:.4f}")
    
    print("\n4. SNR...")
    snr = stimulus_snr(u_true, u_hat)
    print(f"  SNR: {snr:.2f} dB")
    
    print("\n5. Functional map consistency...")
    func_metrics = functional_map_consistency(u_hat[0], functional_map, top_k=10)
    for key, value in func_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n6. All metrics...")
    all_metrics = compute_all_metrics(u_true, u_hat, W, functional_map)
    print(f"Total metrics computed: {len(all_metrics)}")
    
    print("\nAll statistical functions tested successfully!")
