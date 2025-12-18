"""
PDE-based brain dynamics simulation.

This module generates synthetic brain activity using partial differential equations
to model spatiotemporal neural field dynamics with diffusion and coupling.

Scientific background:
- Neural field model with reaction-diffusion dynamics
- Captures spatial smoothness of cortical activity
- Task stimuli u(x,t) drive activity in spatial regions
- Connectivity defines long-range coupling between regions
- More realistic for modeling spatially extended cortical dynamics
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Dict, Optional


def neural_field_pde_step(X: np.ndarray, W: np.ndarray, u: np.ndarray,
                          dt: float = 0.1, diffusion: float = 0.1,
                          tau: float = 1.0, alpha: float = 0.5) -> np.ndarray:
    """
    One time step of neural field PDE.
    
    Implements: dX/dt = -X/tau + alpha*W@X + diffusion*∇²X + u
    
    Parameters
    ----------
    X : np.ndarray
        Current state [N]
    W : np.ndarray
        Connectivity matrix [N, N]
    u : np.ndarray
        Current stimulus [N]
    dt : float
        Time step size
    diffusion : float
        Diffusion coefficient (spatial smoothing)
    tau : float
        Time constant
    alpha : float
        Coupling strength
        
    Returns
    -------
    np.ndarray
        Next state [N]
    """
    N = len(X)
    
    # Reaction term: -X/tau + tanh(W @ X + u)
    coupling = alpha * (W @ X)
    activation = np.tanh(coupling + u)
    reaction = -X / tau + activation
    
    # Diffusion term: approximate Laplacian using spatial smoothing
    # For 1D spatial arrangement
    laplacian = np.zeros_like(X)
    for i in range(1, N-1):
        laplacian[i] = X[i+1] - 2*X[i] + X[i-1]
    # Boundary conditions (Neumann)
    laplacian[0] = X[1] - X[0]
    laplacian[-1] = X[-2] - X[-1]
    
    # Update
    dXdt = reaction + diffusion * laplacian
    X_new = X + dt * dXdt
    
    return X_new


def generate_pde_data(N: int = 50, T: int = 100, n_samples: int = 100,
                      dt: float = 0.1, diffusion: float = 0.05,
                      tau: float = 1.0, alpha: float = 0.5,
                      stimulus_regions: Optional[np.ndarray] = None,
                      stimulus_strength: float = 2.0,
                      noise_level: float = 0.1,
                      random_seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Generate synthetic brain dynamics using PDE-based neural field model.
    
    Parameters
    ----------
    N : int
        Number of spatial points (brain regions)
    T : int
        Number of time points
    n_samples : int
        Number of samples to generate
    dt : float
        Time step for PDE integration
    diffusion : float
        Diffusion coefficient for spatial smoothing
    tau : float
        Time constant of neural dynamics
    alpha : float
        Coupling strength
    stimulus_regions : np.ndarray, optional
        Indices of regions receiving stimulus. If None, random regions chosen.
    stimulus_strength : float
        Amplitude of task stimulus
    noise_level : float
        Observation noise standard deviation
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'X': State time series [n_samples, N, T]
        - 'u': Stimulus time series [n_samples, N, T]
        - 'W': Connectivity matrix [N, N]
        - 'meta': Metadata dictionary
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate connectivity matrix with spatial structure
    SPATIAL_DECAY_FACTOR = 4  # Controls spatial extent of connectivity
    CONNECTIVITY_STRENGTH = 0.2
    
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                # Distance-dependent connectivity
                distance = abs(i - j)
                W[i, j] = np.exp(-distance**2 / (2 * (N/SPATIAL_DECAY_FACTOR)**2)) * np.random.randn() * CONNECTIVITY_STRENGTH
    W = (W + W.T) / 2  # Make symmetric
    
    # Initialize storage
    X = np.zeros((n_samples, N, T))
    u = np.zeros((n_samples, N, T))
    
    for sample in range(n_samples):
        # Generate spatially smooth stimulus
        if stimulus_regions is None:
            # Select spatially contiguous region
            stim_center = np.random.randint(N//4, 3*N//4)
            stim_width = np.random.randint(N//10, N//5)
            stim_regions = np.arange(max(0, stim_center - stim_width//2),
                                    min(N, stim_center + stim_width//2))
        else:
            stim_regions = stimulus_regions
        
        # Create temporal stimulus pattern
        for region in stim_regions:
            # Random task blocks
            n_blocks = np.random.randint(2, 5)
            for _ in range(n_blocks):
                onset = np.random.randint(0, T - 10)
                duration = np.random.randint(5, 15)
                end = min(onset + duration, T)
                # Spatial Gaussian weighting
                center = np.mean(stim_regions)
                spatial_weight = np.exp(-(region - center)**2 / (2 * (len(stim_regions)/2)**2))
                u[sample, region, onset:end] = stimulus_strength * spatial_weight * np.random.uniform(0.5, 1.0)
        
        # Smooth stimulus in time
        for region in range(N):
            u[sample, region, :] = gaussian_filter1d(u[sample, region, :], sigma=1.5)
        
        # Integrate PDE
        X_current = np.random.randn(N) * 0.1  # Initial condition
        X[sample, :, 0] = X_current
        
        for t in range(1, T):
            X_current = neural_field_pde_step(X_current, W, u[sample, :, t-1],
                                             dt=dt, diffusion=diffusion, tau=tau, alpha=alpha)
            X[sample, :, t] = X_current
        
        # Add observation noise
        X[sample] += np.random.randn(N, T) * noise_level
    
    meta = {
        'N': N,
        'T': T,
        'n_samples': n_samples,
        'dt': dt,
        'diffusion': diffusion,
        'tau': tau,
        'alpha': alpha,
        'stimulus_strength': stimulus_strength,
        'noise_level': noise_level,
        'model': 'neural_field_pde',
        'description': 'PDE-based neural field model with reaction-diffusion dynamics'
    }
    
    return {
        'X': X.astype(np.float32),
        'u': u.astype(np.float32),
        'W': W.astype(np.float32),
        'meta': meta
    }


if __name__ == '__main__':
    # Example usage
    print("Generating PDE-based brain dynamics data...")
    data = generate_pde_data(N=50, T=100, n_samples=10, random_seed=42)
    
    print(f"Generated data shapes:")
    print(f"  X (state): {data['X'].shape}")
    print(f"  u (stimulus): {data['u'].shape}")
    print(f"  W (connectivity): {data['W'].shape}")
    
    # Save example
    output_path = 'data/sim/pde_example.npz'
    np.savez(output_path, **data)
    print(f"\nSaved to: {output_path}")
