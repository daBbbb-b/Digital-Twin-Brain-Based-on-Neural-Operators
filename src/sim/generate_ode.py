"""
ODE-based brain dynamics simulation.

This module generates synthetic brain activity data using ordinary differential equations
to model coupled neural populations with external task stimuli.

Scientific background:
- Models neural populations as coupled oscillators
- Task stimuli u(t) drive specific regions
- Connectivity matrix W defines inter-regional coupling
- Outputs realistic temporal dynamics similar to fMRI signals
"""

import numpy as np
from scipy.integrate import odeint
from typing import Tuple, Dict, Optional


def wilson_cowan_ode(state: np.ndarray, t: float, W: np.ndarray, u: np.ndarray, 
                     time_points: np.ndarray, tau: float = 1.0, 
                     alpha: float = 1.0) -> np.ndarray:
    """
    Wilson-Cowan neural mass model.
    
    Parameters
    ----------
    state : np.ndarray
        Current state of all regions [N]
    t : float
        Current time point
    W : np.ndarray
        Connectivity matrix [N, N]
    u : np.ndarray
        Stimulus time series [N, T]
    time_points : np.ndarray
        Time grid for stimulus interpolation [T]
    tau : float
        Time constant
    alpha : float
        Coupling strength
        
    Returns
    -------
    np.ndarray
        Time derivative dx/dt [N]
    """
    N = len(state)
    # Interpolate stimulus at current time
    idx = np.searchsorted(time_points, t)
    idx = min(idx, len(time_points) - 1)
    u_t = u[:, idx]
    
    # Wilson-Cowan dynamics: dx/dt = -x/tau + sigmoid(W @ x + u)
    coupling = alpha * (W @ state)
    activation = np.tanh(coupling + u_t)
    dxdt = (-state / tau) + activation
    
    return dxdt


def generate_ode_data(N: int = 50, T: int = 100, n_samples: int = 100,
                      tau: float = 1.0, alpha: float = 0.5,
                      stimulus_regions: Optional[np.ndarray] = None,
                      stimulus_strength: float = 2.0,
                      noise_level: float = 0.1,
                      random_seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Generate synthetic brain dynamics using ODE-based neural mass model.
    
    Parameters
    ----------
    N : int
        Number of brain regions
    T : int
        Number of time points
    n_samples : int
        Number of samples to generate
    tau : float
        Time constant of neural dynamics
    alpha : float
        Coupling strength
    stimulus_regions : np.ndarray, optional
        Indices of regions receiving stimulus. If None, random regions are chosen.
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
    
    # Generate connectivity matrix (small-world-like structure)
    CONNECTIVITY_STRENGTH = 0.1
    SPATIAL_DECAY_FACTOR = 4  # Controls distance-dependent decay
    
    W = np.random.randn(N, N) * CONNECTIVITY_STRENGTH
    # Add distance-dependent decay
    for i in range(N):
        for j in range(N):
            distance = abs(i - j)
            W[i, j] *= np.exp(-distance / (N / SPATIAL_DECAY_FACTOR))
    W = (W + W.T) / 2  # Make symmetric
    np.fill_diagonal(W, 0)
    
    # Initialize storage
    X = np.zeros((n_samples, N, T))
    u = np.zeros((n_samples, N, T))
    
    time_points = np.linspace(0, 10, T)
    
    for sample in range(n_samples):
        # Generate stimulus pattern
        if stimulus_regions is None:
            n_stim_regions = np.random.randint(2, max(3, N // 10))
            stim_regions = np.random.choice(N, n_stim_regions, replace=False)
        else:
            stim_regions = stimulus_regions
        
        # Create stimulus time course (boxcar with some variation)
        for region in stim_regions:
            # Random task blocks
            n_blocks = np.random.randint(2, 5)
            for _ in range(n_blocks):
                onset = np.random.randint(0, T - 10)
                duration = np.random.randint(5, 15)
                end = min(onset + duration, T)
                u[sample, region, onset:end] = stimulus_strength * np.random.uniform(0.5, 1.0)
        
        # Smooth stimulus
        from scipy.ndimage import gaussian_filter1d
        for region in range(N):
            u[sample, region, :] = gaussian_filter1d(u[sample, region, :], sigma=1.0)
        
        # Integrate ODE
        x0 = np.random.randn(N) * 0.1  # Initial condition
        sol = odeint(wilson_cowan_ode, x0, time_points, 
                     args=(W, u[sample], time_points, tau, alpha))
        X[sample] = sol.T
        
        # Add observation noise
        X[sample] += np.random.randn(N, T) * noise_level
    
    meta = {
        'N': N,
        'T': T,
        'n_samples': n_samples,
        'tau': tau,
        'alpha': alpha,
        'stimulus_strength': stimulus_strength,
        'noise_level': noise_level,
        'model': 'wilson_cowan_ode',
        'description': 'ODE-based neural mass model simulating task-driven brain dynamics'
    }
    
    return {
        'X': X.astype(np.float32),
        'u': u.astype(np.float32),
        'W': W.astype(np.float32),
        'meta': meta
    }


if __name__ == '__main__':
    # Example usage
    print("Generating ODE-based brain dynamics data...")
    data = generate_ode_data(N=50, T=100, n_samples=10, random_seed=42)
    
    print(f"Generated data shapes:")
    print(f"  X (state): {data['X'].shape}")
    print(f"  u (stimulus): {data['u'].shape}")
    print(f"  W (connectivity): {data['W'].shape}")
    
    # Save example
    output_path = 'data/sim/ode_example.npz'
    np.savez(output_path, **data)
    print(f"\nSaved to: {output_path}")
