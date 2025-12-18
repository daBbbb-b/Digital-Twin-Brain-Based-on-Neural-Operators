"""
Surrogate model for effective connectivity inference.

This module provides methods to estimate effective connectivity (EC) between brain
regions from observed dynamics. EC represents causal influence between regions.

Scientific background:
- Effective connectivity captures directed causal interactions
- Can be estimated from time series using various methods
- Used as input to neural operator models alongside observed activity
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.linear_model import Ridge


class SurrogateEC:
    """
    Surrogate model for estimating effective connectivity from brain dynamics.
    
    Uses regularized regression to infer directed connectivity that explains
    observed temporal dynamics.
    """
    
    def __init__(self, method: str = 'ridge', alpha: float = 1.0):
        """
        Initialize surrogate EC model.
        
        Parameters
        ----------
        method : str
            Method for connectivity estimation ('ridge', 'granger', or 'correlation')
        alpha : float
            Regularization strength for ridge regression
        """
        self.method = method
        self.alpha = alpha
        self.W_estimated = None
    
    def estimate_connectivity_ridge(self, X: np.ndarray, lag: int = 1) -> np.ndarray:
        """
        Estimate connectivity using ridge regression.
        
        Fits: X(t) = W @ X(t-lag) + noise
        
        Parameters
        ----------
        X : np.ndarray
            Time series data [N, T]
        lag : int
            Time lag for autoregressive model
            
        Returns
        -------
        np.ndarray
            Estimated connectivity matrix [N, N]
        """
        N, T = X.shape
        W = np.zeros((N, N))
        
        # For each target region, fit ridge regression
        for i in range(N):
            # Target: X[i, lag:]
            # Predictors: X[:, :-lag]
            y = X[i, lag:]
            X_predictors = X[:, :-lag].T
            
            # Ridge regression
            ridge = Ridge(alpha=self.alpha, fit_intercept=False)
            ridge.fit(X_predictors, y)
            W[i, :] = ridge.coef_
        
        return W
    
    def estimate_connectivity_correlation(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate connectivity using correlation.
        
        Parameters
        ----------
        X : np.ndarray
            Time series data [N, T]
            
        Returns
        -------
        np.ndarray
            Correlation-based connectivity matrix [N, N]
        """
        N, T = X.shape
        # Compute correlation matrix
        W = np.corrcoef(X)
        # Remove self-connections
        np.fill_diagonal(W, 0)
        return W
    
    def estimate_connectivity_granger(self, X: np.ndarray, max_lag: int = 3) -> np.ndarray:
        """
        Estimate connectivity using simplified Granger causality.
        
        Parameters
        ----------
        X : np.ndarray
            Time series data [N, T]
        max_lag : int
            Maximum lag to consider
            
        Returns
        -------
        np.ndarray
            Granger causality-based connectivity matrix [N, N]
        """
        N, T = X.shape
        W = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                
                # Test if X[j] Granger-causes X[i]
                # Fit model with and without X[j]
                y = X[i, max_lag:]
                
                # Full model: use both X[i] and X[j] history
                X_full = []
                for lag in range(1, max_lag + 1):
                    X_full.append(X[i, max_lag-lag:-lag])
                    X_full.append(X[j, max_lag-lag:-lag])
                X_full = np.column_stack(X_full)
                
                # Reduced model: use only X[i] history
                X_reduced = []
                for lag in range(1, max_lag + 1):
                    X_reduced.append(X[i, max_lag-lag:-lag])
                X_reduced = np.column_stack(X_reduced)
                
                # Fit models and compare
                ridge_full = Ridge(alpha=self.alpha, fit_intercept=False)
                ridge_reduced = Ridge(alpha=self.alpha, fit_intercept=False)
                
                ridge_full.fit(X_full, y)
                ridge_reduced.fit(X_reduced, y)
                
                # Compute residuals
                rss_full = np.sum((y - ridge_full.predict(X_full))**2)
                rss_reduced = np.sum((y - ridge_reduced.predict(X_reduced))**2)
                
                # Granger causality measure
                if rss_full > 0:
                    W[i, j] = max(0, (rss_reduced - rss_full) / rss_reduced)
        
        return W
    
    def fit(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Estimate effective connectivity from observed dynamics.
        
        Parameters
        ----------
        X : np.ndarray
            Observed brain activity time series [N, T] or [batch, N, T]
        **kwargs
            Additional arguments passed to estimation method
            
        Returns
        -------
        np.ndarray
            Estimated connectivity matrix [N, N]
        """
        # Handle batch dimension
        if X.ndim == 3:
            # Average over samples
            W_list = []
            for sample in range(X.shape[0]):
                if self.method == 'ridge':
                    W = self.estimate_connectivity_ridge(X[sample], **kwargs)
                elif self.method == 'correlation':
                    W = self.estimate_connectivity_correlation(X[sample])
                elif self.method == 'granger':
                    W = self.estimate_connectivity_granger(X[sample], **kwargs)
                else:
                    raise ValueError(f"Unknown method: {self.method}")
                W_list.append(W)
            self.W_estimated = np.mean(W_list, axis=0)
        else:
            if self.method == 'ridge':
                self.W_estimated = self.estimate_connectivity_ridge(X, **kwargs)
            elif self.method == 'correlation':
                self.W_estimated = self.estimate_connectivity_correlation(X)
            elif self.method == 'granger':
                self.W_estimated = self.estimate_connectivity_granger(X, **kwargs)
            else:
                raise ValueError(f"Unknown method: {self.method}")
        
        return self.W_estimated
    
    def get_connectivity(self) -> Optional[np.ndarray]:
        """
        Get estimated connectivity matrix.
        
        Returns
        -------
        np.ndarray or None
            Estimated connectivity matrix [N, N], or None if not fitted
        """
        return self.W_estimated


if __name__ == '__main__':
    # Example usage
    print("Testing SurrogateEC...")
    
    # Generate synthetic data
    N, T = 20, 100
    # Create true connectivity
    W_true = np.random.randn(N, N) * 0.1
    np.fill_diagonal(W_true, 0)
    
    # Generate dynamics
    X = np.zeros((N, T))
    X[:, 0] = np.random.randn(N) * 0.1
    for t in range(1, T):
        X[:, t] = np.tanh(W_true @ X[:, t-1]) + np.random.randn(N) * 0.1
    
    # Estimate connectivity
    surrogate = SurrogateEC(method='ridge', alpha=0.1)
    W_est = surrogate.fit(X, lag=1)
    
    print(f"True connectivity shape: {W_true.shape}")
    print(f"Estimated connectivity shape: {W_est.shape}")
    print(f"Correlation with true W: {np.corrcoef(W_true.flatten(), W_est.flatten())[0, 1]:.3f}")
