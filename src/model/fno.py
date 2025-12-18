"""
Fourier Neural Operator (FNO) for stimulus inversion.

This module implements a Fourier Neural Operator that learns to invert the mapping
from observed brain activity (X) and connectivity (W) to task stimuli (u).

Scientific background:
- FNO operates in Fourier space for efficient global convolutions
- Can capture long-range dependencies in spatiotemporal data
- Learns operator mapping between function spaces
- Suitable for inverse problems in brain dynamics

Model input: (X, W) where X is [batch, N, T], W is [N, N]
Model output: u_hat [batch, N, T] (estimated stimulus)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class SpectralConv1d(nn.Module):
    """
    1D Fourier layer for time series.
    """
    
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        modes : int
            Number of Fourier modes to keep
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Fourier weights for multiplication in frequency domain
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spectral convolution.
        
        Parameters
        ----------
        x : torch.Tensor
            Input [batch, in_channels, length]
            
        Returns
        -------
        torch.Tensor
            Output [batch, out_channels, length]
        """
        batch, channels, length = x.shape
        
        # FFT
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # Truncate to modes
        x_ft = x_ft[:, :, :self.modes]
        
        # Multiply in Fourier space
        out_ft = torch.zeros(batch, self.out_channels, x_ft.shape[-1],
                             dtype=torch.cfloat, device=x.device)
        
        for i in range(self.in_channels):
            for j in range(self.out_channels):
                out_ft[:, j, :] += x_ft[:, i, :] * self.weights[i, j, :]
        
        # Pad back to original size
        out_ft_padded = torch.zeros(batch, self.out_channels, length // 2 + 1,
                                    dtype=torch.cfloat, device=x.device)
        out_ft_padded[:, :, :self.modes] = out_ft
        
        # Inverse FFT
        x = torch.fft.irfft(out_ft_padded, n=length, dim=-1)
        
        return x


class FNOBlock(nn.Module):
    """
    Single FNO block with spectral and local convolutions.
    """
    
    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.spectral_conv = SpectralConv1d(channels, channels, modes)
        self.conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input [batch, channels, length]
            
        Returns
        -------
        torch.Tensor
            Output [batch, channels, length]
        """
        x1 = self.spectral_conv(x)
        x2 = self.conv(x)
        x = x1 + x2
        x = self.activation(x)
        return x


class FNO(nn.Module):
    """
    Fourier Neural Operator for stimulus inversion from brain dynamics.
    
    Takes as input:
    - X: observed brain activity [batch, N, T]
    - W: connectivity matrix [N, N] or [batch, N, N]
    
    Outputs:
    - u_hat: estimated stimulus [batch, N, T]
    """
    
    def __init__(self, N: int, T: int, hidden_channels: int = 32,
                 n_layers: int = 4, modes: int = 16):
        """
        Initialize FNO model.
        
        Parameters
        ----------
        N : int
            Number of brain regions
        T : int
            Number of time points
        hidden_channels : int
            Number of hidden channels in FNO layers
        n_layers : int
            Number of FNO blocks
        modes : int
            Number of Fourier modes to use
        """
        super().__init__()
        
        self.N = N
        self.T = T
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.modes = modes
        
        # Encode connectivity as node features
        self.W_encoder = nn.Linear(N, hidden_channels // 2)
        
        # Initial projection: concatenate X and W features
        self.input_proj = nn.Conv1d(1 + hidden_channels // 2, hidden_channels, kernel_size=1)
        
        # FNO layers
        self.fno_layers = nn.ModuleList([
            FNOBlock(hidden_channels, modes) for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels // 2, 1, kernel_size=1)
        )
    
    def forward(self, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: invert stimulus from activity and connectivity.
        
        Parameters
        ----------
        X : torch.Tensor
            Observed brain activity [batch, N, T]
        W : torch.Tensor
            Connectivity matrix [N, N] or [batch, N, N]
            
        Returns
        -------
        torch.Tensor
            Estimated stimulus [batch, N, T]
        """
        batch, N, T = X.shape
        
        # Handle W dimension
        if W.dim() == 2:
            W = W.unsqueeze(0).expand(batch, -1, -1)  # [batch, N, N]
        
        # Encode connectivity: aggregate incoming connections for each node
        # W_features: [batch, N, hidden_channels // 2]
        W_features = self.W_encoder(W)  # [batch, N, hidden_channels // 2]
        
        # Expand W features across time
        W_features = W_features.unsqueeze(-1).expand(-1, -1, -1, T)  # [batch, N, hidden//2, T]
        
        # Prepare X
        X_expanded = X.unsqueeze(2)  # [batch, N, 1, T]
        
        # Concatenate X and W features
        x = torch.cat([X_expanded, W_features], dim=2)  # [batch, N, 1+hidden//2, T]
        
        # Reshape for processing: treat each spatial location independently
        # Flatten batch and spatial dimensions
        x = x.reshape(batch * N, 1 + self.hidden_channels // 2, T)
        
        # Initial projection
        x = self.input_proj(x)  # [batch*N, hidden_channels, T]
        
        # Apply FNO layers
        for layer in self.fno_layers:
            x = layer(x) + x  # Residual connection
        
        # Output projection
        x = self.output_proj(x)  # [batch*N, 1, T]
        
        # Reshape back
        u_hat = x.reshape(batch, N, T)
        
        return u_hat


if __name__ == '__main__':
    # Example usage
    print("Testing FNO model...")
    
    # Create model
    N, T = 50, 100
    model = FNO(N=N, T=T, hidden_channels=32, n_layers=4, modes=16)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy input
    batch_size = 4
    X = torch.randn(batch_size, N, T)
    W = torch.randn(N, N)
    
    # Forward pass
    u_hat = model(X, W)
    
    print(f"Input X shape: {X.shape}")
    print(f"Input W shape: {W.shape}")
    print(f"Output u_hat shape: {u_hat.shape}")
    print(f"Output range: [{u_hat.min():.3f}, {u_hat.max():.3f}]")
