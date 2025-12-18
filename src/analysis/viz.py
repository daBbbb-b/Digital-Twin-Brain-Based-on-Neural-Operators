"""
Visualization utilities for brain dynamics and stimulus maps.

This module provides functions to visualize stimulus maps, brain activity,
connectivity matrices, and training curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from pathlib import Path


def visualize_stimulus_map(u: np.ndarray,
                           u_hat: Optional[np.ndarray] = None,
                           title: str = "Stimulus Map",
                           cmap: str = 'RdBu_r',
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
    """
    Visualize stimulus map as spatiotemporal heatmap.
    
    Parameters
    ----------
    u : np.ndarray
        True stimulus [N, T]
    u_hat : np.ndarray, optional
        Estimated stimulus [N, T]
    title : str
        Plot title
    cmap : str
        Colormap name
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    if u_hat is not None:
        fig, axes = plt.subplots(1, 3, figsize=(figsize[0]*1.5, figsize[1]))
        
        # True stimulus
        im1 = axes[0].imshow(u, aspect='auto', cmap=cmap, interpolation='nearest')
        axes[0].set_title('True Stimulus')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Brain Region')
        plt.colorbar(im1, ax=axes[0])
        
        # Estimated stimulus
        vmin, vmax = u.min(), u.max()
        im2 = axes[1].imshow(u_hat, aspect='auto', cmap=cmap, interpolation='nearest',
                            vmin=vmin, vmax=vmax)
        axes[1].set_title('Estimated Stimulus')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Brain Region')
        plt.colorbar(im2, ax=axes[1])
        
        # Difference
        diff = u - u_hat
        im3 = axes[2].imshow(diff, aspect='auto', cmap='seismic', interpolation='nearest')
        axes[2].set_title('Difference (True - Est)')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Brain Region')
        plt.colorbar(im3, ax=axes[2])
        
        fig.suptitle(title, fontsize=14, y=1.02)
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(u, aspect='auto', cmap=cmap, interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Brain Region')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_timecourse(X: np.ndarray,
                   roi_indices: Optional[List[int]] = None,
                   title: str = "Brain Activity Time Course",
                   save_path: Optional[str] = None,
                   figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot time courses for selected ROIs.
    
    Parameters
    ----------
    X : np.ndarray
        Brain activity [N, T]
    roi_indices : list of int, optional
        ROI indices to plot. If None, plots first 5 ROIs.
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    N, T = X.shape
    
    if roi_indices is None:
        roi_indices = list(range(min(5, N)))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    time = np.arange(T)
    for i in roi_indices:
        ax.plot(time, X[i, :], label=f'ROI {i}', alpha=0.7)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Activity')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_connectivity_matrix(W: np.ndarray,
                            title: str = "Connectivity Matrix",
                            cmap: str = 'coolwarm',
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (8, 7)) -> plt.Figure:
    """
    Visualize connectivity matrix as heatmap.
    
    Parameters
    ----------
    W : np.ndarray
        Connectivity matrix [N, N]
    title : str
        Plot title
    cmap : str
        Colormap name
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot connectivity matrix
    vmax = max(abs(W.min()), abs(W.max()))
    im = ax.imshow(W, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
    
    ax.set_xlabel('Source Region')
    ax.set_ylabel('Target Region')
    ax.set_title(title)
    
    plt.colorbar(im, ax=ax, label='Connection Strength')
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_training_curves(history: dict,
                         title: str = "Training Curves",
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 5)) -> plt.Figure:
    """
    Plot training and validation loss curves.
    
    Parameters
    ----------
    history : dict
        Training history with keys 'train_loss', 'val_loss', 'epoch'
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = history['epoch']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    
    ax.plot(epochs, train_loss, label='Train Loss', marker='o', linewidth=2)
    ax.plot(epochs, val_loss, label='Val Loss', marker='s', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add best validation loss marker
    if val_loss:
        best_epoch = np.argmin(val_loss)
        best_val = val_loss[best_epoch]
        ax.axvline(epochs[best_epoch], color='red', linestyle='--', alpha=0.5,
                  label=f'Best Val (epoch {epochs[best_epoch]})')
        ax.plot(epochs[best_epoch], best_val, 'r*', markersize=15)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_comparison_panel(X: np.ndarray,
                         u_true: np.ndarray,
                         u_hat: np.ndarray,
                         W: np.ndarray,
                         sample_idx: int = 0,
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
    """
    Create comprehensive comparison panel with activity, stimuli, and connectivity.
    
    Parameters
    ----------
    X : np.ndarray
        Brain activity [n_samples, N, T] or [N, T]
    u_true : np.ndarray
        True stimulus [n_samples, N, T] or [N, T]
    u_hat : np.ndarray
        Estimated stimulus [n_samples, N, T] or [N, T]
    W : np.ndarray
        Connectivity matrix [N, N]
    sample_idx : int
        Sample index to visualize (if batch input)
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Handle batch dimension
    if X.ndim == 3:
        X = X[sample_idx]
        u_true = u_true[sample_idx]
        u_hat = u_hat[sample_idx]
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Brain activity
    ax1 = fig.add_subplot(gs[0, :])
    im1 = ax1.imshow(X, aspect='auto', cmap='viridis', interpolation='nearest')
    ax1.set_title('Observed Brain Activity (X)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Region')
    plt.colorbar(im1, ax=ax1)
    
    # True stimulus
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(u_true, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax2.set_title('True Stimulus (u)', fontsize=11)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Region')
    plt.colorbar(im2, ax=ax2)
    
    # Estimated stimulus
    ax3 = fig.add_subplot(gs[1, 1])
    vmin, vmax = u_true.min(), u_true.max()
    im3 = ax3.imshow(u_hat, aspect='auto', cmap='RdBu_r', interpolation='nearest',
                    vmin=vmin, vmax=vmax)
    ax3.set_title('Estimated Stimulus (û)', fontsize=11)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Region')
    plt.colorbar(im3, ax=ax3)
    
    # Difference
    ax4 = fig.add_subplot(gs[1, 2])
    diff = u_true - u_hat
    im4 = ax4.imshow(diff, aspect='auto', cmap='seismic', interpolation='nearest')
    ax4.set_title('Error (u - û)', fontsize=11)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Region')
    plt.colorbar(im4, ax=ax4)
    
    # Connectivity matrix
    ax5 = fig.add_subplot(gs[2, 0])
    vmax_w = max(abs(W.min()), abs(W.max()))
    im5 = ax5.imshow(W, cmap='coolwarm', vmin=-vmax_w, vmax=vmax_w, aspect='auto')
    ax5.set_title('Connectivity (W)', fontsize=11)
    ax5.set_xlabel('Source')
    ax5.set_ylabel('Target')
    plt.colorbar(im5, ax=ax5)
    
    # Time courses
    ax6 = fig.add_subplot(gs[2, 1:])
    time = np.arange(u_true.shape[1])
    # Select a few regions to plot
    N = u_true.shape[0]
    roi_indices = np.linspace(0, N-1, min(5, N), dtype=int)
    for i in roi_indices:
        ax6.plot(time, u_true[i, :], '--', alpha=0.6, label=f'True ROI {i}')
        ax6.plot(time, u_hat[i, :], '-', alpha=0.8, label=f'Est ROI {i}')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Stimulus')
    ax6.set_title('Stimulus Time Courses', fontsize=11)
    ax6.legend(fontsize=8, ncol=2)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Brain Dynamics and Stimulus Inversion', fontsize=14, fontweight='bold')
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


if __name__ == '__main__':
    # Example usage
    print("Testing visualization functions...")
    
    # Create synthetic data
    N, T = 50, 100
    X = np.random.randn(N, T)
    u_true = np.random.randn(N, T)
    u_hat = u_true + np.random.randn(N, T) * 0.3
    W = np.random.randn(N, N) * 0.2
    np.fill_diagonal(W, 0)
    
    # Test stimulus map
    print("\n1. Stimulus map...")
    fig1 = visualize_stimulus_map(u_true, u_hat, title="Test Stimulus Map")
    
    # Test time course
    print("2. Time course...")
    fig2 = plot_timecourse(X, roi_indices=[0, 10, 20], title="Test Time Course")
    
    # Test connectivity
    print("3. Connectivity matrix...")
    fig3 = plot_connectivity_matrix(W, title="Test Connectivity")
    
    # Test training curves
    print("4. Training curves...")
    history = {
        'epoch': list(range(10)),
        'train_loss': [1.0 - 0.05*i + np.random.rand()*0.1 for i in range(10)],
        'val_loss': [1.0 - 0.04*i + np.random.rand()*0.1 for i in range(10)]
    }
    fig4 = plot_training_curves(history, title="Test Training Curves")
    
    # Test comparison panel
    print("5. Comparison panel...")
    fig5 = plot_comparison_panel(X, u_true, u_hat, W)
    
    print("\nAll visualizations created successfully!")
    plt.show()
