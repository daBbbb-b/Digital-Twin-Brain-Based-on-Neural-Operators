"""
Training module for neural operator models.

This module provides a Trainer class for training FNO models to invert stimuli
from brain dynamics, with support for validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Optional, Callable
from pathlib import Path
import json
from tqdm import tqdm


class BrainDynamicsDataset(Dataset):
    """
    Dataset for brain dynamics with stimulus inversion task.
    """
    
    def __init__(self, X: np.ndarray, u: np.ndarray, W: np.ndarray):
        """
        Parameters
        ----------
        X : np.ndarray
            Observed brain activity [n_samples, N, T]
        u : np.ndarray
            Task stimuli [n_samples, N, T]
        W : np.ndarray
            Connectivity matrix [N, N]
        """
        self.X = torch.from_numpy(X).float()
        self.u = torch.from_numpy(u).float()
        self.W = torch.from_numpy(W).float()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'X': self.X[idx],
            'u': self.u[idx],
            'W': self.W
        }


class Trainer:
    """
    Trainer for neural operator models.
    """
    
    def __init__(self, model: nn.Module, 
                 train_data: Dict[str, np.ndarray],
                 val_data: Optional[Dict[str, np.ndarray]] = None,
                 learning_rate: float = 1e-3,
                 batch_size: int = 16,
                 device: str = 'cpu',
                 checkpoint_dir: str = 'experiments/checkpoints'):
        """
        Initialize trainer.
        
        Parameters
        ----------
        model : nn.Module
            Neural operator model (e.g., FNO)
        train_data : dict
            Training data with keys 'X', 'u', 'W'
        val_data : dict, optional
            Validation data with keys 'X', 'u', 'W'
        learning_rate : float
            Learning rate for optimizer
        batch_size : int
            Batch size for training
        device : str
            Device to train on ('cpu' or 'cuda')
        checkpoint_dir : str
            Directory to save checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Create datasets
        self.train_dataset = BrainDynamicsDataset(
            train_data['X'], train_data['u'], train_data['W']
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        
        if val_data is not None:
            self.val_dataset = BrainDynamicsDataset(
                val_data['X'], val_data['u'], val_data['W']
            )
            self.val_loader = DataLoader(
                self.val_dataset, batch_size=batch_size, shuffle=False
            )
        else:
            self.val_loader = None
        
        # Optimizer and loss
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns
        -------
        float
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(self.train_loader, desc='Training', leave=False):
            X = batch['X'].to(self.device)
            u = batch['u'].to(self.device)
            W = batch['W'].to(self.device)
            
            # Forward pass
            u_hat = self.model(X, W)
            loss = self.criterion(u_hat, u)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> float:
        """
        Validate on validation set.
        
        Returns
        -------
        float
            Average validation loss
        """
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation', leave=False):
                X = batch['X'].to(self.device)
                u = batch['u'].to(self.device)
                W = batch['W'].to(self.device)
                
                u_hat = self.model(X, W)
                loss = self.criterion(u_hat, u)
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, n_epochs: int, 
              save_best: bool = True,
              early_stopping_patience: Optional[int] = None) -> Dict[str, list]:
        """
        Train model for multiple epochs.
        
        Parameters
        ----------
        n_epochs : int
            Number of epochs to train
        save_best : bool
            Whether to save best model based on validation loss
        early_stopping_patience : int, optional
            Stop if validation loss doesn't improve for this many epochs
            
        Returns
        -------
        dict
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['epoch'].append(epoch)
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_model.pt', epoch, val_loss)
                print(f"Saved best model (val_loss: {val_loss:.6f})")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break
        
        # Save final model
        self.save_checkpoint('final_model.pt', n_epochs, val_loss)
        
        return self.history
    
    def save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        """
        Save model checkpoint.
        
        Parameters
        ----------
        filename : str
            Checkpoint filename
        epoch : int
            Current epoch
        val_loss : float
            Validation loss
        """
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }, checkpoint_path)
    
    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint.
        
        Parameters
        ----------
        filename : str
            Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.6f}")


if __name__ == '__main__':
    # Example usage
    print("Testing Trainer...")
    
    from src.model.fno import FNO
    
    # Create synthetic data
    N, T = 50, 100
    n_train, n_val = 100, 20
    
    X_train = np.random.randn(n_train, N, T).astype(np.float32)
    u_train = np.random.randn(n_train, N, T).astype(np.float32)
    W = np.random.randn(N, N).astype(np.float32)
    
    X_val = np.random.randn(n_val, N, T).astype(np.float32)
    u_val = np.random.randn(n_val, N, T).astype(np.float32)
    
    train_data = {'X': X_train, 'u': u_train, 'W': W}
    val_data = {'X': X_val, 'u': u_val, 'W': W}
    
    # Create model and trainer
    model = FNO(N=N, T=T, hidden_channels=16, n_layers=2, modes=8)
    trainer = Trainer(model, train_data, val_data, batch_size=16, learning_rate=1e-3)
    
    # Train for a few epochs
    print("\nTraining model...")
    history = trainer.train(n_epochs=3, save_best=True)
    
    print(f"\nTraining complete!")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
