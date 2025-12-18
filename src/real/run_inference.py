"""
Run inference on real fMRI task data.

This module provides a complete pipeline to load real fMRI data, preprocess it,
load a trained neural operator model, and infer task stimuli from brain activity.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional
from src.real.preprocess import prepare_for_inference
from src.model.inference import load_model_and_infer, save_inference_results


def infer_from_real_data(fmri_data_path: str,
                        checkpoint_path: str,
                        model_class: type,
                        model_kwargs: Dict,
                        output_path: str,
                        connectivity_path: Optional[str] = None,
                        TR: float = 2.0,
                        preprocess: bool = True,
                        device: str = 'cpu') -> Dict[str, np.ndarray]:
    """
    Complete pipeline for inferring stimuli from real fMRI data.
    
    Parameters
    ----------
    fmri_data_path : str
        Path to fMRI data file (.npz with key 'X' for time series [N, T] or [n_samples, N, T])
    checkpoint_path : str
        Path to trained model checkpoint
    model_class : type
        Model class (e.g., FNO)
    model_kwargs : dict
        Model initialization arguments
    output_path : str
        Path to save inference results
    connectivity_path : str, optional
        Path to connectivity matrix file (.npz with key 'W'). If None, will be estimated.
    TR : float
        Repetition time in seconds
    preprocess : bool
        Whether to preprocess fMRI data
    device : str
        Device for inference ('cpu' or 'cuda')
        
    Returns
    -------
    dict
        Dictionary with 'X', 'u_hat', 'W'
    """
    print(f"Loading fMRI data from {fmri_data_path}...")
    fmri_data = np.load(fmri_data_path)
    X = fmri_data['X']
    
    print(f"Input data shape: {X.shape}")
    
    # Load connectivity if provided
    W = None
    if connectivity_path is not None:
        print(f"Loading connectivity from {connectivity_path}...")
        conn_data = np.load(connectivity_path)
        W = conn_data['W']
        print(f"Connectivity shape: {W.shape}")
    
    # Preprocess and prepare data
    print("\nPreparing data for inference...")
    data = prepare_for_inference(X, W=W, TR=TR, preprocess=preprocess)
    X_proc = data['X']
    W = data['W']
    
    print(f"Preprocessed X shape: {X_proc.shape}")
    print(f"Connectivity W shape: {W.shape}")
    
    # Run inference
    print("\nRunning inference...")
    u_hat = load_model_and_infer(
        checkpoint_path=checkpoint_path,
        model_class=model_class,
        model_kwargs=model_kwargs,
        X=X_proc,
        W=W,
        device=device
    )
    
    print(f"Estimated stimulus shape: {u_hat.shape}")
    print(f"Stimulus range: [{u_hat.min():.3f}, {u_hat.max():.3f}]")
    
    # Save results
    print(f"\nSaving results to {output_path}...")
    save_inference_results(
        output_path=output_path,
        X=X_proc,
        u_hat=u_hat,
        W=W,
        meta={
            'fmri_data_path': fmri_data_path,
            'checkpoint_path': checkpoint_path,
            'TR': TR,
            'preprocessed': preprocess
        }
    )
    
    results = {
        'X': X_proc,
        'u_hat': u_hat,
        'W': W
    }
    
    return results


def batch_infer_from_directory(fmri_dir: str,
                               checkpoint_path: str,
                               model_class: type,
                               model_kwargs: Dict,
                               output_dir: str,
                               pattern: str = '*.npz',
                               **kwargs) -> None:
    """
    Run inference on all fMRI files in a directory.
    
    Parameters
    ----------
    fmri_dir : str
        Directory containing fMRI data files
    checkpoint_path : str
        Path to trained model checkpoint
    model_class : type
        Model class (e.g., FNO)
    model_kwargs : dict
        Model initialization arguments
    output_dir : str
        Directory to save results
    pattern : str
        File pattern to match (default: '*.npz')
    **kwargs
        Additional arguments passed to infer_from_real_data
    """
    fmri_dir = Path(fmri_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fmri_files = sorted(fmri_dir.glob(pattern))
    
    print(f"Found {len(fmri_files)} fMRI files in {fmri_dir}")
    
    for i, fmri_file in enumerate(fmri_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing file {i}/{len(fmri_files)}: {fmri_file.name}")
        print(f"{'='*60}")
        
        output_file = output_dir / f"inference_{fmri_file.stem}.npz"
        
        try:
            infer_from_real_data(
                fmri_data_path=str(fmri_file),
                checkpoint_path=checkpoint_path,
                model_class=model_class,
                model_kwargs=model_kwargs,
                output_path=str(output_file),
                **kwargs
            )
            print(f"✓ Successfully processed {fmri_file.name}")
        except Exception as e:
            print(f"✗ Error processing {fmri_file.name}: {str(e)}")
            continue


if __name__ == '__main__':
    # Example usage
    print("Testing inference on real data...")
    
    from src.model.fno import FNO
    
    # Create synthetic "real" fMRI data
    import tempfile
    
    N, T = 50, 100
    X_real = np.random.randn(N, T).astype(np.float32)
    
    # Save as temporary file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.npz', delete=False) as f:
        temp_fmri_path = f.name
    np.savez(temp_fmri_path, X=X_real)
    print(f"Created test fMRI data: {X_real.shape}")
    
    # Create a simple model (not trained)
    model = FNO(N=N, T=T, hidden_channels=16, n_layers=2, modes=8)
    
    # Save model checkpoint
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pt', delete=False) as f:
        temp_checkpoint_path = f.name
    torch.save({
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {},
        'val_loss': 0.0,
        'history': {}
    }, temp_checkpoint_path)
    print(f"Created test checkpoint")
    
    # Run inference
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.npz', delete=False) as f:
        temp_output_path = f.name
    results = infer_from_real_data(
        fmri_data_path=temp_fmri_path,
        checkpoint_path=temp_checkpoint_path,
        model_class=FNO,
        model_kwargs={'N': N, 'T': T, 'hidden_channels': 16, 'n_layers': 2, 'modes': 8},
        output_path=temp_output_path,
        TR=2.0,
        preprocess=True,
        device='cpu'
    )
    
    print("\nInference results:")
    print(f"  X shape: {results['X'].shape}")
    print(f"  u_hat shape: {results['u_hat'].shape}")
    print(f"  W shape: {results['W'].shape}")
    
    # Verify saved file
    loaded = np.load(temp_output_path, allow_pickle=True)
    print(f"\nSaved file contains keys: {list(loaded.keys())}")
