#!/usr/bin/env python
"""
Complete end-to-end example of the Neural Operator pipeline.

This script demonstrates the full workflow:
1. Generate synthetic brain dynamics
2. Train a Fourier Neural Operator
3. Run inference and evaluate results
4. Visualize and analyze outputs
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import modules
from src.sim.generate_ode import generate_ode_data
from src.model.fno import FNO
from src.model.trainer import Trainer
from src.model.inference import run_inference, evaluate_reconstruction
from src.analysis.viz import plot_comparison_panel, plot_training_curves
from src.analysis.stats import compute_all_metrics


def main():
    """Run complete pipeline example."""
    
    # Configuration
    N = 50              # Number of brain regions
    T = 100             # Number of time points
    n_train = 500       # Training samples
    n_val = 100         # Validation samples
    n_test = 100        # Test samples
    n_epochs = 30       # Training epochs
    device = 'cpu'      # Device for training
    
    print("="*60)
    print("Neural Operator Brain Dynamics Pipeline")
    print("="*60)
    
    # Step 1: Generate synthetic data
    print("\n[Step 1/5] Generating synthetic brain dynamics...")
    print(f"  N={N} regions, T={T} time points")
    
    data = generate_ode_data(
        N=N,
        T=T,
        n_samples=n_train + n_val + n_test,
        tau=1.0,
        alpha=0.5,
        stimulus_strength=2.0,
        noise_level=0.1,
        random_seed=42
    )
    
    print(f"  Generated {data['X'].shape[0]} samples")
    
    # Split data
    train_data = {
        'X': data['X'][:n_train],
        'u': data['u'][:n_train],
        'W': data['W']
    }
    
    val_data = {
        'X': data['X'][n_train:n_train+n_val],
        'u': data['u'][n_train:n_train+n_val],
        'W': data['W']
    }
    
    test_data = {
        'X': data['X'][n_train+n_val:],
        'u': data['u'][n_train+n_val:],
        'W': data['W']
    }
    
    print(f"  Train: {len(train_data['X'])}, Val: {len(val_data['X'])}, Test: {len(test_data['X'])}")
    
    # Step 2: Create and train model
    print(f"\n[Step 2/5] Training Fourier Neural Operator...")
    
    model = FNO(
        N=N,
        T=T,
        hidden_channels=32,
        n_layers=4,
        modes=12
    )
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = Trainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        learning_rate=1e-3,
        batch_size=32,
        device=device,
        checkpoint_dir='experiments/checkpoints'
    )
    
    history = trainer.train(
        n_epochs=n_epochs,
        save_best=True,
        early_stopping_patience=10
    )
    
    print(f"  Best validation loss: {min(history['val_loss']):.6f}")
    
    # Step 3: Load best model and run inference on test set
    print(f"\n[Step 3/5] Running inference on test set...")
    
    trainer.load_checkpoint('best_model.pt')
    u_hat = run_inference(
        model=trainer.model,
        X=test_data['X'],
        W=test_data['W'],
        device=device,
        batch_size=64
    )
    
    print(f"  Generated predictions for {len(u_hat)} test samples")
    
    # Step 4: Evaluate results
    print(f"\n[Step 4/5] Evaluating reconstruction quality...")
    
    # Basic metrics
    basic_metrics = evaluate_reconstruction(test_data['u'], u_hat)
    print("\n  Basic Metrics:")
    for key, value in basic_metrics.items():
        print(f"    {key:30s}: {value:.4f}")
    
    # Comprehensive metrics
    all_metrics = compute_all_metrics(
        u_true=test_data['u'],
        u_hat=u_hat,
        W=test_data['W']
    )
    
    print("\n  Advanced Metrics:")
    print(f"    SNR                           : {all_metrics['snr_db']:.2f} dB")
    print(f"    Spatial consistency           : {all_metrics['spatial_consistency']:.4f}")
    print(f"    Temporal consistency          : {all_metrics['temporal_consistency']:.4f}")
    
    # Step 5: Visualize results
    print(f"\n[Step 5/5] Creating visualizations...")
    
    # Create output directory
    output_dir = Path('experiments/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training curves
    fig1 = plot_training_curves(
        history,
        title="FNO Training Progress",
        save_path=output_dir / 'training_curves.png'
    )
    print(f"  ✓ Saved training curves")
    
    # Plot comparison for a few test samples
    for i in range(min(3, len(test_data['X']))):
        fig2 = plot_comparison_panel(
            X=test_data['X'][i],
            u_true=test_data['u'][i],
            u_hat=u_hat[i],
            W=test_data['W'],
            save_path=output_dir / f'comparison_sample_{i}.png'
        )
        print(f"  ✓ Saved comparison panel {i}")
        plt.close(fig2)
    
    # Save results
    results_path = output_dir / 'test_results.npz'
    np.savez(
        results_path,
        X=test_data['X'],
        u_true=test_data['u'],
        u_hat=u_hat,
        W=test_data['W'],
        metrics=all_metrics,
        history=history
    )
    print(f"  ✓ Saved results to {results_path}")
    
    # Summary
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    print("\nSummary:")
    print(f"  • Generated {n_train + n_val + n_test} samples of brain dynamics")
    print(f"  • Trained FNO model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  • Achieved test correlation: {basic_metrics['correlation']:.4f}")
    print(f"  • Achieved test R²: {basic_metrics['r2']:.4f}")
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - training_curves.png")
    print(f"  - comparison_sample_*.png")
    print(f"  - test_results.npz")
    print("\nCheckpoints saved to: experiments/checkpoints")
    print(f"  - best_model.pt")
    print(f"  - final_model.pt")
    

if __name__ == '__main__':
    main()
