#!/usr/bin/env python
"""Quick verification that the pipeline works end-to-end."""

import numpy as np
from src.sim.generate_ode import generate_ode_data
from src.model.fno import FNO
from src.model.trainer import Trainer
from src.model.inference import run_inference, evaluate_reconstruction

print("Quick Pipeline Verification")
print("="*50)

# Generate small dataset
print("\n1. Generating data...")
data = generate_ode_data(N=20, T=50, n_samples=50, random_seed=42)
print(f"   Generated: X={data['X'].shape}, u={data['u'].shape}")

# Split data
train_data = {'X': data['X'][:30], 'u': data['u'][:30], 'W': data['W']}
val_data = {'X': data['X'][30:40], 'u': data['u'][30:40], 'W': data['W']}
test_data = {'X': data['X'][40:], 'u': data['u'][40:], 'W': data['W']}

# Create and train model
print("\n2. Training model (3 epochs)...")
model = FNO(N=20, T=50, hidden_channels=16, n_layers=2, modes=8)
print(f"   Model params: {sum(p.numel() for p in model.parameters()):,}")

trainer = Trainer(model, train_data, val_data, batch_size=10, learning_rate=1e-3)
history = trainer.train(n_epochs=3)
print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
print(f"   Final val loss: {history['val_loss'][-1]:.4f}")

# Test inference
print("\n3. Running inference...")
trainer.load_checkpoint('best_model.pt')
u_hat = run_inference(trainer.model, test_data['X'], test_data['W'])
print(f"   Predicted: {u_hat.shape}")

# Evaluate
print("\n4. Evaluation metrics:")
metrics = evaluate_reconstruction(test_data['u'], u_hat)
for key, val in metrics.items():
    print(f"   {key:30s}: {val:.4f}")

print("\n" + "="*50)
print("✓ Pipeline verification complete!")
