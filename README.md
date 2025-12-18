# Digital Twin Brain Based on Neural Operators

A research repository for modeling brain dynamics using neural operators, with applications to task stimulus inversion from fMRI data.

## Overview

This repository implements a complete pipeline for:
1. **Generating synthetic brain dynamics** using ODE and PDE models
2. **Training neural operators** (Fourier Neural Operator) to invert latent task stimuli from observed brain activity
3. **Applying trained models** to real fMRI task data
4. **Visualizing and evaluating** stimulus maps and their consistency with functional brain organization

### Scientific Background

The goal is to learn an inverse mapping from observed brain activity **X** (fMRI BOLD signals) and connectivity structure **W** to underlying task stimuli **u**:

```
(X, W) → u_hat
```

Where:
- **X**: Brain activity time series [N regions × T time points]
- **W**: Effective connectivity matrix [N × N]
- **u**: Task stimulus/drive [N × T]

This is a challenging inverse problem because multiple stimulus patterns can produce similar brain dynamics, and the mapping involves complex nonlinear neural dynamics.

## Repository Structure

```
.
├── data/
│   ├── raw/              # Raw fMRI data
│   └── sim/              # Simulated brain dynamics
├── src/
│   ├── sim/              # Data generation
│   │   ├── generate_ode.py    # ODE-based simulation
│   │   └── generate_pde.py    # PDE-based simulation
│   ├── ec/               # Effective connectivity
│   │   └── surrogate.py       # EC estimation methods
│   ├── model/            # Neural operator models
│   │   ├── fno.py            # Fourier Neural Operator
│   │   ├── trainer.py        # Training utilities
│   │   └── inference.py      # Inference pipeline
│   ├── real/             # Real data processing
│   │   ├── preprocess.py     # fMRI preprocessing
│   │   └── run_inference.py  # Apply models to real data
│   └── analysis/         # Analysis and visualization
│       ├── viz.py            # Visualization tools
│       └── stats.py          # Statistical analysis
├── experiments/
│   └── configs/          # Configuration files
│       └── default_config.yaml
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

### Requirements
- Python 3.9+
- PyTorch 1.10+

### Setup

```bash
# Clone the repository
git clone https://github.com/daBbbb-b/Digital-Twin-Brain-Based-on-Neural-Operators.git
cd Digital-Twin-Brain-Based-on-Neural-Operators

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Synthetic Data

```python
from src.sim.generate_ode import generate_ode_data
import numpy as np

# Generate ODE-based brain dynamics
data = generate_ode_data(
    N=50,              # Number of brain regions
    T=100,             # Number of time points
    n_samples=1000,    # Number of samples
    random_seed=42
)

# Save data
np.savez('data/sim/training_data.npz', **data)
```

### 2. Train Neural Operator Model

```python
from src.model.fno import FNO
from src.model.trainer import Trainer
import numpy as np

# Load data
data = np.load('data/sim/training_data.npz', allow_pickle=True)
train_data = {
    'X': data['X'][:800],
    'u': data['u'][:800],
    'W': data['W']
}
val_data = {
    'X': data['X'][800:],
    'u': data['u'][800:],
    'W': data['W']
}

# Create model
N, T = 50, 100
model = FNO(N=N, T=T, hidden_channels=64, n_layers=4, modes=16)

# Train
trainer = Trainer(
    model=model,
    train_data=train_data,
    val_data=val_data,
    learning_rate=1e-3,
    batch_size=32,
    device='cpu'
)

history = trainer.train(n_epochs=100, save_best=True, early_stopping_patience=20)
```

### 3. Run Inference on Real Data

```python
from src.real.run_inference import infer_from_real_data
from src.model.fno import FNO

# Run inference
results = infer_from_real_data(
    fmri_data_path='data/raw/subject01_task.npz',
    checkpoint_path='experiments/checkpoints/best_model.pt',
    model_class=FNO,
    model_kwargs={'N': 50, 'T': 100, 'hidden_channels': 64, 'n_layers': 4, 'modes': 16},
    output_path='experiments/results/subject01_inference.npz',
    TR=2.0,
    preprocess=True,
    device='cpu'
)
```

### 4. Visualize Results

```python
from src.analysis.viz import plot_comparison_panel
from src.analysis.stats import compute_consistency_metrics
import numpy as np
import matplotlib.pyplot as plt

# Load results
results = np.load('experiments/results/subject01_inference.npz', allow_pickle=True)
X = results['X']
u_hat = results['u_hat']
W = results['W']

# Visualize (if ground truth available)
if 'u_true' in results:
    u_true = results['u_true']
    
    # Create comparison panel
    fig = plot_comparison_panel(X, u_true, u_hat, W)
    plt.savefig('experiments/results/comparison.png', dpi=300)
    
    # Compute metrics
    metrics = compute_consistency_metrics(u_true, u_hat)
    print("Reconstruction metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
```

## Complete Pipeline Example

Here's a complete example running the entire pipeline:

```python
import numpy as np
from src.sim.generate_ode import generate_ode_data
from src.model.fno import FNO
from src.model.trainer import Trainer
from src.model.inference import run_inference, evaluate_reconstruction
from src.analysis.viz import plot_comparison_panel
import matplotlib.pyplot as plt

# 1. Generate synthetic data
print("Generating synthetic brain dynamics...")
data = generate_ode_data(N=50, T=100, n_samples=1000, random_seed=42)

# Split into train/val/test
train_data = {
    'X': data['X'][:700],
    'u': data['u'][:700],
    'W': data['W']
}
val_data = {
    'X': data['X'][700:900],
    'u': data['u'][700:900],
    'W': data['W']
}
test_data = {
    'X': data['X'][900:],
    'u': data['u'][900:],
    'W': data['W']
}

# 2. Train model
print("\nTraining FNO model...")
model = FNO(N=50, T=100, hidden_channels=64, n_layers=4, modes=16)
trainer = Trainer(model, train_data, val_data, learning_rate=1e-3, batch_size=32)
history = trainer.train(n_epochs=50, save_best=True, early_stopping_patience=10)

# 3. Test model
print("\nTesting model...")
trainer.load_checkpoint('best_model.pt')
u_hat = run_inference(model, test_data['X'], test_data['W'])

# 4. Evaluate
print("\nEvaluating reconstruction...")
metrics = evaluate_reconstruction(test_data['u'], u_hat)
for key, value in metrics.items():
    print(f"  {key}: {value:.4f}")

# 5. Visualize
print("\nVisualizing results...")
fig = plot_comparison_panel(
    test_data['X'][0],
    test_data['u'][0],
    u_hat[0],
    test_data['W']
)
plt.savefig('results.png', dpi=300)
print("Saved visualization to results.png")
```

## Data Format

All data files are saved as `.npz` (compressed NumPy archives) with the following structure:

### Simulation Data
- **X**: State/activity time series `[n_samples, N, T]` - observed brain dynamics
- **u**: Stimulus time series `[n_samples, N, T]` - ground truth task drive
- **W**: Connectivity matrix `[N, N]` - structural/effective connectivity
- **meta**: Dictionary with simulation parameters

### Real fMRI Data
- **X**: Preprocessed fMRI time series `[N, T]` or `[n_samples, N, T]`
- **W**: Connectivity matrix `[N, N]` (optional, will be estimated if not provided)

### Model Output
- **X**: Input brain activity
- **u_hat**: Estimated stimulus `[n_samples, N, T]`
- **W**: Connectivity used
- **metrics**: Evaluation metrics (if ground truth available)
- **meta**: Metadata

## Model Architecture

### Fourier Neural Operator (FNO)

The FNO is a neural operator that learns mappings between function spaces. Key features:

- **Global receptive field**: Fourier transforms capture long-range dependencies
- **Efficient**: O(N log N) complexity via FFT
- **Mesh-invariant**: Can handle different spatial/temporal resolutions

**Architecture overview:**
1. **Input encoding**: Concatenate brain activity X with connectivity-derived features
2. **Fourier layers**: Multiple spectral convolution layers operating in frequency domain
3. **Skip connections**: Residual connections between layers
4. **Output projection**: Map to stimulus space

**Input**: (X, W) where X is [batch, N, T], W is [N, N]  
**Output**: u_hat [batch, N, T]

## Preprocessing Pipeline

For real fMRI data, the following preprocessing steps are applied:

1. **Detrending**: Remove linear trends
2. **Bandpass filtering**: 0.01-0.1 Hz (typical for task fMRI)
3. **Normalization**: Z-score normalization per ROI
4. **Optional temporal smoothing**: Gaussian smoothing

Connectivity can be:
- Provided externally (structural connectivity from DTI)
- Estimated from data using ridge regression, correlation, or Granger causality

## Evaluation Metrics

The repository computes comprehensive metrics:

### Reconstruction Quality
- **MSE, RMSE, MAE**: Error metrics
- **Pearson/Spearman correlation**: Linear/rank correlation
- **R²**: Coefficient of determination
- **SNR**: Signal-to-noise ratio

### Consistency Metrics
- **Spatial consistency**: Agreement between stimulus and connectivity structure
- **Temporal consistency**: Smoothness and autocorrelation
- **Functional map consistency**: Overlap with known functional organization

## Configuration

Experiments can be configured using YAML files in `experiments/configs/`. See `default_config.yaml` for all available options.

## Examples

Additional examples are provided in each module's `__main__` section. Run any module directly to see example usage:

```bash
# Test ODE simulation
python -m src.sim.generate_ode

# Test PDE simulation
python -m src.sim.generate_pde

# Test FNO model
python -m src.model.fno

# Test training
python -m src.model.trainer

# Test preprocessing
python -m src.real.preprocess

# Test visualization
python -m src.analysis.viz

# Test statistics
python -m src.analysis.stats
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{digital_twin_brain_neural_operators,
  title = {Digital Twin Brain Based on Neural Operators},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/daBbbb-b/Digital-Twin-Brain-Based-on-Neural-Operators}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Fourier Neural Operator architecture based on Li et al. (2020)
- Brain dynamics models inspired by neural mass and neural field literature
- fMRI preprocessing follows standard neuroimaging practices

## References

1. Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020). Fourier neural operator for parametric partial differential equations. arXiv preprint arXiv:2010.08895.

2. Deco, G., Jirsa, V. K., & McIntosh, A. R. (2011). Emerging concepts for the dynamical organization of resting-state activity in the brain. Nature Reviews Neuroscience, 12(1), 43-56.

3. Friston, K. J., Harrison, L., & Penny, W. (2003). Dynamic causal modelling. NeuroImage, 19(4), 1273-1302.

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.