# GNN Protein Atom Exposure Prediction

A Graph Neural Network (GNN) project using PyTorch Geometric to predict atom exposure levels in protein structures.

## Overview

This project predicts the burial depth of individual atoms within protein structures using Graph Neural Networks. Proteins are represented as graphs where atoms are nodes and bonds are edges, enriched with biochemical features.

**Current Best Result**: R² = 0.9408, MAE = 0.0655 (Phase 15 + Global Pooling)

## Quick Start

```bash
# Clone the repository
git clone [repository-url]
cd GNN_atom_exposure

# Install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train the best model
python main.py --config configs/phase15_globalpool.yaml

# Evaluate only (skip training)
python main.py --config configs/phase15_globalpool.yaml --eval-only
```

For detailed installation instructions, see [Getting Started](docs/GETTING_STARTED.md).

## Documentation

- **[Getting Started](docs/GETTING_STARTED.md)** - Installation, setup, and basic usage
- **[Architecture](docs/ARCHITECTURE.md)** - Model design, features, and design decisions
- **[Dataset](docs/DATASET.md)** - Data structure, features, and statistics
- **[History](docs/HISTORY.md)** - Project timeline and experimental milestones

## Project Structure

```
GNN_atom_exposure/
├── main.py                    # Training and evaluation entry point
├── requirements.txt           # Dependencies
├── configs/                   # Configuration files
│   ├── phase15_globalpool.yaml          # Best model config (R² = 0.9408)
│   └── phase15_globalpool_gaussian.yaml # + Gaussian noise regularization
├── src/                       # Source code
│   ├── data/                  # Dataset and feature engineering
│   │   ├── dataset_fixed.py             # ProteinAtomDataset class
│   │   ├── feature_engineering.py       # Feature extraction pipeline
│   │   ├── backbone_angles.py           # Phi/psi dihedral angles
│   │   └── aggregated_transforms.py     # Alternative feature transforms
│   ├── models/                # GNN architectures
│   │   └── gnn.py                       # AtomExposureGNN (GATv2, GCN, GAT, GIN, GINE)
│   ├── training/              # Training and evaluation
│   │   ├── train.py                     # Trainer class, loss functions
│   │   └── evaluate.py                  # Metrics (raw + clamped)
│   └── utils/                 # Utilities
│       ├── config.py                    # YAML config loader
│       ├── visualization.py             # Plots (heatmap, error, training curves)
│       └── data_validation.py           # Dataset integrity checks
├── experiments/               # Experiment results
│   └── phase15_globalpool/    # Best model results and plots
├── dataset/                   # Dataset (not in git, see docs/DATASET.md)
└── docs/                      # Documentation
```

## Current Results

| Phase | Model | R² | MAE | Key Change |
|-------|-------|-----|-----|------------|
| 3 | GCN | 0.484 | 0.204 | Baseline |
| 5 | GINE | 0.5684 | 0.1829 | Edge features + weighted loss |
| 12 | GATv2 | 0.8817 | 0.0866 | contact_count_10A feature |
| 13 | GATv2 | 0.8837 | 0.0846 | Range-Specific Weighted Loss |
| 15 | GATv2 | 0.8892 | 0.0838 | Optuna-tuned hyperparameters |
| **15+GP** | **GATv2** | **0.9408** | **0.0655** | **Dynamic Global Pooling** |

### Key Findings
- **Dynamic Global Pooling**: +5.8% R² (0.8892 -> 0.9408)
- **contact_count_10A is critical**: removing it drops R² from 0.8892 to 0.5689 (-36%)
- **+94% total improvement** from baseline GCN (R² 0.484 -> 0.9408)

## Architecture

**Best model**: GATv2 with Dynamic Global Pooling

- **93 node features** (biochemical + geometric + backbone angles + contact_count_10A)
- **12 edge features** (bond types, distances, radius graph)
- **4 GATv2 layers** with 4 attention heads, 136 hidden channels
- **Gated global pooling** injected after every layer (mean pooling)
- **Range-Specific Weighted Loss** for exposure distribution bias
- **OneCycleLR scheduler** with 50-epoch warmup
- **Early stopping on R²** for model selection
- **Gaussian noise injection** (optional, σ=0.02 on input features during training)

## Usage

### Training

```bash
# Train best model
python main.py --config configs/phase15_globalpool.yaml

# Train with Gaussian noise regularization
python main.py --config configs/phase15_globalpool_gaussian.yaml
```

### Evaluation

```bash
# Evaluate and regenerate plots
python main.py --config configs/phase15_globalpool.yaml --eval-only

# Evaluate with specific checkpoint
python main.py --config configs/phase15_globalpool.yaml --eval-only --checkpoint path/to/best_model.pt
```

### Output

Evaluation generates:
- `predictions_raw.png` - 2D density heatmap of raw predictions (shows negatives)
- `predictions_clamped.png` - 2D density heatmap of clamped predictions (exposure >= 0)
- `error_distribution.png` - Error histogram and box plot
- `error_by_exposure_range.png` - MAE, bias, and distribution by exposure range
- `training_curves.png` - Loss and R² curves
- `test_metrics.json` - Numerical metrics (MAE, RMSE, R², Pearson, etc.)

## Technology Stack

- **PyTorch** + **PyTorch Geometric** - GNN framework
- **Graphein** - Protein graph construction
- **scikit-learn** / **SciPy** - Evaluation metrics
- **Matplotlib** / **Seaborn** - Visualization

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- CUDA-enabled GPU (recommended)
- See [requirements.txt](requirements.txt) for complete list
