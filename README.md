# GNN Protein Atom Exposure Prediction

A Graph Neural Network (GNN) for predicting atom burial depth in protein structures using PyTorch Geometric.

## Overview

This project predicts the exposure level of individual atoms within protein structures. Proteins are represented as graphs where atoms are nodes and bonds are edges.

Two model architectures are available:

| Model | R² | Parameters | Features | Config |
|-------|-----|-----------|----------|--------|
| **GATv2 + Global Pooling** | **0.9408** | ~500K | 93 node + 12 edge | `configs/phase15_globalpool.yaml` |
| **MinimalGCN + Global Pooling** | **0.87-0.89** | ~16K | 5 node only | `configs/minimal_gcn_globalpool.yaml` |

## Quick Start

```bash
# Install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train the best model (GATv2)
python main.py --config configs/phase15_globalpool.yaml

# Train the minimal model (GCN)
python main.py --config configs/minimal_gcn_globalpool.yaml

# Evaluate only (skip training)
python main.py --config configs/phase15_globalpool.yaml --eval-only
python main.py --config configs/minimal_gcn_globalpool.yaml --eval-only
```

## Model Architectures

### GATv2 + Dynamic Global Pooling (Best)

- **93 node features** (biochemical + geometric + backbone angles + contact_count_10A)
- **12 edge features** (bond types, distances, radius graph)
- **4 GATv2 layers** with 4 attention heads, 136 hidden channels
- **Gated global pooling** injected after every layer (mean pooling)
- **Range-Specific Weighted Loss** for exposure distribution bias
- **OneCycleLR scheduler** with 50-epoch warmup
- **Early stopping on R²** for model selection
- **Gaussian noise injection** (optional, σ=0.02 on input features during training)

### MinimalGCN (Lightweight)

A streamlined model derived from systematic ablation of the GATv2 architecture:

- **5 geometric features** per atom (contact_count_10A dominates)
- **2 GCN layers**, 64 hidden channels
- **No edge features** (ablation showed 0 importance)
- **Gated global pooling** for protein-level context
- **~16K parameters** (~30x fewer than GATv2)

#### Features Used (MinimalGCN)

| Feature | Description | Importance |
|---------|-------------|------------|
| `contact_count_10A` | Neighbor atoms within 10Å | **Dominant** |
| `dist_to_center` | Distance to protein center | High |
| `radial_position` | Normalized radial position | High |
| `3rd_nearest_dist` | Distance to 3rd nearest atom | Medium |
| `std_dist` | Std dev of neighbor distances | Medium |

## Project Structure

```
GNN_atom_exposure/
├── main.py                    # Training and evaluation entry point
├── requirements.txt           # Dependencies
├── normalizer_train.pkl       # Feature normalization
├── configs/                   # Configuration files
│   ├── phase15_globalpool.yaml          # GATv2 best model (R² = 0.9408)
│   ├── phase15_globalpool_gaussian.yaml # GATv2 + Gaussian noise regularization
│   ├── minimal_gcn_globalpool.yaml      # MinimalGCN with global pooling
│   └── minimal_gcn.yaml                # MinimalGCN without global pooling
├── src/                       # Source code
│   ├── data/                  # Dataset and feature engineering
│   │   ├── dataset_fixed.py             # ProteinAtomDataset class
│   │   ├── feature_engineering.py       # Feature extraction pipeline
│   │   ├── backbone_angles.py           # Phi/psi dihedral angles
│   │   └── aggregated_transforms.py     # Alternative feature transforms
│   ├── models/                # GNN architectures
│   │   └── gnn.py                       # AtomExposureGNN, MinimalGCN, SimpleGCN
│   ├── training/              # Training and evaluation
│   │   ├── train.py                     # Trainer class, loss functions
│   │   └── evaluate.py                  # Metrics (raw + clamped)
│   └── utils/                 # Utilities
│       ├── config.py                    # YAML config loader
│       ├── visualization.py             # Plots (heatmap, error, training curves)
│       └── data_validation.py           # Dataset integrity checks
├── experiments/               # Experiment results
│   ├── phase15_globalpool/    # GATv2 results and plots
│   └── minimal_gcn_globalpool/# MinimalGCN results, plots, and best model checkpoint
├── dataset/                   # Dataset (not in git, see docs/DATASET.md)
└── docs/                      # Documentation
    ├── ARCHITECTURE.md        # Model design details
    ├── DATASET.md             # Data format and features
    ├── GETTING_STARTED.md     # Setup and installation
    └── HISTORY.md             # Project timeline
```

## Results

| Phase | Model | R² | MAE | Key Change |
|-------|-------|-----|-----|------------|
| 3 | GCN | 0.484 | 0.204 | Baseline |
| 5 | GINE | 0.5684 | 0.1829 | Edge features + weighted loss |
| 12 | GATv2 | 0.8817 | 0.0866 | contact_count_10A feature |
| 13 | GATv2 | 0.8837 | 0.0846 | Range-Specific Weighted Loss |
| 15 | GATv2 | 0.8892 | 0.0838 | Optuna-tuned hyperparameters |
| **15+GP** | **GATv2** | **0.9408** | **0.0655** | **Dynamic Global Pooling** |
| MinimalGCN | GCN | 0.87-0.89 | — | 5 features, 16K params |

### Key Findings

- **Dynamic Global Pooling**: +5.8% R² (0.8892 → 0.9408)
- **contact_count_10A is critical**: removing it drops R² from 0.8892 to 0.5689 (-36%)
- **+94% total improvement** from baseline GCN (R² 0.484 → 0.9408)
- **GCN ≈ GAT**: Simple GCN matches GATv2 attention performance for this task
- **Edge features have 0 importance**: Removing edge features has no impact
- **5 features suffice**: Reduced from 93 features with minimal accuracy loss

## Usage

### Training

```bash
# Train GATv2 best model
python main.py --config configs/phase15_globalpool.yaml

# Train with Gaussian noise regularization
python main.py --config configs/phase15_globalpool_gaussian.yaml

# Train MinimalGCN
python main.py --config configs/minimal_gcn_globalpool.yaml

# Custom parameters
python main.py --config configs/minimal_gcn_globalpool.yaml --batch-size 64 --epochs 200 --lr 0.0005
```

### Evaluation

```bash
# Evaluate GATv2
python main.py --config configs/phase15_globalpool.yaml --eval-only

# Evaluate MinimalGCN with specific checkpoint
python main.py --config configs/minimal_gcn_globalpool.yaml --eval-only --checkpoint experiments/minimal_gcn_globalpool/best_model.pt
```

### Python API

```python
from src.data.dataset_fixed import ProteinAtomDataset
from src.models.gnn import create_model
from torch_geometric.loader import DataLoader

# Load data
dataset = ProteinAtomDataset(
    root='dataset/',
    split='test',
    feature_config={'use_minimal_features': True}
)
loader = DataLoader(dataset, batch_size=64)

# Create MinimalGCN model
model = create_model({
    'model_type': 'minimal',
    'in_channels': 5,
    'hidden_channels': 64,
    'num_layers': 2,
    'use_global_pool': True,
    'global_pool_type': 'mean'
})
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

## Documentation

- **[Getting Started](docs/GETTING_STARTED.md)** - Installation, setup, and basic usage
- **[Architecture](docs/ARCHITECTURE.md)** - Model design, features, and design decisions
- **[Dataset](docs/DATASET.md)** - Data structure, features, and statistics
- **[History](docs/HISTORY.md)** - Project timeline and experimental milestones
