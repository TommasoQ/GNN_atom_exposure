# GNN Protein Atom Exposure Prediction

A Graph Neural Network (GNN) for predicting atom burial depth in protein structures using PyTorch Geometric.

## Overview

This project predicts the exposure level of individual atoms within protein structures. Proteins are represented as graphs where atoms are nodes and bonds are edges.

**Model**: MinimalGCN with Global Pooling
**Performance**: R² = 0.87-0.89, ~16K parameters
**Features**: 5 geometric features only

## Quick Start

```bash
# Install dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train
python main.py --config configs/minimal_gcn_globalpool.yaml

# Evaluate only
python main.py --config configs/minimal_gcn_globalpool.yaml --eval-only
```

## Model Architecture

The MinimalGCN is a streamlined 2-layer GCN that achieves strong performance with minimal complexity:

- **Input**: 5 geometric features per atom
- **Architecture**: 2 GCN layers, 64 hidden channels
- **Global Pooling**: Gated protein-level context injection
- **Parameters**: ~16,000

### Features Used

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
├── main.py                     # Training entry point
├── requirements.txt            # Dependencies
├── normalizer_train.pkl        # Feature normalization
├── configs/
│   ├── minimal_gcn_globalpool.yaml   # Primary config
│   └── minimal_gcn.yaml              # Without global pooling
├── src/
│   ├── data/
│   │   ├── dataset_fixed.py    # Data loading
│   │   └── feature_engineering.py
│   ├── models/
│   │   └── gnn.py              # MinimalGCN model
│   ├── training/
│   │   ├── train.py            # Trainer class
│   │   └── evaluate.py         # Evaluation
│   └── utils/
│       ├── config.py           # Config loading
│       └── visualization.py    # Plotting
├── experiments/
│   └── minimal_gcn_globalpool/ # Best model checkpoint
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATASET.md
│   └── GETTING_STARTED.md
└── dataset/                    # (not in git)
    ├── sadic_data/             # Protein graphs
    ├── depth_indexes_dict.pkl  # Ground truth
    └── processed/              # PyG cache
```

## Usage

### Training

```bash
# Default training
python main.py

# Custom parameters
python main.py --batch-size 64 --epochs 200 --lr 0.0005

# Specific config
python main.py --config configs/minimal_gcn_globalpool.yaml
```

### Evaluation

```bash
# Evaluate best model
python main.py --eval-only

# Evaluate specific checkpoint
python main.py --eval-only --checkpoint experiments/minimal_gcn_globalpool/best_model.pt
```

### Python API

```python
from src.data.dataset_fixed import ProteinAtomDataset
from src.models.gnn import MinimalGCN, create_model
from torch_geometric.loader import DataLoader

# Load data
dataset = ProteinAtomDataset(
    root='dataset/',
    split='test',
    feature_config={'use_minimal_features': True}
)
loader = DataLoader(dataset, batch_size=64)

# Create model
model = create_model({
    'in_channels': 5,
    'hidden_channels': 64,
    'num_layers': 2,
    'use_global_pool': True,
    'global_pool_type': 'mean'
})
```

## Key Findings

Through extensive experimentation, we discovered:

1. **Contact count dominates**: The `contact_count_10A` feature alone explains most of the variance
2. **GCN ≈ GAT**: Simple GCN matches GATv2 attention performance for this task
3. **Edge features have 0 importance**: Removing edge features has no impact
4. **5 features suffice**: Reduced from 93 features with minimal accuracy loss

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- See [requirements.txt](requirements.txt)

## Documentation

- [Getting Started](docs/GETTING_STARTED.md) - Setup and installation
- [Architecture](docs/ARCHITECTURE.md) - Model design details
- [Dataset](docs/DATASET.md) - Data format and features

## License

[Your license here]
