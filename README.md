# GNN Protein Atom Exposure Prediction

A Graph Neural Network (GNN) project using PyTorch Geometric to predict atom exposure levels in protein structures.

## Overview

This project predicts the burial depth of individual atoms within protein structures using Graph Neural Networks. Proteins are represented as graphs where atoms are nodes and bonds are edges, enriched with biochemical features.

**Current Best Result**: R² = 0.8817, MAE = 0.0866 (Phase 12 - GATv2 + Contact Count)

## Quick Start

```bash
# Clone the repository
git clone [repository-url]
cd GNN_atom_exposure

# Install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
bash install_simple.sh

# Train a model
python main.py
```

For detailed installation instructions, see [Getting Started](docs/GETTING_STARTED.md).

## Documentation

### Core Documentation
- **[Getting Started](docs/GETTING_STARTED.md)** - Installation, setup, and basic usage
- **[Architecture](docs/ARCHITECTURE.md)** - Model design, features, and design decisions
- **[Dataset](docs/DATASET.md)** - Data structure, features, and statistics
- **[Experiments](docs/EXPERIMENTS.md)** - Complete experimental history and results
- **[Changelog](docs/CHANGELOG.md)** - Project timeline and milestones

### Analysis & Reference
- **[docs/analysis/](docs/analysis/)** - Feature analysis, error analysis, bug reports
- **[docs/archive/](docs/archive/)** - Historical documentation and completed phases

### Current Status
See [experiments/STATUS.md](experiments/STATUS.md) for current phase and next steps.

## Project Structure

```
GNN_atom_exposure/
├── main.py                    # Training entry point
├── requirements.txt           # Dependencies
├── docs/                      # Documentation
│   ├── GETTING_STARTED.md
│   ├── ARCHITECTURE.md
│   ├── DATASET.md
│   ├── EXPERIMENTS.md
│   ├── CHANGELOG.md
│   ├── analysis/              # Analysis documents
│   └── archive/               # Historical docs
├── configs/                   # Configuration files
│   └── config.yaml
├── src/                       # Source code
│   ├── data/                  # Dataset and feature engineering
│   ├── models/                # GNN architectures (GCN, GAT, GIN, GINE, GATv2)
│   ├── training/              # Training and evaluation
│   └── utils/                 # Utilities
├── experiments/               # Experiment results and tracking
│   ├── STATUS.md              # Current progress
│   ├── baselines/             # Baseline experiments
│   ├── grid_search/           # Hyperparameter tuning
│   ├── analysis/              # Result analysis
│   ├── checkpoints/           # Model checkpoints
│   └── logs/                  # Training logs
├── dataset/                   # Dataset (not in git)
│   ├── sadic_data/            # Pre-processed protein graphs
│   ├── depth_indexes.pkl      # Ground truth labels
│   └── processed/             # PyG cache
└── notebooks/                 # Jupyter notebooks
```

## Current Results

| Phase | Model | R² | MAE | Status |
|-------|-------|-----|-----|--------|
| 3 | GCN | 0.484 | 0.204 | ✅ Baseline |
| 4 | GCN (tuned) | 0.5456 | - | ✅ Complete |
| 5 | GINE | 0.5684 | 0.1829 | ✅ Complete |
| 8 | GINE + Backbone Angles | 0.607 | 0.14 | ✅ Complete |
| 9 | Extended Training (200 epochs) | 0.6028 | - | ✅ Complete |
| 11 | GATv2 Transition | 0.598 | - | ✅ Complete |
| **12** | **GATv2 + Contact Count** | **0.8817** | **0.0866** | ✅ **Current Best** |

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for complete experimental history.

## Features

### Supported GNN Architectures
- **GCN** (Graph Convolutional Network)
- **GAT** (Graph Attention Network)
- **GIN** (Graph Isomorphism Network)
- **GINE** (GIN with Edge features)
- **GATv2** (Improved attention mechanism) - **Current best**

### Key Features
- 93 carefully selected node features (biochemical + geometric + backbone angles + contact count)
- 12 edge features (bond types, distances, radius graph)
- GATv2 attention with internal residual connections
- Weighted loss for class imbalance
- OneCycleLR scheduler
- Comprehensive experiment tracking

## Usage

### Training

```bash
# Train with default configuration
python main.py

# Train with custom parameters
python main.py --batch-size 16 --epochs 50 --lr 0.0005

# Use specific config file
python main.py --config configs/gatv2_config.yaml
```

### Evaluation

```bash
# Evaluate trained model
python main.py --eval-only --checkpoint experiments/checkpoints/phase5_best_model.pt

# With visualizations
python main.py --eval-only --checkpoint experiments/checkpoints/phase5_best_model.pt --visualize
```

### Using Python API

```python
from src.data.dataset_fixed import ProteinAtomDataset
from src.models.gnn import AtomExposureGNN
from torch_geometric.loader import DataLoader

# Load data
dataset = ProteinAtomDataset(root='dataset/', split='train')
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Create model (Phase 12 best config)
model = AtomExposureGNN(
    in_channels=93,          # 88 + 4 backbone + 1 contact_count
    hidden_channels=128,
    num_layers=3,
    conv_type='gatv2',       # GATv2 attention
    edge_dim=12,             # 11 + 1 radius graph
    dropout=0.25
)

# Train (see docs/GETTING_STARTED.md for complete example)
```

## Key Achievements

- **+82% improvement** from baseline (R² 0.484 → 0.8817)
- **Phase 12 breakthrough**: +46% R² from contact_count_10A feature
- **Critical bug fixes** that improved performance by 3-4x
- **93 optimized features** (88 base + 4 backbone angles + contact count)
- **Weighted loss** effectively addresses class imbalance
- **Comprehensive documentation** and experiment tracking

## Technology Stack

- **PyTorch** - Deep learning framework
- **PyTorch Geometric** - Graph neural network library
- **Graphein** - Protein graph construction
- **NumPy/Pandas** - Data manipulation
- **scikit-learn** - Evaluation metrics
- **Matplotlib/Seaborn** - Visualization

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- See [requirements.txt](requirements.txt) for complete list

## Contributing

See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for development setup.

## License

[Specify your license here]

## Citation

If you use this project in your research, please cite:

```bibtex
@software{gnn_atom_exposure,
  title={GNN Protein Atom Exposure Prediction},
  author={[Your Name]},
  year={2026},
  url={https://github.com/[your-repo]}
}
```

## Contact

For questions or collaboration: [your-email@domain.com]

## Acknowledgments

- PyTorch Geometric team for the excellent library
- Graphein developers for protein graph preprocessing tools
- Reference implementation that inspired Phase 7 experiments
