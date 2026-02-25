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
│   ├── config.yaml            # Default config
│   ├── phase15.yaml           # Phase 15 (current best)
│   ├── phase15_globalpool.yaml        # Phase 15 + Dynamic Global Pooling
│   └── phase15_noCC_globalpool.yaml   # Without contact_count (ablation)
├── src/                       # Source code
│   ├── data/                  # Dataset and feature engineering
│   ├── models/                # GNN architectures (GCN, GAT, GIN, GINE, GATv2)
│   ├── training/              # Training and evaluation
│   └── utils/                 # Utilities
├── experiments/               # Experiment results and tracking
│   ├── STATUS.md              # Current progress
│   ├── baselines/             # Baseline experiments
│   ├── grid_search/           # Hyperparameter tuning
│   ├── analysis/              # Analysis scripts
│   │   ├── attention_analysis.py
│   │   ├── edge_importance.py
│   │   └── feature_importance.py
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
| 12 | GATv2 + Contact Count | 0.8817 | 0.0866 | ✅ Complete |
| 13 | Extended Training + Range Loss | 0.8837 | 0.0846 | ✅ Complete |
| 14 | Hyperparameter Tuning (Optuna) | 0.8856 | 0.0843 | ✅ Complete |
| 15 | Optimized Architecture | 0.8892 | 0.0838 | ✅ Complete |
| 15 noCC | Without contact_count | 0.5689 | 0.1646 | ⚠️ Proves contact_count critical |
| **15 + GP** | **Global Pooling** | **0.9408** | **0.0655** | ✅ **Current Best** |

### Key Findings
- **Dynamic Global Pooling boost**: +5.8% R² improvement (0.8892 → 0.9408)
- **contact_count_10A is CRITICAL**: Removing it drops R² from 0.8892 → 0.5689 (-36%)
- **+94% improvement** from baseline GCN (R² 0.484 → 0.9408)

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for complete experimental history.

## Features

### Supported GNN Architectures
- **GCN** (Graph Convolutional Network)
- **GAT** (Graph Attention Network)
- **GIN** (Graph Isomorphism Network)
- **GINE** (GIN with Edge features)
- **GATv2** (Improved attention mechanism) - **Current best**

### Advanced Architecture Options
- **Virtual Global Node**: Adds a virtual node connected to all atoms for global protein context
- **Dynamic Global Pooling**: Aggregates global information via gated mechanism after each GNN layer
  - Pooling types: `mean`, `max`, `both`
  - Configurable injection points: `every`, `middle`, `last` layer

### Key Features
- **93 node features** (biochemical + geometric + backbone angles + contact_count_10A)
- **12 edge features** (bond types, distances, radius graph)
- **GATv2 attention** with internal residual connections (4 heads, 4 layers)
- **Range-Specific Weighted Loss** for targeting bias in different exposure ranges
- **Early stopping on R²** for optimal model selection
- **OneCycleLR scheduler** with configurable parameters
- **Comprehensive experiment tracking** with TensorBoard logging

## Usage

### Training

```bash
# Train with default configuration
python main.py

# Train Phase 15 (current best)
python main.py --config configs/phase15.yaml

# Train with Dynamic Global Pooling
python main.py --config configs/phase15_globalpool.yaml

# Train without contact_count (ablation study)
python main.py --config configs/phase15_noCC_globalpool.yaml
```

### Evaluation

```bash
# Evaluate trained model
python main.py --eval-only --checkpoint experiments/checkpoints/phase15/best_model.pt

# With visualizations
python main.py --eval-only --checkpoint experiments/checkpoints/phase15/best_model.pt --visualize
```

### Analysis Scripts

```bash
# Feature importance analysis (permutation-based)
python experiments/analysis/feature_importance.py

# Attention pattern analysis
python experiments/analysis/attention_analysis.py

# Edge importance analysis
python experiments/analysis/edge_importance.py
```

### Using Python API

```python
from src.data.dataset_fixed import ProteinAtomDataset
from src.models.gnn import AtomExposureGNN
from torch_geometric.loader import DataLoader

# Load data
dataset = ProteinAtomDataset(root='dataset/', split='train')
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Create model (Phase 15 best config)
model = AtomExposureGNN(
    in_channels=93,          # 88 base + 4 backbone + 1 contact_count
    hidden_channels=136,
    num_layers=4,
    conv_type='gatv2',       # GATv2 attention
    edge_dim=12,             # 11 + 1 radius graph
    dropout=0.26,
    heads=4
)

# With Dynamic Global Pooling
model = AtomExposureGNN(
    in_channels=93,
    hidden_channels=136,
    num_layers=4,
    conv_type='gatv2',
    edge_dim=12,
    dropout=0.26,
    use_global_pool=True,
    global_pool_type='mean',    # 'mean', 'max', or 'both'
    global_pool_layers='every'  # 'every', 'middle', or 'last'
)

# Train (see docs/GETTING_STARTED.md for complete example)
```

## Key Achievements

- **+94% improvement** from baseline (R² 0.484 → 0.9408)
- **Phase 15 + GP breakthrough**: Dynamic Global Pooling adds +5.8% R² (0.8892 → 0.9408)
- **Phase 12 breakthrough**: +46% R² from contact_count_10A feature alone
- **Phase 15 optimization**: Optuna-tuned hyperparameters (4 layers, 136 hidden, 4 heads)
- **Critical discovery**: contact_count_10A is essential (removing it drops R² by 36%)
- **93 optimized features** (88 base + 4 backbone angles + contact_count)
- **Range-Specific Weighted Loss** for handling exposure distribution bias
- **Dynamic Global Pooling** provides protein-level context via gated mechanism
- **Comprehensive documentation** and experiment tracking

## Technology Stack

- **PyTorch** - Deep learning framework
- **PyTorch Geometric** - Graph neural network library
- **Graphein** - Protein graph construction
- **Optuna** - Hyperparameter optimization
- **NumPy/Pandas** - Data manipulation
- **scikit-learn** - Evaluation metrics
- **Matplotlib/Seaborn** - Visualization
- **TensorBoard** - Training visualization

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
