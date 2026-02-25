# Documentation

GNN Protein Atom Exposure Prediction documentation index.

## Contents

| Document | Description |
|----------|-------------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Installation, setup, and usage |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Model design, features, global pooling, loss functions |
| [DATASET.md](DATASET.md) | Data structure, 93 features, splits, loading |
| [HISTORY.md](HISTORY.md) | Experimental phases and milestones |

## Current Status

**Best model**: Phase 15 + Global Pooling

- **R² = 0.9408**, MAE = 0.0655
- GATv2 (4 layers, 136 hidden, 4 heads) + Dynamic Global Pooling
- 93 node features, 12 edge features
- Range-Specific Weighted Loss, OneCycleLR scheduler

## Quick Navigation

### For New Users
1. [GETTING_STARTED.md](GETTING_STARTED.md) - Installation and first training run
2. [DATASET.md](DATASET.md) - Understand the data
3. [ARCHITECTURE.md](ARCHITECTURE.md) - Understand the model

### For Researchers
1. [ARCHITECTURE.md](ARCHITECTURE.md) - Technical details and design decisions
2. [HISTORY.md](HISTORY.md) - Complete experimental evolution
3. [../experiments/](../experiments/) - Results and plots

```
docs/
├── README.md              # This file
├── GETTING_STARTED.md     # Installation and usage
├── ARCHITECTURE.md        # Model design and features
├── DATASET.md             # Data structure and loading
└── HISTORY.md             # Project timeline
```
