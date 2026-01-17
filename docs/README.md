# Documentation

Welcome to the GNN Protein Atom Exposure Prediction documentation.

## Core Documentation

### Getting Started
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Installation, setup, and usage
  - Prerequisites and dependencies
  - Installation methods (automated and manual)
  - Quick start guide
  - Troubleshooting

### Architecture
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Model design and technical details
  - Graph representation
  - Feature engineering (88 features)
  - GNN architectures (GCN, GAT, GIN, GINE, GATv2)
  - Training strategy and hyperparameters
  - Design decisions and trade-offs

### Dataset
- **[DATASET.md](DATASET.md)** - Data structure and features
  - Dataset statistics (4,769 proteins)
  - File structure and formats
  - Node and edge features
  - Data splits and loading
  - Data quality and integrity

### Experiments
- **[EXPERIMENTS.md](EXPERIMENTS.md)** - Complete experimental history
  - All phases (1-8)
  - Results and metrics
  - Methodology and protocols
  - Key findings
  - Performance evolution

### Changelog
- **[CHANGELOG.md](CHANGELOG.md)** - Project timeline and milestones
  - Chronological changes
  - Major improvements
  - Bug fixes
  - Failed experiments and learnings

## Analysis & Reference

### Analysis Documents
Located in [analysis/](analysis/):

- **[STRATEGIC_ANALYSIS.md](analysis/STRATEGIC_ANALYSIS.md)** - Deep dive into model behavior
- **[FEATURE_REDUCTION.md](analysis/FEATURE_REDUCTION.md)** - Feature selection process
- **[DATASET_INTEGRITY.md](analysis/DATASET_INTEGRITY.md)** - Data quality analysis
- **[BUG_DISCOVERY.md](analysis/BUG_DISCOVERY.md)** - Critical bugs and fixes
- **[FOLDER_STRUCTURE.md](analysis/FOLDER_STRUCTURE.md)** - Project organization

### Historical Documentation
Located in [archive/](archive/):

- **[phase1_complete.md](archive/phase1_complete.md)** - Phase 1 summary
- **[phase2_complete.md](archive/phase2_complete.md)** - Phase 2 summary
- **[phase3_complete.md](archive/phase3_complete.md)** - Phase 3 summary (bug fixes)
- **[README.md](archive/README.md)** - Archive index and explanation

## Current Status

For the latest project status, see:
- **[../experiments/STATUS.md](../experiments/STATUS.md)** - Current phase and roadmap
- **[../experiments/README.md](../experiments/README.md)** - Experiments overview

## Quick Navigation

### For New Users
1. Start with [GETTING_STARTED.md](GETTING_STARTED.md) for setup
2. Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand the model
3. Review [DATASET.md](DATASET.md) to understand the data
4. Check [EXPERIMENTS.md](EXPERIMENTS.md) for current results

### For Contributors
1. Review [GETTING_STARTED.md](GETTING_STARTED.md) for development setup
2. Read [ARCHITECTURE.md](ARCHITECTURE.md) for design rationale
3. Check [../experiments/STATUS.md](../experiments/STATUS.md) for current work
4. See [CHANGELOG.md](CHANGELOG.md) for recent changes

### For Researchers
1. Review [EXPERIMENTS.md](EXPERIMENTS.md) for complete methodology
2. Read [analysis/](analysis/) for detailed analysis
3. Check [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
4. See [CHANGELOG.md](CHANGELOG.md) for project evolution

## Project Overview

**Goal**: Predict atom exposure depth in protein structures using Graph Neural Networks

**Current Best**: R² = 0.5684, MAE = 0.1829 (Phase 5 - GINE with Weighted Loss)

**Key Achievements**:
- +17.4% improvement from baseline
- Critical bug fixes improving performance by 3-4x
- Systematic feature reduction (100+ → 88 features)
- Comprehensive documentation and tracking

## External Links

- [Main README](../README.md) - Project overview
- [GitHub Repository](#) - Source code
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - Framework docs
- [Graphein](https://github.com/a-r-j/graphein) - Protein graph library

## Documentation Structure

```
docs/
├── README.md                  # This file - documentation index
├── GETTING_STARTED.md         # Installation and setup
├── ARCHITECTURE.md            # Model design and features
├── DATASET.md                 # Data structure and loading
├── EXPERIMENTS.md             # Complete experimental history
├── CHANGELOG.md               # Project timeline
├── analysis/                  # Analysis documents
│   ├── STRATEGIC_ANALYSIS.md
│   ├── FEATURE_REDUCTION.md
│   ├── DATASET_INTEGRITY.md
│   ├── BUG_DISCOVERY.md
│   └── FOLDER_STRUCTURE.md
└── archive/                   # Historical documentation
    ├── README.md
    ├── phase1_complete.md
    ├── phase2_complete.md
    └── phase3_complete.md
```

## Contributing to Documentation

When adding or updating documentation:
1. Keep docs concise but complete
2. Use clear headings and structure
3. Include code examples where helpful
4. Link to related documents
5. Update this README if adding new docs
6. Follow existing formatting conventions

## Questions or Issues

- Check [GETTING_STARTED.md](GETTING_STARTED.md) for troubleshooting
- Review [analysis/BUG_DISCOVERY.md](analysis/BUG_DISCOVERY.md) for known issues
- See [../experiments/STATUS.md](../experiments/STATUS.md) for current work
- Contact project maintainers for additional help

---

Last updated: 2026-01-16
