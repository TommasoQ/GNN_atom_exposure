# Experiments

This directory contains all experimental results, analysis, and tracking for the GNN protein atom exposure prediction project.

## Structure

```
experiments/
├── README.md                           # This file
├── STATUS.md                           # Current phase status and roadmap
├── baselines/                          # Baseline experiment runs
│   ├── README.md                       # Index of all baseline experiments
│   ├── _archived_invalid/              # Pre-bugfix experiments (invalid)
│   ├── experiment_3.5_gcn_baseline/    # First valid baseline (R² 0.484)
│   ├── experiment_3.6_gat_edges/       # GAT comparison (R² 0.477)
│   ├── experiment_5.0_gine_best/       # Current best (R² 0.5684)
│   └── FAILED_phase6_aggregated/       # Failed aggregation experiment
├── grid_search/                        # Phase 4 hyperparameter tuning
│   ├── results/                        # 18 experiment runs
│   ├── analysis/                       # Comparison plots
│   ├── scripts/                        # Grid search scripts
│   └── summary.csv                     # All results table
├── analysis/                           # Result analysis
│   ├── error_analysis/                 # Error breakdown by type
│   └── feature_analysis/               # Feature importance/correlation
├── checkpoints/                        # Model checkpoints
│   └── phase5_best_model.pt            # Current best model
├── logs/                               # Training logs
│   └── latest/                         # Most recent training logs
└── progress/                           # Historical planning docs
    ├── _archived_pre_bugfix/           # Pre-Phase 3 docs
    └── _archived_phase4_complete/      # Phase 4 planning docs
```

## Quick Links

- **[STATUS.md](STATUS.md)** - Current phase, next steps, and project roadmap
- **[baselines/README.md](baselines/README.md)** - Index of all baseline experiments
- **[grid_search/](grid_search/)** - Phase 4 hyperparameter tuning results

## Valid Experiments (Post-Bugfix)

All experiments before 3.5 are invalid due to critical bugs fixed in Phase 3.

| Exp | Model | R² | MAE | RMSE | Pearson | Date | Status |
|-----|-------|-----|-----|------|---------|------|--------|
| 3.5 | GCN | 0.484 | 0.204 | 0.255 | 0.696 | 2026-01-09 | ✅ Valid baseline |
| 3.6 | GAT | 0.477 | 0.204 | 0.255 | 0.691 | 2026-01-09 | ✅ Complete |
| 4.x | GCN tuned | 0.5456 | - | - | - | 2026-01-12 | ✅ Grid search |
| **5.0** | **GINE** | **0.5684** | **0.1829** | **0.2321** | **0.7551** | **2026-01-14** | ✅ **BEST** |
| 6.0 | GINE agg | 0.4002 | - | - | - | 2026-01-15 | ❌ Failed |
| 7.0 | GINE low reg | 0.5607 | - | - | - | 2026-01-16 | ❌ Failed |

## Current Best Model

**Experiment 5.0 - GINE with Weighted Loss**

- **Architecture**: GINE (3 layers, 96 hidden channels)
- **Loss**: Exposure-weighted MSE (α=1.5, threshold=0.8)
- **Scheduler**: OneCycleLR (max_lr=0.003, pct_start=0.3)
- **R² Score**: 0.5684
- **MAE**: 0.1829
- **RMSE**: 0.2321
- **Pearson**: 0.7551
- **Checkpoint**: [checkpoints/phase5_best_model.pt](checkpoints/phase5_best_model.pt)

## Using Experiment Results

### Load Best Model

```python
import torch
from src.models.gnn import AtomExposureGNN

# Load checkpoint
checkpoint = torch.load('experiments/checkpoints/phase5_best_model.pt')

# Create model
model = AtomExposureGNN(
    in_channels=88,
    hidden_channels=96,
    num_layers=3,
    conv_type='gine',
    dropout=0.35
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Reproduce Results

```bash
# Reproduce Phase 5 (current best)
python main.py --config experiments/baselines/experiment_5.0_gine_best/config_snapshot.yaml --epochs 200
```

### View Training History

Each experiment folder contains:
- `RESULTS.md` - Detailed results and metrics
- `training_history.csv` - Per-epoch train/val metrics
- `config_snapshot.yaml` - Exact configuration used
- `best_model.pt` - Saved checkpoint (if available)

## Experiment Tracking

### Adding New Experiments

1. Create experiment folder in `baselines/` with clear naming (e.g., `experiment_X.Y_description/`)
2. Save configuration snapshot as `config_snapshot.yaml`
3. Document results in `RESULTS.md`
4. Save checkpoint and training history
5. Update `baselines/README.md` with new entry
6. Update `STATUS.md` if starting a new phase

### Failed Experiments

Failed experiments are kept for reference:
- Prefix with `FAILED_` (e.g., `FAILED_phase6_aggregated/`)
- Include `NOTES.md` explaining why it failed
- Document learnings in `STATUS.md` and [docs/CHANGELOG.md](../docs/CHANGELOG.md)

## Analysis

### Error Analysis
- **Location**: `analysis/error_analysis/`
- **Contents**: Error breakdown by atom type, exposure range, protein size
- **Purpose**: Understand model weaknesses and improvement opportunities

### Feature Analysis
- **Location**: `analysis/feature_analysis/`
- **Contents**: Feature importance, correlation analysis
- **Purpose**: Guide feature engineering decisions

## Historical Documentation

### Archived Pre-Bugfix (Invalid)
- **Location**: `progress/_archived_pre_bugfix/`
- **Date**: Pre-2026-01-09
- **Status**: All experiments invalid due to ~1000x gradient bug

### Archived Phase 4 Planning
- **Location**: `progress/_archived_phase4_complete/`
- **Status**: Planning docs superseded by completed Phase 4 results

## See Also

- [../docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md) - Complete experimental history and analysis
- [../docs/CHANGELOG.md](../docs/CHANGELOG.md) - Project timeline and milestones
- [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) - Model design and features
- [STATUS.md](STATUS.md) - Current phase and next steps
