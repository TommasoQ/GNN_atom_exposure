# Experiments Documentation

This document provides a comprehensive overview of all experiments conducted in this project, including results, methodologies, and key findings.

## Current Status

**Current Best Model**: Phase 5 - GINE with Weighted Loss
- **R² Score**: 0.5684
- **MAE**: 0.1829
- **RMSE**: 0.2321
- **Improvement**: +17.4% from baseline

## Experimental Timeline

### Phase 1: Initial Exploration ✅ Complete
**Goal**: Understand dataset and establish baselines
**Duration**: Early development
**Outcome**: Dataset structure understood, initial models tested

### Phase 2: Architecture Testing ✅ Complete
**Goal**: Test different GNN architectures
**Outcome**: GCN and GAT tested, discovered critical bugs

### Phase 3: Bug Fixes and Valid Baseline ✅ Complete
**Goal**: Fix critical bugs and establish valid baseline
**Date**: January 9, 2026
**Critical Bugs Fixed**:
1. Loss weighting bug (~1000x gradient error)
2. Config mismatches
3. Edge feature handling
4. Target padding issues
5. Dataset caching problems

**Results**:
- **Experiment 3.5 (GCN)**: R² 0.484, MAE 0.204 - **First valid baseline**
- **Experiment 3.6 (GAT)**: R² 0.477, MAE 0.204 - No improvement over GCN

**Key Finding**: GCN and GAT perform equivalently; GCN recommended as baseline

### Phase 4: Hyperparameter Tuning ✅ Complete
**Goal**: Optimize learning rate and scheduler
**Method**: Grid search over 18 configurations
**Variables Tested**:
- Learning rate: [0.001, 0.003, 0.005]
- Schedulers: OneCycleLR vs Cosine
- Warmup: [0.1, 0.3, 0.5]

**Results**:
- **Best Config**: OneCycleLR, max_lr=0.003, pct_start=0.3
- **R² Score**: 0.5456
- **Improvement**: +12.6% over Phase 3 baseline

**Key Finding**: OneCycleLR outperforms Cosine scheduler

### Phase 5: GINE + Weighted Loss ✅ Complete - **CURRENT BEST**
**Goal**: Leverage edge features and address class imbalance
**Date**: January 2026
**Changes**:
- Architecture: GINE (Graph Isomorphism Network with Edge features)
- Loss: Exposure-weighted MSE (α=1.5, threshold=0.8)
- Scheduler: OneCycleLR (from Phase 4)
- Deterministic: seed=42

**Results**:
- **R² Score**: 0.5684
- **MAE**: 0.1829
- **RMSE**: 0.2321
- **Pearson**: 0.7551
- **Improvement**: +17.4% from Phase 3 baseline

**Key Finding**: Edge features + weighted loss significantly improve performance

### Phase 6: Aggregated Features ❌ FAILED
**Goal**: Test aggregated geometric features
**Date**: January 2026
**Changes**:
- Aggregated neighbor features (mean, std, max)
- Centroid-based geometric descriptors

**Results**:
- **R² Score**: 0.4002
- **Performance**: -29% worse than baseline
- **Status**: FAILED

**Why It Failed**:
- Lost important predictive information through over-aggregation
- Centroid-based features inferior to neighbor-based
- Too much information compression

**Lesson Learned**: Feature engineering must preserve local geometric details

### Phase 7: Low Regularization ❌ FAILED
**Goal**: Reduce regularization based on colleague's insights
**Date**: January 2026
**Changes**:
- Reduced weight decay
- Lower dropout rate

**Results**:
- **R² Score**: 0.5604-0.5607
- **Performance**: Didn't beat Phase 5 (0.5684)
- **Status**: FAILED to improve

**Lesson Learned**: Current regularization is already well-tuned

### Phase 8: Advanced Features 🚧 READY TO RUN
**Goal**: Test backbone angles and improved attention
**Date**: January 2026
**Planned Experiments**:
1. **GATv2 + Attention**: Dynamic attention mechanism
2. **GINE + Backbone Angles**: Add φ/ψ dihedral angles

**Status**: Configs ready, awaiting execution

## Complete Experiment Table

| Phase | Experiment | Model | R² | MAE | RMSE | Pearson | Status | Notes |
|-------|-----------|-------|-----|-----|------|---------|--------|-------|
| 3 | 3.1-3.4 | Various | 0.10-0.15 | 0.26-0.32 | - | 0.45-0.50 | ❌ Invalid | Pre-bugfix |
| 3 | 3.5 | GCN | **0.484** | 0.204 | 0.255 | 0.696 | ✅ Valid | **First baseline** |
| 3 | 3.6 | GAT | 0.477 | 0.204 | 0.255 | 0.691 | ✅ Valid | No improvement |
| 4 | Grid Search | GCN | 0.5456 | - | - | - | ✅ Complete | 18 configs tested |
| 5 | Baseline | GINE | **0.5684** | **0.1829** | **0.2321** | **0.7551** | ✅ Best | **Current best** |
| 6 | Aggregated | GINE | 0.4002 | - | - | - | ❌ Failed | Over-aggregation |
| 7 | Low Reg | GINE | 0.5607 | - | - | - | ❌ Failed | No improvement |
| 8 | - | GATv2/GINE | - | - | - | - | 🚧 Ready | Pending |

## Methodology

### Evaluation Metrics

All experiments use consistent metrics:
- **R² (Coefficient of Determination)**: Proportion of variance explained
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **RMSE (Root Mean Squared Error)**: Standard deviation of errors
- **Pearson Correlation**: Linear correlation between predictions and ground truth
- **Median Absolute Error**: Robust metric less affected by outliers
- **Mean Error**: Bias in predictions

### Experiment Protocol

1. **Data Splits**: 70% train, 15% val, 15% test (protein-level)
2. **Seeds**: Deterministic with seed=42 for reproducibility
3. **Hardware**: GPU-accelerated training
4. **Checkpointing**: Save best validation loss model
5. **Early Stopping**: Monitor validation loss

### Training Configuration

**Standard Settings** (Phase 5):
```yaml
model:
  conv_type: gine
  hidden_channels: 96
  num_layers: 3
  dropout: 0.35

training:
  batch_size: 8
  epochs: 200
  learning_rate: 0.003  # OneCycle max_lr
  scheduler: OneCycleLR
  pct_start: 0.3
  weight_decay: 0.0

loss:
  type: weighted_mse
  alpha: 1.5
  threshold: 0.8
```

## Key Findings

### 1. GCN = GAT for This Task
Despite GAT's attention mechanism, it performs equivalently to GCN for atom exposure prediction. **Recommendation**: Use GCN as baseline due to simplicity.

### 2. Edge Features Matter
GINE (with edge features) outperforms GCN/GAT by 15%. Bond types and distances provide valuable information.

### 3. Weighted Loss Critical
Addressing class imbalance with weighted loss significantly improves performance on exposed atoms (minority class).

### 4. OneCycleLR > Cosine
OneCycleLR scheduler consistently outperforms Cosine annealing across multiple configurations.

### 5. Feature Engineering Trade-offs
- **Too much aggregation** loses information (Phase 6 failure)
- **Too many redundant features** causes overfitting
- **88 features** is the current sweet spot

### 6. Regularization Already Optimal
Attempts to reduce regularization (Phase 7) didn't improve results, suggesting current settings are well-tuned.

## Performance Evolution

```
Phase 3 (Baseline): R² = 0.484
    ↓ +12.6%
Phase 4 (Tuning):   R² = 0.5456
    ↓ +4.2%
Phase 5 (GINE):     R² = 0.5684  ← Current Best
```

**Total Improvement**: +17.4% from baseline

## Experiment Files Location

```
experiments/
├── baselines/
│   ├── _archived_invalid/         # Pre-bugfix experiments (invalid)
│   ├── gcn_bugs_fixed/            # Experiment 3.5 (baseline)
│   ├── gat_with_edges/            # Experiment 3.6
│   ├── phase5_gine_weighted_loss/ # Current best
│   └── FAILED_phase6_aggregated/  # Failed experiment
├── grid_search/                   # Phase 4 hyperparameter tuning
│   ├── results/                   # 18 experiment runs
│   ├── analysis/                  # Comparison plots
│   └── summary.csv                # All results table
├── analysis/
│   ├── error_analysis/            # Error breakdown by atom type
│   └── feature_analysis/          # Feature importance/correlation
├── checkpoints/
│   └── phase5_best_model.pt       # Current best model checkpoint
└── logs/
    └── latest/                    # Recent training logs
```

## Reproducing Results

### Phase 5 (Current Best)

```bash
# Load exact configuration
python main.py --config experiments/baselines/phase5_gine_weighted_loss/config_snapshot.yaml --epochs 200

# Ensure deterministic results
# Config should have: deterministic: true, seed: 42
```

### Phase 4 Grid Search

```bash
# Re-run grid search
python experiments/grid_search/scripts/run_grid_search.py
```

## Future Experiments

### Planned (Phase 8)
1. **GATv2 with dynamic attention**
2. **Backbone dihedral angles** (φ/ψ)
3. **Enhanced geometric features**

### Potential Future Work
- [ ] E(3)-equivariant networks
- [ ] Pre-training on larger datasets
- [ ] Multi-task learning (exposure + secondary structure)
- [ ] Ensemble methods
- [ ] Attention visualization
- [ ] Transfer learning from protein language models

## Analysis Documents

For detailed analysis, see:
- [Error Analysis](analysis/error_analysis/) - Breakdown by atom type, exposure range
- [Feature Analysis](analysis/feature_analysis/) - Correlation and importance
- [Strategic Analysis](analysis/STRATEGIC_ANALYSIS.md) - Deep dive into model behavior
- [Bug Discovery](analysis/BUG_DISCOVERY.md) - Critical bugs and fixes

## See Also

- [Architecture Documentation](ARCHITECTURE.md) - Model design
- [Dataset Documentation](DATASET.md) - Data structure
- [Getting Started](GETTING_STARTED.md) - Installation and usage
