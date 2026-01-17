# Changelog

All notable changes, milestones, and experimental results for this project.

## [Phase 8] - 2026-01-16 - READY

### Planned
- **GATv2 with Attention**: Testing dynamic attention mechanisms
- **Backbone Dihedral Angles**: Adding φ/ψ angles as additional features
- **Target**: Achieve R² > 0.58

## [Phase 7] - 2026-01-16 - FAILED

### Changed
- Reduced regularization (dropout 0.35 → 0.1, weight_decay 1e-4 → 1e-5)
- Increased model capacity (96 → 128 hidden, 3 → 4 layers)
- Smaller batch size (32 → 16)
- Removed weighted loss

### Results
- R² = 0.5604-0.5607 (didn't beat Phase 5's 0.5684)
- **Conclusion**: Current regularization is already well-tuned
- **Status**: FAILED to improve

## [Phase 6] - 2026-01-15 - FAILED

### Changed
- Implemented aggregated feature engineering approach
- Reduced features from 88 to 50
- Aggregated hydrophobicity scales into mean/std
- Used centroid-based geometric features
- Changed categorical encoding to label indices

### Results
- R² = 0.4002 (29% drop from Phase 5)
- **Conclusion**: Over-aggregation lost important predictive information
- **Status**: FAILED
- **Lesson**: Neighbor-based geometric features superior to centroid-based

## [Phase 5] - 2026-01-14 - CURRENT BEST

### Added
- GINE architecture (Graph Isomorphism Network with Edge features)
- Exposure-weighted MSE loss (α=1.5, threshold=0.8)
- Deterministic training (seed=42)

### Changed
- Architecture: GCN → GINE
- Hidden channels: 96
- Layers: 3
- Dropout: 0.35
- Scheduler: OneCycleLR (max_lr=0.003, pct_start=0.3)

### Results
- **R² = 0.5684** (NEW BEST)
- **MAE = 0.1829**
- **RMSE = 0.2321**
- **Pearson = 0.7551**
- **Improvement**: +17.4% from Phase 3 baseline
- **Status**: Current best deterministic baseline

## [Phase 4] - 2026-01-12 - Complete

### Added
- Systematic hyperparameter grid search (18 configurations)
- OneCycleLR vs Cosine scheduler comparison
- Learning rate sweep [0.001, 0.003, 0.005]
- Warmup ratio testing [0.1, 0.3, 0.5]

### Results
- **Best Config**: OneCycleLR, max_lr=0.003, pct_start=0.3
- **R² = 0.5456**
- **Improvement**: +12.6% from Phase 3
- **Key Finding**: OneCycleLR consistently outperforms Cosine

## [Phase 3] - 2026-01-09 - Critical Bug Fixes

### Fixed - CRITICAL BUGS
1. **Loss Weighting Bug**: Fixed `batch.num_graphs` → `batch.num_nodes` (~1000x gradient error)
2. **Config Mismatch**: Corrected feature dimensions
3. **Edge Features**: Added edge features to GAT layers
4. **Target Padding**: Changed silent 0.0 padding to explicit failure
5. **Dataset Caching**: Fixed triple processing issue

### Changed
- Feature reduction: 100+ features → 88 features
- Removed highly correlated features (>0.95)
- Removed low-importance features

### Results - First Valid Experiments
- **Experiment 3.5 (GCN)**: R² = 0.484, MAE = 0.204 (First valid baseline)
- **Experiment 3.6 (GAT)**: R² = 0.477, MAE = 0.204
- **Key Finding**: GCN and GAT perform equivalently
- **Improvement**: 3-4x better than pre-bugfix experiments

### Performance vs Pre-Fix
| Metric | Pre-Fix (Exp 3.1-3.4) | Post-Fix (Exp 3.5) | Improvement |
|--------|----------------------|-------------------|-------------|
| MAE | 0.263-0.317 | 0.204 | 22-36% better |
| R² | 0.10-0.15 | 0.484 | 3-4x better |
| Pearson | 0.45-0.50 | 0.696 | 40% better |

### Deprecated
- All experiments 3.1-3.4 marked as INVALID
- Archived to `experiments/baselines/_archived_invalid_experiments/`

## [Phase 2] - 2026-01-08 - Dataset Quality

### Added
- Dataset integrity report
- HETATM analysis (identified 38 proteins with >50% HETATM)
- Missing protein investigation (231 proteins)

### Changed
- Dataset: 5,000 proteins listed → 4,769 available structures
- Feature engineering pipeline established

### Results
- Confirmed dataset quality
- Documented data quirks
- Established loading pipeline

## [Phase 1] - 2026-01-08 - Initial Exploration

### Added
- Exploratory data analysis
- Feature correlation analysis
- Feature importance ranking
- Dataset structure documentation

### Results
- **Key Finding**: B-factor is top predictor of atom exposure
- Identified 100+ initial features
- Established baseline understanding

## [Initial Setup] - 2026-01-08

### Added
- Project structure
- Dataset loading (ProteinAtomDataset class)
- GNN model implementations (GCN, GAT, GIN, GINE, GATv2)
- Training and evaluation pipeline
- Configuration management
- Documentation framework

### Infrastructure
- PyTorch Geometric integration
- Graphein for graph construction
- Experiment tracking system
- Checkpoint management

---

## Performance Summary

### Best Results by Phase

| Phase | Model | R² | MAE | Status |
|-------|-------|-----|-----|--------|
| 3 | GCN | 0.484 | 0.204 | ✅ Baseline |
| 4 | GCN (tuned) | 0.5456 | - | ✅ Complete |
| **5** | **GINE** | **0.5684** | **0.1829** | ✅ **BEST** |
| 6 | GINE (aggregated) | 0.4002 | - | ❌ Failed |
| 7 | GINE (low reg) | 0.5607 | - | ❌ Failed |

### Overall Progress

```
Phase 3 (Baseline): R² = 0.484
    ↓ Bug fixes + 88 features
Phase 4 (Tuning):   R² = 0.5456 (+12.6%)
    ↓ GINE + Weighted loss
Phase 5 (GINE):     R² = 0.5684 (+17.4% total)
```

---

## Key Learnings

### What Worked ✅
1. **GINE architecture**: Edge features matter (+15% over GCN)
2. **Weighted loss**: Addresses class imbalance effectively
3. **OneCycleLR**: Better than Cosine scheduler
4. **88 features**: Optimal balance after reducing from 100+
5. **Systematic debugging**: Critical bug fixes led to 3-4x improvement

### What Didn't Work ❌
1. **GAT vs GCN**: Attention provides no benefit for this task
2. **Aggregated features**: Over-simplification loses information
3. **Low regularization**: Current settings already optimal
4. **Centroid geometry**: Neighbor-based features superior

### Critical Insights
- **Feature engineering**: More isn't always better (100+ → 88)
- **Architecture choice**: Edge features matter more than attention
- **Class imbalance**: Must be addressed explicitly with weighted loss
- **Debugging**: Small bugs can cause massive performance degradation
- **Hyperparameters**: Can matter as much as architecture

---

## Next Steps

- [ ] Execute Phase 8 experiments (GATv2, backbone angles)
- [ ] Explore E(3)-equivariant networks
- [ ] Pre-training on larger protein datasets
- [ ] Multi-task learning
- [ ] Attention visualization
- [ ] Production deployment

---

## See Also

- [Architecture Documentation](ARCHITECTURE.md) - Model design
- [Experiments Documentation](EXPERIMENTS.md) - Detailed results
- [Dataset Documentation](DATASET.md) - Data structure
