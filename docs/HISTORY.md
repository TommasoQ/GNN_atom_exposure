# Project History

Chronological record of experimental phases and milestones.

## Phase 15 + Global Pooling - CURRENT BEST

**R² = 0.9408, MAE = 0.0655** (174 epochs, early stopped)

- Added Dynamic Global Pooling with gated mechanism
- Mean pooling injected after every GATv2 layer
- +5.8% R² improvement over Phase 15 alone
- Pearson correlation: 0.9707

## Phase 15 - Optimized Architecture

**R² = 0.8892, MAE = 0.0838**

- Optuna hyperparameter optimization
- Final config: 4 layers, 136 hidden, 4 heads, dropout 0.26
- OneCycleLR with 50-epoch warmup, max_lr 0.0007
- Early stopping on R² (patience 60)

## Phase 15 noCC - Ablation Study

**R² = 0.5689, MAE = 0.1646**

- Removed contact_count_10A to measure its importance
- Proved contact_count is critical: -36% R² without it

## Phase 14 - Hyperparameter Tuning

**R² = 0.8856, MAE = 0.0843**

- Optuna-based search over architecture and training params
- Identified optimal hidden_channels=136, num_layers=4

## Phase 13 - Range-Specific Loss

**R² = 0.8837, MAE = 0.0846**

- Introduced Range-Specific Weighted MSE loss
- Different weights for buried/semi-buried/intermediate/semi-exposed/exposed
- Asymmetric penalty (1.5x) for under-prediction

## Phase 12 - Contact Count Breakthrough

**R² = 0.8817, MAE = 0.0866**

- Added contact_count_10A feature (+46% R² from Phase 11)
- Switched to GATv2 with edge features
- Single most impactful change in project history

## Phase 11 - GATv2 Transition

**R² = 0.598**

- Migrated from GINE to GATv2
- Dynamic attention mechanism with edge feature support

## Phase 9 - Extended Training

**R² = 0.6028**

- Extended to 200 epochs with OneCycleLR

## Phase 8 - Backbone Angles

**R² = 0.607, MAE = 0.14**

- Added phi/psi backbone dihedral angles (sin/cos encoded)
- 4 new features -> 93 total (with later additions)

## Phase 5 - GINE with Weighted Loss

**R² = 0.5684, MAE = 0.1829**

- Switched to GINE architecture (edge features)
- Exposure-weighted MSE loss
- +17.4% R² from baseline

## Phase 4 - Hyperparameter Grid Search

**R² = 0.5456**

- 18-configuration grid search
- OneCycleLR vs Cosine comparison -> OneCycleLR wins

## Phase 3 - Critical Bug Fixes

**R² = 0.484, MAE = 0.204** (first valid baseline)

- Fixed `batch.num_graphs` -> `batch.num_nodes` (~1000x gradient error)
- Feature reduction: 100+ -> 88 features
- All experiments before Phase 3 are invalid

## Phases 1-2 - Setup and Dataset

- Exploratory data analysis
- Dataset integrity verification (4,769/5,000 proteins available)
- Feature correlation analysis
- Project structure established

---

## Summary

```
Phase 3  (GCN baseline):     R² = 0.484
Phase 5  (GINE + loss):      R² = 0.568  (+17%)
Phase 12 (contact_count):    R² = 0.882  (+82%)
Phase 15 (optimized):        R² = 0.889  (+84%)
Phase 15+GP (global pool):   R² = 0.941  (+94%)
```

## Key Learnings

### What Worked
1. **contact_count_10A**: Single most important feature (+46% R²)
2. **GATv2 attention**: Dynamic attention with edge features
3. **Dynamic Global Pooling**: Protein-level context via gating (+5.8% R²)
4. **Range-Specific Weighted Loss**: Addresses exposure distribution bias
5. **OneCycleLR**: Better convergence than Cosine scheduler

### What Didn't Work
1. **PNA (Principal Neighbourhood Aggregation)**: 4.6x parameters, no clear benefit
2. **GAT vs GCN (early phases)**: Attention alone didn't help without edge features
3. **Aggregated features (Phase 6)**: Over-simplification lost information
4. **Low regularization (Phase 7)**: Default settings were already optimal
