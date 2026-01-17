# Phase 3: Baseline Training & Bug Discovery

**Status**: ✅ BUGS FIXED, READY FOR VALID TRAINING
**Date**: 2026-01-08 to 2026-01-09

---

## Summary

Phase 3 began as baseline GNN training but became a critical debugging phase. After 4 failed experiments with consistently poor results, systematic investigation revealed **4 critical bugs** in the training pipeline. All bugs have been fixed.

---

## Experiments 3.1-3.4: Invalid (Pre-Bug-Fix)

**⚠️ These results are invalid due to Bug #1 (loss weighting error)**

| Exp | Model | Config | MAE | R² | Status |
|-----|-------|--------|-----|-----|--------|
| 3.1 | GCN | 3L, 128H, d0.2, 50ep | 0.263 | 0.155 | ❌ Invalid |
| 3.2 | GAT | 3L, 128H, d0.2, 10ep | 0.279 | 0.134 | ❌ Invalid |
| 3.3 | GCN | 2L, 64H, d0.6 | 0.317 | -0.104 | ❌ Invalid |
| 3.3b | GCN | 2L, 96H, d0.4 | 0.275 | 0.126 | ❌ Invalid |
| 3.4 | GAT | 2L, 96H, d0.4 | 0.283 | 0.099 | ❌ Invalid |

**Pattern**: ALL experiments showed R² ~0.10-0.15 and MAE ~0.26-0.32, regardless of architecture or regularization. This indicated systematic bugs, not modeling issues.

---

## Bug Discovery (2026-01-09)

### Critical Observation:

Changing architecture (GCN/GAT), layers (2/3), hidden dims (64/96/128), dropout (0.2/0.4/0.6), and weight decay (1e-5/1e-4/5e-4) had **minimal effect** on performance. This pattern suggested infrastructure bugs.

### Investigation Method:

Launched deep code exploration to examine:
- Data loading and feature extraction
- Model forward pass and dimensions
- Training loop and loss calculation
- Config consistency

### Bugs Discovered:

1. **Bug #1 (CRITICAL)**: Loss weighting by `num_graphs` instead of `num_nodes` → ~1000x gradient reduction
2. **Bug #2 (CRITICAL)**: Config `in_channels: 100` but actual features: 101
3. **Bug #3 (IMPORTANT)**: GAT not using edge features (distances)
4. **Bug #4 (IMPORTANT)**: Silent target padding with 0.0 for missing atoms

See [bug_discovery.md](bug_discovery.md) for full details.

---

## Fixes Applied (2026-01-09)

### Code Changes:

1. **src/training/train.py** (lines 85-86, 121-122):
   - Changed `batch.num_graphs` → `batch.num_nodes`

2. **configs/config.yaml** (line 11):
   - Changed `in_channels: 100` → `in_channels: 101`

3. **src/models/gnn.py** (lines 50-56, 102):
   - Added `edge_dim=1` to GATConv initialization
   - Added `edge_attr=edge_attr` to GAT forward pass

4. **src/data/dataset_fixed.py** (lines 316-322):
   - Replaced silent padding with explicit error on missing labels

### Documentation Cleanup:

- Archived invalid experiments → `baselines/_archived_invalid_experiments/`
- Archived outdated analysis → `progress/_archived_pre_bugfix/`
- Created `bug_discovery.md` with full bug report
- Updated this summary

### Cache Cleanup:

- Deleted `processed/` folder to force regeneration with correct logic

---

## Experiment 3.5: First Valid Training ✅ COMPLETE

**Date**: 2026-01-09
**Status**: SUCCESS - First valid baseline established!

**Config**:
```yaml
Model: GCN
Layers: 3
Hidden: 96
Dropout: 0.3
in_channels: 100 (CORRECTED: 34+58+8)
All bugs: FIXED
```

**Actual Performance**:
- Test MAE: **0.2040** ✅ (target: < 0.24)
- Test RMSE: 0.2539
- Test R²: **0.4835** ✅ (target: > 0.40)
- Pearson: **0.6963** ✅
- Training: Smooth convergence, no issues

**Success Criteria**:
- ✅ R² > 0.40 (achieved 0.48)
- ✅ MAE < 0.24 (achieved 0.20)
- ✅ Training/val curves healthy
- ✅ No dimension errors or NaN values

**Results**: All success criteria met! Bug fixes successfully restored model learning capability. Performance improved 3-4x compared to buggy experiments (R² 0.48 vs 0.10-0.15).

**Full details**: [baselines/gcn_bugs_fixed/RESULTS.md](../baselines/gcn_bugs_fixed/RESULTS.md)

---

## Lessons Learned

1. **Systematic poor performance → look for infrastructure bugs**
   - When hyperparameters have no effect, it's not a tuning problem

2. **Per-node tasks need per-node accounting**
   - Weight and average by `num_nodes`, not `num_graphs`

3. **Dimension mismatches can be silent**
   - Always verify actual feature counts match config

4. **Never pad targets silently**
   - Missing data should fail loudly

5. **Debug methodology matters**
   - Systematic code investigation found bugs that manual inspection missed

---

---

## Experiment 3.6: GAT with Edge Features ✅ COMPLETE

**Date**: 2026-01-10
**Status**: COMPLETE - Fair comparison with GCN completed

**Config**:
```yaml
Model: GAT (with edge features enabled)
Layers: 3
Hidden: 96
Dropout: 0.3
Attention heads: 4
Edge features: ENABLED (distance information)
All bugs: FIXED
```

**Actual Performance**:
- Test MAE: **0.2035**
- Test RMSE: 0.2556
- Test R²: **0.4765**
- Pearson: **0.6907**
- Training: Smooth convergence, similar to GCN

**Comparison with GCN (Exp 3.5)**:
- MAE: 0.2035 vs 0.2040 (0.2% better - negligible)
- R²: 0.4765 vs 0.4835 (1.4% worse - negligible)
- Pearson: 0.6907 vs 0.6963 (0.8% worse - negligible)

**Conclusion**: GAT performs **virtually identically** to GCN. The attention mechanism and edge features do not provide meaningful improvement for this task. **GCN is the preferred baseline** due to simpler architecture, faster training, and equivalent performance.

**Full details**: [baselines/gat_with_edges/RESULTS.md](../baselines/gat_with_edges/RESULTS.md)

---

## Phase 3 Summary

**Status**: ✅ **COMPLETE**

### Valid Experiments (Post-Bug-Fix):
- **Experiment 3.5**: GCN (3L, 96H, d0.3) → MAE 0.204, R² 0.484 ✅
- **Experiment 3.6**: GAT (3L, 96H, d0.3) → MAE 0.204, R² 0.477 ✅

### Key Findings:
1. **Bug fixes were critical** - Performance improved 3-4x (R² 0.15 → 0.48)
2. **GCN = GAT performance** - No advantage from attention or edge features
3. **GCN is preferred baseline** - Simpler, faster, equivalent results
4. **Performance plateau** - Both models reach ~R² 0.48

### Recommendations for Phase 4:
1. **Deeper GCN** (4-5 layers) - Capture longer-range interactions
2. **Hyperparameter tuning** - Learning rate, dropout optimization
3. **Feature engineering** - Analyze feature importance, add domain features
4. **Alternative architectures** - GIN, GraphSAGE if needed

---

## Experiment 3.7: GIN Architecture ✅ COMPLETE

**Date**: 2026-01-10
**Status**: COMPLETE - Best performing architecture so far

**Config**:
```yaml
Model: GIN (Graph Isomorphism Network)
Layers: 3
Hidden: 96
Dropout: 0.3
Features: 100 (pre-reduction)
```

**Actual Performance**:
- Test MAE: **0.1987**
- Test R²: **0.4985**
- Pearson: **0.7081**

**Comparison**: GIN outperforms both GCN and GAT by small margin.

---

## Experiment 3.8: Feature Reduction ✅ COMPLETE

**Date**: 2026-01-10
**Status**: COMPLETE - Feature reduction validated

**Config**:
```yaml
Model: GIN
Layers: 3
Hidden: 96
Dropout: 0.3
Features: 88 (reduced from 100 - removed 12 redundant features)
```

**Actual Performance**:
- Test MAE: **0.2004**
- Test RMSE: 0.2488
- Test R²: **0.5041** ✅ (new best!)
- Pearson: **0.7143** ✅ (new best!)
- Median AE: 0.1722

**Comparison with Exp 3.7** (100 features):
| Metric | 100 feat | 88 feat | Change |
|--------|----------|---------|--------|
| R² | 0.4985 | **0.5041** | +1.1% |
| MAE | 0.1987 | 0.2004 | +0.9% |
| Pearson | 0.7081 | **0.7143** | +0.9% |

**Conclusion**: Feature reduction successful. 88-feature model achieves slightly better R² while being 12% faster to train. Removed features were confirmed redundant.

---

## Phase 3 Final Summary

**Status**: ✅ **COMPLETE**

### All Valid Experiments:
| Exp | Model | Features | MAE | R² | Pearson |
|-----|-------|----------|-----|-----|---------|
| 3.5 | GCN | 100 | 0.204 | 0.484 | 0.696 |
| 3.6 | GAT | 100 | 0.204 | 0.477 | 0.691 |
| 3.7 | GIN | 100 | 0.199 | 0.499 | 0.708 |
| **3.8** | **GIN** | **88** | **0.200** | **0.504** | **0.714** |

### Key Findings:
1. **GIN > GCN ≈ GAT** - GIN provides ~2% improvement
2. **Feature reduction works** - 88 features match/exceed 100 features
3. **Performance plateau** - All architectures hit ~R² 0.50 ceiling
4. **Geometric features dominate** - 80% of importance from 8 features

### Recommendations for Phase 4:
1. **Deeper GIN** (4-5 layers) - Capture longer-range interactions
2. **Hyperparameter tuning** - LR, dropout, width optimization
3. **Skip connections** - Help gradient flow in deeper networks

---

## Next Steps

1. ✅ All bugs fixed
2. ✅ Documentation cleaned up
3. ✅ GCN baseline (Exp 3.5) - R² 0.484
4. ✅ GAT comparison (Exp 3.6) - R² 0.477
5. ✅ GIN architecture (Exp 3.7) - R² 0.499
6. ✅ Feature reduction (Exp 3.8) - R² 0.504
7. ⏳ **Phase 4: Deeper networks + hyperparameter tuning**

**Phase 3 COMPLETE**: Best model is GIN with 88 features at R² 0.504.

---

**Files**:
- Bug details: [bug_discovery.md](bug_discovery.md)
- Invalid experiments: [baselines/_archived_invalid_experiments/](../baselines/_archived_invalid_experiments/)
- Archived analysis: [_archived_pre_bugfix/](_archived_pre_bugfix/)
- GCN results: [baselines/gcn_bugs_fixed/RESULTS.md](../baselines/gcn_bugs_fixed/RESULTS.md)
- GAT results: [baselines/gat_with_edges/RESULTS.md](../baselines/gat_with_edges/RESULTS.md)
