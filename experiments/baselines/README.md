# Baseline Experiments

This folder contains baseline GNN training experiments for atom exposure prediction.

## Folder Structure

- **`_archived_invalid_experiments/`** - Experiments 3.1-3.4 (invalid due to Bug #1)
  - All had loss weighting bug causing ~1000x gradient reduction
  - Preserved for historical reference only
  - See [README](_archived_invalid_experiments/README.md) for details

- **`experiment_3.5_gcn_baseline/`** (Experiment 3.5) ✅ **COMPLETE**
  - First valid experiment with all bugs fixed
  - GCN: 3 layers, 96 hidden, dropout 0.3
  - Results: MAE 0.204, R² 0.484, Pearson 0.696
  - Status: **First valid baseline established!**
  - [View results](experiment_3.5_gcn_baseline/RESULTS.md)

- **`experiment_3.6_gat_edges/`** (Experiment 3.6) ✅ **COMPLETE**
  - GAT with edge features enabled (fair comparison)
  - GAT: 3 layers, 96 hidden, dropout 0.3, 4 heads
  - Results: MAE 0.204, R² 0.477, Pearson 0.691
  - Status: **Equivalent to GCN, no improvement**
  - [View results](experiment_3.6_gat_edges/RESULTS.md)

- **`experiment_5.0_gine_best/`** (Phase 5) ✅ **COMPLETE**
  - GINE: 3 layers × 96 hidden, dropout 0.35, exposure-weighted MSE (α=1.5)
  - Scheduler: OneCycle (max_lr=0.003, warmup=15), deterministic seed=42
  - Results: MAE 0.1829, R² 0.5684, Pearson 0.7551
  - Status: Previous best (superseded by Phase 12)
  - [View results](experiment_5.0_gine_best/RESULTS.md)

- **`phase12_gatv2_radius_graph/`** (Phase 12) ✅ **CURRENT BEST**
  - GATv2: 3 layers × 128 hidden, dropout 0.20, 4 heads, internal residual + ELU
  - Features: 93 node features (+contact_count_10A), 12 edge features (+in_radius)
  - Results: MAE 0.0866, R² **0.8817**, Pearson 0.9393
  - Status: **NEW BEST! +46% R² over Phase 9**
  - [View results](phase12_gatv2_radius_graph/RESULTS.md)

- **`FAILED_phase6_aggregated/`** (Phase 6) ❌ **FAILED**
  - Attempted aggregated feature engineering (100+ → 50 features)
  - Results: R² 0.4002 (29% drop from Phase 5)
  - Reason: Over-aggregation lost important predictive information
  - [View failure analysis](FAILED_phase6_aggregated/NOTES.md)

## Valid Experiments (Post-Bug-Fix)

| Exp | Model | Config | MAE | R² | Pearson | Status |
|-----|-------|--------|-----|-----|---------|--------|
| 3.5 | GCN | 3L, 96H, d0.3 | 0.204 | 0.484 | 0.696 | ✅ Complete |
| 3.6 | GAT | 3L, 96H, d0.3, 4h | 0.204 | 0.477 | 0.691 | ✅ Complete |
| 5.0 | GINE + Weighted Loss | 3L, 96H, d0.35, α=1.5 | 0.183 | 0.568 | 0.755 | ✅ Complete |
| **12** | **GATv2 + Radius+Contact** | 3L, 128H, d0.2, 93 features | **0.087** | **0.882** | **0.939** | **✅ BEST** |

### Key Finding (Phase 12)
Adding **local density features** (`contact_count_10A`, `in_radius`) provides massive improvement (+46% R²). The GATv2 attention mechanism, combined with these features, can now effectively distinguish buried vs exposed atoms.

Start from Experiment 3.5 onwards. All experiments before 3.5 are invalid due to critical bugs.

## Bug Information

- **Full bug report**: `../../docs/analysis/BUG_DISCOVERY.md`
- **Phase 3 summary**: `../../docs/archive/phase3_complete.md`

## Key Bugs Fixed:

1. **Loss weighting**: Changed `batch.num_graphs` → `batch.num_nodes` (~1000x gradient fix)
2. **Config mismatch**: Verified `in_channels: 100` (34+58+8 features)
3. **Edge features**: Added to GAT layers
4. **Target padding**: Fail instead of silent 0.0 padding
5. **Dataset caching**: Fixed triple processing issue

**Date fixed**: 2026-01-09

## Performance Comparison

| Metric | Pre-Fix (Exp 3.1-3.4) | Post-Fix (Exp 3.5) | Improvement |
|--------|----------------------|-------------------|-------------|
| MAE | 0.263-0.317 | **0.204** | 22-36% better |
| R² | 0.10-0.15 | **0.484** | 3-4x better |
| Pearson | ~0.45-0.50 | **0.696** | 40% better |
