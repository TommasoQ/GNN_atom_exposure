# Baseline Experiments

This folder contains baseline GNN training experiments for atom exposure prediction.

## Folder Structure

- **`_archived_invalid_experiments/`** - Experiments 3.1-3.4 (invalid due to Bug #1)
  - All had loss weighting bug causing ~1000x gradient reduction
  - Preserved for historical reference only
  - See [README](_archived_invalid_experiments/README.md) for details

- **`gcn_bugs_fixed/`** (Experiment 3.5) ✅ **COMPLETE**
  - First valid experiment with all bugs fixed
  - GCN: 3 layers, 96 hidden, dropout 0.3
  - Results: MAE 0.204, R² 0.484, Pearson 0.696
  - Status: **First valid baseline established!**
  - [View results](gcn_bugs_fixed/RESULTS.md)

- **`gat_with_edges/`** (Experiment 3.6) ✅ **COMPLETE**
  - GAT with edge features enabled (fair comparison)
  - GAT: 3 layers, 96 hidden, dropout 0.3, 4 heads
  - Results: MAE 0.204, R² 0.477, Pearson 0.691
  - Status: **Equivalent to GCN, no improvement**
  - [View results](gat_with_edges/RESULTS.md)

## Valid Experiments (Post-Bug-Fix)

| Exp | Model | Config | MAE | R² | Pearson | Status |
|-----|-------|--------|-----|-----|---------|--------|
| 3.5 | GCN | 3L, 96H, d0.3 | 0.204 | 0.484 | 0.696 | ✅ Complete |
| 3.6 | GAT | 3L, 96H, d0.3, 4h | 0.204 | 0.477 | 0.691 | ✅ Complete |

### Key Finding
**GCN = GAT** in performance. The attention mechanism and edge features provide no meaningful advantage for atom exposure prediction. **GCN is the recommended baseline** due to simpler architecture and faster training.

Start from Experiment 3.5 onwards. All experiments before 3.5 are invalid due to critical bugs.

## Bug Information

- **Full bug report**: `../progress/bug_discovery.md`
- **Phase 3 summary**: `../progress/phase3_summary.md`

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
