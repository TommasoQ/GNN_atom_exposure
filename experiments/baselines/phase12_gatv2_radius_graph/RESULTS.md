# Phase 12 - GATv2 + Radius Graph + Contact Count

**NEW BEST MODEL** - R² = 0.8817 (+46.3% vs Phase 9)

## Key Changes from Phase 9 (GINE baseline)

| Aspect | Phase 9 | Phase 12 |
|--------|---------|----------|
| Architecture | GINE | GATv2 with residual + ELU |
| Node features | 92 | 93 (+`contact_count_10A`) |
| Edge features | 11 | 12 (+`in_radius` @ 8Å) |
| Hidden channels | 96 | 128 |
| Dropout | 0.35 | 0.20 |

## Test Metrics

| Metric | Value |
|--------|-------|
| **R²** | **0.8817** |
| MAE | 0.0866 |
| RMSE | 0.1215 |
| Pearson | 0.9393 |
| Median AE | 0.0642 |
| Mean Error | +0.0061 |
| Std Error | 0.1213 |

## Comparison with Previous Best (Phase 9)

| Metric | Phase 9 | Phase 12 | Improvement |
|--------|---------|----------|-------------|
| R² | 0.6028 | 0.8817 | **+46.3%** |
| MAE | ~0.18 | 0.0866 | **-52%** |
| RMSE | ~0.23 | 0.1215 | **-47%** |
| Pearson | ~0.78 | 0.9393 | **+20%** |

## Files

- `best_model.pt` - Checkpoint saved at epoch 42 (lowest val loss)
- `config_snapshot.yaml` - Exact configuration for reproduction
- `training_history.csv` - Per-epoch train/val metrics
- `training_curves.png` - Loss and R² over epochs

## Reproduce

```bash
python main.py --config experiments/baselines/phase12_gatv2_radius_graph/config_snapshot.yaml
```

## Analysis Notes

The dramatic improvement (+46% R²) came from two key additions:

1. **`contact_count_10A` node feature**: Number of atoms within 10Å radius - provides direct local density information that the GNN can use without needing to aggregate over multiple hops.

2. **`in_radius` edge feature**: Binary indicator for edges with distance < 8Å - helps the attention mechanism distinguish between close contacts and distant interactions.

Combined with GATv2's dynamic attention (vs GINE's fixed message passing), the model can now selectively weight information from the local environment based on learned relevance.
