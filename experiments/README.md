# Experiments

Experimental results for the GNN protein atom exposure prediction project.

## Structure

```
experiments/
├── README.md                      # This file
├── checkpoints/                   # Model checkpoints (not in git)
├── logs/                          # Latest training logs and plots
└── gatv2_globalpool/            # Best model results
    ├── best_model.pt              # Trained model weights (not in git)
    ├── test_metrics.json          # Test set metrics
    ├── training_curves.png        # Loss and R² over epochs
    ├── predictions_raw.png        # Raw predictions heatmap
    ├── predictions_clamped.png    # Clamped predictions heatmap
    ├── error_distribution.png     # Error histogram
    └── error_by_exposure_range.png # Error by exposure range
```

## Best Model: Phase 15 + Global Pooling

| Metric | Value |
|--------|-------|
| R² | 0.9409 |
| MAE | 0.0654 |
| RMSE | 0.0859 |
| Pearson | 0.9707 |
| Mean Error (Bias) | -0.0128 |
| Median AE | 0.0514 |

**Architecture**: GATv2 (4 layers, 136 hidden, 4 heads) + Dynamic Global Pooling (mean, every layer)

**Training**: OneCycleLR, 200 epochs (early stopped at 174), Range-Specific Weighted MSE

**Config**: [configs/gatv2_globalpool.yaml](../configs/gatv2_globalpool.yaml)

## Reproduce Results

```bash
# Train from scratch
python main.py --config configs/gatv2_globalpool.yaml

# Evaluate only (regenerate plots)
python main.py --config configs/gatv2_globalpool.yaml --eval-only
```

## Output Files

Each experiment run generates:
- `test_metrics.json` - Numerical metrics (clamped predictions)
- `training_curves.png` - Loss and R² over training
- `predictions_raw.png` - 2D density heatmap, raw predictions (shows negatives)
- `predictions_clamped.png` - 2D density heatmap, clamped predictions (exposure >= 0)
- `error_distribution.png` - Error histogram and box plot
- `error_by_exposure_range.png` - MAE, bias, and distribution per exposure range

## See Also

- [Architecture](../docs/ARCHITECTURE.md) - Model design
- [History](../docs/HISTORY.md) - All experimental phases
