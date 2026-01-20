# Phase 13a Best Model - Archive

**Date**: 2026-01-19
**Status**: Best baseline model

---

## Test Metrics

| Metric | Value |
|--------|-------|
| **R²** | **0.8876** |
| MAE | 0.0857 |
| RMSE | 0.1184 |
| Pearson Correlation | 0.9423 |
| Median AE | 0.0647 |
| Mean Error (Bias) | -0.0003 |

---

## Configuration

- **Architecture**: GATv2 (4 heads, 128 hidden channels, 3 layers)
- **Features**: 93 node features, 12 edge features
- **Dropout**: 0.25
- **Training**: 200 epochs, OneCycle (max_lr=0.001)
- **Early Stopping**: R²-based, patience=50
- **Best Checkpoint**: Epoch 159

---

## Key Improvements over Phase 12

| Metric | Phase 12 | Phase 13a | Change |
|--------|----------|-----------|--------|
| R² | 0.8817 | 0.8876 | +0.67% |
| MAE | 0.0866 | 0.0857 | -1.0% |
| RMSE | 0.1215 | 0.1184 | -2.5% |

---

## Changes from Phase 12

1. **Lower max_lr**: 0.002 → 0.001 (reduced oscillation)
2. **Longer warmup**: 25 → 35 epochs
3. **Increased patience**: 35 → 50 epochs
4. **Higher dropout**: 0.20 → 0.25
5. **Stronger weight decay**: 5e-5 → 1e-4
6. **Early stopping on R²** instead of val_loss

---

## Files in this Archive

- `best_model.pt` - PyTorch model checkpoint
- `config.yaml` - Full configuration
- `test_metrics.json` - Detailed test metrics
- `training_history.csv` - Per-epoch training metrics
- `training_curves.png` - Loss and R² curves
- `predictions_vs_actual.png` - Scatter plot
- `error_distribution.png` - Error histogram
- `error_by_exposure_range.png` - Error analysis by target value

---

## Usage

```python
import torch
from src.models.gnn import AtomExposureGNN

# Load model
checkpoint = torch.load('best_model.pt')
model = AtomExposureGNN(
    in_channels=93,
    hidden_channels=128,
    num_layers=3,
    conv_type='gatv2',
    edge_dim=12,
    dropout=0.25
)
model.load_state_dict(checkpoint['model_state_dict'])
```
