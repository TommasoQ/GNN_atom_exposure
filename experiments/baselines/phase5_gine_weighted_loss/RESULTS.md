# Phase 5 – GINE (96×3) + Exposure-Weighted Loss

Deterministic baseline run (seed=42) with GINE, 3 layers × 96 hidden channels, exposure-weighted MSE (alpha=1.5, threshold=0.8). Scheduler: OneCycleLR with max_lr=0.003, warmup_epochs=15, div_factor=25, final_div_factor=10000. AMP enabled.

## Test Metrics
| Metric | Value |
| --- | --- |
| R² | **0.5684** |
| MAE | 0.1829 |
| RMSE | 0.2321 |
| Pearson | 0.7551 |
| Median AE | 0.1504 |
| Mean Error | +0.0114 |

## Files
- `best_model.pt` – checkpoint saved at epoch 139 (lowest val loss 0.0648)
- `training_history.csv` – per-epoch train/val metrics
- `config_snapshot.yaml` – exact configuration for reproduction

## Reproduce / Resume
```bash
python main.py --config experiments/baselines/phase5_gine_weighted_loss/config_snapshot.yaml --epochs 150
```
Ensure `deterministic: true` in the config so results match.
