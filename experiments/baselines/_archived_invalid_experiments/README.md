# Archived Invalid Experiments

**Date Archived**: 2026-01-09

## ⚠️ WARNING: These Results Are Invalid

All experiments in this folder were conducted with **critical Bug #1** that caused ~1000x gradient reduction.

### Bug #1: Loss Weighting Error

**Location**: `src/training/train.py` lines 85-86 and 121-122

**The Problem**:
```python
# WRONG (what was used):
total_loss += loss.item() * batch.num_graphs  # Multiplying by 8 graphs
num_samples += batch.num_graphs

# CORRECT (fixed):
total_loss += loss.item() * batch.num_nodes   # Multiplying by ~8000 atoms
num_samples += batch.num_nodes
```

**Impact**:
- Gradients were ~1000x too small
- Model barely learned despite appearing to train
- All experiments showed R² ~0.10-0.15 and MAE ~0.26-0.32
- Changing architecture/regularization had minimal effect

### Archived Experiments:

| Folder | Description | Results |
|--------|-------------|---------|
| `gcn_overfit/` | Exp 3.1: GCN baseline (3L, 128H, d0.2, 50ep) | MAE 0.263, R² 0.155 |
| `gat_fair/` | Exp 3.2: GAT fair test (3L, 128H, d0.2, 10ep) | MAE 0.279, R² 0.134 |
| `gcn_overreg/` | Exp 3.3: GCN over-regularized (2L, 64H, d0.6) | MAE 0.317, R² -0.104 |
| `gcn_moderate/` | Exp 3.3b: GCN moderate (2L, 96H, d0.4) | MAE 0.275, R² 0.126 |

### Why Preserved:

These experiments are kept for historical reference to document:
1. The bug discovery process
2. How systematic issues manifest (consistent poor performance)
3. The importance of proper loss calculation in node-level tasks

### Valid Experiments:

See parent folder for experiments conducted after bug fixes:
- Exp 3.5+ (with fixed loss weighting, correct in_channels, etc.)

### More Information:

- Full bug report: `../progress/bug_discovery.md`
- Phase 3 summary: `../progress/phase3_summary.md`
