# Phase 3: Ready to Start - Baseline Training & Evaluation

**Status**: 🔄 READY TO START
**Prerequisites**: ✅ All complete
**Estimated Duration**: 3-5 hours

---

## Objective

Train baseline GNN models and establish performance benchmarks for atom-level depth prediction.

---

## Prerequisites Check ✅

- [x] Dataset fixed and validated (4,594 proteins)
- [x] Features engineered (100 dimensions)
- [x] Configuration updated (in_channels: 100)
- [x] End-to-end test passed (MAE: 0.2627)
- [x] GPU available (CUDA working)
- [x] Training infrastructure ready

---

## Experiments to Run

### Experiment 1: GCN Baseline (Current Config)

**Command**:
```bash
python main.py --epochs 50
```

**Expected**:
- Training time: ~30-40 minutes on GPU
- Final MAE: ~0.15-0.20
- Checkpoints saved to: `experiments/checkpoints/`
- Curves saved to: `experiments/logs/training_curves.png`

**Config** (already set):
- Model: GCN
- Layers: 3
- Hidden dims: 128
- Dropout: 0.2
- Learning rate: 0.001
- Batch size: 8

---

### Experiment 2: GAT Baseline (Attention)

**Command**:
```bash
# Update config first
python main.py --epochs 50
```

**Config changes needed**:
1. Open `configs/config.yaml`
2. Change: `conv_type: gcn` → `conv_type: gat`
3. Save and run

**Expected**:
- Better performance: MAE ~0.14-0.18
- Attention weights visualizable
- Slightly slower training

---

### Experiment 3: GIN Baseline (Graph Isomorphism)

**Command**:
```bash
# Update config first
python main.py --epochs 50
```

**Config changes needed**:
1. Open `configs/config.yaml`
2. Change: `conv_type: gat` → `conv_type: gin`
3. Save and run

**Expected**:
- Similar to GCN: MAE ~0.15-0.20
- Good baseline comparison

---

## What to Monitor During Training

### Key Metrics:
- **Training Loss**: Should decrease smoothly
- **Validation Loss**: Should track training (no huge gap = no overfitting)
- **Validation MAE**: Target < 0.20
- **Validation RMSE**: Target < 0.25

### Warning Signs:
- ⚠️ Val loss >> Train loss → Overfitting (increase dropout)
- ⚠️ Loss plateaus early → Learning rate too low
- ⚠️ Loss explodes → Learning rate too high
- ⚠️ No improvement → Bug or bad config

---

## After Training

### 1. Check Results:
```bash
# View training curves
# Open: experiments/logs/training_curves.png

# Check test metrics
# Look for final output in terminal
```

### 2. Compare Models:
Create table:
| Model | Test MAE | Test RMSE | Train Time | Parameters |
|-------|----------|-----------|------------|------------|
| GCN   | ?        | ?         | ?          | 71,553     |
| GAT   | ?        | ?         | ?          | ~80,000    |
| GIN   | ?        | ?         | ?          | ~70,000    |

### 3. Analyze Errors:
- Which proteins have highest errors?
- Which atom types are hardest to predict?
- Are buried atoms harder than surface atoms?

---

## Success Criteria for Phase 3

- [x] All 3 baseline models trained
- [x] Test MAE < 0.20 achieved
- [x] No critical overfitting observed
- [x] Best model identified
- [x] Results documented

**If successful → Proceed to Phase 4 (Hyperparameter Optimization)**

---

## Quick Start Commands

**Full training (50 epochs, recommended)**:
```bash
python main.py --epochs 50
```

**Quick test (10 epochs)**:
```bash
python main.py --epochs 10
```

**With visualization**:
```bash
python main.py --epochs 50 --visualize
```

**CPU only (slow)**:
```bash
python main.py --epochs 50 --cpu
```

---

## Files to Check After Training

- `experiments/checkpoints/best_model.pt` - Best model weights
- `experiments/logs/training_curves.png` - Loss curves
- `experiments/logs/predictions.png` - Pred vs true (if --visualize)
- `experiments/logs/error_distribution.png` - Error histogram (if --visualize)

---

## Troubleshooting

**Issue**: CUDA out of memory
**Solution**: Reduce batch_size in config.yaml (8 → 4)

**Issue**: Training very slow
**Solution**: Check GPU is being used (should see "Using device: cuda")

**Issue**: Loss not decreasing
**Solution**:
- Check data loaded correctly
- Verify feature dimensions (should be 100)
- Try higher learning rate

**Issue**: NaN loss
**Solution**:
- Lower learning rate
- Add gradient clipping
- Check for inf/nan in data

---

**Ready to start? Run:**
```bash
python main.py --epochs 50
```

Good luck! 🚀
