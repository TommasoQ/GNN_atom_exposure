# Phase 3: Overfitting Analysis - GCN Baseline

**Date**: 2026-01-09
**Model**: GCN (3 layers, 128 hidden, dropout 0.2)
**Training**: 50 epochs complete
**Status**: 🚨 **SEVERE OVERFITTING DETECTED**

---

## Training Curve Analysis

### Observed Behavior:

**Training Loss (Blue)**:
- Initial: ~0.08
- Final: ~0.06
- Pattern: Smooth hyperbolic decrease, very stable
- ✅ Healthy training behavior

**Validation Loss (Red)**:
- Initial: ~0.10
- Final: ~0.25-0.48 (with spike to 0.48 at epoch 50)
- Pattern: Increases over time with high variance
- ❌ **Validation loss is 5-8x higher than training loss**

### Overfitting Indicators:

1. ✅ **Train loss decreases while val loss increases** - Primary indicator
2. ✅ **Large train/val gap** (5-8x difference)
3. ✅ **High validation variance** (oscillates 0.18-0.48)
4. ✅ **Gap grows over time** (diverges after epoch 10)

**Diagnosis**: The model is **memorizing training proteins** rather than learning generalizable patterns.

---

## Root Causes

### 1. Model Complexity vs Dataset Size
- **Model capacity**: 3 layers × 128 hidden × 100 features ≈ **70K parameters**
- **Training samples**: 3,675 proteins
- **Ratio**: ~19 samples per parameter (generally want >100)
- **Issue**: Model has too much capacity relative to data

### 2. Insufficient Regularization
- **Current dropout**: 0.2 (20%)
- **Weight decay**: 1e-5 (very small)
- **Issue**: Not enough regularization to prevent memorization

### 3. No Early Stopping
- **Training ran**: Full 50 epochs
- **Best epoch likely**: ~10-15 (when val loss was lowest ~0.18)
- **Issue**: Continued training degraded generalization

### 4. Small Validation Set
- **Validation size**: 459 proteins
- **Issue**: High variance in validation metrics (explains oscillations)

### 5. Dataset Characteristics
- **Possible issue**: Proteins in train/val/test may be too similar
- **Need to verify**: Are splits stratified by protein families?

---

## Evidence from Curves

### Epoch-by-Epoch Breakdown:

| Epoch Range | Train Loss | Val Loss | Gap | Interpretation |
|-------------|------------|----------|-----|----------------|
| 1-10 | 0.08 → 0.07 | 0.10 → 0.18 | 2.5x | Initial learning, acceptable gap |
| 11-20 | 0.07 → 0.065 | 0.18 → 0.28 | 4x | Overfitting begins |
| 21-40 | 0.065 → 0.06 | 0.25 → 0.35 | 5-6x | Severe overfitting |
| 41-50 | 0.06 (flat) | 0.30 → 0.48 | 6-8x | Critical overfitting |

**Best model likely saved around epoch 12-15** when validation loss was minimized.

---

## Recommendations

### Immediate Actions (Phase 3 Completion):

1. **Evaluate Test Set** 🔴 HIGH PRIORITY
   - Run: `python main.py --eval-only --checkpoint experiments/checkpoints/best_model.pt`
   - Check if test MAE/RMSE matches validation loss
   - If test performance is also poor → confirms overfitting
   - If test performance is good → validation set may be unusual

2. **Check Best Model Epoch**
   - Verify which epoch saved best_model.pt
   - Likely epoch 10-15 (lowest validation loss)

### Short-Term Fixes (Phase 3 Retry):

3. **Increase Dropout** (easiest fix)
   ```yaml
   # config.yaml
   model:
     dropout: 0.5  # Increase from 0.2
   ```

4. **Enable Early Stopping** (prevent overtraining)
   ```yaml
   # config.yaml
   training:
     early_stopping: true
     early_stopping_patience: 15  # Stop if no improvement for 15 epochs
   ```

5. **Add Gradient Clipping** (stabilize training)
   ```yaml
   # config.yaml
   training:
     gradient_clip: 1.0
   ```

### Medium-Term Fixes (Phase 4):

6. **Reduce Model Capacity**
   ```yaml
   # Option A: Fewer layers
   model:
     num_layers: 2  # Reduce from 3
     hidden_channels: 128

   # Option B: Smaller hidden dims
   model:
     num_layers: 3
     hidden_channels: 64  # Reduce from 128

   # Option C: Both
   model:
     num_layers: 2
     hidden_channels: 64
   ```

7. **Increase Weight Decay**
   ```yaml
   # config.yaml
   training:
     weight_decay: 1.0e-4  # Increase from 1e-5
   ```

8. **Try Data Augmentation**
   - Random coordinate perturbations
   - Feature dropout during training
   - Edge dropout

### Long-Term Improvements (Phase 5+):

9. **Verify Train/Val/Test Splits**
   - Ensure proteins are split by family/similarity
   - Current split may be random (proteins too similar)

10. **Add More Training Data**
    - 3,675 proteins may not be enough for 70K parameters
    - Consider semi-supervised learning or pre-training

11. **Try Simpler Architecture**
    - Linear baseline to establish lower bound
    - 1-layer GCN to verify complexity issue

---

## Expected Outcomes After Fixes

### With Increased Dropout (0.5) + Early Stopping:

**Expected**:
- Train loss: ~0.10-0.12 (higher than before, but that's OK)
- Val loss: ~0.15-0.20 (much closer to train)
- Gap: <2x (healthy)
- Test MAE: ~0.18-0.22

### With Reduced Model (2 layers, 64 hidden):

**Expected**:
- Train loss: ~0.12-0.15 (higher, less memorization)
- Val loss: ~0.16-0.22 (better generalization)
- Gap: <1.5x (excellent)
- Test MAE: ~0.20-0.25

---

## Next Steps

### Phase 3 Completion Checklist:

- [ ] Evaluate test set with best_model.pt
- [ ] Document test metrics (MAE, RMSE, R², Pearson)
- [ ] Identify which epoch was "best"
- [ ] Create scatter plot (predictions vs ground truth)
- [ ] Analyze per-protein errors

### Phase 3 Retry Plan:

**Experiment 3.2: GCN with Regularization**
```yaml
Config changes:
- dropout: 0.2 → 0.5
- early_stopping_patience: 15 (new)
- gradient_clip: 1.0 (new)

Expected improvement:
- Val loss decreases to ~0.15-0.20
- Train/val gap < 2x
- Test MAE < 0.22
```

**Experiment 3.3: Lightweight GCN**
```yaml
Config changes:
- num_layers: 3 → 2
- hidden_channels: 128 → 64
- dropout: 0.5

Expected improvement:
- Better generalization
- Val loss ~0.16-0.22
- Test MAE < 0.24
```

---

## Lessons Learned

### What Worked:
✅ Dataset loading and feature engineering pipeline
✅ Training infrastructure and checkpointing
✅ Model can learn (train loss decreases)

### What Didn't Work:
❌ Model too complex for dataset size
❌ Insufficient regularization
❌ No early stopping mechanism
❌ Dropout too aggressive only on training, not validation

### Key Insights:
1. **More parameters ≠ better performance** without sufficient data
2. **Validation curves are critical** - should have stopped at epoch 15
3. **Regularization is essential** for small molecular datasets
4. **Early stopping prevents overtraining**

---

## Comparison to Initial 1-Epoch Test

| Metric | 1-Epoch Test | 50-Epoch Training |
|--------|--------------|-------------------|
| Train Loss | 0.088 | 0.06 |
| Val Loss | ? | 0.48 (final) |
| Test MAE | 0.2627 | ? (pending) |
| Test RMSE | 0.3132 | ? (pending) |

**Hypothesis**: 1-epoch model may actually perform BETTER on test set than 50-epoch model due to less overfitting.

---

## References

**Overfitting Indicators**:
- Train loss << Val loss (gap > 2x)
- Val loss increases while train loss decreases
- High variance in validation metrics

**Common Causes**:
- Model too complex
- Insufficient regularization
- Training too long
- Small dataset

**Standard Fixes**:
- Increase dropout (0.3-0.5 for molecular data)
- Reduce model capacity
- Early stopping
- Data augmentation
- Stronger weight decay

---

**Status**: Analysis complete, awaiting test set evaluation
**Next**: Evaluate best_model.pt on test set to confirm overfitting hypothesis

