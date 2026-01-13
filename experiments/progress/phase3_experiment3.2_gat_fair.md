# Phase 3 Experiment 3.2: GAT vs GCN Fair Comparison

**Date**: 2026-01-09
**Status**: 🔄 Ready to Run
**Type**: Controlled experiment - isolate architecture effect

---

## Experimental Design

### Research Question:
**Does GAT architecture perform better than GCN when using identical hyperparameters?**

### Hypothesis:
GAT's attention mechanism will better capture atom burial patterns than GCN's uniform neighbor aggregation, even without optimized regularization.

### Control Variables (Kept Identical):
- ✅ Layers: 3
- ✅ Hidden channels: 128
- ✅ Dropout: 0.2
- ✅ Weight decay: 1e-5
- ✅ Learning rate: 0.001
- ✅ Batch size: 8
- ✅ Training epochs: 10
- ✅ Dataset splits: Same
- ✅ Random seed: 42

### Independent Variable (Only Change):
- **Conv type**: gcn → **gat**

---

## Configuration Comparison

| Parameter | Exp 3.1 (GCN) | Exp 3.2 (GAT) | Status |
|-----------|---------------|---------------|--------|
| **conv_type** | gcn | **gat** | ⚠️ CHANGED |
| in_channels | 100 | 100 | ✅ Same |
| hidden_channels | 128 | 128 | ✅ Same |
| num_layers | 3 | 3 | ✅ Same |
| dropout | 0.2 | 0.2 | ✅ Same |
| weight_decay | 1e-5 | 1e-5 | ✅ Same |
| learning_rate | 0.001 | 0.001 | ✅ Same |
| num_epochs | 50* | 10 | ℹ️ Different |

*Note: We'll compare to GCN's first 10 epochs for fair comparison

---

## Expected Outcomes

### Scenario A: GAT is Better ✅
**Expected metrics (10 epochs)**:
- GAT MAE: ~0.24-0.26
- GCN MAE: ~0.27 (from Exp 3.1 at epoch 10)
- Improvement: 4-10%

**Interpretation**: Attention mechanism helps, proceed to Exp 3.3 (optimized GAT)

### Scenario B: GAT is Same ≈
**Expected metrics (10 epochs)**:
- GAT MAE: ~0.26-0.27
- GCN MAE: ~0.27
- Difference: <2%

**Interpretation**: Architecture doesn't matter much, focus on regularization instead

### Scenario C: GAT is Worse ❌
**Expected metrics (10 epochs)**:
- GAT MAE: >0.27
- GCN MAE: ~0.27
- Difference: GAT worse

**Interpretation**: GAT may need different hyperparameters or is poorly suited for this task

---

## Success Criteria

### Primary Goal:
- [ ] Determine if GAT architecture provides any benefit over GCN

### Secondary Goals:
- [ ] Compare training stability (loss curves)
- [ ] Compare overfitting behavior (train/val gap)
- [ ] Compare training time per epoch

### Quantitative Thresholds:

**GAT is "better" if**:
- Test MAE < GCN's MAE by >3%
- OR R² > GCN's R² by >5%
- OR Pearson > GCN's Pearson by >5%

**GAT is "same" if**:
- All metrics within ±3% of GCN

**GAT is "worse" if**:
- Any metric worse than GCN by >3%

---

## Comparison Plan

### Metrics to Compare:

**From GCN Baseline (Exp 3.1) at Epoch 10**:
- Training loss: ?
- Validation loss: ~0.18 (estimated from curve)
- Test MAE: ? (will need to check)
- Test RMSE: ?
- R²: ?

**From GAT (Exp 3.2) at Epoch 10**:
- Training loss: ?
- Validation loss: ?
- Test MAE: ?
- Test RMSE: ?
- R²: ?

### Visualization:
- Side-by-side training curves (first 10 epochs)
- Bar chart comparing test metrics
- Scatter plot: GCN predictions vs GAT predictions

---

## Why This Matters

### Scientific Rigor:
- Changing one variable at a time is fundamental to experimental design
- Allows us to attribute performance changes to specific factors
- Prevents confounding variables

### Practical Implications:

**If GAT wins**:
- Confirms attention mechanism is valuable
- Justifies proceeding with GAT optimization (Exp 3.3)
- Suggests attention-based architectures for future work

**If GCN wins or ties**:
- Suggests architecture isn't the bottleneck
- Focus should shift to regularization, not architecture
- May try optimizing GCN instead of GAT (cheaper computation)

---

## Training Protocol

### Command:
```bash
cd C:\Users\Edoardo\GNN_protein_exposure\GNN_atom_exposure
venv\Scripts\activate
python main.py --epochs 10
```

### Expected Duration:
- ~15-20 minutes (10 epochs × ~1.5-2 min/epoch)

### What to Monitor:
1. **Training curves** - Do they look similar to GCN's first 10 epochs?
2. **Validation metrics** - Is val loss lower than GCN at epoch 10?
3. **Training time** - Is GAT slower than GCN? (attention has overhead)

---

## After Training

### Immediate Analysis:

1. **Extract GCN metrics at epoch 10** from previous run
   - Check saved logs/checkpoints
   - Need train loss, val loss at epoch 10

2. **Run test evaluation**:
   ```bash
   python main.py --eval-only --checkpoint experiments/checkpoints/best_model.pt
   ```

3. **Compare metrics**:
   - Create comparison table
   - Calculate % differences
   - Decide if improvement is meaningful

### Decision Tree:

```
GAT better by >3%?
├─ YES → Proceed to Exp 3.3 (Optimized GAT)
│         - Add regularization (dropout 0.6, weight decay 5e-4)
│         - Reduce model size (2 layers, 64 hidden)
│         - Train for 50 epochs with early stopping
│
├─ NO (same performance) → Try Exp 3.3' (Optimized GCN)
│         - Keep GCN architecture
│         - Add regularization
│         - May be simpler/faster than GAT
│
└─ WORSE → Debug
          - Check if GAT needs different LR
          - Try GAT with smaller model first
          - Consider GCN optimization instead
```

---

## Files to Create After Experiment

1. **experiments/baselines/gat_fair/results.md**
   - Full metrics and analysis
   - Training curves
   - Comparison to GCN

2. **experiments/progress/phase3_comparison.md**
   - Side-by-side comparison of Exp 3.1 vs 3.2
   - Decision for next experiment
   - Justification

---

## Notes

### Why 10 Epochs?
- Fast iteration (~20 mins)
- Enough to see if architecture helps
- GCN showed learning in first 10 epochs (before severe overfitting)
- Won't waste time if GAT is no better

### Why Not Optimize Yet?
- Need to know if GAT is inherently better
- If we optimize GCN instead, we save computation (GCN is faster)
- Prevents confirmation bias (changing everything and hoping it works)

### Model Complexity:
- GAT has slightly more parameters than GCN (attention weights)
- Estimated: GCN ~70K params, GAT ~80K params
- Should mention in comparison

---

## Configuration File

**configs/config.yaml**:
```yaml
model:
  model_type: gnn
  in_channels: 100
  hidden_channels: 128  # Same as GCN
  num_layers: 3         # Same as GCN
  dropout: 0.2          # Same as GCN
  conv_type: gat        # ONLY CHANGE

training:
  num_epochs: 10        # Short run
  learning_rate: 0.001
  weight_decay: 1.0e-05 # Same as GCN
  scheduler: reduce_on_plateau
  patience: 10
  min_lr: 1.0e-06
  # No early stopping or gradient clip (same as GCN)

experiment:
  name: gat_fair_comparison
  checkpoint_dir: experiments/checkpoints
  log_dir: experiments/logs
  save_best: true
  seed: 42
```

---

## Potential Issues

### Issue 1: GAT Training Instability
- GAT can be less stable than GCN
- May need gradient clipping even for fair test
- If training diverges, may need to restart with clip

### Issue 2: Memory Usage
- GAT uses more memory (attention matrices)
- May need to reduce batch size if OOM
- Would affect fair comparison

### Issue 3: Speed
- GAT is ~20-30% slower per epoch than GCN
- Expected: ~2 min/epoch vs GCN's ~1.5 min/epoch
- Not a problem, just FYI

---

**Status**: Configuration set for fair comparison
**Ready to train**: Yes
**Estimated time**: 15-20 minutes
**Next step**: Run training and compare to GCN epoch 10 metrics

