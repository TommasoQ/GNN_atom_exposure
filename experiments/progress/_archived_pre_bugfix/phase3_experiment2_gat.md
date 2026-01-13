# Phase 3 Experiment 2: GAT with Strong Regularization

**Date**: 2026-01-09
**Status**: 🔄 Ready to Run
**Previous**: GCN Baseline (severe overfitting)

---

## Motivation

Experiment 1 (GCN) showed severe overfitting:
- Train loss: 0.06
- Val loss: 0.48 (8x higher!)
- Test MAE: 0.2672 (worse than 1-epoch baseline)
- R²: 0.1552 (only 15.5% variance explained)

**Root causes identified**:
1. Model too complex (70K parameters, 3 layers, 128 hidden)
2. Insufficient regularization (dropout 0.2)
3. No early stopping (trained all 50 epochs)
4. GCN may not capture burial patterns well

---

## Changes from Experiment 1

### Model Architecture:

| Parameter | GCN Baseline | GAT Regularized | Change |
|-----------|--------------|-----------------|--------|
| **Conv Type** | GCN | **GAT** | ✅ Attention mechanism |
| **Layers** | 3 | **2** | ✅ -33% depth |
| **Hidden Channels** | 128 | **64** | ✅ -50% capacity |
| **Dropout** | 0.2 | **0.6** | ✅ 3x stronger |
| **Total Params** | ~70K | **~22K** | ✅ -68% parameters |

### Training Configuration:

| Parameter | GCN Baseline | GAT Regularized | Change |
|-----------|--------------|-----------------|--------|
| **Weight Decay** | 1e-5 | **5e-4** | ✅ 50x stronger |
| **Gradient Clip** | None | **1.0** | ✅ NEW |
| **Early Stopping** | None | **Patience 10** | ✅ NEW |
| **Max Epochs** | 50 | 50 | (will stop early) |

---

## Updated Configuration

**File**: `configs/config.yaml`

```yaml
model:
  model_type: gnn
  in_channels: 100
  hidden_channels: 64  # Reduced from 128
  num_layers: 2  # Reduced from 3
  dropout: 0.6  # Increased from 0.2
  conv_type: gat  # Changed from gcn

training:
  num_epochs: 50
  learning_rate: 0.001
  weight_decay: 5.0e-04  # Increased from 1e-5
  scheduler: reduce_on_plateau
  patience: 10
  early_stopping_patience: 10  # NEW
  min_lr: 1.0e-06
  gradient_clip: 1.0  # NEW

experiment:
  name: gat_regularized
  checkpoint_dir: experiments/checkpoints
  log_dir: experiments/logs
  save_best: true
  seed: 42
```

---

## Code Changes

### 1. Added Gradient Clipping
**File**: `src/training/train.py:78-80`

```python
# Gradient clipping if enabled
if self.gradient_clip is not None:
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
```

### 2. Added Early Stopping
**File**: `src/training/train.py:190-196`

```python
# Early stopping check
if self.early_stopping_patience is not None:
    if self.epochs_without_improvement >= self.early_stopping_patience:
        print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
        print(f"   No improvement for {self.early_stopping_patience} consecutive epochs")
        break
```

### 3. Updated Trainer Initialization
**File**: `main.py:113-114`

```python
gradient_clip=getattr(config.training, 'gradient_clip', None),
early_stopping_patience=getattr(config.training, 'early_stopping_patience', None)
```

---

## Expected Improvements

### Training Behavior:

**Expected curves**:
- Train loss: ~0.10-0.15 (higher than GCN due to dropout)
- Val loss: ~0.15-0.20 (much closer to train)
- Train/val gap: <2x (healthy, not 8x like before)
- Early stop: Around epoch 15-25 (not full 50)

**Why this should work**:
1. ✅ **GAT attention** - Better captures which neighbors matter for burial
2. ✅ **Simpler model** - Less capacity = less memorization
3. ✅ **Strong dropout** - Forces robustness, prevents overfitting
4. ✅ **Early stopping** - Won't overtrain
5. ✅ **Gradient clipping** - Training stability
6. ✅ **Strong L2 reg** - Weight decay discourages large weights

### Test Performance Goals:

| Metric | GCN Baseline | GAT Target | Improvement |
|--------|--------------|------------|-------------|
| **MAE** | 0.2672 | **0.20-0.24** | 10-25% better |
| **RMSE** | 0.3247 | **0.25-0.30** | 15-25% better |
| **R² Score** | 0.1552 | **0.35-0.50** | 2-3x better |
| **Pearson** | 0.4580 | **0.60-0.75** | 30-65% better |

**Success threshold**:
- MAE < 0.24 (better than GCN)
- R² > 0.30 (meaningful improvement)
- Train/val gap < 2x (no overfitting)

---

## Why GAT?

### Graph Attention Networks (GAT) Benefits:

1. **Attention Mechanism**:
   - Learns which neighbors are important
   - Surface atoms attend to nearby surface atoms
   - Buried atoms attend to buried core

2. **Better for Heterogeneous Graphs**:
   - Different atom types (C, N, O, S)
   - Different bond types (covalent, H-bonds)
   - Attention can distinguish importance

3. **Interpretability**:
   - Can visualize attention weights
   - Understand model decisions
   - Debug errors better

4. **Proven Track Record**:
   - Better than GCN on many molecular tasks
   - Widely used in protein modeling

### GCN Limitations (Why it Failed):

- Treats all neighbors equally (weighted only by degree)
- Can't distinguish important vs unimportant edges
- May over-smooth features in deep networks
- No way to focus on relevant structural context

---

## Experiment Protocol

### To Run:

```bash
cd C:\Users\Edoardo\GNN_protein_exposure\GNN_atom_exposure
venv\Scripts\activate
python main.py --epochs 50
```

### What to Monitor:

1. **Early stopping trigger**:
   - Should stop around epoch 15-25
   - If stops earlier (<10) → increase patience
   - If runs full 50 → not enough regularization

2. **Training curves**:
   - Val loss should decrease (not increase!)
   - Train/val gap should be <2x
   - No wild oscillations in val loss

3. **Epoch progress**:
   - Look for "✓ New best model saved" messages
   - Count epochs without improvement

4. **Final metrics**:
   - MAE, RMSE, R², Pearson on test set

### After Training:

```bash
# Evaluate best model
python main.py --eval-only --checkpoint experiments/checkpoints/best_model.pt
```

---

## Comparison Plan

After experiment completes, we'll create a comparison document:

**experiments/baselines/comparison.md**:
- Side-by-side training curves
- Test metrics comparison table
- Analysis of improvements
- Recommendations for Phase 4

---

## Risk Assessment

### Low Risk Changes:
✅ GAT architecture - well-tested, should work
✅ Dropout 0.6 - standard for molecular graphs
✅ Early stopping - prevents overtraining

### Medium Risk:
⚠️ Weight decay 5e-4 - quite strong, may underfit
⚠️ Hidden 64 - might be too small for 100 features
⚠️ 2 layers - may lack expressiveness

### Mitigation:
- If underfitting (train loss too high):
  - Reduce dropout to 0.5
  - Increase to 3 layers
  - Try hidden 96

- If still overfitting:
  - Increase dropout to 0.7
  - Reduce to 1 layer (!!)
  - Try hidden 32

---

## Success Criteria

### Minimum (Must Achieve):
- [ ] No overfitting (train/val gap < 2x)
- [ ] Better than GCN (MAE < 0.2672)
- [ ] Meaningful R² (> 0.25)

### Target (Should Achieve):
- [ ] MAE < 0.24
- [ ] R² > 0.35
- [ ] Pearson > 0.60
- [ ] Early stopping triggers (proves effectiveness)

### Stretch (Would Love):
- [ ] MAE < 0.22
- [ ] R² > 0.45
- [ ] Pearson > 0.70
- [ ] Smooth training curves (no oscillations)

---

## If This Fails

If GAT with regularization still overfits or performs poorly:

**Next steps**:
1. Check if train/val/test splits are stratified (proteins may be too similar)
2. Try even simpler model (1-layer GCN/GAT)
3. Revisit feature engineering (100 features may have redundancy)
4. Consider data augmentation
5. Try different GNN architectures (GraphSAGE, TransformerConv)

**Nuclear option**:
- Linear baseline (no GNN, just MLP on node features)
- If linear does better → graph structure not helping
- May need to rethink approach

---

## Files

**Modified**:
- `configs/config.yaml` - GAT config
- `src/training/train.py` - Added gradient clip & early stopping
- `main.py` - Pass new parameters to Trainer

**Created**:
- `experiments/baselines/gcn_overfit/` - Archived GCN results
- `experiments/baselines/gcn_overfit/results.md` - GCN analysis

**Will Create**:
- `experiments/baselines/gat_regularized/` - GAT results (after training)
- `experiments/progress/phase3_results.md` - Final comparison

---

**Status**: Configuration complete, ready to train
**Command**: `python main.py --epochs 50`
**Expected Duration**: 1-2 hours (may stop early with early stopping)

