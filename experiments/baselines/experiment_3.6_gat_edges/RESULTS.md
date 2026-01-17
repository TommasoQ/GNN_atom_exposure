# Experiment 3.6: GAT with Edge Features

**Date**: 2026-01-10
**Status**: ✅ COMPLETE
**Model**: GAT (Graph Attention Network)

---

## Configuration

### Model Architecture
```yaml
Type: GAT
Layers: 3
Hidden dimensions: 96
Dropout: 0.3
Attention heads: 4
Input features: 100 (34 numerical + 58 categorical + 8 geometric)
Edge features: ENABLED (distance information)
Output: Single value per atom (exposure depth)
```

### Training Setup
```yaml
Epochs: 50 (with early stopping)
Batch size: 8
Learning rate: 0.001
Weight decay: 1e-4
Optimizer: Adam
Scheduler: ReduceLROnPlateau (patience=10)
Early stopping: patience=15
Gradient clipping: 1.0
```

### Critical Fixes Applied
This experiment includes all fixes from Experiment 3.5, plus GAT-specific improvements:

1. **Bug #1 (CRITICAL)**: Fixed loss weighting - `batch.num_nodes` (not `num_graphs`)
2. **Bug #2 (CRITICAL)**: Verified input dimensions - 100 features
3. **Bug #3 (CRITICAL)**: **Edge features ENABLED** - GAT now uses distance information
4. **Bug #4 (IMPORTANT)**: Explicit error for missing atoms (no silent padding)
5. **Bug #5 (PERFORMANCE)**: Fixed triple processing issue

See [bug_discovery.md](../../progress/bug_discovery.md) for full details.

---

## Results

### Test Set Performance

| Metric | Value | vs GCN (Exp 3.5) | Difference |
|--------|-------|------------------|------------|
| **MAE** | **0.2035** | 0.2040 | **-0.0005** (0.2% better) |
| **RMSE** | **0.2556** | 0.2539 | **+0.0017** (0.7% worse) |
| **R²** | **0.4765** | 0.4835 | **-0.0070** (1.4% worse) |
| **Pearson** | **0.6907** | 0.6963 | **-0.0056** (0.8% worse) |
| Median AE | 0.1674 | 0.1742 | -0.0068 (3.9% better) |
| Mean Error | 0.0080 | -0.0030 | +0.0110 (slight positive bias) |
| Std Error | 0.2555 | 0.2539 | +0.0016 |

### Performance Assessment

**Unexpected Result**: GAT performs **virtually identically** to GCN, with only minor differences:
- MAE: 0.2035 vs 0.2040 (negligible 0.2% improvement)
- R²: 0.4765 vs 0.4835 (negligible 1.4% decrease)
- Pearson: 0.6907 vs 0.6963 (negligible 0.8% decrease)

**Interpretation**: The differences are within statistical noise - GAT and GCN are essentially equivalent on this task.

---

## Analysis

### Why GAT Doesn't Outperform GCN

Several possible explanations:

1. **Distance information may not be critical**
   - Protein structure is highly regular (covalent bonds)
   - Topology (connectivity) is more important than exact distances
   - GCN captures sufficient spatial information through graph structure

2. **Task may not require attention mechanism**
   - Atom exposure is primarily a local structural property
   - All neighbors contribute relatively equally
   - Attention mechanism adds complexity without benefit

3. **Similar model capacity**
   - Both models have ~96 hidden dimensions per layer
   - GAT: 4 heads × 24 dims = 96 total
   - GCN: 96 dims directly
   - Effective capacity is comparable

4. **Limited depth (3 layers)**
   - Receptive field only extends 3 hops
   - Not enough depth for attention patterns to emerge
   - Deeper models might show GAT advantage

5. **Training dynamics**
   - GAT may need different learning rate or regularization
   - Attention weights may need more epochs to stabilize
   - Early stopping might favor GCN's simpler optimization

### Performance Context

Despite GAT not improving over GCN, both models achieve solid performance:
- **Target range**: 0.0 - ~1.9 (exposure depth values)
- **MAE 0.20**: On average, predictions are off by 0.20 units (~10% of range)
- **R² 0.48**: Model explains 48% of variance in atom exposure
- **Pearson 0.69**: Strong positive correlation with ground truth

This is a reasonable baseline for a complex 3D structural prediction task.

---

## Comparison: GAT vs GCN

### Architectural Differences

**GCN** (Experiment 3.5):
- Simple neighborhood aggregation: mean of neighbors
- No learnable attention weights
- Fewer parameters per layer
- Faster training (simpler computations)
- More stable training (no attention instability)

**GAT** (Experiment 3.6):
- Attention-weighted aggregation: learned importance per edge
- 4 attention heads for multi-scale features
- More parameters (attention weights + edge features)
- Slower training (attention computations)
- Potentially less stable (attention optimization)

### Performance Comparison

| Model | MAE | R² | Pearson | Parameters | Speed |
|-------|-----|-----|---------|------------|-------|
| GCN   | 0.204 | 0.484 | 0.696 | ~68K | Faster |
| GAT   | 0.204 | 0.477 | 0.691 | ~75K | Slower |

**Verdict**: GCN is the **preferred model** for this task:
- Equivalent performance to GAT
- Simpler architecture (easier to interpret)
- Faster training and inference
- More stable optimization

---

## Training Dynamics

### Observations
- Training loss decreased smoothly and consistently
- Validation loss tracked training loss appropriately
- No obvious overfitting (dropout 0.3 was appropriate)
- Learning rate scheduler activated appropriately
- Convergence behavior very similar to GCN

### Model Behavior
- Predictions have slight positive bias (mean error +0.008 vs GCN -0.003)
- Median error (0.167) is lower than mean (0.204), similar to GCN
- Standard deviation of errors matches RMSE (0.256)
- Error distribution appears normal

---

## Conclusions

### Key Findings

1. **GAT does not outperform GCN on this task**
   - Edge features (distances) don't provide expected advantage
   - Attention mechanism doesn't improve predictions
   - Simpler GCN architecture is sufficient

2. **Graph topology > Edge attributes**
   - Connectivity pattern (which atoms are bonded) matters most
   - Exact distances less critical than structural context
   - Protein graphs may be too regular for attention to help

3. **GCN is the recommended baseline**
   - Equivalent performance with simpler architecture
   - Faster training and inference
   - Easier to interpret and debug

4. **Performance ceiling reached?**
   - Both models plateau around R² 0.48
   - May need different approaches to improve further:
     - Deeper networks (5+ layers)
     - Different architectures (GIN, GraphSAGE, Transformer)
     - Better features or feature engineering
     - Multi-scale or hierarchical models

### Implications for Future Work

**Architecture exploration**:
- Deeper GCN (4-5 layers) may capture longer-range interactions
- Different aggregation schemes (max, attention, learned)
- Residual connections and skip connections

**Feature engineering**:
- More sophisticated geometric features (angles, torsions)
- Secondary structure information
- Solvent accessibility predictions as auxiliary targets

**Training improvements**:
- Hyperparameter tuning (learning rate, dropout, batch size)
- Different optimizers or schedules
- Data augmentation (coordinate perturbations)

**Advanced techniques**:
- Pre-training on related tasks
- Multi-task learning (predict multiple properties)
- Ensemble methods (combine GCN + GAT)

---

## Next Steps

### Immediate
1. ✅ Document GAT results (this file)
2. ✅ Compare with GCN baseline
3. ⏳ Update phase3_summary.md
4. ⏳ Update baselines/README.md

### Phase 3 Completion
- ✅ Experiment 3.5: GCN baseline (R² 0.48)
- ✅ Experiment 3.6: GAT with edges (R² 0.48)
- ✅ Fair comparison completed
- **Recommendation**: GCN is the preferred baseline

### Future Experiments (Phase 4+)

**Option 1: Deeper GCN** (Recommended)
- Test 4-5 layers instead of 3
- Increase receptive field
- May capture longer-range structural interactions

**Option 2: Wider networks**
- Increase hidden dimensions (128-256)
- More model capacity
- Risk: overfitting with current data

**Option 3: Different architectures**
- GIN (Graph Isomorphism Network) - stronger expressiveness
- GraphSAGE - different aggregation
- Transformer - global attention

**Option 4: Hyperparameter tuning**
- Systematic grid search or Bayesian optimization
- Learning rate, dropout, weight decay
- May yield 5-10% improvement

**Option 5: Feature engineering**
- Review feature importance
- Add domain-specific features
- Remove redundant features

---

## Files

- Model checkpoint: `experiments/checkpoints/gat_with_edges/best_model.pt`
- Training logs: `experiments/logs/gat_with_edges/`
- Config: [config.yaml](../../../configs/config.yaml)
- Bug report: [bug_discovery.md](../../progress/bug_discovery.md)
- Phase 3 summary: [phase3_summary.md](../../progress/phase3_summary.md)

---

## Reproducibility

To reproduce this experiment:
```bash
# Ensure all bug fixes are applied
# Set config: 3 layers, 96 hidden, 0.3 dropout, GAT with edge features

python main.py --config configs/config.yaml
```

**Environment**:
- Python 3.10.6
- PyTorch 2.5.1+cu121
- PyTorch Geometric 2.7.0
- Dataset: 4,594 proteins (after filtering 406 without labels)
- Features: 100 (34 numerical + 58 categorical + 8 geometric)
- Edge features: Distance (1D)

---

**Experiment conducted by**: Claude Code (automated GNN training)
**Dataset**: Protein atom exposure depth prediction
**Task**: Node-level regression on protein graphs
