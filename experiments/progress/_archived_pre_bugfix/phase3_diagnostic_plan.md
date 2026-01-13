# Phase 3 Diagnostic: Why Are All Models Failing?

**Date**: 2026-01-09
**Status**: 🔴 Critical Analysis Needed

---

## Problem Statement

After 4 experiments, ALL models have R² < 0.16 (explaining <16% of variance):

| Experiment | Model | MAE | R² | Notes |
|------------|-------|-----|-----|-------|
| 3.1 (Baseline) | GCN | 0.263 | 0.155 | Best so far ⭐ |
| 3.2 | GAT | 0.279 | 0.134 | Undertrained (10 epochs) |
| 3.3 | GCN | 0.317 | -0.104 | Over-regularized |
| 3.3b | GCN | 0.275 | 0.126 | Moderate reg |
| 3.4 | GAT | 0.283 | 0.099 | Moderate reg |

**Key insight**: Regularization changes don't help. All approaches fail similarly.

---

## Possible Root Causes

### 1. Feature Quality Issues ⚠️
**Hypothesis**: 100 features don't contain sufficient information for burial prediction

**Evidence**:
- All models plateau at R² ~0.10-0.15
- Doesn't matter if we use GCN, GAT, 2 layers, 3 layers
- Suggests information ceiling

**Test**: Train simple MLP (no graph) on node features only
- If MLP ≈ GNN → graph structure isn't helping
- If MLP << GNN → at least graph helps, need better features

### 2. Label Noise ⚠️
**Hypothesis**: `depth_indexes` ground truth has errors or inconsistencies

**Evidence**:
- High error variance (std error ~0.32-0.34)
- Poor R² across all models
- May have measurement noise

**Test**:
- Visualize predictions vs ground truth for specific proteins
- Check if errors are random or systematic
- Compute inter-protein label consistency

### 3. Dataset Split Issues ⚠️
**Hypothesis**: Train/val/test splits not stratified, proteins too similar

**Evidence**:
- Random 80/10/10 split by protein ID
- No stratification by protein family or size
- Models may memorize protein-specific patterns

**Test**:
- Check protein similarity between splits
- Try stratified splitting by protein family
- Compute protein embedding distances

### 4. Architectural Limitations ⚠️
**Hypothesis**: 2-3 layer GNNs can't capture global protein structure

**Evidence**:
- Atom burial depends on overall protein shape
- 3-layer GNN sees 3-hop neighborhood (~10-15 Å)
- Protein structures span 30-50 Å
- Local info may be insufficient

**Test**:
- Try deeper networks (4-5 layers)
- Try virtual nodes (global pooling + broadcast)
- Try hierarchical/coarse-graining approaches

---

## Diagnostic Experiments

### Experiment D1: MLP Baseline (No Graph) 🔴 HIGH PRIORITY

**Goal**: Determine if graph structure helps at all

**Setup**:
```yaml
model: simple_mlp  # Just use node features, ignore edges
layers: [100, 128, 64, 1]
dropout: 0.3
```

**Expected outcomes**:
- If MLP MAE ≈ 0.26-0.28 → **Graph doesn't help, feature problem**
- If MLP MAE > 0.35 → Graph helps, need better GNN architecture

### Experiment D2: Feature-Target Correlation Analysis 🔴 HIGH PRIORITY

**Goal**: Identify which features actually predict depth

**Setup**:
- Compute Pearson correlation: each feature vs depth_index
- Rank features by correlation strength
- Identify if ANY features have strong correlation (>0.5)

**Expected outcomes**:
- If max correlation < 0.3 → Features don't capture burial well
- If some features > 0.5 → Focus on those, drop weak ones

### Experiment D3: Error Analysis by Atom Type

**Goal**: Understand where model fails

**Setup**:
- Compute MAE separately for:
  - Buried atoms (depth < 0.3)
  - Surface atoms (depth > 0.7)
  - Different atom types (C, N, O, S)
  - Different residue types

**Expected outcomes**:
- Identify systematic biases
- May reveal labeling issues
- Guide feature engineering

### Experiment D4: Visualization

**Goal**: See what model learned

**Setup**:
- Pick 3-5 representative proteins
- Plot predicted vs actual depth per atom
- Color by atom type
- Check if patterns make sense

---

## Decision Tree

```
Run D1 (MLP Baseline)
├─ MLP ≈ GNN (within 5%)
│  └─> Problem: Features or labels, not architecture
│     ├─ Run D2 (Feature correlation)
│     │  ├─ Weak correlations (<0.3) → Need better features
│     │  └─ Some strong (>0.5) → Feature selection helps
│     │
│     └─ Run D3 (Error analysis)
│        └─ Systematic bias → Labeling issues
│
└─ MLP >> GNN (MLP worse by >10%)
   └─> GNN is helping, need better architecture
       ├─ Try deeper networks (4-5 layers)
       ├─ Try virtual nodes
       └─ Try different GNN types (TransformerConv)
```

---

## Immediate Action: Run MLP Baseline

This is the most important diagnostic:

**Why**: Tells us if we're solving the wrong problem (architecture) when the real issue is features/labels

**How**:
1. Check if `simple` model type exists in codebase
2. If not, create simple MLP model
3. Train with same train/val/test splits
4. Compare to GCN baseline

**Time**: 30 minutes

**Decision**: Based on MLP results, we know if architecture optimization is even worth pursuing

---

## If MLP Shows Graph Doesn't Help...

### Then the problem is:

1. **Features are insufficient**
   - Missing critical spatial/chemical information
   - Need to add:
     - Global protein properties (size, shape)
     - Secondary structure annotations
     - Solvent accessibility (if available)
     - Better geometric features

2. **Labels are noisy**
   - depth_indexes may have measurement error
   - Need to understand how labels were created
   - May need to smooth/denoise labels

3. **Task is inherently difficult**
   - Burial prediction from local features alone may be impossible
   - May need multi-scale approach
   - May need to accept limited performance

---

## Success Criteria

### For MLP Diagnostic:
- Run completes without errors
- Can compare MAE, R², Pearson to GNN
- Clear answer on whether graph helps

### For Continuing Phase 3:
If graph helps (MLP >> GNN):
- Continue architecture optimization
- Try deeper/wider networks
- Experiment with different GNN types

If graph doesn't help (MLP ≈ GNN):
- **STOP architecture optimization**
- Focus on feature engineering
- Consider revisiting Phase 2 (feature selection)

---

## Status: Awaiting Decision

**Options**:
1. Run MLP baseline (Exp D1) - 30 mins
2. Do error analysis first (Exp D3) - 1 hour
3. Accept baseline GCN (3.1) as best and document limitations
4. Try one more architecture (3 layers, 96 hidden, dropout 0.3)

**My recommendation**: Run MLP baseline first. It's quick and tells us if we're on the right track.

