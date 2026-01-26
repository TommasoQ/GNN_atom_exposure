# Phase 15 noCC Analysis Scripts

## Overview

Gli script di analisi sono stati aggiornati per analizzare il modello **Phase 15 noCC**:
- **92 features** (contact_count_10A rimosso)
- **136 hidden channels**
- **4 layers**
- **dropout 0.26**
- **GATv2 architecture**

## Model Configuration

```python
model = AtomExposureGNN(
    in_channels=92,       # 92 features (no contact_count)
    hidden_channels=136,  # Phase 15 size
    num_layers=4,         # Phase 15 depth
    conv_type='gatv2',
    edge_dim=12,
    dropout=0.26          # Phase 15 dropout
)
```

**Total Parameters**: 253,777

## Scripts Available

### 1. Attention Analysis
**File**: `attention_analysis.py`

Analizza i pattern di attenzione appresi dai 4 layer GATv2.

**Usage**:
```bash
cd GNN_atom_exposure
.venv/Scripts/python.exe experiments/analysis/attention_analysis.py
```

**Optional arguments**:
```bash
# Specify different checkpoint
--checkpoint experiments/checkpoints/phase15_noCC/best_model.pt

# Analyze more proteins (default: 50)
--num-proteins 100

# Custom output directory
--output-dir experiments/analysis/phase15_noCC_results
```

**Output**:
```
experiments/analysis/attention_analysis/
├── attention_statistics.csv          # Stats per layer/head
├── attention_distribution.png        # Distribution of attention weights
├── attention_vs_distance.png         # Correlation with distance
├── attention_by_edge_type.png        # By edge type
├── attention_by_exposure.csv         # By exposure level
├── attention_by_exposure.png         # Visualization
└── attention_heatmap.png            # Heatmap (if small protein found)
```

**Note**: 4 layers analyzed (0, 1, 2, 3)

---

### 2. Edge Feature Importance
**File**: `edge_importance.py`

Calcola l'importanza delle 12 edge features tramite zero-out method.

**Usage**:
```bash
cd GNN_atom_exposure
.venv/Scripts/python.exe experiments/analysis/edge_importance.py
```

**Optional arguments**:
```bash
# Specify different checkpoint
--checkpoint experiments/checkpoints/phase15_noCC/best_model.pt

# Custom output directory
--output-dir experiments/analysis/phase15_noCC_results
```

**Output**:
```
experiments/analysis/
├── edge_importance.csv              # Importance per feature
└── edge_importance.png              # Bar plot
```

**Edge features analyzed** (12 total):
- Bond types (7): covalent, peptide, hydrophobic, aromatic, hbond, ionic, ring
- Distances (4): distance, bond_length, normalized_dist, relative_dist
- Radius flag (1): in_radius

---

### 3. Node Feature Importance
**File**: `feature_importance.py`

Calcola l'importanza delle **92 node features** tramite permutation importance.

**Usage**:
```bash
cd GNN_atom_exposure
.venv/Scripts/python.exe experiments/analysis/feature_importance.py
```

**Optional arguments**:
```bash
# Specify different checkpoint
--checkpoint experiments/checkpoints/phase15_noCC/best_model.pt

# More repeats for stability (default: 3)
--n-repeats 5

# Custom output directory
--output-dir experiments/analysis/phase15_noCC_results
```

**Output**:
```
experiments/analysis/
├── feature_importance.csv           # All 92 features ranked
└── feature_importance.png           # Top 30 features plot
```

**Feature groups analyzed** (92 total):
1. Numerical (24): b_factor, hbond_donors, hydrophobicity scales, etc.
2. Atom types (31): N, CA, C, O, CB, CG, etc.
3. Elements (5): C, N, O, S, OTHER
4. Residues (21): ALA, ARG, ASN, ..., VAL
5. **Geometric (7)**: mean_dist, min_dist, max_dist, std_dist, 3rd_nearest, dist_to_center, radial_position
   - **NOTE**: contact_count_10A NOT INCLUDED (removed in Phase 15 noCC)
6. Backbone angles (4): sin_phi, cos_phi, sin_psi, cos_psi

---

## Key Differences from Phase 14 Analysis

| Aspect | Phase 14 Scripts | **Phase 15 noCC Scripts** |
|--------|------------------|---------------------------|
| **in_channels** | 93 | **92** |
| **hidden_channels** | 176 | **136** |
| **num_layers** | 5 | **4** |
| **dropout** | 0.28 | **0.26** |
| **Checkpoint default** | phase14_large_model | **phase15_noCC** |
| **Layers analyzed** | 0-4 (5 total) | **0-3 (4 total)** |
| **Geometric features** | 8 (with contact_count) | **7 (no contact_count)** |

---

## Expected Results for Phase 15 noCC

### ⚠️ Important Context
Phase 15 noCC has **significantly degraded performance** compared to Phase 15 original:

| Metric | Phase 15 (93 feat) | Phase 15 noCC (92 feat) | Impact |
|--------|-------------------|------------------------|--------|
| R² | 0.8892 | 0.5689 | -36% ❌ |
| MAE | 0.087 | 0.179 | +106% ❌ |

**Reason**: `contact_count_10A` is a **critical feature** for predicting exposure.

### What to Look For in Analysis

#### 1. Attention Analysis
**Expected**:
- Attention patterns may be different than Phase 15 original
- Model may rely more heavily on distance features
- Attention might be less focused (model struggling without contact_count)

**Key questions**:
- Do attention weights compensate for missing contact_count?
- Is attention more diffuse or more concentrated?

#### 2. Edge Feature Importance
**Expected**:
- Similar importance ranking as Phase 15 original
- Edge features alone cannot replace node-level contact_count

**Key questions**:
- Are distance features more important now?
- Does in_radius feature become more critical?

#### 3. Node Feature Importance
**Critical Analysis**:
- **Top features should be examined**: Which features try to compensate for missing contact_count?
- **Radial_position** might become more important
- **Numerical features** (b_factor, buried_residues) might gain importance

**Expected top features** (hypothesis):
1. `b_factor` (always strong)
2. `geom_radial_position` (next best burial indicator)
3. `expasy:buriedresidues`, `expasy:accessibleresidues`
4. Other geometric features compensating

**Key question**: Can we identify which remaining features the model relies on most?

---

## Running Full Analysis

To run all three analyses sequentially:

```bash
cd GNN_atom_exposure

# 1. Attention analysis
.venv/Scripts/python.exe experiments/analysis/attention_analysis.py \
    --checkpoint experiments/checkpoints/phase15_noCC/best_model.pt \
    --num-proteins 50

# 2. Edge importance
.venv/Scripts/python.exe experiments/analysis/edge_importance.py \
    --checkpoint experiments/checkpoints/phase15_noCC/best_model.pt

# 3. Feature importance (WARNING: takes 30-45 min on CPU)
.venv/Scripts/python.exe experiments/analysis/feature_importance.py \
    --checkpoint experiments/checkpoints/phase15_noCC/best_model.pt \
    --n-repeats 3
```

**Estimated time**:
- Attention: ~5-10 min
- Edge importance: ~5-8 min
- Feature importance: ~30-45 min
- **Total**: ~40-60 min on CPU, ~15-20 min on GPU

---

## Troubleshooting

### Error: "size mismatch for convs.0.lin_l.weight"
**Cause**: Trying to load Phase 14 checkpoint (93 features) with Phase 15 noCC config (92 features)

**Solution**: Ensure checkpoint matches:
```python
# Phase 15 noCC expects 92 features
model = AtomExposureGNN(in_channels=92, hidden_channels=136, num_layers=4, ...)
```

### Error: "Expected 4 layers, got 5"
**Cause**: Script configured for wrong model

**Solution**: Verify script has correct configuration (should be fixed now)

### Feature importance takes too long
**Solution**: Reduce repeats or test subset
```bash
--n-repeats 1  # Faster but less stable
```

---

## Comparison with Phase 15 Original

To compare Phase 15 noCC with Phase 15 original, you would need to:

1. **Keep Phase 15 original checkpoint** separate
2. **Re-run analysis scripts** with Phase 15 original config:
   ```python
   # Phase 15 original
   model = AtomExposureGNN(in_channels=93, hidden_channels=136, num_layers=4, dropout=0.26)
   ```
3. **Compare results** to understand impact of removing contact_count

**Key comparisons**:
- Which features become more/less important?
- How do attention patterns change?
- Can other features compensate for contact_count?

---

## Conclusion

These scripts are configured for **Phase 15 noCC** analysis. Given the dramatic performance drop (R² 0.89 → 0.57), the analysis will reveal:

1. **What the model relies on** without contact_count
2. **Whether other features compensate** (spoiler: they don't, based on R²)
3. **How the model's behavior changes** when a critical feature is removed

This is a valuable **ablation study** demonstrating the importance of contact_count for exposure prediction.
