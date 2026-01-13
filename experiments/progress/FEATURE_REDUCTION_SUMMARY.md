# Feature Reduction Summary

**Date**: 2026-01-10
**Analysis**: Correlation + Permutation Importance on GIN baseline (Exp 3.7)
**Decision**: Conservative reduction from 100 → 88 features

---

## Analysis Performed

### 1. Correlation Analysis
- **Script**: `experiments/analysis/feature_correlation_analysis.py`
- **Method**: Computed correlation matrix for all 100 features + correlation with target
- **Key findings**:
  - Identified high inter-correlation (r > 0.90) among feature groups
  - Found redundant hydrophobicity scales (eisenberg/janin/chothia r=0.93)
  - Found redundant beta sheet propensities (totalbeta/sheetfasman/sheetroux r=0.95-0.97)

### 2. Permutation Importance Analysis
- **Script**: `experiments/analysis/feature_importance.py`
- **Method**: Shuffle each feature individually, measure R² drop on validation set
- **Key findings**:
  - Geometric features dominate (top 8 features, 80% of importance)
  - Many biochemical features have near-zero importance for current GIN model
  - Some features perfectly duplicated (geom_min_dist = geom_nearest_dist, r=1.0)

---

## Features Removed (12 total)

### Group 1: Perfect Duplicates (1 feature)

**1. `geom_nearest_dist`**
- **Correlation with `geom_min_dist`**: 1.0 (identical)
- **GIN importance**: 0.336 (but same as geom_min_dist: 0.102)
- **Justification**: Exact duplicate feature with different name
- **Action**: Keep `geom_min_dist`, drop `geom_nearest_dist`

---

### Group 2: Redundant Hydrophobicity Scales (3 features)

**Cluster: Eisenberg/Janin/Chothia (keep Eisenberg only)**

All three measure buried hydrophobicity with near-identical values:
- eisenberg ↔ janin: r = 0.93
- eisenberg ↔ chothia: r = 0.90
- janin ↔ chothia: r = 0.92

**2. `expasy:hphob_janin`**
- **Target correlation**: -0.353
- **GIN importance**: 0.002
- **Redundant with**: `hphob_eisenberg` (r=0.93)
- **Justification**: Eisenberg scale is more widely used and established

**3. `expasy:hphob_chothia`**
- **Target correlation**: -0.354
- **GIN importance**: 0.002
- **Redundant with**: `hphob_eisenberg` (r=0.90)
- **Justification**: Nearly identical to eisenberg scale

**4. `expasy:hphob_woods`**
- **Target correlation**: +0.376
- **GIN importance**: 0.003
- **Redundant with**: `transmembranetendency` (r=-0.93), `polaritygrantham` (r=+0.90)
- **Justification**: Highly redundant with two stronger features we're keeping

**Kept hydrophobicity scales**:
- `hphob_eisenberg` (most established, r=-0.355 target)
- `hphob_rose` (best target correlation r=-0.403)
- `hphob_guy` (complementary, r=+0.378 target)

---

### Group 3: Redundant Beta Sheet Propensities (3 features)

**Cluster: Total/Fasman/Roux/Antiparallel (keep totalbeta_strand only)**

All measure beta sheet tendency with very high correlation:
- totalbeta ↔ sheetroux: r = 0.97
- totalbeta ↔ sheetfasman: r = 0.95
- totalbeta ↔ antiparallel: r = 0.95

**5. `expasy:beta_sheetfasman`**
- **Target correlation**: -0.356
- **GIN importance**: 0.011
- **Redundant with**: `totalbeta_strand` (r=0.95)
- **Justification**: Fasman is a specific prediction method; totalbeta is more general

**6. `expasy:beta_sheetroux`**
- **Target correlation**: -0.344
- **GIN importance**: 0.002
- **Redundant with**: `totalbeta_strand` (r=0.97)
- **Justification**: Roux is another specific method; totalbeta captures same info

**7. `expasy:antiparallelbeta_strand`**
- **Target correlation**: -0.327
- **GIN importance**: 0.009
- **Redundant with**: `totalbeta_strand` (r=0.95)
- **Justification**: Subset of total beta strand information

**Kept beta features**:
- `totalbeta_strand` (general measure, r=-0.355 target)
- `parallelbeta_strand` (complementary to total, r=-0.318 target)
- `beta_turnfasman`, `beta_turnroux` (turn vs strand distinction)

---

### Group 4: Low-Value Numerical Features (3 features)

**8. `expasy:refractivity`**
- **Target correlation**: -0.127 (weak)
- **GIN importance**: 0.005
- **Redundant with**: `molecularweight` (r=0.92), `averageburied` (r=0.87)
- **Justification**: Derived from molecular weight; redundant

**9. `expasy:molecularweight`**
- **Target correlation**: -0.012 (essentially zero)
- **GIN importance**: 0.019
- **Redundant with**: Residue type (categorical features implicitly encode MW)
- **Justification**: Residue identity provides same information; near-zero target correlation

**10. `expasy:isoelectric_points`**
- **Target correlation**: +0.060 (very weak)
- **GIN importance**: 0.0004 (negligible)
- **Redundant with**: `meiler:dim_5` (r=0.91)
- **Justification**: Highly correlated with Meiler descriptor; weak predictive value

---

### Group 5: Low-Value Meiler Descriptor (1 feature)

**11. `meiler:dim_6`**
- **Target correlation**: -0.024 (very weak)
- **GIN importance**: 0.003
- **Redundant with**: `coilroux` (r=0.83)
- **Justification**: PCA component with weak predictive value; coilroux more interpretable

**Kept Meiler descriptors**: dim_1, dim_4, dim_5, dim_7 (all r > 0.04 or importance > 0.003)

---

### Group 6: Rare/Empty Categorical Features (1 feature)

**12. `element_P`**
- **Target correlation**: NaN (no data - phosphorus extremely rare)
- **GIN importance**: 0.0
- **Justification**: Virtually no phosphorus atoms in dataset; provides no information

**Note**: `element_OTHER` and `residue_OTHER` were initially considered for removal but retained for technical reasons (required as fallback categories in one-hot encoding). These features have no data and will be all zeros, which is harmless.

---

## Features Retained: 88 (down from 100)

### Numerical Features: 24 (down from 34)

**Core biochemical (3)**:
- b_factor
- hbond_donors
- hbond_acceptors

**Meiler descriptors (4)**: dim_1, dim_4, dim_5, dim_7

**Hydrophobicity (3)**: eisenberg, rose, guy

**Structural propensities (9)**:
- buriedresidues, accessibleresidues, averageburied
- averageflexibility, transmembranetendency
- totalbeta_strand, parallelbeta_strand
- beta_turnfasman, beta_turnroux, coilroux

**Polarity/properties (5)**:
- polarityzimmerman, polaritygrantham
- bulkiness, ratioside

### Categorical Features: 57 (down from 58)

**Atom types (31)**: All standard protein atom types (N, CA, C, O, CB, CG, etc.)

**Elements (5)**: C, N, O, S, OTHER (dropped P only; OTHER retained for encoding but has no data)

**Residues (21)**: 20 standard amino acids + OTHER (OTHER retained for encoding but has no data)

### Geometric Features: 7 (down from 8)

- geom_mean_dist
- geom_min_dist (duplicate geom_nearest_dist removed)
- geom_max_dist
- geom_std_dist
- geom_3rd_nearest_dist
- geom_dist_to_center
- geom_radial_position

---

## Expected Impact

### Performance:
- **Expected R² loss**: < 0.5%
- **Rationale**: All removed features have:
  - GIN importance < 0.02, OR
  - Perfect duplicates (r=1.0), OR
  - Very low target correlation (|r| < 0.15) + redundancy

### Training Speed:
- **Speedup**: ~12% faster (88/100 = 0.88)
- **Per epoch**: ~30 min → ~26-27 min
- **50 epochs**: 25 hours → 22 hours (save ~3 hours)

### Benefits:
- Reduced multicollinearity
- Better regularization
- Cleaner model interpretation
- Maintains flexibility for different architectures

---

## Validation Plan

**Experiment 3.8**: Retrain GIN with 88 features
- Same config as Exp 3.7 (3L, 96H, d0.3)
- Compare R², MAE, Pearson with baseline
- **Success criteria**: R² drop < 1% (i.e., R² > 0.494)

If successful → Use 88 features for all Phase 4 experiments
If unsuccessful (R² drop > 1%) → Revert to 100 features

---

## Notes

### Why Conservative?

This reduction is **intentionally conservative** because:
1. **Model-specific importance**: GIN may not use features that GAT/deeper networks would
2. **Correlation is model-agnostic**: High-correlation features retained even if GIN doesn't use them
3. **Future flexibility**: Preserves features for Phase 4 architecture/hyperparameter experiments

### Future Opportunities

If Phase 4 testing shows certain architectures don't improve with the retained features, could consider:
- **More aggressive reduction**: Drop features with importance < 0.01 (would remove ~40 more)
- **Remove empty categoricals**: Drop element_OTHER and residue_OTHER (would save 2 more features → 86 total)
- **Architecture-specific feature sets**: Different features for GCN vs GAT vs GIN
- **Learned feature selection**: Use attention mechanisms to identify important features

---

**Files Modified**:
- `src/data/feature_engineering.py` - Updated feature lists (removed 12 features)
- `configs/config.yaml` - Updated in_channels: 100 → 88
- `experiments/progress/FEATURE_REDUCTION_SUMMARY.md` - This file

**Technical Note**: `element_OTHER` and `residue_OTHER` were retained (not removed as initially planned) because they're required as fallback categories in the one-hot encoding functions. These features have no data and contribute nothing to the model, but removing them would require refactoring the encoding logic.

**Analysis Data**:
- `experiments/analysis/feature_correlation_matrix.csv`
- `experiments/analysis/feature_target_correlation.csv`
- `experiments/analysis/high_correlation_pairs.csv`
- `experiments/analysis/feature_importance_scores.csv`
