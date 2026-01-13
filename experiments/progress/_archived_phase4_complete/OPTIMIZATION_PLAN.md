# Optimization Plan: Path to Maximum R²

**Date**: 2026-01-10
**Goal**: Achieve highest possible R² through systematic optimization
**Strategy**: Small incremental changes, prioritize highest impact

---

## Phase 3 Completion: Architecture Comparison

### Completed Experiments (3 layers, 96 hidden, dropout 0.3)

| Exp | Model | Config | MAE | R² | Pearson | Status |
|-----|-------|--------|-----|-----|---------|--------|
| 3.5 | GCN | 3L, 96H, d0.3 | 0.2040 | 0.4835 | 0.6963 | ✅ Complete |
| 3.6 | GAT | 3L, 96H, d0.3, 4h | 0.2035 | 0.4765 | 0.6907 | ✅ Complete |
| 3.7 | GIN | 3L, 96H, d0.3 | - | - | - | 🔄 Running |

**Key Finding**: GCN = GAT (no advantage from attention/edge features)

**Next**: Test if GIN's higher expressiveness helps

---

## Phase 4: Hyperparameter Optimization

**Strategy**: Once best architecture identified, optimize hyperparameters

### Hyperparameter Grid (8-10 experiments)

**Architecture**: [Winner of Exp 3.5-3.7]

#### Primary Variables:
1. **Depth**: [3, 4, 5] layers
2. **Learning Rate**: [0.0005, 0.001, 0.002]
3. **Dropout**: [0.2, 0.3, 0.4]
4. **Hidden Dims**: [96, 128, 192]

#### Experiment Queue:

**Baseline** (Exp 3.x):
- Current best from Phase 3

**Depth Sweep** (Exp 4.1-4.2):
- 4L, 96H, lr=0.001, d=0.3
- 5L, 96H, lr=0.001, d=0.3

**Learning Rate Sweep** (Exp 4.3-4.4):
- 3L, 96H, lr=0.0005, d=0.3 (lower LR)
- 3L, 96H, lr=0.002, d=0.3 (higher LR)

**Dropout Sweep** (Exp 4.5-4.6):
- 3L, 96H, lr=0.001, d=0.2 (less regularization)
- 3L, 96H, lr=0.001, d=0.4 (more regularization)

**Width Sweep** (Exp 4.7-4.8):
- 3L, 128H, lr=0.001, d=0.3 (wider)
- 3L, 192H, lr=0.001, d=0.3 (much wider)

**Best Combo** (Exp 4.9-4.10):
- Combine best settings from above
- Example: 4L, 128H, lr=0.001, d=0.2

**Total**: ~10 experiments × 50 epochs × 30 min = **5-6 hours**

---

## Phase 4b: Re-test All Architectures with Optimal Hyperparameters

**Once optimal hyperparameters found**:

Re-run GCN, GAT, GIN with new config to ensure winner still wins.

Example (if optimal = 4L, 128H, lr=0.001, d=0.2):
- Exp 4.11: GCN with optimal config
- Exp 4.12: GAT with optimal config
- Exp 4.13: GIN with optimal config

**Ensures**: Fair comparison with optimized settings

---

## Phase 5: Feature Engineering (Bundled with Hyperparameter Analysis)

**Triggered if**: R² plateaus < 0.55 after hyperparameter tuning

### Step 5.1: Feature Importance Analysis

**Method**: Permutation importance on best model

**Questions**:
1. Which of 100 features actually contribute?
2. Are 20 hydrophobicity scales redundant? (likely yes)
3. Which categorical features matter most?
4. Are geometric features helping?

**Action**: Create ranked feature importance list

### Step 5.2: Feature Selection

**Goal**: Reduce from 100 → 50-60 features

**Process**:
1. Remove features with importance < threshold
2. Check correlation matrix (remove multicollinear features)
3. Keep at least:
   - Top 20 numerical features
   - All categorical features (atom/element/residue type)
   - Top 5 geometric features

**Expected**: Faster training, less overfitting, potentially better R²

### Step 5.3: Add Missing Domain Features (If Possible)

**Potentially impactful missing features**:

1. **Secondary structure** (helix, sheet, coil)
   - Source: DSSP analysis of protein structure
   - Impact: HIGH (directly related to exposure)

2. **Residue SASA** (solvent-accessible surface area)
   - Source: FreeSASA or DSSP
   - Impact: VERY HIGH (nearly direct target proxy)

3. **Local density**
   - Number of atoms within 5Å, 10Å, 15Å shells
   - Impact: MEDIUM (already have geometric features)

4. **Residue polarity category**
   - Charged (+/-), polar, hydrophobic, special
   - Impact: MEDIUM (may be redundant with hydrophobicity)

**Challenge**: These require additional computation on protein structures
**Worth it if**: R² improvement > 5%

### Step 5.4: Retrain with Selected Features

After feature engineering:
- Retrain best architecture with reduced feature set
- Compare R² with original 100 features
- If better → adopt new features
- If worse → revert to 100 features

---

## Decision Framework

### When to Move to Next Phase:

**Phase 3 → Phase 4**:
- ✅ All three architectures tested (GCN, GAT, GIN)
- ✅ Best architecture identified
- ⏳ Start hyperparameter optimization

**Phase 4 → Phase 4b**:
- ✅ Optimal hyperparameters found
- ⏳ Re-test all architectures with optimal settings

**Phase 4b → Phase 5**:
- **If R² > 0.55**: Success! Document and analyze
- **If R² < 0.55**: Proceed to feature engineering

**Phase 5 → Complete**:
- **If R² > 0.60**: Excellent! Stop optimization
- **If R² 0.55-0.60**: Good! Diminishing returns likely
- **If R² < 0.55**: Investigate label quality or task definition

### Stopping Criteria:

**Stop optimizing when**:
1. R² > 0.65 (diminishing returns, likely hitting label noise ceiling)
2. Three consecutive experiments show < 1% improvement
3. Time/effort exceeds value of marginal gains

**Current Target**: R² 0.55-0.60 is realistic and valuable

---

## Naming Convention for Experiments

**Format**: `{architecture}_{variant}_{key_params}`

**Examples**:
- `gcn_bugs_fixed` (Exp 3.5) - baseline GCN
- `gat_with_edges` (Exp 3.6) - baseline GAT
- `gin_baseline` (Exp 3.7) - baseline GIN
- `gcn_deep4` (Exp 4.1) - GCN with 4 layers
- `gcn_deep5` (Exp 4.2) - GCN with 5 layers
- `gcn_lr_low` (Exp 4.3) - GCN with lr=0.0005
- `gcn_wide128` (Exp 4.7) - GCN with 128 hidden
- `gcn_optimal` (Exp 4.9) - GCN with best hyperparameters
- `gin_optimal` (Exp 4.13) - GIN with best hyperparameters
- `gcn_features_reduced` (Exp 5.1) - GCN with 50 features

**Archived experiments**: All pre-fix experiments moved to `_archived_invalid_experiments/`

---

## Tracking Progress

### Baseline Performance (Phase 3)

| Architecture | R² | Notes |
|--------------|-----|-------|
| GCN | 0.4835 | Preferred baseline |
| GAT | 0.4765 | No advantage |
| GIN | TBD | Testing now |

### Best Performance So Far

| Phase | Best Model | R² | MAE | Config |
|-------|-----------|-----|-----|--------|
| 3 | GCN | 0.4835 | 0.2040 | 3L, 96H, d0.3 |
| 4 | TBD | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD | TBD |

**Goal**: R² > 0.55

---

## Current Status

**Phase 3**: 🔄 **IN PROGRESS**
- ✅ Exp 3.5: GCN baseline
- ✅ Exp 3.6: GAT baseline
- 🔄 Exp 3.7: GIN baseline (running now)

**Next Action**: Wait for GIN results → identify best architecture → start Phase 4

---

## Notes on Feature Engineering

**Current features (100 total)**:
- 34 numerical (b_factor + Meiler + hydrophobicity + structural)
- 58 categorical (31 atom types + 6 elements + 21 residues)
- 8 geometric (distances, radial position)

**Potential redundancy**:
- **20 hydrophobicity scales** → likely highly correlated
- **Multiple structural propensities** → may overlap
- **Some Meiler descriptors** → PCA-derived, may be redundant

**Analysis needed**:
1. Correlation matrix of numerical features
2. Permutation importance ranking
3. Ablation studies (remove feature groups)

**Bundled approach**: Do feature analysis AFTER finding optimal hyperparameters
- Separates architecture/hyperparameter effects from feature effects
- Ensures fair comparison

---

**Last Updated**: 2026-01-10
**Current Experiment**: 3.7 (GIN baseline)
**Overall Goal**: Maximum R² through systematic optimization
