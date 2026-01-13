# Feature Analysis & Selection Plan

**Date**: 2026-01-10
**Goal**: Analyze current 100 features and identify most important subset
**Strategy**: Reduce features → Faster training + Better generalization

---

## Current Feature Breakdown

### Total: 100 Features

**Numerical (34)**:
- 1 b_factor (very important!)
- 2 H-bonding (hbond_donors, hbond_acceptors)
- 7 Meiler descriptors (PCA-derived physicochemical properties)
- 24 ExPASy features:
  - **20 hydrophobicity scales** ← Likely VERY redundant!
  - 4 structural propensities

**Categorical (58)**:
- 31 atom types (N, CA, C, O, CB, etc.)
- 6 elements (C, N, O, S, P, OTHER)
- 21 residues (20 amino acids + OTHER)

**Geometric (8)**:
- Mean/min/max/std distance to neighbors
- Nearest neighbor distances
- Distance to protein center
- Radial position

---

## Hypothesis: Redundancy in Features

### Suspected Redundant Features

**1. Hydrophobicity Scales (20 total)** ← **PRIMARY TARGET**

Current scales in feature_engineering.py:
```python
'expasy:hphob_eisenberg'
'expasy:hphob_janin'
'expasy:hphob_rose'
'expasy:hphob_guy'
'expasy:hphob_woods'
'expasy:hphob_chothia'
# ... plus 14 more!
```

**Issue**: These are highly correlated!
- All measure "hydrophobicity" but with different methods
- Correlation likely > 0.8 between many pairs
- Contribute redundant information

**Action**: Keep top 3-5, drop the rest

**Expected impact**:
- Reduce 20 → 5 scales = **-15 features**
- Minimal performance loss (< 1%)
- Faster training (~20% faster)
- Better generalization (less overfitting)

**2. Structural Propensities** (lower priority)

Current features:
```python
'expasy:buriedresidues'
'expasy:accessibleresidues'
'expasy:averageburied'
'expasy:averageflexibility'
'expasy:transmembranetendency'
# ... plus more beta/alpha propensities
```

Some may be redundant with burial/exposure (our target!).

**Action**: Check correlation with target and each other

---

## Feature Analysis Workflow

### **Step 1: Feature Correlation Analysis** (30 minutes)

**Script to create**: `experiments/analysis/feature_correlation_analysis.py`

**What it does**:
1. Load best GIN model (Exp 3.7)
2. Load validation dataset
3. Compute correlation matrix for all 100 features
4. Identify highly correlated feature pairs (r > 0.8)
5. Compute correlation of each feature with target (exposure depth)
6. Generate visualizations:
   - Correlation heatmap (100×100)
   - Feature-target correlation bar plot
   - Identify redundant clusters

**Output**:
- `feature_correlation_matrix.csv`
- `feature_target_correlation.csv`
- `feature_correlation_heatmap.png`
- List of redundant feature groups

---

### **Step 2: Feature Importance Analysis** (45 minutes)

**Method**: Permutation Importance

**Script to create**: `experiments/analysis/feature_importance.py`

**What it does**:
1. Load best GIN model
2. Evaluate baseline performance on validation set
3. For each feature:
   - Randomly shuffle that feature
   - Re-evaluate performance
   - Measure drop in R²
4. Rank features by importance (drop in R²)
5. Generate importance scores

**Output**:
- `feature_importance_scores.csv`
- `feature_importance_plot.png`
- Top 50 features ranked

**Expected findings**:
- **b_factor** will be most important (we know this from Phase 1)
- **Hydrophobicity scales**: Only 3-5 will matter, rest ~0 importance
- **Categorical features**: Atom type and residue type likely important
- **Geometric features**: Distance features likely important

---

### **Step 3: Feature Selection** (30 minutes)

**Strategy**: Keep features that are:
1. High importance (top 60-70)
2. Low correlation with others (r < 0.8)
3. Domain knowledge essential (b_factor, atom type, etc.)

**Selection Criteria**:

**KEEP** (estimated ~50-60 features):
- All 34 categorical atom types, elements, residues (essential!)
- All 8 geometric features (cheap to compute, informative)
- b_factor (critical!)
- H-bonding features (2)
- All 7 Meiler descriptors (PCA-derived, already compact)
- Top 5 hydrophobicity scales (highest importance)
- Top 5-10 structural propensities (if important)

**DROP** (estimated ~40-50 features):
- 15 redundant hydrophobicity scales (keep top 5)
- Low-importance structural features
- Highly correlated features

**Target**: **50-60 features** (down from 100)

---

### **Step 4: Retrain GIN with Reduced Features** (30 minutes)

**Experiment 3.8**: GIN with selected features

**Config**:
```yaml
Model: GIN
Layers: 3
Hidden: 96
Dropout: 0.3
in_channels: 55  # Reduced from 100
```

**Expected results**:
- **Best case**: R² 0.50-0.51 (no loss, maybe slight gain!)
- **Likely case**: R² 0.49-0.50 (< 1% loss)
- **Worst case**: R² 0.48-0.49 (if we dropped something important)

**Benefits**:
- Faster training (20-30% faster)
- Less overfitting risk
- Cleaner model
- Easier to interpret

---

### **Step 5: Compare Full vs Reduced** (5 minutes)

| Metric | GIN Full (100 feat) | GIN Reduced (~55 feat) | Change |
|--------|---------------------|------------------------|--------|
| R² | 0.4985 | TBD | TBD |
| MAE | 0.1987 | TBD | TBD |
| Training time | ~30 min | ~20-25 min | -20-30% |

**Decision**:
- If R² drop < 1%: **Use reduced features** for Phase 4
- If R² drop > 1%: **Keep full features** but note which are most important

---

## Detailed Feature Reduction Plan

### **Hydrophobicity Scales: 20 → 5**

**Keep these 5** (based on literature + likely importance):
1. `hphob_eisenberg` - Most widely used, burial-focused
2. `hphob_janin` - Specifically for buried residues
3. `hphob_rose` - Accessibility scale
4. `hphob_guy` - Different hydrophobicity measure (positive correlation in Phase 1)
5. `hphob_woods` - Complementary scale

**Drop these 15**:
- All other hphob_* scales (likely redundant)

**Rationale**:
- Phase 1 correlation analysis showed top 5 scales have highest correlation with exposure
- Others are likely < 0.01 importance difference

---

### **Structural Propensities: Review Individually**

**Definitely keep** (directly relevant):
- `buriedresidues` - Directly related to burial!
- `accessibleresidues` - Directly related to exposure!
- `averageburied` - Related to burial depth
- `averageflexibility` - Mobile regions tend to be surface

**Evaluate** (may be redundant):
- Beta sheet propensities (multiple)
- Alpha helix propensities (multiple)
- Coil propensities

**Strategy**: Keep if importance > threshold, drop if redundant

---

### **Categorical Features: Keep ALL**

**Why**:
- Atom type (31 features) - Essential chemical identity
- Element (6 features) - Fundamental property
- Residue (21 features) - Critical for protein context

**Total**: 58 features - **non-negotiable**

Even though this is many features, they're:
- One-hot encoded (sparse)
- Fundamental to the task
- Not redundant with each other

---

### **Geometric Features: Keep ALL**

**Why**:
- Only 8 features
- Cheap to compute
- Capture spatial context
- Likely important for GIN (neighbor counting)

**Total**: 8 features - **keep all**

---

## Expected Final Feature Set

**After reduction**:

| Category | Before | After | Reduction |
|----------|--------|-------|-----------|
| Numerical | 34 | ~20 | -14 |
| Categorical | 58 | 58 | 0 |
| Geometric | 8 | 8 | 0 |
| **TOTAL** | **100** | **~86** | **-14** |

Wait, this is only 14 features dropped if we keep top 5 hydrophobicity scales...

**More aggressive option**:

Keep only **top 3 hydrophobicity scales** + **top 5 structural propensities**:
- Drop 17 hydrophobicity scales (keep 3)
- Drop 5-10 structural features (keep top 5)

**Result**: ~70-75 features (25-30% reduction)

---

## Implementation Scripts

### **Script 1: Correlation Analysis**

```python
# experiments/analysis/feature_correlation_analysis.py

import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.data.dataset_fixed import ProteinAtomDataset

# Load validation dataset
dataset = ProteinAtomDataset(root='dataset', split='val')

# Extract all features and targets
all_features = []
all_targets = []

for data in dataset:
    all_features.append(data.x.cpu().numpy())
    all_targets.append(data.y.cpu().numpy())

features = np.vstack(all_features)
targets = np.concatenate(all_targets)

# Feature names (get from feature_engineering.py)
from src.data.feature_engineering import (
    SELECTED_NUMERICAL_FEATURES,
    STANDARD_ATOM_TYPES,
    STANDARD_ELEMENTS,
    STANDARD_RESIDUES
)

feature_names = (
    SELECTED_NUMERICAL_FEATURES +
    [f'atom_{a}' for a in STANDARD_ATOM_TYPES] +
    [f'element_{e}' for e in STANDARD_ELEMENTS] +
    [f'residue_{r}' for r in STANDARD_RESIDUES] +
    ['geom_mean_dist', 'geom_min_dist', 'geom_max_dist', 'geom_std_dist',
     'geom_nearest_dist', 'geom_3rd_nearest_dist',
     'geom_dist_to_center', 'geom_radial_position']
)

# Compute correlation matrix
df = pd.DataFrame(features, columns=feature_names)
corr_matrix = df.corr()

# Save correlation matrix
corr_matrix.to_csv('experiments/analysis/feature_correlation_matrix.csv')

# Compute correlation with target
target_corr = df.corrwith(pd.Series(targets))
target_corr.to_csv('experiments/analysis/feature_target_correlation.csv')

# Visualize correlation heatmap (numerical features only)
plt.figure(figsize=(20, 16))
sns.heatmap(
    corr_matrix.iloc[:34, :34],  # Just numerical features
    cmap='coolwarm', center=0,
    xticklabels=feature_names[:34],
    yticklabels=feature_names[:34]
)
plt.title('Numerical Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('experiments/analysis/numerical_feature_correlation.png', dpi=150)

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append({
                'feature1': feature_names[i],
                'feature2': feature_names[j],
                'correlation': corr_matrix.iloc[i, j]
            })

pd.DataFrame(high_corr_pairs).to_csv(
    'experiments/analysis/high_correlation_pairs.csv',
    index=False
)

print(f"Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.8)")
```

---

### **Script 2: Permutation Importance**

```python
# experiments/analysis/feature_importance.py

import torch
import numpy as np
from src.data.dataset_fixed import ProteinAtomDataset
from src.models.gnn import AtomExposureGNN
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score

# Load best GIN model
model = AtomExposureGNN(
    in_channels=100,
    hidden_channels=96,
    num_layers=3,
    dropout=0.3,
    conv_type='gin'
)
checkpoint = torch.load('experiments/checkpoints/gin_baseline/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load validation dataset
val_dataset = ProteinAtomDataset(root='dataset', split='val')
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Baseline performance
def evaluate(model, loader):
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            predictions.append(out.cpu().numpy())
            targets.append(batch.y.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(targets)

preds_base, targets = evaluate(model, val_loader)
r2_base = r2_score(targets, preds_base)
print(f"Baseline R²: {r2_base:.4f}")

# Permutation importance
feature_names = [...]  # Same as correlation script

importances = []

for feat_idx in range(100):
    # Create permuted dataset
    val_dataset_permuted = ProteinAtomDataset(root='dataset', split='val')

    # Permute feature
    for data in val_dataset_permuted:
        perm = torch.randperm(data.x.size(0))
        data.x[:, feat_idx] = data.x[perm, feat_idx]

    loader_perm = DataLoader(val_dataset_permuted, batch_size=8, shuffle=False)
    preds_perm, _ = evaluate(model, loader_perm)
    r2_perm = r2_score(targets, preds_perm)

    importance = r2_base - r2_perm
    importances.append({
        'feature': feature_names[feat_idx],
        'importance': importance,
        'r2_drop': importance
    })

    print(f"Feature {feat_idx+1}/100: {feature_names[feat_idx]}: {importance:.6f}")

# Save results
import pandas as pd
df_importance = pd.DataFrame(importances).sort_values('importance', ascending=False)
df_importance.to_csv('experiments/analysis/feature_importance_scores.csv', index=False)

print("\nTop 20 features:")
print(df_importance.head(20))
```

---

## Timeline

**Total Time**: ~2.5 hours

| Step | Time | Task |
|------|------|------|
| 1 | 30 min | Correlation analysis |
| 2 | 45 min | Permutation importance |
| 3 | 30 min | Feature selection (manual review) |
| 4 | 30 min | Retrain GIN with reduced features |
| 5 | 5 min | Compare results |

**Output**: Optimized feature set for Phase 4

---

## Decision Framework

### If Reduced Features Perform Well (R² drop < 1%)

**Action**: Use reduced features for ALL Phase 4 experiments
**Benefits**:
- 20-30% faster training
- All hyperparameter experiments benefit
- Total Phase 4 time: 9 hours → **6-7 hours** (saved 2-3 hours!)

### If Reduced Features Hurt Performance (R² drop > 2%)

**Action**: Keep full 100 features
**Lesson**: All features contribute, model needs full information

---

## Next Steps After Feature Analysis

Once we have optimal feature set:

**Phase 4: Hyperparameter Tuning**

Test all 3 architectures (GCN, GAT, GIN) with:
1. **4 layers** (first priority)
2. If promising, test **5 layers** on best architecture
3. Optimize other hyperparameters on winner

**Timeline** (with reduced features):
- 3 architectures × 4 layers = 3 experiments × 25 min = 1.25 hours
- Winner × hyperparameter sweep = 8 experiments × 25 min = 3.5 hours
- **Total**: ~5 hours (vs 9 hours with full features!)

---

**Status**: Ready to start feature analysis
**First Action**: Create correlation analysis script
**Expected Outcome**: ~50-75 features (down from 100), comparable performance
