# Feature Update: 93 → 92 Features

## Change Summary

**Date**: 2026-01-21
**Reason**: `contact_count_10A` feature was too dominant/heavy compared to other geometric features
**Impact**: Node features reduced from 93 to 92

## What Was Removed

### Removed Feature
- **`geom_contact_count_10A`** (Feature #8 of geometric features)
  - **Description**: Number of atoms within 10Å radius (excluding self)
  - **Computation**: Used `scipy.spatial.distance.cdist` for all pairwise distances
  - **Issue**: Too dominant/heavy weight compared to other geometric features in feature importance analysis

## Updated Feature Breakdown

### Before (93 features)
```
24 Numerical features     (biochemical properties)
31 Atom types            (one-hot)
5  Elements              (one-hot)
21 Residues              (one-hot)
8  Geometric features    (3D structure) ← CHANGED
4  Backbone angles       (φ/ψ sin/cos)
────────────────────────
93 TOTAL
```

### After (92 features)
```
24 Numerical features     (biochemical properties)
31 Atom types            (one-hot)
5  Elements              (one-hot)
21 Residues              (one-hot)
7  Geometric features    (3D structure) ← REDUCED
4  Backbone angles       (φ/ψ sin/cos)
────────────────────────
92 TOTAL
```

## Geometric Features Details

### Removed: 8 features → 7 features

**Kept (7 features)**:
1. `geom_mean_dist` - Mean distance to graph neighbors
2. `geom_min_dist` - Minimum distance to a neighbor
3. `geom_max_dist` - Maximum distance to a neighbor
4. `geom_std_dist` - Standard deviation of neighbor distances
5. `geom_3rd_nearest_dist` - Distance to 3rd nearest neighbor
6. `geom_dist_to_center` - Distance to protein center of mass
7. `geom_radial_position` - Normalized radial position (0=center, 1=surface)

**Removed**:
- ~~`geom_contact_count_10A`~~ - Count of atoms within 10Å

## Files Modified

### 1. Core Feature Engineering
- **`src/data/feature_engineering.py`**
  - `compute_geometric_features()`: Removed contact_count calculation
  - Changed return shape from `(n_atoms, 8)` to `(n_atoms, 7)`
  - Removed `'geom_contact_count_10A'` from feature names
  - Updated `get_feature_dimensions()`: `8` → `7` for geometric features

### 2. Configuration Files (in_channels: 93 → 92)
- `configs/phase14_large_model.yaml`
- `configs/phase15_optimized.yaml`
- `configs/phase16_enhanced.yaml`
- `configs/phase17_balanced.yaml`

### 3. Analysis Scripts (in_channels: 93 → 92)
- `experiments/analysis/attention_analysis.py`
- `experiments/analysis/feature_importance.py`
- `experiments/analysis/edge_importance.py`
- `experiments/analysis/PHASE14_ANALYSIS_README.md`

## Model Parameter Changes

With 92 input features instead of 93:

| Configuration | Params Before | Params After | Δ |
|---------------|---------------|--------------|---|
| **Phase 14** (176 hidden, 5 layers) | 512,689 | 512,513 | -176 |
| **Phase 15/17** (136 hidden, 4 layers) | 253,913 | 253,777 | -136 |
| **Phase 16** (152 hidden, 4 layers) | 314,185 | 314,033 | -152 |

**Calculation**: Removing 1 input feature removes `1 × hidden_channels` parameters in the first layer.

## Dataset Processing Impact

### No Impact On
- ✓ Dataset splits (train/val/test)
- ✓ Graph structure (nodes, edges)
- ✓ Edge features (still 12)
- ✓ Target values (atom exposure)
- ✓ Normalization statistics (only numerical features normalized)

### Requires Reprocessing
- ⚠️ **Cached processed files** must be regenerated
- **Reason**: Feature vectors change from 93 to 92 dimensions
- **Action**: Delete `processed/` directory or let dataset auto-reprocess

## How to Use

### For New Training
No action needed - configs already updated:
```bash
.venv/Scripts/python.exe main.py --config configs/phase17_balanced.yaml
```

### For Loading Old Checkpoints
**WARNING**: Old checkpoints (trained with 93 features) are **incompatible** with new code!

If you need to load Phase 14/15/16 checkpoints:
1. Revert `in_channels` to 93 in model loading code
2. Or retrain from scratch with 92 features

### Force Reprocessing Dataset
```bash
# Delete processed cache
rm -rf dataset/processed/

# Or rename it
mv dataset/processed/ dataset/processed_93_features_backup/

# Next run will auto-reprocess with 92 features
.venv/Scripts/python.exe main.py --config configs/phase17_balanced.yaml
```

## Expected Impact on Performance

### Hypothesis
- **Slight performance improvement** possible if contact_count was causing overfitting or redundancy
- **Minimal performance change** more likely - other geometric features (especially radial_position) already capture burial information
- **Training slightly faster** - 1 less feature to compute per atom

### To Verify
Compare Phase 17 (92 features) with Phase 15 (93 features):
- R² change < 0.001: Feature was redundant (good removal)
- R² improves: Feature was harmful (excellent removal)
- R² drops > 0.005: Feature was important (consider reverting)

## Validation

### Quick Check
```python
from src.data.feature_engineering import get_feature_dimensions

dims = get_feature_dimensions(
    use_reduced_features=False,
    include_atom_type=True,
    include_geometric=True,
    include_backbone_angles=True
)

print(f"Total features: {dims['total']}")  # Should print 92
print(f"Geometric: {dims['geometric']}")    # Should print 7
```

### Model Initialization Check
```python
from src.models.gnn import AtomExposureGNN

model = AtomExposureGNN(
    in_channels=92,  # NEW
    hidden_channels=136,
    num_layers=4,
    conv_type='gatv2',
    edge_dim=12,
    dropout=0.26
)

print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
# Phase 15/17: Should print 253,777 (was 253,913)
```

## Rollback Instructions

If you need to revert this change:

1. **Revert feature_engineering.py**:
   ```python
   # In compute_geometric_features():
   geom_features = np.zeros((n_atoms, 8), dtype=np.float32)

   # Before the for loop, add back:
   pairwise_distances = cdist(coords, coords, metric='euclidean')
   contact_counts = (pairwise_distances < 10.0).sum(axis=1) - 1
   geom_features[:, 7] = contact_counts.astype(np.float32)

   # In full_geometric_names, add back:
   'geom_contact_count_10A'

   # In get_feature_dimensions():
   geometric_count = ... if use_reduced_features else 8
   ```

2. **Revert all config files**: `in_channels: 92` → `93`

3. **Revert analysis scripts**: `in_channels=92` → `93`

4. **Delete processed cache** to force reprocessing with 8 geometric features

## Notes

- This change is **backward incompatible** with existing checkpoints
- Phase 17 will be the first phase trained with 92 features
- If Phase 17 performs worse, we have clear evidence contact_count was important
- If Phase 17 performs equal/better, we've simplified the feature set without loss

---

**Summary**: Removed `contact_count_10A` geometric feature, reducing from 93 to 92 node features. All configs and scripts updated. Old checkpoints incompatible with new code.
