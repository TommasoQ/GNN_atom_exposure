# Project History

Brief summary of key learnings from the development of this GNN model.

## Evolution

The project went through 17+ experimental phases, evolving from a complex GATv2 model with 93 features to a minimal GCN with just 5 features.

| Phase | Model | Features | R² | Key Finding |
|-------|-------|----------|-----|-------------|
| 3 | GCN | 80 | 0.48 | Baseline established |
| 5 | GINE | 88 | 0.57 | Edge features added |
| 12 | GATv2 | 93 | 0.88 | Contact count breakthrough |
| 15 | GATv2 | 93 | 0.94 | Global pooling added |
| **Final** | **MinimalGCN** | **5** | **0.87-0.89** | **Simplified architecture** |

## Key Discoveries

### 1. Contact Count Dominates (Phase 12)

The `contact_count_10A` feature (number of atoms within 10Å radius) alone explains most of the variance in atom exposure. This makes intuitive sense: buried atoms have more neighbors.

**Feature importance analysis** showed contact_count at 40-60% importance, with geometric features (distance to center, radial position) following at 10-20% each.

### 2. GCN ≈ GAT Performance

Switching from complex GATv2 attention (4 heads, edge features) to simple GCN had **negligible impact** on accuracy. The attention mechanism wasn't learning anything useful beyond what simple message passing captures.

This finding reduced model complexity from ~500K to ~16K parameters.

### 3. Edge Features Have 0 Importance

Ablation studies showed edge features (bond types, distances) contribute **exactly 0%** to model performance. Removing the 12 edge features entirely had no impact on R².

### 4. Global Pooling Helps

Adding gated global pooling (protein-level context injected at each layer) improved R² by 2-5%. This allows atoms to "see" the overall protein size and shape.

### 5. 5 Features Suffice

Reduced from 93 features to just 5 geometric features with only ~2% accuracy drop:
- `contact_count_10A` - Neighbor count within 10Å
- `dist_to_center` - Distance to protein center of mass
- `radial_position` - Normalized position (0=center, 1=surface)
- `3rd_nearest_dist` - Local packing density proxy
- `std_dist` - Variation in neighbor distances

## Failed Experiments

- **Phase 6**: Over-aggregated features destroyed spatial information
- **Phase 7**: Reduced regularization caused overfitting (regularization was already optimal)
- **Backbone angles**: Phi/psi dihedral angles had <1% importance

## Critical Bug Fixes

- **1000x gradient error**: Loss function was computing mean twice, causing vanishing gradients
- **Edge index mismatch**: Radius graph edges weren't aligned with edge features

## Final Architecture

The MinimalGCN represents the distilled knowledge from all experiments:

```
Input (5 features) → Linear(64) → ReLU
    → GCNConv(64) → BatchNorm → ReLU → Dropout → [Global Pool Gate]
    → GCNConv(64) → BatchNorm → ReLU → Dropout → [Global Pool Gate] + Residual
    → Linear(32) → ReLU → Dropout → Linear(1)
```

~16,000 parameters, R² 0.87-0.89
