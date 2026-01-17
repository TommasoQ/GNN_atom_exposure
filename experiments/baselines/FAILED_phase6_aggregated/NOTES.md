# Phase 6: Aggregated Features - FAILED

**Date**: 2026-01-16
**Result**: R² = 0.4002 (worse than baseline)

## What Was Tried

Implemented aggregated feature transforms based on a reference implementation:
- Aggregated 13 hydrophobicity scales → 2 features (mean, std)
- Aggregated secondary structure propensities → 3 features
- Custom geometric features (centroid-relative, spherical coords)
- Embedding layers for element/residue instead of one-hot

**Total: 31 numerical + 8 element embed + 11 residue embed = 50 features**

## Why It Failed

1. **Lost important predictive features**: The aggregation dropped features like `transmembranetendency`, `averageburied`, individual beta-strand features that were predictive in Phase 5.

2. **Different geometric approach**: Centroid-based features (distance from protein center) don't capture local neighborhood information as well as neighbor-based features.

3. **Over-generalization**: Averaging 13 hydrophobicity scales into mean/std loses nuance - some scales correlate differently with the target.

## Lesson Learned

Feature engineering choices matter more than feature count. The reference's 50 features weren't magic - they were tuned for their specific setup. Our 88 features with Phase 5's selection work better for this task.

## Files Created

- `src/data/aggregated_transforms.py` - Aggregated feature engineering (kept for reference)
- Model/training code updated to support embeddings (kept for future use)
