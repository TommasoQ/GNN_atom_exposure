# GNN Protein Atom Exposure Prediction - Project Roadmap

**Project Start**: 2026-01-08
**Last Updated**: 2026-01-11
**Current Status**: Phase 4 In Progress

---

## Quick Status

| Phase | Status | Best Result |
|-------|--------|-------------|
| Phase 1: EDA | COMPLETE | Identified b_factor as top predictor |
| Phase 2: Dataset Fixes | COMPLETE | 4,594 proteins, 88 features |
| Phase 3: Architecture Comparison | COMPLETE | GIN best (R² 0.504) |
| Phase 4: Hyperparameter Tuning | IN PROGRESS | R² 0.516 (Exp 4.2) |
| Phase 5: Feature/Edge Analysis | NEXT | - |

---

## Phase 3 Summary (COMPLETE)

**Experiments Run**:
| Exp | Model | Features | R² | Pearson | MAE |
|-----|-------|----------|-----|---------|-----|
| 3.5 | GCN | 100 | 0.484 | 0.696 | 0.204 |
| 3.6 | GAT | 100 | 0.477 | 0.691 | 0.204 |
| 3.7 | GIN | 100 | 0.499 | 0.708 | 0.199 |
| 3.8 | GIN | 88 | 0.504 | 0.714 | 0.200 |

**Key Findings**:
- GIN > GCN > GAT for this task
- Feature reduction (100→88) slightly improved performance
- Geometric features dominate (80% of importance from 7 features)

---

## Phase 4 Progress (IN PROGRESS)

**Experiments Run**:
| Exp | Config | R² | Pearson | MAE | Notes |
|-----|--------|-----|---------|-----|-------|
| 4.1 | GIN 4L, 128H | 0.507 | 0.715 | ~0.20 | Depth didn't help much (+0.5%) |
| 4.2 | GIN 3L, LR=0.0005, D=0.35 | **0.516** | **0.720** | **0.197** | Best so far! |

**Key Insights**:
1. Deeper networks provide diminishing returns (over-smoothing)
2. Lower LR (0.0005) + higher dropout (0.35) works better than depth
3. RMSE not improving as much as MAE → persistent outliers exist
4. **GIN ignores edge features** (distances) - potential improvement area

**Next Experiments**:
- Exp 4.3: GINE (GIN with Edge features) - use distance information
- Exp 4.4: GAT with edge features (already implemented)
- Exp 4.5: Ablation without geometric features

---

## Phase 5 Plan (Feature/Edge Analysis)

**Objectives**:
1. Compare edge-aware models (GINE, GAT) vs edge-ignoring (GIN, GCN)
2. Ablation: train without geometric features to force learning from chemistry
3. Error analysis: identify which atoms have persistent high errors
4. Investigate if chemical features can capture cases geometry misses

**Expected Duration**: 2-3 days

---

## Key Files

**Core Code**:
- `src/data/dataset_fixed.py` - Dataset with all fixes
- `src/data/feature_engineering.py` - 88 features (24 numerical + 57 categorical + 7 geometric)
- `src/models/gnn.py` - GCN, GAT, GIN architectures
- `configs/config.yaml` - Current experiment config

**Documentation**:
- `experiments/progress/phase3_summary.md` - Detailed Phase 3 results
- `experiments/progress/bug_discovery.md` - Critical bug fixes
- `experiments/progress/FEATURE_REDUCTION_SUMMARY.md` - Feature selection rationale

**Analysis**:
- `experiments/analysis/` - Correlation and importance analysis scripts

---

## Performance Ceiling Analysis

Current best: R² = 0.516, MAE = 0.197

**Why we might be hitting a ceiling**:
1. Geometric features capture "easy" cases (radial position → exposure)
2. Hard cases: surface atoms in local pockets, bent inward
3. GIN/GCN ignore edge features (inter-atomic distances)
4. Missing information: secondary structure, sequence context

**Potential Improvements**:
- Use edge features (GINE, GAT) for local density
- Add secondary structure labels if available
- Multi-resolution graph (residue + atom level)

---

## Commands Reference

```bash
# Train with current config
python main.py

# Evaluate only (uses best checkpoint)
python main.py --eval-only --visualize

# Override config
python main.py --epochs 50 --lr 0.0005

# Check feature dimensions
python src/data/feature_engineering.py
```

---

**Next Action**: Run GINE and GAT comparison to test edge feature utilization
