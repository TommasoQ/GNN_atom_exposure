# GNN Protein Atom Exposure Prediction - Project Roadmap

**Project Start**: 2026-01-08
**Last Updated**: 2026-01-13
**Current Status**: Phase 5 IN PROGRESS - Error Analysis Next

---

## Quick Status

| Phase | Status | Best Result |
|-------|--------|-------------|
| Phase 1: EDA | ✅ COMPLETE | Identified b_factor as top predictor |
| Phase 2: Dataset Fixes | ✅ COMPLETE | 4,556 proteins, 88 features |
| Phase 3: Architecture Comparison | ✅ COMPLETE | GIN best (R² 0.504) |
| Phase 4: Hyperparameter Tuning | ✅ COMPLETE | R² 0.5456 (OneCycle scheduler) |
| Phase 5: Extended Training | ✅ **R² 0.5680** | **New best! +4.1% over Phase 4** |
| Phase 5: Error Analysis | 🔄 IN PROGRESS | Next step |

---

## Phase 5 Progress (2026-01-13)

### Extended Training Results - NEW BEST! 🎉

After fixing OneCycle scheduler warmup scaling, achieved **R² = 0.5680**

| Metric | Phase 4 Best | Phase 5 (150ep) | Improvement |
|--------|--------------|-----------------|-------------|
| **R² Score** | 0.5456 | **0.5680** | +4.1% |
| **MAE** | 0.1887 | **0.1831** | -3.0% |
| **RMSE** | 0.2381 | **0.2322** | -2.5% |
| **Pearson** | 0.7398 | **0.7540** | +1.9% |
| **Median AE** | 0.1567 | **0.1506** | -3.9% |
| **Mean Error** | - | -0.0066 | Near zero bias |

### Critical Bug Fix: OneCycle Warmup Scaling

**Problem**: When increasing epochs from 50→150, `pct_start=0.3` meant:
- 50 epochs → 15 epoch warmup (good)
- 150 epochs → 45 epoch warmup (too long! model barely learning)

**Solution**: Changed config from `pct_start` to `warmup_epochs`:
```yaml
# OLD (breaks when epochs change)
pct_start: 0.3

# NEW (adaptive to any epoch count)
warmup_epochs: 15  # pct_start calculated as warmup_epochs/num_epochs
```

Code updated in `main.py` to calculate `pct_start = warmup_epochs / num_epochs`.

### Dataset Cleanup

Removed 38 HETATM-dominant proteins from training:
- **Before**: 4,594 proteins
- **After**: 4,556 proteins (-0.8%)
- **Reason**: These have <10 protein atoms, mostly ligands

Filter added to `src/data/dataset_fixed.py` as `HETATM_DOMINANT_PROTEINS` set.

### Infrastructure Improvements

1. **Training History Export**: Now saves `experiments/logs/training_history.csv` with:
   - epoch, train_loss, val_loss, val_mae, val_rmse, learning_rate
   
2. **LR Tracking**: Records learning rate per epoch for debugging scheduler issues

### Target Distribution Analysis

Analyzed exposure target distribution (SADIC values):
```
Range: [0.000, 1.731]  (not full 0-2 range)
Mean: 0.486, Std: 0.349
<0.2 (buried): 27.0%
>1.5 (exposed): 0.2%
Exactly 0.0: only 25 atoms (0.01%) - not a clipping artifact
```

**Conclusion**: Sigmoid activation NOT recommended - linear output is appropriate.

### Next Step: Error Analysis

Goal: Understand where the model fails to guide further improvements
- Error by exposure range (buried vs exposed)
- Error by atom type
- Error by protein size

---

## Phase 4 Summary (COMPLETE - 2026-01-13)

### Grid Search Results

Conducted systematic grid search comparing **Cosine Annealing** vs **One Cycle** schedulers (18 total experiments).

**Winner: One Cycle Scheduler** with `max_lr=0.003, pct_start=0.3`

| Metric | Value |
|--------|-------|
| **R² Score** | 0.5456 |
| **MAE** | 0.1887 |
| **RMSE** | 0.2381 |
| **Pearson** | 0.7398 |
| **Median AE** | 0.1567 |

### Scheduler Comparison

| Scheduler | R² | MAE | Pearson | Avg Score |
|-----------|-----|-----|---------|-----------|
| **One Cycle** | 0.5389 ± 0.0040 | 0.1906 ± 0.0011 | 0.7353 ± 0.0029 | **0.7904** |
| Cosine Annealing | 0.5234 ± 0.0092 | 0.1938 ± 0.0020 | 0.7248 ± 0.0064 | 0.3440 |

**Key Finding**: One Cycle wins by margin of 0.4464 and shows lower variance (more robust).

### Training Optimizations Applied
- AMP (Automatic Mixed Precision) for ~30-50% speedup
- cuDNN benchmark mode
- Batch size: 32 (up from 8)
- DataLoader: pin_memory, persistent_workers

### Grid Search Results Location
- Summary: `experiments/grid_search/results/summary.csv`
- Analysis plots: `experiments/grid_search/analysis/`
- Individual runs: `experiments/grid_search/results/<exp_name>/`

---

## Dataset Integrity Investigation (2026-01-13)

### 38 Proteins with Severe Graphein vs Raw Mismatch

Investigated colleague's report of 38 proteins with Graphein/PDB mismatch.

**Finding**: These are **HETATM-dominant structures** where PDB files have mostly ligands/non-standard residues. Graphein correctly extracts only protein atoms (`ATOM` records), resulting in very small graphs.

**Root Causes**:
- 33/38: HETATM-dominant (e.g., `3mbs`: 888 HETATM, only 4 ATOM)
- 6/38: Heavy alternate conformations (>30% atoms with altloc)
- 3/38: Other structural quirks

**Decision**: Keep in dataset but document. These are valid proteins but contribute minimal training signal.

**Full list**: See `experiments/progress/analyze_38_proteins.py`

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

## Phase 5 Plan (IN PROGRESS)

### Objectives:
1. **Extended Training**: 50 epochs with optimal hyperparameters
2. **Error Analysis**: Identify which atoms are hardest to predict
3. **Improvement Strategies**: Target R² ≥ 0.55-0.56

### Target Metrics:
- R² > 0.55 (stretch: 0.56)
- MAE < 0.185
- Pearson > 0.74

---

## Historical Results Comparison

| Phase | Experiment | Model | R² | MAE | Notes |
|-------|------------|-------|-----|-----|-------|
| 3 | 3.5 | GCN | 0.484 | 0.204 | First valid baseline |
| 3 | 3.6 | GAT | 0.477 | 0.204 | No improvement over GCN |
| 3 | 3.7 | GIN | 0.499 | 0.199 | Best architecture |
| 3 | 3.8 | GIN-88 | 0.504 | 0.200 | With feature reduction |
| 4 | 4.1 | GIN-4L | 0.507 | ~0.20 | Deeper model |
| 4 | 4.2 | GIN-LR | 0.516 | 0.197 | Lower LR + higher dropout |
| 4 | Grid Search | GINE | 0.5456 | 0.1887 | OneCycle scheduler |
| **5** | **150ep** | **GINE** | **0.5680** | **0.1831** | **Current best** |

**Progress**: R² improved from 0.484 → 0.5680 (+17.4% relative improvement)

---

## Key Files

### Core Code
- `src/data/dataset_fixed.py` - Dataset with all fixes
- `src/data/feature_engineering.py` - 88 features
- `src/models/gnn.py` - GCN, GAT, GIN, GINE architectures
- `src/training/train.py` - Trainer with AMP support
- `configs/config.yaml` - Current experiment config

### Grid Search
- `experiments/grid_search/run_grid_search.py` - Grid search orchestrator
- `experiments/grid_search/grid_config.yaml` - Search parameters
- `experiments/grid_search/analysis/compare_results.py` - Analysis & visualization

### Documentation
- `experiments/progress/PROJECT_ROADMAP.md` - This file
- `experiments/progress/phase3_summary.md` - Phase 3 details
- `experiments/progress/bug_discovery.md` - Critical bug fixes
- `experiments/progress/STRATEGIC_ANALYSIS.md` - Deep performance analysis

---

## Commands Reference

```bash
# Train with current config
python main.py

# Train with specific hyperparameters
python main.py --epochs 50 --lr 0.003

# Run grid search
python experiments/grid_search/run_grid_search.py

# Analyze grid search results
python experiments/grid_search/analysis/compare_results.py

# Evaluate checkpoint
python main.py --eval-only --checkpoint <path>
```

---

**Next Action**: Run 50-epoch training with optimal hyperparameters, then error analysis
