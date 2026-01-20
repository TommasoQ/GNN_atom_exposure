# GNN Protein Atom Exposure Prediction - Project Roadmap

**Project Start**: 2026-01-08
**Last Updated**: 2026-01-19
**Current Status**: Phase 12 COMPLETE - **R² = 0.8817** (NEW BEST!)

---

## Quick Status

| Phase | Status | Best Result |
|-------|--------|-------------|
| Phase 1: EDA | ✅ COMPLETE | Identified b_factor as top predictor |
| Phase 2: Dataset Fixes | ✅ COMPLETE | 4,556 proteins, 88 features |
| Phase 3: Architecture Comparison | ✅ COMPLETE | GIN best (R² 0.504) |
| Phase 4: Hyperparameter Tuning | ✅ COMPLETE | R² 0.5456 (OneCycle scheduler) |
| Phase 5: Extended Training | ✅ COMPLETE | R² 0.5684 |
| Phase 6: Aggregated Features | ❌ FAILED | R² 0.4002 (lost important features) |
| Phase 7: Low Regularization | ❌ FAILED | R² 0.5604-0.5607 (didn't beat Phase 5) |
| Phase 8: Backbone Angles | ✅ COMPLETE | R² 0.607 |
| Phase 9: Extended Training (200ep) | ✅ COMPLETE | R² 0.6028 |
| Phase 10: Quadratic Loss | ✅ COMPLETE | R² 0.6022 |
| Phase 11: GATv2 Transition | ✅ COMPLETE | R² 0.598 |
| **Phase 12: GATv2 + Radius Graph** | ✅ **BEST** | **R² 0.8817 (+46% vs Phase 9)** |

---

## Phase 6: Aggregated Feature Transforms - FAILED (2026-01-15)

### Motivation

A reference implementation achieved R² = 0.5723 using only **50 features** compared to our 88 features. Analysis of their `transforms.py` revealed a fundamentally different feature engineering approach.

### Key Differences

| Aspect | Our Approach (Phase 5) | Reference Approach |
|--------|------------------------|-------------------|
| Hydrophobicity | 1 scale (hphob_rose) | Aggregate 13 scales → mean + std |
| Secondary structure | Individual propensities | Aggregate helix/sheet/turn propensities |
| Geometric features | Neighbor-based (min/std/radial) | Centroid-relative + spherical coords |
| Categorical encoding | One-hot (5+21=26 features) | LabelEncoder indices + embeddings |
| Total numerical | 20 | 31 |

### Implementation

Created `aggregated_transforms.py` with 31 numerical + embeddings (50 total features).

### Result: FAILED ❌

**R² = 0.4002** (vs 0.5684 baseline - a 29% drop!)

### Root Cause Analysis

1. **Lost Important Features**: Aggregating hydrophobicity scales into mean/std lost the predictive power of individual scales (especially `hphob_rose`)

2. **Different Geometric Approach**: Centroid-based features are fundamentally different from neighbor-based features. Our Phase 5 geometric features (`min_neighbor_dist`, `neighbor_dist_std`, etc.) capture local environment better for exposure prediction.

3. **Over-Generalization**: Averaging secondary structure propensities may have lost feature-specific information

### Lesson Learned

The reference achieved R²=0.5723 despite simpler features because of **better hyperparameters** (lower dropout, lighter regularization), not because of better feature engineering. Our 88 features are superior but Phase 5 was over-regularized.

### Files Archived

- `experiments/baselines/phase6_aggregated_failed/NOTES.md` - Detailed failure analysis
- `src/data/aggregated_transforms.py` - Kept for reference but disabled

---

## Phase 7: GINE with Low Regularization (2026-01-16)

### Motivation

Phase 5 achieved R²=0.5684 with GINE architecture, but analysis of the reference implementation revealed our model was **over-regularized**:

| Parameter | Phase 5 GINE | Reference | Phase 7 |
|-----------|--------------|-----------|---------|
| dropout | 0.35 | 0.1 | **0.1** |
| weight_decay | 1e-4 | 1e-5 | **1e-5** |
| hidden_channels | 96 | 128 | **128** |
| num_layers | 3 | 4 | **4** |
| batch_size | 32 | 16 | **16** |
| conv_type | gine | gatv2 | **gine** |
| weighted_loss | true | false | **false** |

### Strategy: "Poach" Hyperparameters

Keep our proven GINE architecture and 88 features, but apply the reference's lighter regularization:
- **Lower dropout (0.1)** - allows model to learn more complex patterns
- **Lighter weight decay (1e-5)** - reduces underfitting
- **More capacity (128 hidden, 4 layers)** - captures more nuance
- **Smaller batches (16)** - more gradient updates per epoch

### Config File

`configs/phase7_gine_low_regularization.yaml`

### Expected Results

| Configuration | Expected R² |
|---------------|-------------|
| Phase 7 (GINE + low reg) | **0.58-0.60** |
| Phase 5 baseline | 0.5684 |
| Reference GATv2 | 0.5723 |

### Status

❌ **FAILED** - R² 0.5604-0.5607, did not beat Phase 5 baseline (0.5684)

Lower regularization did not help. The model appears to have reached its capacity with the current feature set.

---

## Phase 8: Backbone Dihedral Angles (2026-01-16)

### Motivation

Previous experiments focused on hyperparameter tuning but all approaches plateaued around R²=0.56-0.57. The next improvement must come from **new structural features**.

Secondary structure (helix/sheet/coil) is already encoded via residue propensities, but these are correlated with residue type. **Backbone dihedral angles (φ/ψ)** provide actual 3D structural information:

- φ (phi): C(i-1) - N(i) - Cα(i) - C(i)
- ψ (psi): N(i) - Cα(i) - C(i) - N(i+1)

These angles determine secondary structure and are directly calculable from 3D coordinates.

### Implementation

Created `src/data/backbone_angles.py`:
- Calculates φ/ψ dihedral angles from atomic coordinates
- Uses **sin/cos encoding** to avoid discontinuity at ±180°
- Outputs 4 features: sin(φ), cos(φ), sin(ψ), cos(ψ)
- **Invalid angles encoded as (0, 0)** with norm=0 (vs valid norm=1)
  - N-terminal: phi=(0,0), psi=valid
  - C-terminal: phi=valid, psi=(0,0)
  - Model can learn: norm=1 → valid angle, norm=0 → terminal residue

**Tested on protein 142l:**
- 99.4% of atoms have computed phi angles
- 99.3% have psi angles
- Statistics are consistent with expected protein backbone conformations

### Config File

`configs/phase8_gine_backbone_angles.yaml`

| Parameter | Value |
|-----------|-------|
| Features | 92 (88 standard + 4 backbone angles) |
| Architecture | GINE (3×96, dropout 0.35) |
| Loss | Weighted MSE (α=1.5, threshold=0.8) |
| Scheduler | OneCycle (same as Phase 5 best) |

### Expected Results

| Configuration | Expected R² |
|---------------|-------------|
| Phase 8 (backbone angles) | **0.58+** |
| Phase 5 baseline | 0.5684 |

### Status

🔄 **READY** - Cache cleared, ready to train

```bash
python main.py --config configs/phase8_gine_backbone_angles.yaml
```

---

## Phase 5 Progress (2026-01-13)

### Extended Training Results - NEW BEST! 🎉

After fixing OneCycle scheduler warmup scaling, achieved **R² = 0.5684 (deterministic)**

| Metric | Phase 4 Best | Phase 5 (150ep) | Improvement |
|--------|--------------|-----------------|-------------|
| **R² Score** | 0.5456 | **0.5684** | +4.2% |
| **MAE** | 0.1887 | **0.1829** | -3.1% |
| **RMSE** | 0.2381 | **0.2321** | -2.5% |
| **Pearson** | 0.7398 | **0.7551** | +2.1% |
| **Median AE** | 0.1567 | **0.1504** | -4.0% |
| **Mean Error** | - | +0.0114 | Slight overprediction |

### Deterministic Baseline Archived (2026-01-14)

- Saved checkpoint + logs under `experiments/baselines/phase5_gine_weighted_loss/`
- Config: GINE (3×96, dropout 0.35), exposure-weighted MSE (α=1.5, threshold 0.8)
- Scheduler: OneCycle (max_lr 0.003, warmup 15, div 25, final_div 10000)
- Determinism: `deterministic: true`, seed 42, CUBLAS workspace set to `:4096:8`
- Files include `best_model.pt`, `training_history.csv`, and `config_snapshot.yaml`

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

**Conclusions**:
- Sigmoid activation NOT recommended - linear output is appropriate
- Target normalization (0-2 → 0-1) provides no benefit without bounded activation

### Error Analysis Results (2026-01-13)

Analyzed prediction errors on test set (1.09M atoms, 456 proteins):

**By Exposure Range:**
| Range | Count | % | MAE | Bias | Issue |
|-------|-------|---|-----|------|-------|
| Buried (0-0.2) | 287K | 26.3% | 0.179 | +0.174 | Overpredicts |
| Semi-buried (0.2-0.5) | 307K | 28.1% | 0.144 | +0.061 | **Best** |
| Intermediate (0.5-0.8) | 265K | 24.2% | 0.166 | -0.079 | Slight under |
| Semi-exposed (0.8-1.2) | 198K | 18.1% | 0.240 | -0.214 | Underpredicts |
| Exposed (1.2+) | 36K | 3.3% | 0.362 | -0.360 | **Worst** |

**Key Finding**: Model regresses to mean - conservative on extremes.

**By Element**: Error driven by exposure range, not chemistry (C/N/O/S similar MAE)

**By Protein Size**: Small proteins (<500 atoms) slightly worse, likely due to edge effects

### Next Step: Weighted Loss for Exposed Atoms

**Rationale**:
- Exposed atoms are 8x rarer (3.3% vs 26%) but have 2.5x higher error
- Biological importance: surface atoms drive protein function (binding, interactions)
- Asymmetric weighting targets the biggest problem without disrupting good predictions

**Implementation Plan**:
1. Asymmetric weighted MSE: `weight = 1 + α * max(0, target - threshold)`
2. Default: α=3.0, threshold=0.8
3. Config parameters: `weighted_loss: true`, `loss_alpha`, `loss_threshold`

**TODO**: Also test symmetric weighting (both extremes) for comparison

### Pending: OneCycle LR Schedule Verification

Need to inspect `training_history.csv` to verify:
- Warmup reaches max_lr at epoch ~15
- Smooth annealing after peak
- No unexpected LR behavior

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
| 5 | 150ep | GINE | 0.5680 | 0.1831 | Weighted loss |
| 8 | Backbone angles | GINE | 0.607 | ~0.17 | +phi/psi features |
| 9 | 200ep | GINE | 0.6028 | ~0.18 | Extended training |
| **12** | **Radius+Contact** | **GATv2** | **0.8817** | **0.0866** | **NEW BEST!** |

**Progress**: R² improved from 0.484 → 0.8817 (+82% relative improvement)

---

## Key Files

### Core Code
- `src/data/dataset_fixed.py` - Dataset with all fixes
- `src/data/feature_engineering.py` - Original 88 features
- `src/data/aggregated_transforms.py` - Aggregated features (50 with embeddings)
- `src/models/gnn.py` - GCN, GAT, GIN, GINE, GATv2 architectures (with embedding support)
- `src/training/train.py` - Trainer with AMP support
- `configs/gatv2_config.yaml` - GATv2 with aggregated features config

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
