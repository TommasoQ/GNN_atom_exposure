# Archived Documentation

This directory contains historical documentation that has been superseded by current project documentation but is preserved for reference.

## Contents

### Phase Summaries (Completed Phases)

- **[phase1_complete.md](phase1_complete.md)** - Phase 1: Initial exploration and EDA
  - Exploratory data analysis
  - Feature correlation analysis
  - Initial model testing
  - Status: Completed 2026-01-08

- **[phase2_complete.md](phase2_complete.md)** - Phase 2: Dataset quality and fixes
  - Dataset integrity analysis
  - HETATM investigation
  - Feature engineering pipeline
  - Status: Completed 2026-01-08

- **[phase3_complete.md](phase3_complete.md)** - Phase 3: Critical bug fixes and valid baseline
  - Fixed ~1000x gradient error
  - Feature reduction (100+ → 88)
  - First valid experiments (3.5, 3.6)
  - Status: Completed 2026-01-09

### Pre-Bugfix Planning (Archived)

Historical planning documents from before the critical bug fixes in Phase 3. These experiments (3.1-3.4) are now invalid due to the loss weighting bug.

**Location**: `experiments/progress/_archived_pre_bugfix/`
**Date**: Pre-2026-01-09
**Status**: Invalid - for historical reference only

### Phase 4 Planning (Archived)

Planning documents from Phase 4 hyperparameter tuning. These have been superseded by completed results.

**Location**: `experiments/progress/_archived_phase4_complete/`
**Date**: 2026-01-12
**Status**: Complete - results documented in [EXPERIMENTS.md](../EXPERIMENTS.md)

## Why Archived?

These documents are preserved to:
1. **Maintain project history**: Show evolution of understanding
2. **Document mistakes**: Learn from failed experiments and bugs
3. **Preserve context**: Understand why certain decisions were made
4. **Reference material**: Useful for similar future investigations

## Current Documentation

For current project status and documentation, see:
- [../EXPERIMENTS.md](../EXPERIMENTS.md) - Complete experimental history
- [../CHANGELOG.md](../CHANGELOG.md) - Chronological project timeline
- [../../experiments/STATUS.md](../../experiments/STATUS.md) - Current phase status

## Note on Invalid Experiments

Experiments 3.1-3.4 (archived in `experiments/baselines/_archived_invalid_experiments/`) should not be used for comparisons or conclusions. All had a critical loss weighting bug that caused ~1000x gradient error.

**Valid experiments start from 3.5 onwards** (Phase 3, post-bugfix).

## Organizational Changes

As of 2026-01-16, the project documentation was reorganized:
- Created centralized `docs/` directory
- Consolidated scattered documentation
- Archived historical/completed phase docs here
- Moved analysis documents to `docs/analysis/`
- Simplified experiments tracking

See [../CHANGELOG.md](../CHANGELOG.md) for details.
