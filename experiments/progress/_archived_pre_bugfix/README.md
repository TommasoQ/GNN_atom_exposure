# Archived Pre-Bug-Fix Analysis Files

**Date Archived**: 2026-01-09

## Context

These analysis files were created during Phase 3 before discovering the critical loss weighting bug. They document various hypotheses about why training was failing (overfitting, regularization, architecture) - all of which turned out to be secondary to Bug #1.

## Files:

- `phase3_overfitting_analysis.md` - Analysis of apparent overfitting (caused by Bug #1)
- `phase3_experiment2_gat.md` - Plan for GAT with heavy regularization (wrong diagnosis)
- `phase3_diagnostic_plan.md` - Diagnostic plan before bug discovery
- `phase3_start.md` - Initial Phase 3 plan

## Why Archived:

These analyses were based on the incorrect assumption that architectural/hyperparameter choices were the problem. After discovering Bug #1 (loss weighting error causing ~1000x gradient reduction), these documents became outdated.

## Current Documentation:

- **Bug report**: `../bug_discovery.md`
- **Phase 3 summary**: `../phase3_summary.md`
- **Valid experiments**: Start from Experiment 3.5+
