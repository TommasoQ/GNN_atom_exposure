# Architecture

Model architecture, features, and design decisions for the GNN Protein Atom Exposure Prediction project.

## Overview

Graph Neural Networks predict atom exposure depth in protein structures. Proteins are represented as graphs where atoms are nodes and bonds are edges, with rich biochemical features at both node and edge levels.

## Graph Representation

### Nodes (Atoms)
Each atom is a node with **93 features** (see Feature Engineering below).

### Edges (Bonds/Contacts)
Edges represent connections between atoms with **12 edge features**:
- Covalent bonds (bond type, bond length)
- Ring structures
- Distance-based connections (radius graph, spatial distance)

## Feature Engineering (93 Node Features)

The 93 features are composed of:

| Group | Count | Description |
|-------|-------|-------------|
| Atom type | 24 | One-hot element encoding, atom identifiers |
| Chemical properties | 31 | Meiler descriptors, ExPASy features, H-bond donors/acceptors |
| Secondary structure | 5 | Alpha-helix, beta-sheet, beta-turn propensities |
| Residue encoding | 21 | One-hot amino acid type |
| Geometric features | 8 | Distances, neighbor counts, **contact_count_10A** |
| Backbone angles | 4 | Phi/psi dihedral angles (sin/cos encoded) |

### Critical Feature: contact_count_10A
The number of atoms within 10A radius. Removing this single feature drops R² from 0.8892 to 0.5689 (-36%), making it the most important feature by far.

## Model Architecture

### Best Model: GATv2 + Dynamic Global Pooling

```
Input (93 features)
    |
[Input Projection] -> hidden_channels (136)
    |
[GATv2 Layer 1] (4 heads x 34 = 136) + BatchNorm + ELU + Dropout
    |--- [Global Mean Pool -> Gated Fusion]
    |
[GATv2 Layer 2] + BatchNorm + ELU + Dropout + Residual
    |--- [Global Mean Pool -> Gated Fusion]
    |
[GATv2 Layer 3] + BatchNorm + ELU + Dropout + Residual
    |--- [Global Mean Pool -> Gated Fusion]
    |
[GATv2 Layer 4] + BatchNorm + ELU + Dropout + Residual
    |--- [Global Mean Pool -> Gated Fusion]
    |
[Output MLP] (136 -> 68 -> 1)
    |
Atom Exposure Prediction
```

### Dynamic Global Pooling

After each GNN layer, a gated mechanism injects protein-level context:

1. **Pool**: Aggregate all node features per graph (mean pooling)
2. **Broadcast**: Expand global representation back to node level
3. **Gate**: Learn per-node gate value via `sigmoid(W * [node || global])`
4. **Fuse**: `node = node + gate * global`

This gives each node access to protein-wide context (e.g., overall protein size/shape) while letting the gate learn how much global information each node needs.

### Gaussian Noise Injection (Optional)

When `input_noise_std > 0`, Gaussian noise (default σ=0.02) is added to input features during training only. This regularizes against overfitting to static crystallographic coordinates.

### Supported GNN Types

The `AtomExposureGNN` class supports multiple convolution types:

| Type | Edge Features | Description | Best R² |
|------|:---:|-------------|---------|
| **GATv2** | Yes | Dynamic attention, 4 heads, residual connections | **0.9408** |
| GINE | Yes | GIN with edge features | 0.5684 |
| GAT | Yes | Standard attention | 0.477 |
| GIN | No | Graph Isomorphism Network | - |
| GCN | No | Basic message passing | 0.484 |

### Hyperparameters (Current Best)

```yaml
model:
  conv_type: gatv2
  in_channels: 93
  hidden_channels: 136
  num_layers: 4
  dropout: 0.26
  edge_dim: 12
  use_global_pool: true
  global_pool_type: mean
  global_pool_layers: every

training:
  learning_rate: 5.0e-04
  weight_decay: 1.2e-04
  scheduler: one_cycle
  max_lr: 0.0007
  warmup_epochs: 50
  early_stopping_patience: 60
  early_stopping_metric: r2
  gradient_clip: 0.5
  use_amp: true
```

## Loss Function

### Range-Specific Weighted MSE

Different exposure ranges get different weights to address distribution imbalance:

| Range | Weight | Rationale |
|-------|--------|-----------|
| Buried (0-0.2) | 1.5x | Under-represented, important |
| Semi-buried (0.2-0.5) | 1.0x | Baseline |
| Intermediate (0.5-0.8) | 1.0x | Baseline |
| Semi-exposed (0.8-1.2) | 1.3x | Transition zone |
| Exposed (1.2+) | 2.0x | Rare, biologically important |

Additionally, an asymmetric penalty (1.5x) penalizes under-prediction more than over-prediction.

## Evaluation

### Raw vs Clamped Metrics

Atom exposure cannot be negative, so evaluation reports both:
- **Raw predictions**: For model diagnosis (shows negative predictions, bias)
- **Clamped predictions**: Final metrics, `max(prediction, 0)`

### Metrics
- MAE, RMSE, R², Pearson correlation
- Mean error (bias), std error, median absolute error
- Negative prediction count (diagnostic)

## Training Strategy

- **Scheduler**: OneCycleLR with 50-epoch warmup (out of 200 total)
- **Early stopping**: On validation R² with patience of 60 epochs
- **Mixed precision** (AMP) for faster training
- **cuDNN benchmark** mode enabled
- **Gradient clipping** at 0.5

### Data Splits

- **Training**: 80% of proteins
- **Validation**: 10% of proteins
- **Test**: 10% of proteins

Split at protein level (not atom level) to prevent data leakage. Deterministic with `seed=42`.

## Design Decisions

### Why GATv2 over GINE?
GATv2's dynamic attention mechanism + edge feature support + internal residual connections significantly outperform GINE (R² 0.88 vs 0.57). The attention heads learn meaningful bond/distance importance patterns.

### Why Global Pooling?
Node-level predictions benefit from protein-wide context. A buried atom in a small protein differs from one in a large protein. Global pooling provides this context with minimal parameter overhead via gating.

### Why Range-Specific Loss?
Exposure distribution is heavily skewed toward buried atoms. Uniform MSE under-weights rare exposed atoms that are biologically important (active sites, binding regions).

## References

- **GATv2**: Brody et al., "How Attentive are Graph Attention Networks?" (2022)
- **PyTorch Geometric**: Fey & Lenssen, "Fast Graph Representation Learning with PyTorch Geometric" (2019)
- **OneCycleLR**: Smith & Topin, "Super-Convergence" (2019)

## See Also

- [Dataset Documentation](DATASET.md)
- [Getting Started](GETTING_STARTED.md)
- [History](HISTORY.md)
