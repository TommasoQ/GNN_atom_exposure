# Architecture

Model architecture and design decisions for the GNN Protein Atom Exposure Prediction project.

## Overview

This project uses a **MinimalGCN** - a streamlined Graph Convolutional Network - to predict atom burial depth in protein structures. The architecture prioritizes simplicity while maintaining strong performance.

## Graph Representation

### Nodes (Atoms)
Each atom is a node with **5 geometric features**:

| Feature | Description | Range |
|---------|-------------|-------|
| `contact_count_10A` | Atoms within 10Å radius | 0-200+ |
| `dist_to_center` | Distance to protein center | 0-50+ Å |
| `radial_position` | Normalized radial position | 0-1 |
| `3rd_nearest_dist` | Distance to 3rd nearest atom | 2-10 Å |
| `std_dist` | Std dev of neighbor distances | 0-5 Å |

### Edges (Bonds)
Edges represent atomic connectivity from the protein structure. **Edge features are not used** - ablation studies showed they contribute 0% to model performance.

## MinimalGCN Architecture

```
Input: 5 features per atom
    ↓
Linear(5 → 64) + ReLU
    ↓
┌─────────────────────────────────────┐
│ GCN Layer 1                         │
│   GCNConv(64 → 64)                  │
│   BatchNorm(64)                     │
│   ReLU                              │
│   Dropout(0.2)                      │
│   [Global Pool Gate] ← optional     │
│   [LayerNorm] ← if global pooling   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ GCN Layer 2 + Residual              │
│   GCNConv(64 → 64)                  │
│   BatchNorm(64)                     │
│   ReLU                              │
│   Dropout(0.2)                      │
│   + Residual from Layer 1           │
│   [Global Pool Gate] ← optional     │
│   [LayerNorm] ← if global pooling   │
└─────────────────────────────────────┘
    ↓
Linear(64 → 32) + ReLU + Dropout(0.2)
    ↓
Linear(32 → 1)
    ↓
Output: Predicted atom exposure
```

### Parameters
- **Total**: ~16,000 parameters
- **Hidden channels**: 64
- **Layers**: 2
- **Dropout**: 0.2

## Global Pooling (Optional)

When enabled, each GCN layer includes a **gated global pooling** mechanism:

1. **Pool**: Aggregate all node features to graph-level (mean pooling)
2. **Broadcast**: Expand graph representation back to node level
3. **Gate**: Learn a soft gate (sigmoid) to control how much global context to add
4. **Inject**: Add gated global context to node features
5. **Normalize**: Apply LayerNorm for training stability

This allows each atom to "see" the overall protein context (size, average density, etc.).

```python
# Pseudocode
global_repr = mean_pool(x, batch)           # (num_graphs, 64)
global_expanded = global_repr[batch]        # (num_nodes, 64)
gate = sigmoid(Linear([x, global_expanded]))# (num_nodes, 64)
x = x + gate * global_expanded              # Gated addition
x = LayerNorm(x)                            # Stabilize
```

## Training Configuration

### Loss Function
**Exposure-Weighted MSE Loss**:
- Higher weight for extreme values (buried/exposed)
- Addresses class imbalance in exposure distribution

```python
weight = 1.0 + alpha * |y - threshold|^power
loss = mean(weight * (pred - y)^2)
```

Default: `alpha=1.5, threshold=0.8, power=1.0`

### Optimizer & Scheduler
- **Optimizer**: Adam with weight decay 1e-4
- **Scheduler**: OneCycleLR
  - Max LR: 0.0005
  - Warmup: 40 epochs
  - Total: 200 epochs

### Regularization
- Dropout: 0.2
- Batch normalization after each GCN layer
- Feature noise: 0.05 Gaussian σ during training
- Gradient clipping: 0.5

## Design Decisions

### Why GCN over GATv2?

Experiments showed GCN matches GATv2 performance:
- Attention weights didn't learn meaningful patterns
- Contact count already captures local importance
- 30x fewer parameters with same accuracy

### Why Only 5 Features?

Feature importance analysis revealed:
- `contact_count_10A`: 40-60% importance
- Geometric features: 30-40% combined
- Other 88 features: <10% combined

Removing low-importance features:
- Reduced overfitting
- Faster training
- No accuracy loss

### Why 2 Layers?

- 2 layers = 2-hop neighborhood aggregation
- Sufficient for capturing local burial context
- 3+ layers showed no improvement
- Fewer parameters, faster training

### Why Global Pooling?

Atom exposure depends on:
1. **Local context**: Nearby atom density (captured by GCN)
2. **Global context**: Protein size, shape (captured by pooling)

Gated pooling improved R² by 2-5%.

## Performance

| Configuration | R² | MAE | Parameters |
|--------------|-----|-----|------------|
| MinimalGCN | 0.85 | 0.10 | 16K |
| MinimalGCN + GlobalPool | **0.87-0.89** | **0.09** | 17K |

## See Also

- [HISTORY.md](HISTORY.md) - Development history and key findings
- [DATASET.md](DATASET.md) - Data format and features
- [GETTING_STARTED.md](GETTING_STARTED.md) - Installation and usage
