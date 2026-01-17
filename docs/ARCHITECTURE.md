# Architecture

This document describes the model architecture, features, and design decisions for the GNN Protein Atom Exposure Prediction project.

## Overview

This project uses Graph Neural Networks (GNNs) to predict atom exposure depth in protein structures. Proteins are represented as graphs where atoms are nodes and bonds are edges, with rich biochemical features at both node and edge levels.

## Graph Representation

### Nodes (Atoms)
Each atom in the protein structure is represented as a node in the graph with 88 features (after feature reduction from initial 100+).

### Edges (Bonds)
Edges represent connections between atoms, including:
- Covalent bonds
- Ring structures
- Distance-based connections

## Feature Engineering

### Current Feature Set (88 features)

After systematic feature reduction analysis, the current optimal feature set includes:

**Structural Features:**
- 3D spatial coordinates (x, y, z)
- B-factor (temperature/disorder factor)
- Residue information (name, number, chain)
- Atom type and element

**Chemical Properties:**
- Hydrogen bond donors/acceptors
- Meiler descriptors (7D physicochemical properties)
- ExPASy features:
  - Hydrophobicity scales (multiple methods)
  - pKa values (COOH, NH3, R-group)
  - Isoelectric point
  - Molecular weight
  - Secondary structure propensities (α-helix, β-sheet, β-turn)
  - Accessibility indices
  - Flexibility and mutability indices

**Geometric Features:**
- Distance to neighbors
- Neighbor counts
- Angular features
- Local geometry descriptors

**Edge Features:**
- Bond type
- Bond length
- Spatial distance

### Feature Reduction History

- **Initial**: 100+ features with significant redundancy
- **Phase 3**: Reduced to 88 features by removing highly correlated (>0.95) and low-importance features
- **Result**: Improved R² from 0.477 to 0.484 and reduced overfitting

## Model Architecture

### Supported GNN Types

The project supports multiple GNN architectures through the `AtomExposureGNN` class:

1. **GCN (Graph Convolutional Network)**
   - Basic message passing
   - Symmetric normalization
   - Fast and simple

2. **GAT (Graph Attention Network)**
   - Attention-weighted message passing
   - Learns edge importance
   - Interpretable attention weights

3. **GIN (Graph Isomorphism Network)**
   - Powerful graph representation
   - Theoretically strong expressiveness
   - Good for complex patterns

4. **GINE (Graph Isomorphism Network with Edge features)**
   - **Current best architecture** (Phase 5)
   - Incorporates edge features
   - Stronger representation than GIN
   - R² = 0.5684, MAE = 0.1829

5. **GATv2 (Graph Attention Network v2)**
   - Dynamic attention mechanism
   - More expressive than GAT
   - Being tested in Phase 8

### Network Structure

```
Input (88 features)
    ↓
[GNN Layer 1] (hidden_channels)
    ↓
[Batch Normalization]
    ↓
[ReLU Activation]
    ↓
[Dropout]
    ↓
[GNN Layer 2] (hidden_channels)
    ↓
[Batch Normalization]
    ↓
[ReLU Activation]
    ↓
[Dropout]
    ↓
[GNN Layer 3] (hidden_channels)
    ↓
[Batch Normalization]
    ↓
[ReLU Activation]
    ↓
[Dropout]
    ↓
[Linear Output Layer] → Atom Exposure Prediction
```

### Hyperparameters (Current Best - Phase 5)

```yaml
model:
  conv_type: gine
  hidden_channels: 128
  num_layers: 3
  dropout: 0.2

training:
  batch_size: 8
  learning_rate: 0.003  # max_lr for OneCycle
  epochs: 200
  scheduler: OneCycleLR
  pct_start: 0.3
  weight_decay: 0.0
```

## Training Strategy

### Loss Function

**Weighted MSE Loss** with inverse frequency weighting:
- Addresses class imbalance (most atoms are buried)
- Gives higher weight to exposed atoms (minority class)
- Improved R² significantly over standard MSE

### Learning Rate Scheduler

**OneCycleLR** (Phase 4 winner over Cosine):
- Warm-up phase: 30% of training
- Peak learning rate: 0.003
- Annealing phase: Gradual decay
- Better convergence than Cosine scheduler

### Regularization

- **Dropout**: 0.2 (20% dropout rate)
- **Batch Normalization**: After each GNN layer
- **Weight Decay**: 0.0 (removed after Phase 7 analysis)

### Data Splits

- **Training**: 70% of proteins
- **Validation**: 15% of proteins
- **Test**: 15% of proteins

Split at protein level (not atom level) to prevent data leakage.

## Performance

### Current Best Results (Phase 5 - GINE)

- **R² Score**: 0.5684
- **MAE**: 0.1829
- **RMSE**: 0.2755
- **Improvement**: +17.4% from baseline (R² 0.484)

### Evolution

| Phase | Architecture | R² | MAE | Key Changes |
|-------|-------------|-----|-----|-------------|
| 3.5 | GCN | 0.484 | 0.2012 | Baseline after bug fixes |
| 3.6 | GAT | 0.477 | 0.2020 | Added attention |
| 4 | GCN | 0.5456 | - | Hyperparameter tuning |
| 5 | GINE | 0.5684 | 0.1829 | Edge features + weighted loss |
| 6 | GINE | 0.4002 | - | FAILED: Over-aggregation |
| 7 | GINE | 0.5607 | - | FAILED: Low regularization |

## Design Decisions

### Why GINE over Other Architectures?

1. **Edge Feature Utilization**: GINE can incorporate bond types and distances
2. **Expressive Power**: Strong theoretical foundations
3. **Empirical Performance**: 15% better than GCN/GAT
4. **Local Geometry**: Captures atomic neighborhoods well

### Why Weighted Loss?

Exposure distribution is heavily skewed:
- Most atoms: buried (depth > 5Å)
- Few atoms: exposed (depth < 2Å)
- Weighted loss ensures model learns exposed atoms

### Why OneCycleLR?

- Faster convergence than step decay
- Better final performance than Cosine
- Helps escape local minima during warm-up
- Proven in Phase 4 grid search

### Why 3 Layers?

- Balances receptive field and overfitting
- 3 layers = 3-hop neighborhood aggregation
- Deeper models (4-5 layers) showed diminishing returns
- Shallower models (1-2 layers) under-fit

## Key Challenges

1. **Graph Size Variability**: Proteins range from 500 to 5000+ atoms
   - **Solution**: Dynamic batching with PyG DataLoader

2. **Feature Engineering**: 100+ initial features with redundancy
   - **Solution**: Correlation analysis and feature importance ranking

3. **Class Imbalance**: Buried atoms >> exposed atoms
   - **Solution**: Weighted loss function

4. **Computational Cost**: Large graphs are memory-intensive
   - **Solution**: Batch size 8, gradient accumulation

5. **Overfitting**: High-capacity models overfit easily
   - **Solution**: Dropout, batch norm, careful regularization

## Future Directions

### Phase 8 Experiments

1. **GATv2 with Attention**: Dynamic attention mechanism
2. **Backbone Dihedral Angles**: Add φ/ψ angles as features
3. **Geometric Features**: Enhanced local geometry descriptors

### Potential Improvements

- [ ] E(3)-equivariant networks for geometric invariance
- [ ] Pre-training on larger protein datasets
- [ ] Multi-task learning (exposure + secondary structure)
- [ ] Residue-level predictions
- [ ] Attention visualization for interpretability
- [ ] Ensemble methods

## References

### Key Papers

- **Graph Isomorphism Network**: Xu et al., "How Powerful are Graph Neural Networks?" (2019)
- **PyTorch Geometric**: Fey & Lenssen, "Fast Graph Representation Learning with PyTorch Geometric" (2019)
- **Atom Depth**: Yuan et al., "Atom depth as a descriptor for the 3D structures of molecules" (2006)

### Libraries

- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **Graphein**: https://github.com/a-r-j/graphein

## See Also

- [Dataset Documentation](DATASET.md) - Data structure and features
- [Experiments Documentation](EXPERIMENTS.md) - Full experimental history
- [Getting Started](GETTING_STARTED.md) - Installation and usage
