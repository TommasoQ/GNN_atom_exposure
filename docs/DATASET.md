# Dataset Documentation

This document describes the protein atom exposure dataset used for GNN training.

## Overview

The dataset contains 4,769 protein structures with pre-processed graph representations. Each protein is represented as a graph where atoms are nodes and bonds are edges, with rich biochemical features.

## Dataset Structure

```
dataset/
├── README.md                      # Dataset overview
├── protein_sample_5000.csv        # List of protein IDs and atom counts
├── depth_indexes.pkl              # Ground truth atom exposure values (330 MB)
├── sadic_data/                    # Pre-processed protein graphs
│   └── {pdb_id}/
│       ├── {pdb_id}__graphein__ATOM_nodes.csv
│       ├── {pdb_id}__graphein__ATOM_edges.csv
│       ├── {pdb_id}__graphein__pdb_df.csv
│       ├── {pdb_id}__graphein__raw_pdb_df.csv
│       └── {pdb_id}__graphein__rgroup_df.csv
└── processed/                     # PyTorch Geometric cache (auto-generated)
```

## Dataset Statistics

- **Total proteins listed**: 5,000
- **Available structures**: 4,769 (231 proteins missing from sadic_data/)
- **Protein size range**: ~500 to 5,000+ atoms per protein
- **Total atoms**: ~7.5M atoms across all proteins
- **Node features**: 88 features per atom (after feature reduction)
- **Edge types**: Covalent bonds, ring structures, distance-based connections

## Files Description

### protein_sample_5000.csv
CSV file containing:
- `pdb_id`: Protein Data Bank identifier (e.g., "1a00", "1a01")
- `atom_count`: Number of atoms in the protein

**Note**: 231 proteins in this file don't have corresponding structures in sadic_data/ and are skipped during loading.

### depth_indexes.pkl (330 MB)
Python pickle file containing ground truth labels:
- **Type**: Dictionary
- **Keys**: PDB IDs (strings)
- **Values**: NumPy arrays of float values representing atom exposure depth
- **Shape**: Each array length matches the atom count for that protein
- **Units**: Angstroms (Å) - distance from atom to protein surface

**Exposure Depth Values**:
- `0.0 Å`: Completely exposed (on surface)
- `5.0+ Å`: Deeply buried
- Distribution is heavily skewed toward buried atoms

### sadic_data/
Pre-processed protein graphs created using [Graphein](https://github.com/a-r-j/graphein):

Each protein directory contains:
- **ATOM_nodes.csv**: Node features (one row per atom)
- **ATOM_edges.csv**: Edge list (bond connections)
- **pdb_df.csv**: Processed PDB dataframe
- **raw_pdb_df.csv**: Original PDB data
- **rgroup_df.csv**: Residue group information

## Node Features (88 features)

After systematic feature reduction in Phase 3, the current feature set includes:

### Structural Features
- **Coordinates**: x, y, z (3D spatial position)
- **B-factor**: Temperature/disorder factor
- **Residue**: name, number, chain ID
- **Atom type**: Element symbol, atom identifier

### Chemical Properties
- **Hydrogen bonding**: Donors and acceptors count
- **Meiler descriptors**: 7D physicochemical properties
- **ExPASy features**: ~60 biochemical properties including:
  - Hydrophobicity scales (multiple methods)
  - pKa values (COOH, NH3, R-group)
  - Isoelectric point
  - Molecular weight
  - Secondary structure propensities (α-helix, β-sheet, β-turn)
  - Accessibility indices
  - Flexibility and mutability indices

### Geometric Features
- **Neighbor counts**: Number of connected atoms
- **Distances**: To neighboring atoms
- **Angular features**: Local geometry descriptors

## Edge Features

- **kind**: Bond type (covalent, RING, distance-based)
- **bond_length**: Covalent bond distance (Å)
- **distance**: Spatial distance between atoms (Å)

## Data Splits

Data is split at the **protein level** (not atom level) to prevent data leakage:

- **Training**: 70% of proteins (~3,340 proteins)
- **Validation**: 15% of proteins (~715 proteins)
- **Test**: 15% of proteins (~714 proteins)

Split is deterministic with `seed=42` for reproducibility.

## Data Loading

### Using the Dataset Class

```python
from src.data.dataset_fixed import ProteinAtomDataset
from torch_geometric.loader import DataLoader

# Load splits
train_dataset = ProteinAtomDataset(root='dataset/', split='train')
val_dataset = ProteinAtomDataset(root='dataset/', split='val')
test_dataset = ProteinAtomDataset(root='dataset/', split='test')

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Access a sample
sample = train_dataset[0]
print(f"PDB ID: {sample.pdb_id}")
print(f"Atoms: {sample.num_nodes}")
print(f"Bonds: {sample.num_edges}")
print(f"Features shape: {sample.x.shape}")  # [num_atoms, 88]
print(f"Targets shape: {sample.y.shape}")   # [num_atoms, 1]
```

### Data Format

Each graph sample is a `torch_geometric.data.Data` object with:
- `x`: Node features tensor [num_atoms, 88]
- `edge_index`: Edge connectivity [2, num_edges]
- `edge_attr`: Edge features [num_edges, feature_dim]
- `y`: Target exposure values [num_atoms, 1]
- `pdb_id`: Protein identifier (string)

## Dataset Quality

### Data Quality Issues Addressed

1. **HETATM dominance**: 38 proteins with >50% HETATM atoms
   - Investigated and documented
   - Decision: Keep in dataset (represent diversity)

2. **Feature correlation**: High correlation (>0.95) between some features
   - Removed redundant features in Phase 3
   - Reduced from 100+ to 88 features

3. **Missing proteins**: 231 proteins listed but not in sadic_data/
   - Automatically skipped during loading
   - No impact on training

### Data Integrity

- ✅ All target values verified to match protein structures
- ✅ No missing features in loaded graphs
- ✅ All edge indices valid
- ✅ Consistent feature dimensions across all proteins

See [DATASET_INTEGRITY_REPORT](analysis/DATASET_INTEGRITY.md) for full analysis.

## Data Preprocessing

### Feature Engineering

Feature engineering is handled by [feature_engineering.py](../src/data/feature_engineering.py):
- Loads raw graph CSVs from sadic_data/
- Applies feature selection (88 features)
- Normalizes features
- Creates PyG Data objects

### Caching

PyTorch Geometric automatically caches processed graphs in `dataset/processed/`:
- `pre_transform.pt`: Preprocessing configuration
- `data_*.pt`: Individual processed graphs

To regenerate cache (if feature engineering changes):
```bash
bash cleanup_folders.sh  # Removes processed cache
python main.py  # Regenerates on next run
```

## Data Acquisition

### Files NOT in Git

Due to GitHub's file size limitations, the following are NOT committed:
- `depth_indexes.pkl` (330 MB)
- `sadic_data/` directory (~1 GB)
- `processed/` cache (auto-generated)

### How to Obtain the Dataset

Contact the project authors for dataset access.

## Target Variable

### Atom Exposure Depth

The target variable represents how "buried" or "exposed" each atom is:

- **Definition**: Distance from atom to protein surface (Å)
- **Range**: 0.0 (exposed) to 10+ (buried)
- **Distribution**: Heavily skewed toward buried atoms
- **Biological significance**:
  - Exposed atoms: Active sites, binding regions
  - Buried atoms: Structural core, hydrophobic interior

### Distribution

```
Exposure Range    | % of Atoms
------------------|------------
0.0 - 2.0 Å      | ~10%  (Highly exposed)
2.0 - 5.0 Å      | ~30%  (Partially exposed)
5.0+ Å           | ~60%  (Buried)
```

This imbalance is addressed with **weighted loss** during training.

## License and Citation

[Add dataset license information]

[Add citation information if applicable]

## See Also

- [Architecture Documentation](ARCHITECTURE.md) - Model design and features
- [Getting Started](GETTING_STARTED.md) - Installation and usage
- [Experiments Documentation](EXPERIMENTS.md) - Training results
