# Dataset

Protein atom exposure dataset for GNN training.

## Structure

```
dataset/
├── README.md                      # This file
├── protein_sample_5000.csv        # Protein IDs and atom counts (in git)
├── depth_indexes.pkl              # Ground truth exposure values (330 MB, NOT in git)
├── sadic_data/                    # Pre-processed protein graphs (NOT in git)
│   └── {pdb_id}/
│       ├── {pdb_id}__graphein__ATOM_nodes.csv
│       ├── {pdb_id}__graphein__ATOM_edges.csv
│       ├── {pdb_id}__graphein__pdb_df.csv
│       ├── {pdb_id}__graphein__raw_pdb_df.csv
│       └── {pdb_id}__graphein__rgroup_df.csv
└── processed/                     # PyG cache, auto-generated (NOT in git)
```

## Statistics

- **Total proteins listed**: 5,000
- **Available structures**: 4,769 (231 missing from sadic_data/)
- **Protein size**: ~500 to 5,000+ atoms each
- **Node features**: 93 per atom
- **Edge features**: 12 per edge

## Files NOT in Git

Large files excluded from version control:
- `depth_indexes.pkl` (330 MB) - Ground truth labels
- `sadic_data/` (~1 GB) - Pre-processed protein graphs
- `processed/` - Auto-generated PyG cache

Contact project authors for dataset access.

## Data Format

### protein_sample_5000.csv
- `pdb_id`: Protein Data Bank identifier
- `atom_count`: Number of atoms in the protein

### depth_indexes.pkl
Python pickle dictionary mapping PDB IDs to NumPy arrays of atom exposure values.

### sadic_data/
Pre-processed protein graphs created with [Graphein](https://github.com/a-r-j/graphein). Each protein directory contains CSV files with node features, edge lists, and structural information.

## Usage

```python
from src.data.dataset_fixed import ProteinAtomDataset

dataset = ProteinAtomDataset(root='dataset/', split='train')
sample = dataset[0]
print(f"Atoms: {sample.num_nodes}, Features: {sample.x.shape}")
```

## See Also

- [Dataset Documentation](../docs/DATASET.md) - Full details on features, splits, and loading
