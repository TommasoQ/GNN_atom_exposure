# Dataset

This directory contains the protein atom exposure dataset for GNN training.

## Dataset Structure

```
dataset/
├── README.md                      # This file
├── protein_sample_5000.csv        # List of protein IDs and atom counts (included in repo)
├── depth_indexes.pkl              # Ground truth atom exposure values (330 MB, NOT in repo)
└── sadic_data/                    # Pre-processed protein graphs (NOT in repo)
    └── {pdb_id}/
        ├── {pdb_id}__graphein__ATOM_nodes.csv
        ├── {pdb_id}__graphein__ATOM_edges.csv
        ├── {pdb_id}__graphein__pdb_df.csv
        ├── {pdb_id}__graphein__raw_pdb_df.csv
        └── {pdb_id}__graphein__rgroup_df.csv
```

## Files NOT Committed to Git

Due to GitHub's file size limitations, the following large files are **not** included in the repository:

- `depth_indexes.pkl` (330 MB) - Ground truth labels
- `sadic_data/` directory (containing 4769 protein structures)

## How to Obtain the Dataset

### Option 1: Download from Original Source
[Add instructions here for where to download the original dataset]

### Option 2: Contact the Authors
[Add contact information or instructions]

### Option 3: Request Access
[Add information about requesting dataset access]

## Dataset Statistics

- **Total proteins**: 5,000
- **Total protein structures in sadic_data**: 4,769
- **Atom counts**: Range from ~500 to 5,000+ atoms per protein
- **Node features**: 80+ biochemical properties per atom
- **Edge types**: Covalent bonds, RING structures, etc.

## Data Format

### protein_sample_5000.csv
CSV file with columns:
- `pdb_id`: Protein Data Bank identifier
- `atom_count`: Number of atoms in the protein

### depth_indexes.pkl
Python pickle file containing a dictionary:
- Keys: PDB IDs (matching protein_sample_5000.csv)
- Values: NumPy arrays of atom exposure/depth values

### sadic_data/
Pre-processed protein graphs using [Graphein](https://github.com/a-r-j/graphein):
- Each protein has its own directory
- Contains CSV files with node features, edge lists, and structural information

## Usage

Once you have the dataset files in place, you can load them using:

```python
from src.data.dataset import ProteinAtomDataset

# Load dataset
dataset = ProteinAtomDataset(root='dataset/', split='train')

# Access a sample
sample = dataset[0]
print(f"PDB ID: {sample.pdb_id}")
print(f"Atoms: {sample.num_nodes}")
print(f"Bonds: {sample.num_edges}")
```

## License and Citation

[Add dataset license information]

[Add citation information if applicable]

## Notes

- The dataset is preprocessed and ready for PyTorch Geometric
- Node features include spatial coordinates, chemical properties, and biochemical descriptors
- Edge features include bond types and distances
- Suitable for graph neural network training on atom-level prediction tasks
