"""
Investigate different dataset versions to verify data integrity and filtering
"""
import pickle
import pandas as pd
from pathlib import Path
import sys

print("="*80)
print("DATASET VERSION INVESTIGATION")
print("="*80)

# Define paths
dataset_extra = Path(r"C:\Users\Edoardo\GNN_protein_exposure\dataset (extra)")
dataset_protein_exp = Path(r"C:\Users\Edoardo\protein_exposure\dataset")
dataset_current = Path(r"C:\Users\Edoardo\GNN_protein_exposure\GNN_atom_exposure\dataset")

print("\n" + "-"*80)
print("LOCATION 1: dataset (extra)")
print("-"*80)

if dataset_extra.exists():
    # Load depth_indexes
    depth_path = dataset_extra / "depth_indexes.pkl"
    protein_csv = dataset_extra / "protein_sample_5000.csv"

    with open(depth_path, 'rb') as f:
        depth_extra = pickle.load(f)

    proteins_extra = pd.read_csv(protein_csv)

    print(f"depth_indexes.pkl:")
    print(f"  Type: {type(depth_extra)}")
    if isinstance(depth_extra, pd.DataFrame):
        print(f"  Shape: {depth_extra.shape}")
        print(f"  Columns: {list(depth_extra.columns)}")
        print(f"  Unique proteins: {depth_extra['pdb_id'].nunique()}")

    print(f"\nprotein_sample_5000.csv:")
    print(f"  Shape: {proteins_extra.shape}")
    print(f"  Columns: {list(proteins_extra.columns)}")
else:
    print("  [NOT FOUND]")

print("\n" + "-"*80)
print("LOCATION 2: protein_exposure/dataset (WITH FILTERED VERSIONS)")
print("-"*80)

if dataset_protein_exp.exists():
    # Original versions
    depth_path_orig = dataset_protein_exp / "depth_indexes.pkl"
    protein_csv_orig = dataset_protein_exp / "protein_sample_5000.csv"

    # Filtered versions
    depth_path_filt = dataset_protein_exp / "depth_indexes_filtered.pkl"
    protein_csv_filt = dataset_protein_exp / "protein_sample_filtered.csv"

    print("\nORIGINAL VERSIONS:")
    print("-"*40)

    with open(depth_path_orig, 'rb') as f:
        depth_orig = pickle.load(f)
    proteins_orig = pd.read_csv(protein_csv_orig)

    print(f"depth_indexes.pkl:")
    print(f"  Type: {type(depth_orig)}")
    if isinstance(depth_orig, pd.DataFrame):
        print(f"  Shape: {depth_orig.shape}")
        print(f"  Unique proteins: {depth_orig['pdb_id'].nunique()}")

    print(f"\nprotein_sample_5000.csv:")
    print(f"  Shape: {proteins_orig.shape}")
    print(f"  Total proteins: {len(proteins_orig)}")

    # Check filtered versions
    if depth_path_filt.exists() and protein_csv_filt.exists():
        print("\n" + "-"*40)
        print("FILTERED VERSIONS:")
        print("-"*40)

        with open(depth_path_filt, 'rb') as f:
            depth_filt = pickle.load(f)
        proteins_filt = pd.read_csv(protein_csv_filt)

        print(f"depth_indexes_filtered.pkl:")
        print(f"  Type: {type(depth_filt)}")
        if isinstance(depth_filt, pd.DataFrame):
            print(f"  Shape: {depth_filt.shape}")
            print(f"  Unique proteins: {depth_filt['pdb_id'].nunique()}")

            # Compare with original
            if isinstance(depth_orig, pd.DataFrame):
                diff_rows = depth_orig.shape[0] - depth_filt.shape[0]
                diff_proteins = depth_orig['pdb_id'].nunique() - depth_filt['pdb_id'].nunique()
                print(f"  Rows removed: {diff_rows:,} ({100*diff_rows/depth_orig.shape[0]:.2f}%)")
                print(f"  Proteins removed: {diff_proteins}")

        print(f"\nprotein_sample_filtered.csv:")
        print(f"  Shape: {proteins_filt.shape}")
        print(f"  Total proteins: {len(proteins_filt)}")
        print(f"  Proteins removed: {len(proteins_orig) - len(proteins_filt)}")

        # Check which proteins were removed
        orig_set = set(proteins_orig['pdb_id'])
        filt_set = set(proteins_filt['pdb_id'])
        removed_proteins = orig_set - filt_set

        print(f"\n  Removed proteins: {len(removed_proteins)}")
        if len(removed_proteins) > 0:
            print(f"  First 20 removed: {sorted(list(removed_proteins))[:20]}")

        # Check if filtered proteins ALL have depth data
        if isinstance(depth_filt, pd.DataFrame):
            depth_proteins = set(depth_filt['pdb_id'].unique())
            proteins_without_depth = filt_set - depth_proteins
            print(f"\n  Filtered proteins without depth data: {len(proteins_without_depth)}")
            if len(proteins_without_depth) > 0:
                print(f"    [ERROR] Filtering was incomplete!")
                print(f"    Missing: {sorted(list(proteins_without_depth))[:10]}")
            else:
                print(f"    [OK] All filtered proteins have depth data!")
    else:
        print("\n  [FILTERED VERSIONS NOT FOUND]")

else:
    print("  [NOT FOUND]")

print("\n" + "-"*80)
print("LOCATION 3: GNN_atom_exposure/dataset (CURRENT PROJECT)")
print("-"*80)

if dataset_current.exists():
    depth_path_curr = dataset_current / "depth_indexes.pkl"
    protein_csv_curr = dataset_current / "protein_sample_5000.csv"

    with open(depth_path_curr, 'rb') as f:
        depth_curr = pickle.load(f)
    proteins_curr = pd.read_csv(protein_csv_curr)

    print(f"depth_indexes.pkl:")
    print(f"  Type: {type(depth_curr)}")
    if isinstance(depth_curr, pd.DataFrame):
        print(f"  Shape: {depth_curr.shape}")
        print(f"  Unique proteins: {depth_curr['pdb_id'].nunique()}")

    print(f"\nprotein_sample_5000.csv:")
    print(f"  Shape: {proteins_curr.shape}")
    print(f"  Total proteins: {len(proteins_curr)}")

    # Check for proteins without depth data
    if isinstance(depth_curr, pd.DataFrame):
        depth_proteins = set(depth_curr['pdb_id'].unique())
        csv_proteins = set(proteins_curr['pdb_id'])
        missing_depth = csv_proteins - depth_proteins

        print(f"\n  Proteins in CSV: {len(csv_proteins)}")
        print(f"  Proteins with depth: {len(depth_proteins)}")
        print(f"  Proteins missing depth: {len(missing_depth)}")

        if len(missing_depth) > 0:
            print(f"    [WARNING] Current dataset has unfiltered data!")
            print(f"    First 20 missing: {sorted(list(missing_depth))[:20]}")
        else:
            print(f"    [OK] All proteins have depth data!")

else:
    print("  [NOT FOUND]")

print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

# Compare file sizes and checksums
print("\nFile size comparison:")

datasets = {
    "dataset (extra)": dataset_extra,
    "protein_exposure": dataset_protein_exp,
    "current (GNN_atom_exposure)": dataset_current
}

for name, path in datasets.items():
    if path.exists():
        depth_file = path / "depth_indexes.pkl"
        protein_file = path / "protein_sample_5000.csv"

        if depth_file.exists() and protein_file.exists():
            depth_size = depth_file.stat().st_size / (1024*1024)  # MB
            protein_size = protein_file.stat().st_size / 1024  # KB

            print(f"\n{name}:")
            print(f"  depth_indexes.pkl: {depth_size:.2f} MB")
            print(f"  protein_sample_5000.csv: {protein_size:.2f} KB")

# Check if filtered version should be used
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

if dataset_protein_exp.exists():
    filt_depth = dataset_protein_exp / "depth_indexes_filtered.pkl"
    filt_proteins = dataset_protein_exp / "protein_sample_filtered.csv"

    if filt_depth.exists() and filt_proteins.exists():
        print("\n[RECOMMENDATION] Use FILTERED versions from protein_exposure/dataset/")
        print("\nThe filtered dataset ensures:")
        print("  1. All proteins in CSV have corresponding depth labels")
        print("  2. No missing data issues")
        print("  3. Cleaner training pipeline")
        print("\nTo use filtered version, copy:")
        print(f"  {filt_depth}")
        print(f"  {filt_proteins}")
        print(f"\nTo: {dataset_current}")
    else:
        print("\n[INFO] Filtered versions not found in protein_exposure/dataset/")
        print("Current dataset will need runtime filtering (already implemented in dataset_fixed.py)")
else:
    print("\n[INFO] protein_exposure/dataset/ not found")
    print("Using current dataset with runtime filtering")

print("\n" + "="*80)
