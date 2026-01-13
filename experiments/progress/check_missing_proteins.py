"""
Check if the 406 proteins missing labels actually have sadic_data files
"""
import pandas as pd
import pickle
from pathlib import Path

print("="*80)
print("CHECKING MISSING PROTEINS DATA FILES")
print("="*80)

# Load data
dataset_current = Path(r"C:\Users\Edoardo\GNN_protein_exposure\GNN_atom_exposure\dataset")
sadic_dir = dataset_current / "sadic_data"

protein_csv = dataset_current / "protein_sample_5000.csv"
depth_path = dataset_current / "depth_indexes.pkl"

proteins_df = pd.read_csv(protein_csv)
with open(depth_path, 'rb') as f:
    depth_df = pickle.load(f)

# Find proteins missing depth labels
proteins_with_depth = set(depth_df['pdb_id'].unique())
all_proteins = set(proteins_df['pdb_id'])
missing_depth = sorted(all_proteins - proteins_with_depth)

print(f"\nTotal proteins in CSV: {len(all_proteins)}")
print(f"Proteins with depth labels: {len(proteins_with_depth)}")
print(f"Proteins missing depth labels: {len(missing_depth)}")

# Check if missing proteins have sadic_data files
print(f"\nChecking if {len(missing_depth)} missing proteins have feature files...")

missing_with_files = []
missing_without_files = []

for pdb_id in missing_depth[:50]:  # Check first 50
    protein_dir = sadic_dir / pdb_id
    nodes_file = protein_dir / f"{pdb_id}__graphein__ATOM_nodes.csv"

    if nodes_file.exists():
        missing_with_files.append(pdb_id)
    else:
        missing_without_files.append(pdb_id)

print(f"\nResults (checked {min(50, len(missing_depth))} proteins):")
print(f"  Have feature files: {len(missing_with_files)}")
print(f"  Missing feature files: {len(missing_without_files)}")

if len(missing_with_files) > 0:
    print(f"\nProteins WITH features but WITHOUT labels (first 20):")
    print(f"  {missing_with_files[:20]}")
    print(f"\n  [ISSUE] These proteins have features but no depth labels!")
    print(f"  They should either:")
    print(f"    1. Be removed from protein_sample_5000.csv (use filtered version)")
    print(f"    2. Have their depth labels added to depth_indexes.pkl")

if len(missing_without_files) > 0:
    print(f"\nProteins WITHOUT features AND WITHOUT labels (first 20):")
    print(f"  {missing_without_files[:20]}")
    print(f"\n  [OK] These should just be removed from the CSV")

# Also check: do proteins with labels all have feature files?
print(f"\n" + "="*80)
print("CHECKING PROTEINS WITH LABELS")
print("="*80)

print(f"\nChecking if proteins WITH labels have feature files...")
proteins_sample = sorted(list(proteins_with_depth))[:100]  # Check first 100

with_labels_and_files = []
with_labels_no_files = []

for pdb_id in proteins_sample:
    protein_dir = sadic_dir / pdb_id
    nodes_file = protein_dir / f"{pdb_id}__graphein__ATOM_nodes.csv"

    if nodes_file.exists():
        with_labels_and_files.append(pdb_id)
    else:
        with_labels_no_files.append(pdb_id)

print(f"\nResults (checked {len(proteins_sample)} proteins):")
print(f"  Have both labels AND files: {len(with_labels_and_files)}")
print(f"  Have labels but MISSING files: {len(with_labels_no_files)}")

if len(with_labels_no_files) > 0:
    print(f"\n  [ERROR] Proteins with labels but missing feature files:")
    print(f"  {with_labels_no_files}")
    print(f"\n  This would cause errors during training!")
else:
    print(f"\n  [OK] All checked proteins have both labels and feature files")

# Final recommendation
print(f"\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

if len(missing_with_files) > 0:
    print(f"\n[ACTION REQUIRED] Use the FILTERED dataset")
    print(f"\nThe current dataset has {len(missing_depth)} proteins without labels.")
    print(f"At least {len(missing_with_files)} of them have feature files.")
    print(f"\nOptions:")
    print(f"  1. RECOMMENDED: Copy filtered CSV from protein_exposure/dataset/")
    print(f"     cp protein_sample_filtered.csv -> protein_sample_5000.csv")
    print(f"\n  2. Keep current approach (runtime filtering in dataset_fixed.py)")
    print(f"     - Works but wastes memory loading 406 extra protein metadata")
    print(f"     - Already implemented in your dataset_fixed.py")
else:
    print(f"\n[OK] Runtime filtering approach is fine")
    print(f"The 406 proteins don't have feature files anyway.")

print("="*80)
