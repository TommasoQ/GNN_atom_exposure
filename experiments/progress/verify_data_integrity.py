"""
Comprehensive data integrity verification
"""
import pandas as pd
import pickle
from pathlib import Path
import os

print("="*80)
print("COMPREHENSIVE DATA INTEGRITY VERIFICATION")
print("="*80)

dataset_current = Path(r"C:\Users\Edoardo\GNN_protein_exposure\GNN_atom_exposure\dataset")
sadic_dir = dataset_current / "sadic_data"

# Load current data
protein_csv = dataset_current / "protein_sample_5000.csv"
depth_path = dataset_current / "depth_indexes.pkl"

proteins_df = pd.read_csv(protein_csv)
with open(depth_path, 'rb') as f:
    depth_df = pickle.load(f)

print("\n" + "-"*80)
print("STEP 1: Categorize all 5000 proteins")
print("-"*80)

proteins_with_depth = set(depth_df['pdb_id'].unique())
all_proteins = set(proteins_df['pdb_id'])

# Check sadic_data for ALL proteins
has_features = set()
for pdb_id in all_proteins:
    nodes_file = sadic_dir / pdb_id / f"{pdb_id}__graphein__ATOM_nodes.csv"
    if nodes_file.exists():
        has_features.add(pdb_id)

# Categorize
cat_complete = proteins_with_depth & has_features  # Both labels and features
cat_only_labels = proteins_with_depth - has_features  # Labels but no features
cat_only_features = has_features - proteins_with_depth  # Features but no labels
cat_neither = all_proteins - proteins_with_depth - has_features  # Missing both

print(f"\nTotal proteins in CSV: {len(all_proteins)}")
print(f"\nCategories:")
print(f"  1. COMPLETE (labels + features):     {len(cat_complete):4d} proteins")
print(f"  2. Labels only (no features):        {len(cat_only_labels):4d} proteins")
print(f"  3. Features only (no labels):        {len(cat_only_features):4d} proteins")
print(f"  4. Neither labels nor features:      {len(cat_neither):4d} proteins")
print(f"  " + "-"*50)
print(f"  Total:                               {len(all_proteins):4d} proteins")

# Verify categories
print(f"\nVerification:")
print(f"  Cat 1 + Cat 2 = {len(cat_complete) + len(cat_only_labels)} (should equal proteins with depth: {len(proteins_with_depth)})")
print(f"  Cat 1 + Cat 3 = {len(cat_complete) + len(cat_only_features)} (should equal proteins with features: {len(has_features)})")

print("\n" + "-"*80)
print("STEP 2: Investigate problematic categories")
print("-"*80)

if len(cat_only_labels) > 0:
    print(f"\nCategory 2: {len(cat_only_labels)} proteins with LABELS but NO FEATURES")
    print(f"  [ERROR] These proteins cannot be used for training!")
    print(f"  First 10: {sorted(list(cat_only_labels))[:10]}")
    print(f"  Recommendation: Remove from depth_indexes.pkl")
else:
    print(f"\nCategory 2: [OK] No proteins with labels but missing features")

if len(cat_only_features) > 0:
    print(f"\nCategory 3: {len(cat_only_features)} proteins with FEATURES but NO LABELS")
    print(f"  [WARNING] These proteins have data but no targets!")
    print(f"  List: {sorted(list(cat_only_features))}")
    print(f"  Recommendation: Either add labels or remove from CSV")
else:
    print(f"\nCategory 3: [OK] No proteins with features but missing labels")

print("\n" + "-"*80)
print("STEP 3: Check data quality for complete proteins")
print("-"*80)

# Sample 20 complete proteins and verify their data
sample_complete = sorted(list(cat_complete))[:20]

print(f"\nChecking {len(sample_complete)} complete proteins...")

valid_count = 0
issues = []

for pdb_id in sample_complete:
    try:
        # Check nodes file
        nodes_file = sadic_dir / pdb_id / f"{pdb_id}__graphein__ATOM_nodes.csv"
        nodes_df_sample = pd.read_csv(nodes_file, index_col=0)

        # Check edges file
        edges_file = sadic_dir / pdb_id / f"{pdb_id}__graphein__ATOM_edges.csv"
        edges_exist = edges_file.exists()

        # Check depth matching
        protein_depth = depth_df[depth_df['pdb_id'] == pdb_id]
        depth_dict = dict(zip(protein_depth['atom_name'], protein_depth['depth_index']))

        matched = sum(1 for atom in nodes_df_sample['original_index'] if atom in depth_dict)
        match_pct = 100 * matched / len(nodes_df_sample)

        if match_pct >= 99:  # Allow 1% mismatch
            valid_count += 1
        else:
            issues.append(f"{pdb_id}: only {match_pct:.1f}% atoms matched")

    except Exception as e:
        issues.append(f"{pdb_id}: {str(e)}")

print(f"\nResults:")
print(f"  Valid proteins: {valid_count}/{len(sample_complete)}")
print(f"  Issues found: {len(issues)}")

if issues:
    print(f"\n  Issues:")
    for issue in issues:
        print(f"    - {issue}")
else:
    print(f"  [OK] All sampled proteins have valid data!")

print("\n" + "-"*80)
print("STEP 4: Check sadic_data directory completeness")
print("-"*80)

# Count total directories in sadic_data
sadic_dirs = [d for d in sadic_dir.iterdir() if d.is_dir()]
print(f"\nTotal directories in sadic_data: {len(sadic_dirs)}")
print(f"Proteins in CSV: {len(all_proteins)}")
print(f"Proteins with features (based on files): {len(has_features)}")

# Find directories not in CSV
dirs_not_in_csv = set(d.name for d in sadic_dirs) - all_proteins
if dirs_not_in_csv:
    print(f"\n[INFO] {len(dirs_not_in_csv)} directories in sadic_data NOT in CSV")
    print(f"  First 10: {sorted(list(dirs_not_in_csv))[:10]}")
    print(f"  These are extra proteins not included in the 5000 sample")

print("\n" + "="*80)
print("FINAL RECOMMENDATIONS")
print("="*80)

print(f"\nUsable proteins for training:")
print(f"  Complete proteins (labels + features): {len(cat_complete)}")

if len(cat_only_features) > 0 or len(cat_only_labels) > 0:
    print(f"\n[RECOMMENDATION 1] Use the filtered CSV from protein_exposure/dataset/")
    print(f"  This removes {len(cat_only_features) + len(cat_neither)} proteins without labels")
    print(f"  Command:")
    print(f"    cp C:\\Users\\Edoardo\\protein_exposure\\dataset\\protein_sample_filtered.csv \\")
    print(f"       C:\\Users\\Edoardo\\GNN_protein_exposure\\GNN_atom_exposure\\dataset\\protein_sample_5000.csv")

    if len(cat_only_labels) > 0:
        print(f"\n[RECOMMENDATION 2] Also clean up depth_indexes.pkl")
        print(f"  Remove {len(cat_only_labels)} proteins that have labels but no features")
        print(f"  These proteins: {sorted(list(cat_only_labels))[:20]}")
else:
    print(f"\n[OK] Current dataset_fixed.py runtime filtering is sufficient")
    print(f"  It will filter out the {len(all_proteins) - len(cat_complete)} incomplete proteins")

print(f"\n[INFO] Your dataset_fixed.py already handles this by:")
print(f"  1. Filtering proteins without depth labels (removes cat 3 & 4)")
print(f"  2. Checking file existence before loading (skips cat 2)")

print("\n" + "="*80)
