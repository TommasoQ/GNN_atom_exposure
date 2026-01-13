"""
Phase 1: Feature Analysis & Selection
Exploration script to understand data structure and compute feature correlations
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append('..')

import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path

print("=" * 80)
print("PHASE 1: FEATURE ANALYSIS & SELECTION")
print("=" * 80)

# Step 1: Load and inspect depth_indexes.pkl structure
print("\n[Step 1] Loading depth_indexes.pkl...")
depth_path = Path('../../dataset/depth_indexes.pkl')

with open(depth_path, 'rb') as f:
    depth_indexes = pickle.load(f)

print(f"Type of depth_indexes: {type(depth_indexes)}")

if isinstance(depth_indexes, pd.DataFrame):
    print("\n[OK] depth_indexes is a DataFrame (as discovered in analysis)")
    print(f"Shape: {depth_indexes.shape}")
    print(f"Columns: {list(depth_indexes.columns)}")
    print(f"\nFirst 10 rows:")
    print(depth_indexes.head(10))

    print(f"\nUnique proteins: {depth_indexes['pdb_id'].nunique()}")
    print(f"Total atoms: {len(depth_indexes)}")

    # Check depth value statistics
    print(f"\nDepth value statistics:")
    print(depth_indexes['depth_index'].describe())

else:
    print(f"\nUnexpected structure: {type(depth_indexes)}")


# Step 2: Load protein list
print("\n" + "=" * 80)
print("[Step 2] Loading protein list...")
protein_df = pd.read_csv('../../dataset/protein_sample_5000.csv')
print(f"Total proteins in CSV: {len(protein_df)}")

# Check how many proteins have depth data
if isinstance(depth_indexes, pd.DataFrame):
    proteins_with_depth = set(depth_indexes['pdb_id'].unique())
    proteins_in_csv = set(protein_df['pdb_id'])

    proteins_with_both = proteins_in_csv & proteins_with_depth
    proteins_missing_depth = proteins_in_csv - proteins_with_depth

    print(f"Proteins with depth data: {len(proteins_with_both)}")
    print(f"Proteins missing depth data: {len(proteins_missing_depth)}")

    if len(proteins_missing_depth) > 0:
        print(f"\nFirst 10 proteins missing depth data:")
        print(list(proteins_missing_depth)[:10])


# Step 3: Load and analyze a sample protein
print("\n" + "=" * 80)
print("[Step 3] Analyzing sample protein...")

# Pick first protein that has depth data
if isinstance(depth_indexes, pd.DataFrame):
    sample_pdb = list(proteins_with_both)[0]
else:
    sample_pdb = protein_df['pdb_id'].iloc[0]

print(f"Sample protein: {sample_pdb}")

protein_dir = Path(f'../../dataset/sadic_data/{sample_pdb}')
if not protein_dir.exists():
    print(f"ERROR: Protein directory not found: {protein_dir}")
    sys.exit(1)

# Load nodes
nodes_path = protein_dir / f'{sample_pdb}__graphein__ATOM_nodes.csv'
nodes_df = pd.read_csv(nodes_path, index_col=0)

print(f"Number of atoms: {len(nodes_df)}")
print(f"Total columns: {len(nodes_df.columns)}")
print(f"\nColumn names:")
for i, col in enumerate(nodes_df.columns):
    print(f"  {i}: {col}")

# Identify feature columns (skip first 6 identifiers)
identifier_cols = ['original_index', 'chain_id', 'residue_name', 'residue_number', 'atom_type', 'element_symbol']
feature_cols = [col for col in nodes_df.columns if col not in identifier_cols]

print(f"\nFeature columns: {len(feature_cols)}")
print("Features:")
for i, col in enumerate(feature_cols):
    print(f"  {i}: {col}")


# Step 4: Match depth values to atoms
print("\n" + "=" * 80)
print("[Step 4] Matching depth values to atoms...")

if isinstance(depth_indexes, pd.DataFrame):
    # Get depth data for this protein
    protein_depth = depth_indexes[depth_indexes['pdb_id'] == sample_pdb]
    print(f"Depth entries for {sample_pdb}: {len(protein_depth)}")

    # Create dict for matching
    depth_dict = dict(zip(protein_depth['atom_name'], protein_depth['depth_index']))

    # Match to nodes
    matched_depths = []
    unmatched = []

    for idx, row in nodes_df.iterrows():
        atom_name = row['original_index']
        if atom_name in depth_dict:
            matched_depths.append(depth_dict[atom_name])
        else:
            unmatched.append(atom_name)
            matched_depths.append(np.nan)

    print(f"Matched atoms: {len(matched_depths) - len(unmatched)}")
    print(f"Unmatched atoms: {len(unmatched)}")

    if len(unmatched) > 0:
        print(f"First 5 unmatched: {unmatched[:5]}")

    # Add depth to nodes_df
    nodes_df['depth'] = matched_depths


# Step 5: Compute feature correlations with depth
print("\n" + "=" * 80)
print("[Step 5] Computing feature correlations...")

if 'depth' in nodes_df.columns:
    # Remove NaN depths
    nodes_with_depth = nodes_df[nodes_df['depth'].notna()].copy()
    print(f"Atoms with valid depth values: {len(nodes_with_depth)}")

    # Get numerical feature columns
    numerical_features = []
    for col in feature_cols:
        if nodes_df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
            numerical_features.append(col)

    print(f"Numerical features: {len(numerical_features)}")

    # Compute correlations
    correlations = {}
    for feat in numerical_features:
        try:
            corr = nodes_with_depth[feat].corr(nodes_with_depth['depth'])
            if not np.isnan(corr):
                correlations[feat] = corr
        except:
            pass

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"\nTop 20 features by absolute correlation with depth:")
    for i, (feat, corr) in enumerate(sorted_corr[:20], 1):
        print(f"  {i:2d}. {feat:40s}: {corr:+.4f}")

    print(f"\nBottom 20 features by absolute correlation:")
    for i, (feat, corr) in enumerate(sorted_corr[-20:], 1):
        print(f"  {i:2d}. {feat:40s}: {corr:+.4f}")

    # Save correlations for later use
    corr_df = pd.DataFrame(sorted_corr, columns=['feature', 'correlation'])
    corr_path = Path('phase1_feature_correlations.csv')
    corr_df.to_csv(corr_path, index=False)
    print(f"\n[OK] Saved correlations to: {corr_path}")


# Step 6: Analyze feature distributions
print("\n" + "=" * 80)
print("[Step 6] Analyzing feature distributions...")

print("\nFeature statistics (first 10 features):")
for feat in numerical_features[:10]:
    values = nodes_df[feat]
    print(f"\n{feat}:")
    print(f"  Mean: {values.mean():.4f}")
    print(f"  Std:  {values.std():.4f}")
    print(f"  Min:  {values.min():.4f}")
    print(f"  Max:  {values.max():.4f}")


# Step 7: Categorical features analysis
print("\n" + "=" * 80)
print("[Step 7] Analyzing categorical features...")

print("\nAtom types:")
print(nodes_df['atom_type'].value_counts().head(15))

print("\nElement types:")
print(nodes_df['element_symbol'].value_counts())

print("\nResidue types:")
print(nodes_df['residue_name'].value_counts().head(20))


# Summary
print("\n" + "=" * 80)
print("PHASE 1 EXPLORATION SUMMARY")
print("=" * 80)
print(f"[OK] Confirmed depth_indexes is DataFrame (not dict)")
print(f"[OK] Found {len(proteins_with_both)} proteins with labels (out of 5000)")
print(f"[OK] Identified {len(feature_cols)} total columns ({len(numerical_features)} numerical)")
print(f"[OK] Computed feature correlations with depth")
print(f"[OK] Identified categorical features (atom_type, element, residue)")
print(f"\nNext step: Create detailed feature analysis notebook")
print("=" * 80)
