"""
Comprehensive check for dataset integrity issues.
Investigates potential Graphein vs PDB alternate conformation mismatches.
"""
import pandas as pd
import os
import pickle
from collections import defaultdict

print("Loading depth_indexes...")
with open('../../dataset/depth_indexes_dict.pkl', 'rb') as f:
    depth_indexes = pickle.load(f)

base = '../../dataset/sadic_data'
proteins_with_altloc = []
proteins_with_atom_diff = []

print(f"Checking {len(depth_indexes)} proteins...")

for i, pdb_id in enumerate(list(depth_indexes.keys())):
    if i % 500 == 0:
        print(f"  Progress: {i}/{len(depth_indexes)}")
    
    raw_path = os.path.join(base, pdb_id, f'{pdb_id}__graphein__raw_pdb_df.csv')
    nodes_path = os.path.join(base, pdb_id, f'{pdb_id}__graphein__ATOM_nodes.csv')
    
    if not os.path.exists(raw_path) or not os.path.exists(nodes_path):
        continue
    
    raw_df = pd.read_csv(raw_path, index_col=0)
    nodes_df = pd.read_csv(nodes_path, index_col=0)
    
    # 1. Check for alternate conformations
    if 'alt_loc' in raw_df.columns:
        non_empty = raw_df['alt_loc'].notna() & (raw_df['alt_loc'] != '') & (raw_df['alt_loc'] != ' ')
        if non_empty.any():
            alt_counts = raw_df[non_empty]['alt_loc'].value_counts()
            proteins_with_altloc.append({
                'pdb': pdb_id,
                'altloc_counts': alt_counts.to_dict(),
                'total_atoms_raw': len(raw_df),
                'total_atoms_graphein': len(nodes_df)
            })
    
    # 2. Check if raw_pdb has more atoms than graphein extracted
    raw_atom_count = len(raw_df[raw_df['record_name'] == 'ATOM']) if 'record_name' in raw_df.columns else len(raw_df)
    graphein_atom_count = len(nodes_df)
    
    if raw_atom_count != graphein_atom_count:
        proteins_with_atom_diff.append({
            'pdb': pdb_id,
            'raw_atoms': raw_atom_count,
            'graphein_atoms': graphein_atom_count,
            'diff': raw_atom_count - graphein_atom_count
        })

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

print(f"\n1. ALTERNATE CONFORMATIONS (altloc)")
print(f"   Proteins with altloc entries: {len(proteins_with_altloc)}")
if proteins_with_altloc:
    print("\n   Examples:")
    for p in proteins_with_altloc[:10]:
        print(f"   {p['pdb']}: altlocs={p['altloc_counts']}, raw={p['total_atoms_raw']}, graphein={p['total_atoms_graphein']}")

print(f"\n2. RAW vs GRAPHEIN ATOM COUNT DIFFERENCES")
print(f"   Proteins with different counts: {len(proteins_with_atom_diff)}")
if proteins_with_atom_diff:
    # Sort by difference
    sorted_diff = sorted(proteins_with_atom_diff, key=lambda x: abs(x['diff']), reverse=True)
    print("\n   Top differences:")
    for p in sorted_diff[:20]:
        print(f"   {p['pdb']}: raw={p['raw_atoms']}, graphein={p['graphein_atoms']}, diff={p['diff']:+d}")

# Check if the 38 number matches anything
if len(proteins_with_altloc) == 38:
    print("\n*** FOUND: 38 proteins with altloc - matches colleague's tip! ***")
elif len(proteins_with_atom_diff) == 38:
    print("\n*** FOUND: 38 proteins with atom count diff - matches colleague's tip! ***")
