"""
Check for atom mismatches between Graphein and depth_indexes.
This investigates the reported 38 protein mismatch issue.
"""
import pickle
import pandas as pd
import os

# Load depth indexes
with open('../../dataset/depth_indexes_dict.pkl', 'rb') as f:
    depth_indexes = pickle.load(f)

sadic_base = '../../dataset/sadic_data'
mismatches = []
perfect_matches = 0

# Check ALL proteins for atom name mismatches
for pdb_id in list(depth_indexes.keys()):
    nodes_path = os.path.join(sadic_base, pdb_id, f'{pdb_id}__graphein__ATOM_nodes.csv')
    if os.path.exists(nodes_path):
        nodes_df = pd.read_csv(nodes_path, index_col=0)
        graphein_atoms = set(nodes_df['original_index'].values)
        depth_atoms = set(depth_indexes[pdb_id].keys())
        
        only_in_graphein = graphein_atoms - depth_atoms
        only_in_depth = depth_atoms - graphein_atoms
        
        if only_in_graphein or only_in_depth:
            mismatches.append({
                'pdb': pdb_id,
                'only_graphein': len(only_in_graphein),
                'only_depth': len(only_in_depth),
                'sample_graphein': list(only_in_graphein)[:5] if only_in_graphein else [],
                'sample_depth': list(only_in_depth)[:5] if only_in_depth else []
            })
        else:
            perfect_matches += 1

print(f"Total proteins checked: {len(depth_indexes)}")
print(f"Perfect matches: {perfect_matches}")
print(f"Proteins with atom mismatches: {len(mismatches)}")

if mismatches:
    print(f"\n{'='*60}")
    print(f"MISMATCHED PROTEINS ({len(mismatches)})")
    print('='*60)
    
    for m in mismatches[:20]:
        print(f"\n{m['pdb']}:")
        print(f"  Only in Graphein: {m['only_graphein']} atoms")
        print(f"  Only in Depth: {m['only_depth']} atoms")
        if m['sample_graphein']:
            print(f"  Graphein extras: {m['sample_graphein']}")
        if m['sample_depth']:
            print(f"  Depth extras: {m['sample_depth']}")
    
    if len(mismatches) > 20:
        print(f"\n... and {len(mismatches) - 20} more proteins with mismatches")
    
    # Save full list
    print("\n\nAll mismatched proteins:")
    for m in mismatches:
        print(f"  {m['pdb']}")
