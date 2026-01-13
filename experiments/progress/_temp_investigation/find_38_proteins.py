"""
Find the 38 proteins with mismatch - colleague's tip investigation.
"""
import pandas as pd
import os
import pickle

with open('../../dataset/depth_indexes_dict.pkl', 'rb') as f:
    depth_indexes = pickle.load(f)

base = '../../dataset/sadic_data'

# Collect all protein stats
stats = []

for pdb_id in list(depth_indexes.keys()):
    raw_path = os.path.join(base, pdb_id, f'{pdb_id}__graphein__raw_pdb_df.csv')
    nodes_path = os.path.join(base, pdb_id, f'{pdb_id}__graphein__ATOM_nodes.csv')
    
    if not os.path.exists(raw_path) or not os.path.exists(nodes_path):
        continue
    
    raw_df = pd.read_csv(raw_path, index_col=0)
    nodes_df = pd.read_csv(nodes_path, index_col=0)
    
    # Check altloc
    has_altloc = False
    altloc_count = 0
    if 'alt_loc' in raw_df.columns:
        non_empty = raw_df['alt_loc'].notna() & (raw_df['alt_loc'] != '') & (raw_df['alt_loc'] != ' ')
        has_altloc = non_empty.any()
        altloc_count = non_empty.sum()
    
    raw_count = len(raw_df)
    graphein_count = len(nodes_df)
    depth_count = len(depth_indexes[pdb_id])
    
    stats.append({
        'pdb': pdb_id,
        'raw': raw_count,
        'graphein': graphein_count,
        'depth': depth_count,
        'has_altloc': has_altloc,
        'altloc_atoms': altloc_count,
        'ratio': raw_count / graphein_count if graphein_count > 0 else 0
    })

# Look for 38
print("Looking for patterns that give 38 proteins...")

# Different criteria
criteria_results = {
    'altloc with >10% atoms affected': len([s for s in stats if s['has_altloc'] and s['altloc_atoms'] > s['graphein'] * 0.1]),
    'altloc with >20% atoms affected': len([s for s in stats if s['has_altloc'] and s['altloc_atoms'] > s['graphein'] * 0.2]),
    'altloc with >30% atoms affected': len([s for s in stats if s['has_altloc'] and s['altloc_atoms'] > s['graphein'] * 0.3]),
    'altloc with >40% atoms affected': len([s for s in stats if s['has_altloc'] and s['altloc_atoms'] > s['graphein'] * 0.4]),
    'altloc with >50% atoms affected': len([s for s in stats if s['has_altloc'] and s['altloc_atoms'] > s['graphein'] * 0.5]),
    'raw/graphein ratio > 1.5': len([s for s in stats if s['ratio'] > 1.5]),
    'raw/graphein ratio > 2.0': len([s for s in stats if s['ratio'] > 2.0]),
    'raw/graphein ratio > 2.5': len([s for s in stats if s['ratio'] > 2.5]),
    'raw/graphein ratio > 3.0': len([s for s in stats if s['ratio'] > 3.0]),
    'raw/graphein ratio > 4.0': len([s for s in stats if s['ratio'] > 4.0]),
    'raw/graphein ratio > 5.0': len([s for s in stats if s['ratio'] > 5.0]),
}

for criterion, count in criteria_results.items():
    marker = "*** MATCH! ***" if count == 38 else ""
    print(f"  {criterion}: {count} {marker}")

# Check specific thresholds around 38
print("\nFine-tuning ratio thresholds:")
for i in range(10, 60, 1):
    threshold = i / 10
    count = len([s for s in stats if s['ratio'] > threshold])
    if 35 <= count <= 41:
        print(f"  ratio > {threshold:.1f}: {count} proteins")

# Also check for proteins where graphein has significantly fewer atoms
print("\nProteins where Graphein lost >50% of raw atoms:")
lost_half = [s for s in stats if s['ratio'] > 2.0]
print(f"  Count: {len(lost_half)}")
if len(lost_half) == 38:
    print("  *** THIS MATCHES 38! ***")
    for s in sorted(lost_half, key=lambda x: x['ratio'], reverse=True):
        print(f"    {s['pdb']}: raw={s['raw']}, graphein={s['graphein']}, ratio={s['ratio']:.2f}")
