"""
Deep analysis of the 38 problematic proteins.
"""
import pandas as pd
import os
import pickle

with open('../../dataset/depth_indexes_dict.pkl', 'rb') as f:
    depth_indexes = pickle.load(f)

base = '../../dataset/sadic_data'

# The 38 proteins with ratio > 3.8
problem_proteins = []
for pdb_id in depth_indexes.keys():
    raw_path = os.path.join(base, pdb_id, f'{pdb_id}__graphein__raw_pdb_df.csv')
    nodes_path = os.path.join(base, pdb_id, f'{pdb_id}__graphein__ATOM_nodes.csv')
    
    if os.path.exists(raw_path) and os.path.exists(nodes_path):
        raw_df = pd.read_csv(raw_path, index_col=0)
        nodes_df = pd.read_csv(nodes_path, index_col=0)
        ratio = len(raw_df) / len(nodes_df) if len(nodes_df) > 0 else 0
        if ratio > 3.8:
            problem_proteins.append(pdb_id)

print(f"Analyzing {len(problem_proteins)} problematic proteins...")

# Analyze causes
nmr_multi_model = []
heavy_altloc = []
hetatm_dominant = []
other = []

for pdb_id in problem_proteins:
    raw_path = os.path.join(base, pdb_id, f'{pdb_id}__graphein__raw_pdb_df.csv')
    raw_df = pd.read_csv(raw_path, index_col=0)
    
    analysis = {'pdb': pdb_id, 'raw_count': len(raw_df)}
    
    # Check for multiple models (NMR structures)
    if 'model_id' in raw_df.columns:
        n_models = raw_df['model_id'].nunique()
        analysis['n_models'] = n_models
        if n_models > 1:
            nmr_multi_model.append(pdb_id)
    
    # Check for HETATM dominance
    if 'record_name' in raw_df.columns:
        record_counts = raw_df['record_name'].value_counts()
        analysis['record_types'] = record_counts.to_dict()
        if 'HETATM' in record_counts and record_counts.get('HETATM', 0) > record_counts.get('ATOM', 0):
            hetatm_dominant.append(pdb_id)
    
    # Check for heavy alternate conformations
    if 'alt_loc' in raw_df.columns:
        non_empty = raw_df['alt_loc'].notna() & (raw_df['alt_loc'] != '') & (raw_df['alt_loc'] != ' ')
        analysis['altloc_count'] = non_empty.sum()
        if non_empty.sum() > len(raw_df) * 0.3:
            heavy_altloc.append(pdb_id)

print(f"\n=== ROOT CAUSES ===")
print(f"NMR multi-model structures: {len(nmr_multi_model)}")
if nmr_multi_model:
    print(f"  {nmr_multi_model[:10]}")
    
print(f"HETATM-dominant: {len(hetatm_dominant)}")
if hetatm_dominant:
    print(f"  {hetatm_dominant[:10]}")
    
print(f"Heavy altloc (>30% atoms): {len(heavy_altloc)}")
if heavy_altloc:
    print(f"  {heavy_altloc[:10]}")

# Check overlap
nmr_set = set(nmr_multi_model)
hetatm_set = set(hetatm_dominant)
altloc_set = set(heavy_altloc)
problem_set = set(problem_proteins)

explained = nmr_set | hetatm_set | altloc_set
unexplained = problem_set - explained

print(f"\nTotal explained by these causes: {len(explained)}")
print(f"Unexplained: {len(unexplained)}")
if unexplained:
    print(f"  {list(unexplained)}")

# Deep dive on worst case
print("\n" + "="*60)
print("DETAILED ANALYSIS: 3mbs (worst case)")
print("="*60)
pdb = '3mbs'
raw_df = pd.read_csv(os.path.join(base, pdb, f'{pdb}__graphein__raw_pdb_df.csv'), index_col=0)
nodes_df = pd.read_csv(os.path.join(base, pdb, f'{pdb}__graphein__ATOM_nodes.csv'), index_col=0)

print(f"Raw atoms: {len(raw_df)}")
print(f"Graphein atoms: {len(nodes_df)}")

if 'model_id' in raw_df.columns:
    print(f"\nModels: {raw_df['model_id'].value_counts().to_dict()}")

if 'record_name' in raw_df.columns:
    print(f"Record types: {raw_df['record_name'].value_counts().to_dict()}")

if 'chain_id' in raw_df.columns:
    print(f"Chains: {raw_df['chain_id'].value_counts().to_dict()}")
