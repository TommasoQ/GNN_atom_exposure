"""
Preprocess depth_indexes.pkl to create a faster-loading dict version.
This is a one-time conversion to speed up dataset initialization.
"""

import pickle
import pandas as pd
from pathlib import Path
import time

print("="*80)
print("PREPROCESSING depth_indexes.pkl")
print("="*80)

# Paths
root_dir = Path(__file__).parent.parent.parent
depth_df_path = root_dir / 'dataset' / 'depth_indexes.pkl'
depth_dict_path = root_dir / 'dataset' / 'depth_indexes_dict.pkl'

print(f"\nInput:  {depth_df_path}")
print(f"Output: {depth_dict_path}")

# Load DataFrame
print("\nLoading DataFrame...")
start = time.time()
with open(depth_df_path, 'rb') as f:
    depth_indexes_df = pickle.load(f)
load_time = time.time() - start
print(f"  Loaded in {load_time:.2f}s")
print(f"  Type: {type(depth_indexes_df)}")
print(f"  Shape: {depth_indexes_df.shape}")

# Convert to nested dict
print("\nConverting to nested dict: {pdb_id: {atom_name: depth_value}}")
start = time.time()

depth_dict = {}
for pdb_id, group in depth_indexes_df.groupby('pdb_id'):
    depth_dict[pdb_id] = dict(zip(group['atom_name'], group['depth_index']))

convert_time = time.time() - start
print(f"  Converted in {convert_time:.2f}s")
print(f"  Number of proteins: {len(depth_dict)}")
print(f"  Total atoms: {sum(len(v) for v in depth_dict.values()):,}")

# Save dict version
print("\nSaving dict version...")
start = time.time()
with open(depth_dict_path, 'wb') as f:
    pickle.dump(depth_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
save_time = time.time() - start
print(f"  Saved in {save_time:.2f}s")

# Test loading dict version
print("\nTesting dict loading speed...")
start = time.time()
with open(depth_dict_path, 'rb') as f:
    test_dict = pickle.load(f)
load_dict_time = time.time() - start
print(f"  Dict loaded in {load_dict_time:.2f}s")
print(f"  Speedup: {load_time / load_dict_time:.1f}x faster than DataFrame")

print("\n" + "="*80)
print("PREPROCESSING COMPLETE")
print("="*80)
print(f"\nSummary:")
print(f"  DataFrame load time: {load_time:.2f}s")
print(f"  Dict load time: {load_dict_time:.2f}s")
print(f"  Conversion time: {convert_time:.2f}s")
print(f"  Total proteins: {len(depth_dict)}")
print(f"\nDataset will now use: {depth_dict_path.name}")
print("="*80)
