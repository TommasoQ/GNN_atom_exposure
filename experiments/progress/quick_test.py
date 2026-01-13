"""
Quick test of dataset_fixed.py without full validation
"""
import sys
sys.path.insert(0, '../../src')

import torch
from data.dataset_fixed import ProteinAtomDataset
from pathlib import Path

print("="*80)
print("QUICK DATASET TEST")
print("="*80)

root_dir = Path(__file__).parent.parent.parent

print(f"\nRoot directory: {root_dir}")

try:
    print("\nInitializing dataset (without normalization)...")
    dataset = ProteinAtomDataset(
        root=str(root_dir),
        split='train',
        normalize_features=False
    )

    print(f"\nDataset size: {len(dataset)}")

    print(f"\nLoading first sample...")
    data = dataset[0]

    print(f"\nFirst sample info:")
    print(f"  Number of nodes: {data.num_nodes}")
    print(f"  Number of edges: {data.num_edges}")
    print(f"  Feature dimension: {data.x.shape[1] if data.x is not None else 'None'}")
    print(f"  Target shape: {data.y.shape if data.y is not None else 'None'}")

    if data.y is not None:
        print(f"\nTarget statistics:")
        print(f"  Mean: {data.y.mean():.4f}")
        print(f"  Std: {data.y.std():.4f}")
        print(f"  Min: {data.y.min():.4f}")
        print(f"  Max: {data.y.max():.4f}")

    # Check for NaN/Inf
    has_nan = torch.isnan(data.x).any() if data.x is not None else False
    has_inf = torch.isinf(data.x).any() if data.x is not None else False

    print(f"\nData quality:")
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")

    if not has_nan and not has_inf:
        print("\n[OK] Dataset test PASSED!")
    else:
        print("\n[ERROR] Dataset has NaN or Inf values!")

except Exception as e:
    print(f"\n[ERROR] Dataset test FAILED:")
    print(f"  {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
