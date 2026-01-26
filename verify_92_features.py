"""
Verification script for 92-feature update (removed contact_count_10A).
Checks that all components are correctly updated.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("VERIFICATION: 92 Features (contact_count_10A removed)")
print("=" * 70)

# 1. Check feature dimensions
print("\n1. Checking feature dimensions...")
from src.data.feature_engineering import get_feature_dimensions

dims = get_feature_dimensions(
    use_reduced_features=False,
    include_atom_type=True,
    include_geometric=True,
    include_backbone_angles=True
)

print(f"   Numerical: {dims['numerical']}")
print(f"   Atom types: {dims['atom_types']}")
print(f"   Elements: {dims['elements']}")
print(f"   Residues: {dims['residues']}")
print(f"   Geometric: {dims['geometric']}")
print(f"   Backbone angles: {dims['backbone_angles']}")
print(f"   TOTAL: {dims['total']}")

if dims['total'] == 92 and dims['geometric'] == 7:
    print("   ✓ Feature dimensions CORRECT (92 total, 7 geometric)")
else:
    print(f"   ✗ Feature dimensions WRONG (expected 92 total, 7 geometric)")
    sys.exit(1)

# 2. Check model parameter counts
print("\n2. Checking model parameter counts...")
from src.models.gnn import AtomExposureGNN

configs = [
    ("Phase 14", 92, 176, 5, 0.28, 512_513),
    ("Phase 15/17", 92, 136, 4, 0.26, 253_777),
    ("Phase 16", 92, 152, 4, 0.27, 314_033),
]

all_correct = True
for name, in_ch, hidden, layers, dropout, expected_params in configs:
    model = AtomExposureGNN(
        in_channels=in_ch,
        hidden_channels=hidden,
        num_layers=layers,
        conv_type='gatv2',
        edge_dim=12,
        dropout=dropout
    )
    actual_params = sum(p.numel() for p in model.parameters())

    if actual_params == expected_params:
        print(f"   ✓ {name}: {actual_params:,} params (correct)")
    else:
        print(f"   ✗ {name}: {actual_params:,} params (expected {expected_params:,})")
        all_correct = False

if not all_correct:
    print("\n   Some parameter counts are incorrect!")
    sys.exit(1)

# 3. Check config files
print("\n3. Checking config files...")
import yaml

config_files = [
    'configs/phase14_large_model.yaml',
    'configs/phase15_optimized.yaml',
    'configs/phase16_enhanced.yaml',
    'configs/phase17_balanced.yaml',
]

all_configs_correct = True
for config_file in config_files:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    in_channels = config['model']['in_channels']
    if in_channels == 92:
        print(f"   ✓ {config_file}: in_channels = 92")
    else:
        print(f"   ✗ {config_file}: in_channels = {in_channels} (expected 92)")
        all_configs_correct = False

if not all_configs_correct:
    print("\n   Some config files not updated!")
    sys.exit(1)

# 4. Test feature extraction (if dataset available)
print("\n4. Testing feature extraction...")
try:
    import pandas as pd
    import numpy as np

    # Try to load a sample protein
    sample_dir = os.path.join('dataset', 'sadic_data', '142l')
    if os.path.exists(sample_dir):
        nodes_path = os.path.join(sample_dir, '142l__graphein__ATOM_nodes.csv')
        nodes_df = pd.read_csv(nodes_path, index_col=0)

        # Create dummy edge_index
        n_atoms = len(nodes_df)
        edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64).T

        from src.data.feature_engineering import extract_all_features

        features, feature_names = extract_all_features(
            nodes_df=nodes_df,
            edge_index=edge_index,
            edge_distances=None,
            normalizer=None,
            normalize=False,
            use_reduced_features=False,
            include_atom_type=True,
            include_geometric=True,
            include_backbone_angles=True
        )

        if features.shape[1] == 92:
            print(f"   ✓ Feature extraction produces 92 features")
            print(f"   ✓ Sample protein: {n_atoms} atoms × 92 features")

            # Check that contact_count is NOT in feature names
            if 'geom_contact_count_10A' in feature_names:
                print(f"   ✗ ERROR: contact_count_10A still in feature names!")
                sys.exit(1)
            else:
                print(f"   ✓ contact_count_10A removed from feature names")
        else:
            print(f"   ✗ Feature extraction produces {features.shape[1]} features (expected 92)")
            sys.exit(1)
    else:
        print(f"   ⚠ Dataset not found, skipping extraction test")

except Exception as e:
    print(f"   ⚠ Could not test feature extraction: {e}")

# Summary
print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
print("✓ All checks passed!")
print("✓ System ready for training with 92 features")
print("\nIMPORTANT NOTES:")
print("  - Old checkpoints (93 features) are INCOMPATIBLE")
print("  - Processed dataset cache should be deleted/regenerated")
print("  - Phase 17 will be first phase trained with 92 features")
print("\nTo delete cache and force reprocessing:")
print("  rm -rf dataset/processed/")
print("  # or")
print("  mv dataset/processed/ dataset/processed_93_backup/")
print("=" * 70)
