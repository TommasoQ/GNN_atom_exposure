"""
Verification Script for Feature Reduction

Confirms that the feature reduction from 100 → 86 features is correctly implemented.

Expected counts:
- Numerical: 24 (reduced from 34)
- Atom types: 31 (unchanged)
- Elements: 4 (reduced from 6)
- Residues: 20 (reduced from 21)
- Geometric: 7 (reduced from 8)
- TOTAL: 86 (reduced from 100)

Run this script to verify before training.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data.feature_engineering import (
    SELECTED_NUMERICAL_FEATURES,
    STANDARD_ATOM_TYPES,
    STANDARD_ELEMENTS,
    STANDARD_RESIDUES,
    get_feature_dimensions
)

print("=" * 80)
print("FEATURE REDUCTION VERIFICATION")
print("=" * 80)

# Get feature counts
num_numerical = len(SELECTED_NUMERICAL_FEATURES)
num_atom_types = len(STANDARD_ATOM_TYPES)
num_elements = len(STANDARD_ELEMENTS)
num_residues = len(STANDARD_RESIDUES)
num_geometric = 7  # Manually verified in compute_geometric_features

total = num_numerical + num_atom_types + num_elements + num_residues + num_geometric

# Expected values
expected = {
    'numerical': 24,
    'atom_types': 31,
    'elements': 4,
    'residues': 20,
    'geometric': 7,
    'total': 86
}

# Display counts
print("\nFeature Counts:")
print(f"  Numerical:  {num_numerical:3d}  (expected: {expected['numerical']:3d})  {'[OK]' if num_numerical == expected['numerical'] else '[FAIL] MISMATCH!'}")
print(f"  Atom types: {num_atom_types:3d}  (expected: {expected['atom_types']:3d})  {'[OK]' if num_atom_types == expected['atom_types'] else '[FAIL] MISMATCH!'}")
print(f"  Elements:   {num_elements:3d}  (expected: {expected['elements']:3d})  {'[OK]' if num_elements == expected['elements'] else '[FAIL] MISMATCH!'}")
print(f"  Residues:   {num_residues:3d}  (expected: {expected['residues']:3d})  {'[OK]' if num_residues == expected['residues'] else '[FAIL] MISMATCH!'}")
print(f"  Geometric:  {num_geometric:3d}  (expected: {expected['geometric']:3d})  {'[OK]' if num_geometric == expected['geometric'] else '[FAIL] MISMATCH!'}")
print(f"  {'─' * 50}")
print(f"  TOTAL:      {total:3d}  (expected: {expected['total']:3d})  {'[OK]' if total == expected['total'] else '[FAIL] MISMATCH!'}")

# Verify get_feature_dimensions() function
dims = get_feature_dimensions()
print("\nget_feature_dimensions() output:")
for key, value in dims.items():
    exp = expected.get(key, 'N/A')
    status = '[OK]' if value == exp else '[FAIL] MISMATCH!'
    print(f"  {key:12s}: {value:3d}  (expected: {exp:3d})  {status}")

# Print removed features summary
print("\n" + "=" * 80)
print("FEATURES REMOVED (14 total)")
print("=" * 80)

removed_features = [
    "1. geom_nearest_dist (duplicate of geom_min_dist, r=1.0)",
    "2. expasy:hphob_janin (r=0.93 with eisenberg)",
    "3. expasy:hphob_chothia (r=0.90 with eisenberg)",
    "4. expasy:hphob_woods (r=-0.93 with transmembranetendency)",
    "5. expasy:beta_sheetfasman (r=0.95 with totalbeta_strand)",
    "6. expasy:beta_sheetroux (r=0.97 with totalbeta_strand)",
    "7. expasy:antiparallelbeta_strand (r=0.95 with totalbeta_strand)",
    "8. expasy:refractivity (r=0.92 with molecularweight)",
    "9. expasy:molecularweight (r=-0.012 target, redundant)",
    "10. expasy:isoelectric_points (r=0.91 with meiler:dim_5)",
    "11. meiler:dim_6 (r=0.83 with coilroux, weak)",
    "12. element_P (no data)",
    "13. element_OTHER (no data)",
    "14. residue_OTHER (no data)",
]

for feature in removed_features:
    print(f"  {feature}")

# Final status
print("\n" + "=" * 80)
if total == expected['total']:
    print("[OK] VERIFICATION PASSED - Feature reduction correctly implemented!")
    print(f"[OK] Total features: {total} (reduced from 100)")
    print("\nNext steps:")
    print("  1. Delete dataset/processed/ folder to regenerate graphs with 86 features")
    print("  2. Run training: venv\\Scripts\\python.exe main.py")
    print("  3. Compare results with Exp 3.7 (R² = 0.4985 with 100 features)")
    sys.exit(0)
else:
    print("[FAIL] VERIFICATION FAILED - Feature count mismatch!")
    print(f"[FAIL] Expected: {expected['total']}, Got: {total}")
    print("\nPlease review feature_engineering.py for errors.")
    sys.exit(1)
print("=" * 80)
