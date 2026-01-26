"""
Verify that analysis scripts are correctly configured for Phase 15 noCC.
"""

import re
import sys

print("=" * 70)
print("VERIFICATION: Analysis Scripts for Phase 15 noCC")
print("=" * 70)

scripts = [
    'experiments/analysis/attention_analysis.py',
    'experiments/analysis/edge_importance.py',
    'experiments/analysis/feature_importance.py'
]

expected_config = {
    'in_channels': 92,
    'hidden_channels': 136,
    'num_layers': 4,
    'dropout': 0.26,
    'checkpoint_default': 'phase15_noCC'
}

print("\nChecking configuration in analysis scripts...")
print("-" * 70)

all_correct = True

for script_path in scripts:
    script_name = script_path.split('/')[-1]
    print(f"\n{script_name}:")

    with open(script_path, 'r') as f:
        content = f.read()

    # Check each expected parameter
    checks = []

    # in_channels
    in_channels_match = re.search(r'in_channels=(\d+)', content)
    if in_channels_match:
        value = int(in_channels_match.group(1))
        status = "OK" if value == expected_config['in_channels'] else f"ERROR (found {value})"
        checks.append(('in_channels', expected_config['in_channels'], value, status))
    else:
        checks.append(('in_channels', expected_config['in_channels'], 'NOT FOUND', 'ERROR'))
        all_correct = False

    # hidden_channels
    hidden_match = re.search(r'hidden_channels=(\d+)', content)
    if hidden_match:
        value = int(hidden_match.group(1))
        status = "OK" if value == expected_config['hidden_channels'] else f"ERROR (found {value})"
        checks.append(('hidden_channels', expected_config['hidden_channels'], value, status))
    else:
        checks.append(('hidden_channels', expected_config['hidden_channels'], 'NOT FOUND', 'ERROR'))
        all_correct = False

    # num_layers
    layers_match = re.search(r'num_layers=(\d+)', content)
    if layers_match:
        value = int(layers_match.group(1))
        status = "OK" if value == expected_config['num_layers'] else f"ERROR (found {value})"
        checks.append(('num_layers', expected_config['num_layers'], value, status))
    else:
        checks.append(('num_layers', expected_config['num_layers'], 'NOT FOUND', 'ERROR'))
        all_correct = False

    # dropout
    dropout_match = re.search(r'dropout=(0\.\d+)', content)
    if dropout_match:
        value = float(dropout_match.group(1))
        status = "OK" if abs(value - expected_config['dropout']) < 0.01 else f"ERROR (found {value})"
        checks.append(('dropout', expected_config['dropout'], value, status))
    else:
        checks.append(('dropout', expected_config['dropout'], 'NOT FOUND', 'ERROR'))
        all_correct = False

    # checkpoint default
    checkpoint_match = re.search(r"default='experiments/checkpoints/([^/]+)/", content)
    if checkpoint_match:
        value = checkpoint_match.group(1)
        status = "OK" if value == expected_config['checkpoint_default'] else f"ERROR (found {value})"
        checks.append(('checkpoint', expected_config['checkpoint_default'], value, status))
    else:
        checks.append(('checkpoint', expected_config['checkpoint_default'], 'NOT FOUND', 'ERROR'))
        all_correct = False

    # Print checks
    for param, expected, actual, status in checks:
        print(f"  {param:<20}: expected {expected:<10} -> {status}")
        if 'ERROR' in status:
            all_correct = False

# Special check for attention_analysis: layer count
print(f"\nSpecial checks:")
print("-" * 70)

with open('experiments/analysis/attention_analysis.py', 'r') as f:
    attention_content = f.read()

# Check for range(4) instead of range(5)
range_4_count = attention_content.count('range(4)')
range_5_count = attention_content.count('range(5)')

print(f"attention_analysis.py:")
print(f"  range(4) occurrences: {range_4_count} (should be > 0)")
print(f"  range(5) occurrences: {range_5_count} (should be 0)")

if range_5_count > 0:
    print(f"  ERROR: Still has range(5) - should be range(4) for 4 layers")
    all_correct = False
else:
    print(f"  OK: Correctly configured for 4 layers")

# Check for 4-layer dictionary initialization
dict_4_layers = '{0: [], 1: [], 2: [], 3: []}' in attention_content
dict_5_layers = '{0: [], 1: [], 2: [], 3: [], 4: []}' in attention_content

print(f"  4-layer dict init: {'found' if dict_4_layers else 'NOT FOUND'}")
print(f"  5-layer dict init: {'found' if dict_5_layers else 'NOT FOUND'}")

if dict_5_layers:
    print(f"  ERROR: Still has 5-layer initialization")
    all_correct = False
elif dict_4_layers:
    print(f"  OK: Correctly uses 4-layer initialization")

# Summary
print("\n" + "=" * 70)
if all_correct:
    print("VERIFICATION PASSED")
    print("=" * 70)
    print("All analysis scripts are correctly configured for Phase 15 noCC!")
    print("\nConfiguration:")
    print("  - 92 features (no contact_count)")
    print("  - 136 hidden channels")
    print("  - 4 layers")
    print("  - dropout 0.26")
    print("  - Default checkpoint: phase15_noCC")
    print("\nTo run analysis:")
    print("  cd GNN_atom_exposure")
    print("  .venv/Scripts/python.exe experiments/analysis/attention_analysis.py")
    print("  .venv/Scripts/python.exe experiments/analysis/edge_importance.py")
    print("  .venv/Scripts/python.exe experiments/analysis/feature_importance.py")
    print("=" * 70)
else:
    print("VERIFICATION FAILED")
    print("=" * 70)
    print("Some scripts are not correctly configured!")
    print("Please check the errors above and fix the scripts.")
    print("=" * 70)
    sys.exit(1)
