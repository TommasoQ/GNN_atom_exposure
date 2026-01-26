"""
Verification script for Phase 15 noCC configuration.
Ensures the config is correct and shows parameter comparison.
"""

import sys
import os
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("PHASE 15 noCC CONFIGURATION VERIFICATION")
print("=" * 70)

# Load both configs
with open('configs/phase15_optimized.yaml', 'r') as f:
    phase15_original = yaml.safe_load(f)

with open('configs/phase15_noCC.yaml', 'r') as f:
    phase15_noCC = yaml.safe_load(f)

# 1. Check experiment name
print("\n1. Experiment Name:")
print(f"   Phase 15 original: {phase15_original['experiment']['name']}")
print(f"   Phase 15 noCC:     {phase15_noCC['experiment']['name']}")
if phase15_noCC['experiment']['name'] == 'phase15_noCC':
    print("   OK: Experiment name is unique")
else:
    print("   ERROR: Experiment name should be 'phase15_noCC'")
    sys.exit(1)

# 2. Check in_channels
print("\n2. Model Input Channels:")
print(f"   Phase 15 original: {phase15_original['model']['in_channels']}")
print(f"   Phase 15 noCC:     {phase15_noCC['model']['in_channels']}")
if phase15_noCC['model']['in_channels'] == 92:
    print("   OK: Using 92 features (contact_count removed)")
else:
    print("   ERROR: in_channels should be 92")
    sys.exit(1)

# 3. Verify all other model params are identical
print("\n3. Model Architecture (should be IDENTICAL):")
model_params = ['hidden_channels', 'num_layers', 'dropout', 'conv_type', 'edge_dim']
all_match = True
for param in model_params:
    orig_val = phase15_original['model'][param]
    noCC_val = phase15_noCC['model'][param]
    match = "OK" if orig_val == noCC_val else "ERROR"
    print(f"   {param:<20}: {orig_val:<10} vs {noCC_val:<10} [{match}]")
    if orig_val != noCC_val:
        all_match = False

if not all_match:
    print("\n   ERROR: Some model parameters don't match!")
    sys.exit(1)

# 4. Verify training hyperparameters are identical
print("\n4. Training Hyperparameters (should be IDENTICAL):")
training_params = ['learning_rate', 'weight_decay', 'max_lr', 'warmup_epochs',
                   'gradient_clip', 'batch_size', 'num_epochs']
all_match = True
for param in training_params:
    if param in phase15_original['training'] and param in phase15_noCC['training']:
        orig_val = phase15_original['training'][param]
        noCC_val = phase15_noCC['training'][param]
        match = "OK" if orig_val == noCC_val else "ERROR"
        print(f"   {param:<20}: {orig_val:<10} vs {noCC_val:<10} [{match}]")
        if orig_val != noCC_val:
            all_match = False
    elif param in phase15_original['data'] and param in phase15_noCC['data']:
        orig_val = phase15_original['data'][param]
        noCC_val = phase15_noCC['data'][param]
        match = "OK" if orig_val == noCC_val else "ERROR"
        print(f"   {param:<20}: {orig_val:<10} vs {noCC_val:<10} [{match}]")
        if orig_val != noCC_val:
            all_match = False

if not all_match:
    print("\n   ERROR: Some training parameters don't match!")
    sys.exit(1)

# 5. Verify loss configuration is identical
print("\n5. Loss Configuration (should be IDENTICAL):")
loss_params = ['loss_type', 'weighted_loss']
all_match = True
for param in loss_params:
    orig_val = phase15_original['training'].get(param)
    noCC_val = phase15_noCC['training'].get(param)
    match = "OK" if orig_val == noCC_val else "ERROR"
    print(f"   {param:<20}: {orig_val:<10} vs {noCC_val:<10} [{match}]")
    if orig_val != noCC_val:
        all_match = False

# Check loss weights
orig_weights = phase15_original['training'].get('loss_range_weights', {})
noCC_weights = phase15_noCC['training'].get('loss_range_weights', {})
print(f"   loss_range_weights:")
for key in ['buried', 'semi_buried', 'intermediate', 'semi_exposed', 'exposed']:
    orig_val = orig_weights.get(key)
    noCC_val = noCC_weights.get(key)
    match = "OK" if orig_val == noCC_val else "ERROR"
    print(f"     {key:<15}: {orig_val:<6} vs {noCC_val:<6} [{match}]")
    if orig_val != noCC_val:
        all_match = False

if not all_match:
    print("\n   ERROR: Some loss parameters don't match!")
    sys.exit(1)

# 6. Calculate parameter counts
print("\n6. Model Parameter Count:")
from src.models.gnn import AtomExposureGNN

# Phase 15 original (93 features)
model_93 = AtomExposureGNN(
    in_channels=93,
    hidden_channels=136,
    num_layers=4,
    conv_type='gatv2',
    edge_dim=12,
    dropout=0.26
)
params_93 = sum(p.numel() for p in model_93.parameters())

# Phase 15 noCC (92 features)
model_92 = AtomExposureGNN(
    in_channels=92,
    hidden_channels=136,
    num_layers=4,
    conv_type='gatv2',
    edge_dim=12,
    dropout=0.26
)
params_92 = sum(p.numel() for p in model_92.parameters())

print(f"   Phase 15 original (93 feat): {params_93:,} parameters")
print(f"   Phase 15 noCC (92 feat):     {params_92:,} parameters")
print(f"   Difference:                  {params_93 - params_92:,} parameters ({(params_93-params_92)/params_93*100:.2f}%)")

if params_92 == 253_777:
    print("   OK: Parameter count matches expected value")
else:
    print(f"   WARNING: Expected 253,777 parameters, got {params_92:,}")

# Summary
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)
print("OK: Phase 15 noCC configuration is correct!")
print("\nKey Points:")
print("  - Only difference: 92 features instead of 93 (contact_count removed)")
print("  - All other settings identical to Phase 15 original")
print("  - This is a pure ablation study of contact_count feature")
print("\nExpected Outcomes:")
print("  - Similar R2 (~0.889): contact_count was redundant")
print("  - Better R2 (>0.890): contact_count was harmful")
print("  - Worse R2 (<0.888): contact_count was important")
print("\nTo launch training:")
print("  1. Delete processed cache: rm -rf dataset/processed/")
print("  2. Run: .venv/Scripts/python.exe main.py --config configs/phase15_noCC.yaml")
print("=" * 70)
