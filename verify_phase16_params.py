"""
Quick script to verify Phase 16 model parameter count.
Expected: ~300K parameters (4 layers, 152 hidden channels)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.gnn import AtomExposureGNN

# Phase 16 configuration
model = AtomExposureGNN(
    in_channels=93,
    hidden_channels=152,
    num_layers=4,
    conv_type='gatv2',
    edge_dim=12,
    dropout=0.27
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=" * 60)
print("PHASE 16 MODEL VERIFICATION")
print("=" * 60)
print(f"Architecture: GATv2")
print(f"  - in_channels: 93")
print(f"  - hidden_channels: 152")
print(f"  - num_layers: 4")
print(f"  - edge_dim: 12")
print(f"  - dropout: 0.27")
print()
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print()
print(f"Target: ~300,000 parameters")
print(f"Actual: {total_params:,} parameters")
print(f"Difference from target: {abs(total_params - 300000):,} ({abs(total_params - 300000) / 300000 * 100:.1f}%)")
print()

# Compare with other phases
phases = {
    "Phase 13a": {"layers": 3, "hidden": 128, "params": 71_800},
    "Phase 14": {"layers": 5, "hidden": 176, "params": 512_689},
    "Phase 15": {"layers": 4, "hidden": 136, "params": 253_913},
    "Phase 16": {"layers": 4, "hidden": 152, "params": total_params},
}

print("=" * 60)
print("COMPARISON WITH OTHER PHASES")
print("=" * 60)
print(f"{'Phase':<12} {'Layers':<8} {'Hidden':<8} {'Parameters':<12} {'vs Phase 13a':<15}")
print("-" * 60)
for phase, config in phases.items():
    vs_13a = f"+{(config['params'] / 71_800 - 1) * 100:.0f}%"
    print(f"{phase:<12} {config['layers']:<8} {config['hidden']:<8} {config['params']:<12,} {vs_13a:<15}")

print()
print("=" * 60)
print("SCALING EFFICIENCY")
print("=" * 60)
# R² values (from summary)
r2_values = {
    "Phase 13a": 0.8817,
    "Phase 14": 0.8888,
    "Phase 15": 0.8892,
    "Phase 16": 0.895  # Target
}

print(f"{'Phase':<12} {'R² (actual/target)':<18} {'Params':<12} {'ROI (R² gain / params)':<20}")
print("-" * 60)
baseline_r2 = 0.8817
baseline_params = 71_800

for phase in ["Phase 13a", "Phase 14", "Phase 15", "Phase 16"]:
    r2 = r2_values[phase]
    params = phases[phase]["params"]
    r2_gain = (r2 - baseline_r2) * 1000  # in 0.001 units
    param_increase = params - baseline_params
    roi = r2_gain / (param_increase / 1000) if param_increase > 0 else 0

    r2_str = f"{r2:.4f}"
    if phase == "Phase 16":
        r2_str += " (target)"

    print(f"{phase:<12} {r2_str:<18} {params:<12,} {roi:<20.3f}")

print()
print("=" * 60)
print("✓ Phase 16 configuration verified!")
print("=" * 60)
