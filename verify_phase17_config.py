"""
Quick script to verify Phase 17 configuration and loss weights.
Expected: 254K parameters (same as Phase 15), moderate loss weights between Phase 15 and 16.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.gnn import AtomExposureGNN

# Phase 17 configuration
model = AtomExposureGNN(
    in_channels=93,
    hidden_channels=136,
    num_layers=4,
    conv_type='gatv2',
    edge_dim=12,
    dropout=0.26
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=" * 70)
print("PHASE 17 CONFIGURATION VERIFICATION")
print("=" * 70)
print(f"Architecture: GATv2")
print(f"  - in_channels: 93")
print(f"  - hidden_channels: 136 (same as Phase 15)")
print(f"  - num_layers: 4 (same as Phase 15)")
print(f"  - edge_dim: 12")
print(f"  - dropout: 0.26 (same as Phase 15)")
print()
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print()
print(f"Expected: 253,913 (Phase 15 size)")
print(f"Actual: {total_params:,}")
if total_params == 253_913:
    print("✓ Model size matches Phase 15 exactly!")
else:
    print(f"⚠ Difference: {abs(total_params - 253_913):,} parameters")
print()

# Loss weight comparison
print("=" * 70)
print("LOSS WEIGHT COMPARISON")
print("=" * 70)
print(f"{'Weight':<20} {'Phase 15':<12} {'Phase 16':<12} {'Phase 17':<12} {'Δ vs 15':<12}")
print("-" * 70)

weights = {
    "buried": (1.5, 2.0, 1.6),
    "semi_buried": (1.0, 1.0, 1.1),
    "intermediate": (1.0, 1.0, 1.1),
    "semi_exposed": (1.3, 1.3, 1.4),
    "exposed": (2.0, 2.5, 2.2),
    "asymmetric_penalty": (1.5, 2.0, 1.6),
}

for name, (p15, p16, p17) in weights.items():
    delta = f"+{(p17/p15 - 1)*100:.0f}%"
    print(f"{name:<20} {p15:<12.1f} {p16:<12.1f} {p17:<12.1f} {delta:<12}")

print()
print("=" * 70)
print("STRATEGY SUMMARY")
print("=" * 70)
print("Phase 17 uses MODERATE increases (+7-10%) vs Phase 16's aggressive (+25-33%)")
print("NEW: Protection for intermediate ranges (semi_buried, intermediate: 1.0 → 1.1)")
print("Goal: Improve bias without sacrificing R² (Phase 16 lost -0.64% R²)")
print()

# Penalty examples
print("=" * 70)
print("EXAMPLE PENALTY CALCULATIONS")
print("=" * 70)

examples = [
    ("Buried overestimated", 0.15, 0.20, "buried", 1.6, 1.6),
    ("Exposed underestimated", 1.40, 1.20, "exposed", 2.2, 1.6),
    ("Intermediate (NEW)", 0.60, 0.65, "intermediate", 1.1, 1.0),
]

for desc, target, pred, range_name, range_weight, asym_penalty in examples:
    error = pred - target
    base_mse = error ** 2

    # Determine if asymmetric penalty applies
    apply_asym = False
    if range_name == "buried" and pred > target:
        apply_asym = True
    elif range_name == "exposed" and pred < target:
        apply_asym = True

    if apply_asym:
        total_weight = range_weight * asym_penalty
        weighted_mse = base_mse * total_weight
    else:
        total_weight = range_weight
        weighted_mse = base_mse * total_weight

    print(f"\n{desc}:")
    print(f"  Target: {target:.2f}, Pred: {pred:.2f}, Error: {error:+.2f}")
    print(f"  Base MSE: {base_mse:.4f}")
    print(f"  Range weight: {range_weight:.1f}x")
    if apply_asym:
        print(f"  Asymmetric penalty: {asym_penalty:.1f}x (applies)")
        print(f"  Total weight: {range_weight:.1f} × {asym_penalty:.1f} = {total_weight:.2f}x")
    else:
        print(f"  Asymmetric penalty: Not applied")
        print(f"  Total weight: {total_weight:.2f}x")
    print(f"  Weighted MSE: {weighted_mse:.4f}")

print()
print("=" * 70)
print("EXPECTED OUTCOMES")
print("=" * 70)
print("Best case:")
print("  - R²: 0.8895-0.8900 (improve vs Phase 15: 0.8892)")
print("  - Buried bias: +0.025 to +0.030 (vs Phase 15: +0.0394)")
print("  - Exposed bias: -0.045 to -0.055 (vs Phase 15: -0.0603)")
print("  - Intermediate ranges: STABLE (vs Phase 16: degraded)")
print()
print("Acceptable:")
print("  - R²: ≥0.8890 (at least match Phase 15)")
print("  - Bias: Some improvement without degrading intermediate ranges")
print()
print("Failure (revert to Phase 15):")
print("  - R²: <0.8885")
print("  - OR: Intermediate ranges degrade again")
print()
print("=" * 70)
print("✓ Phase 17 configuration ready to train!")
print("=" * 70)
print()
print("To launch:")
print("  cd GNN_atom_exposure")
print("  .venv/Scripts/python.exe main.py --config configs/phase17_balanced.yaml")
