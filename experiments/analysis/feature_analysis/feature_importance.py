"""
Permutation Feature Importance Analysis Script

Purpose:
- Load best GIN model (Exp 3.7)
- Compute baseline performance on validation set
- For each feature, permute it and measure R² drop
- Rank features by importance (R² drop)
- Generate importance scores and visualizations

Usage:
    python experiments/analysis/feature_importance.py

Note: This will take ~1-2 hours to run (permuting 100 features)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error
from torch_geometric.loader import DataLoader

from src.data.dataset_fixed import ProteinAtomDataset
from src.models.gnn import AtomExposureGNN

print("=" * 80)
print("PERMUTATION FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Load best GIN model
print("\n1. Loading best GIN model (Exp 3.7)...")
model = AtomExposureGNN(
    in_channels=100,
    hidden_channels=96,
    num_layers=3,
    dropout=0.3,
    conv_type='gin'
).to(device)

# Try multiple possible checkpoint locations
possible_paths = [
    'experiments/checkpoints/gin_baseline/best_model.pt',
    'experiments/checkpoints/best_model.pt',
]

checkpoint_path = None
for path in possible_paths:
    if os.path.exists(path):
        checkpoint_path = path
        break

if checkpoint_path is None:
    print(f"\nERROR: Checkpoint not found. Tried:")
    for path in possible_paths:
        print(f"  - {path}")
    print("Please ensure you have trained the GIN baseline model (Exp 3.7)")
    sys.exit(1)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"   Model loaded from: {checkpoint_path}")
print(f"   Best validation R²: {checkpoint.get('best_val_r2', 'N/A')}")

# Load validation dataset
print("\n2. Loading validation dataset...")
# Note: root is the dataset directory
val_dataset = ProteinAtomDataset(root='dataset', split='val')
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
print(f"   Loaded {len(val_dataset)} proteins")

# Get feature names
print("\n3. Loading feature names...")
from src.data.feature_engineering import (
    SELECTED_NUMERICAL_FEATURES,
    STANDARD_ATOM_TYPES,
    STANDARD_ELEMENTS,
    STANDARD_RESIDUES
)

feature_names = (
    SELECTED_NUMERICAL_FEATURES +
    [f'atom_{a}' for a in STANDARD_ATOM_TYPES] +
    [f'element_{e}' for e in STANDARD_ELEMENTS] +
    [f'residue_{r}' for r in STANDARD_RESIDUES] +
    ['geom_mean_dist', 'geom_min_dist', 'geom_max_dist', 'geom_std_dist',
     'geom_nearest_dist', 'geom_3rd_nearest_dist',
     'geom_dist_to_center', 'geom_radial_position']
)

print(f"   Feature names loaded: {len(feature_names)}")
num_features = len(feature_names)

# Evaluation function
def evaluate(model, loader, device):
    """Evaluate model and return predictions and targets"""
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            predictions.append(out.cpu().numpy())
            targets.append(batch.y.cpu().numpy())

    return np.concatenate(predictions), np.concatenate(targets)

# Baseline performance
print("\n4. Computing baseline performance...")
preds_base, targets = evaluate(model, val_loader, device)
r2_base = r2_score(targets, preds_base)
mae_base = mean_absolute_error(targets, preds_base)

print(f"   Baseline R²: {r2_base:.4f}")
print(f"   Baseline MAE: {mae_base:.4f}")
print(f"   Total atoms evaluated: {len(targets):,}")

# Permutation importance
print("\n5. Computing permutation importance...")
print(f"   This will take approximately {num_features * 1.2:.0f} minutes ({num_features} features)")
print("   Progress will be saved incrementally to avoid data loss\n")

importances = []

for feat_idx in tqdm(range(num_features), desc="   Permuting features"):
    feature_name = feature_names[feat_idx]

    # Permute the feature by shuffling predictions (more efficient)
    # This avoids reloading the dataset from disk
    # Method: Shuffle the feature values across all atoms in the validation set

    # Collect all feature values for this feature index
    all_feature_values = []
    for batch in val_loader:
        all_feature_values.append(batch.x[:, feat_idx].clone())
    all_feature_values = torch.cat(all_feature_values)

    # Shuffle the feature values
    perm_global = torch.randperm(len(all_feature_values))
    shuffled_values = all_feature_values[perm_global]

    # Create modified batches with permuted feature
    predictions_perm = []
    idx_start = 0
    for batch in val_loader:
        batch_size = batch.x.size(0)
        batch_modified = batch.clone()

        # Replace the feature with shuffled values
        batch_modified.x[:, feat_idx] = shuffled_values[idx_start:idx_start + batch_size].to(device)
        idx_start += batch_size

        # Evaluate on modified batch
        batch_modified = batch_modified.to(device)
        with torch.no_grad():
            out = model(batch_modified.x, batch_modified.edge_index,
                       batch_modified.edge_attr, batch_modified.batch)
            predictions_perm.append(out.cpu().numpy())

    preds_perm = np.concatenate(predictions_perm)
    r2_perm = r2_score(targets, preds_perm)
    mae_perm = mean_absolute_error(targets, preds_perm)

    # Importance = drop in performance
    r2_drop = r2_base - r2_perm
    mae_increase = mae_perm - mae_base

    importances.append({
        'feature': feature_name,
        'importance_r2': r2_drop,  # Higher is more important
        'importance_mae': mae_increase,  # Higher MAE increase = more important
        'r2_with_permuted': r2_perm,
        'mae_with_permuted': mae_perm
    })

    # Save incrementally every 10 features
    if (feat_idx + 1) % 10 == 0:
        df_temp = pd.DataFrame(importances)
        df_temp.to_csv('experiments/analysis/feature_importance_partial.csv', index=False)

# Save final results
print("\n6. Saving results...")
df_importance = pd.DataFrame(importances).sort_values('importance_r2', ascending=False)
df_importance.to_csv('experiments/analysis/feature_importance_scores.csv', index=False)
print("   Saved: experiments/analysis/feature_importance_scores.csv")

# Remove partial file
if os.path.exists('experiments/analysis/feature_importance_partial.csv'):
    os.remove('experiments/analysis/feature_importance_partial.csv')

# Display top features
print("\n" + "=" * 80)
print("TOP 30 MOST IMPORTANT FEATURES (by R² drop)")
print("=" * 80)
print(df_importance.head(30).to_string(index=False))

# Display least important features
print("\n" + "=" * 80)
print("BOTTOM 20 LEAST IMPORTANT FEATURES (candidates for removal)")
print("=" * 80)
print(df_importance.tail(20).to_string(index=False))

# Visualizations
print("\n7. Generating visualizations...")

# Feature importance bar plot (top 30)
print("   Creating feature importance plot...")
top_30 = df_importance.head(30)

plt.figure(figsize=(12, 10))
plt.barh(range(len(top_30)), top_30['importance_r2'].values)
plt.yticks(range(len(top_30)), top_30['feature'].values)
plt.xlabel('Importance (R² drop when permuted)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 30 Most Important Features', fontsize=14, pad=20)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('experiments/analysis/feature_importance_top30.png', dpi=150, bbox_inches='tight')
print("   Saved: experiments/analysis/feature_importance_top30.png")
plt.close()

# Distribution of importance scores
print("   Creating importance distribution plot...")
importance_values = df_importance['importance_r2'].values

plt.figure(figsize=(10, 6))
plt.hist(importance_values, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Importance (R² drop)', fontsize=12)
plt.ylabel('Number of Features', fontsize=12)
plt.title('Distribution of Feature Importances', fontsize=14)
plt.axvline(x=importance_values.mean(), color='red', linestyle='--',
            label=f'Mean: {importance_values.mean():.6f}')
plt.axvline(x=np.median(importance_values), color='orange', linestyle='--',
            label=f'Median: {np.median(importance_values):.6f}')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('experiments/analysis/importance_distribution.png', dpi=150, bbox_inches='tight')
print("   Saved: experiments/analysis/importance_distribution.png")
plt.close()

# Cumulative importance plot
print("   Creating cumulative importance plot...")
cumulative = np.cumsum(df_importance['importance_r2'].values)
cumulative_pct = 100 * cumulative / cumulative[-1]

plt.figure(figsize=(12, 6))
plt.plot(range(1, len(cumulative_pct) + 1), cumulative_pct, linewidth=2)
plt.axhline(y=80, color='red', linestyle='--', label='80% of total importance')
plt.axhline(y=90, color='orange', linestyle='--', label='90% of total importance')
plt.xlabel('Number of Features (ranked by importance)', fontsize=12)
plt.ylabel('Cumulative Importance (%)', fontsize=12)
plt.title('Cumulative Feature Importance', fontsize=14)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('experiments/analysis/cumulative_importance.png', dpi=150, bbox_inches='tight')
print("   Saved: experiments/analysis/cumulative_importance.png")
plt.close()

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"\nBaseline performance:")
print(f"  R²: {r2_base:.4f}")
print(f"  MAE: {mae_base:.4f}")

print(f"\nImportance statistics:")
print(f"  Total features: {num_features}")
print(f"  Mean importance: {importance_values.mean():.6f}")
print(f"  Median importance: {np.median(importance_values):.6f}")
print(f"  Max importance: {importance_values.max():.6f}")
print(f"  Min importance: {importance_values.min():.6f}")

# Identify low-importance features
threshold_low = 0.001  # R² drop < 0.001
low_importance = df_importance[df_importance['importance_r2'] < threshold_low]
print(f"\nFeatures with importance < {threshold_low} (candidates for removal):")
print(f"  Count: {len(low_importance)}")

if len(low_importance) > 0:
    print(f"\n  Low-importance features:")
    for _, row in low_importance.iterrows():
        print(f"    - {row['feature']}: {row['importance_r2']:.6f}")

# Identify high-importance features
threshold_high = 0.01  # R² drop > 0.01
high_importance = df_importance[df_importance['importance_r2'] > threshold_high]
print(f"\nFeatures with importance > {threshold_high} (critical features):")
print(f"  Count: {len(high_importance)}")

# Find how many features needed for 80% and 90% importance
total_importance = importance_values.sum()
cumsum = np.cumsum(importance_values)
n_80 = np.argmax(cumsum >= 0.8 * total_importance) + 1
n_90 = np.argmax(cumsum >= 0.9 * total_importance) + 1

print(f"\nCumulative importance analysis:")
print(f"  Top {n_80} features capture 80% of total importance")
print(f"  Top {n_90} features capture 90% of total importance")
print(f"\n  Recommendation: Consider keeping top {n_80}-{n_90} features")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nOutput files created:")
print("  - experiments/analysis/feature_importance_scores.csv")
print("  - experiments/analysis/feature_importance_top30.png")
print("  - experiments/analysis/importance_distribution.png")
print("  - experiments/analysis/cumulative_importance.png")
print("\nNext steps:")
print("  1. Review feature_importance_scores.csv")
print("  2. Cross-reference with correlation analysis")
print("  3. Select features to keep (top 70-80 features recommended)")
print("  4. Update feature_engineering.py with selected features")
print("  5. Retrain GIN with reduced feature set")
print("=" * 80)
