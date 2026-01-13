"""
Feature Correlation Analysis Script

Purpose:
- Compute correlation matrix for all 100 features
- Identify highly correlated feature pairs (redundant features)
- Compute correlation of each feature with target (exposure depth)
- Generate visualizations and reports

Usage:
    python experiments/analysis/feature_correlation_analysis.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.data.dataset_fixed import ProteinAtomDataset

print("=" * 80)
print("FEATURE CORRELATION ANALYSIS")
print("=" * 80)

# Load validation dataset
print("\n1. Loading validation dataset...")
# Note: root is the dataset directory
dataset = ProteinAtomDataset(root='dataset', split='val')
print(f"   Loaded {len(dataset)} proteins")

# Extract all features and targets
print("\n2. Extracting features and targets from all proteins...")
all_features = []
all_targets = []

for data in tqdm(dataset, desc="   Processing"):
    all_features.append(data.x.cpu().numpy())
    all_targets.append(data.y.cpu().numpy())

features = np.vstack(all_features)
targets = np.concatenate(all_targets)

print(f"   Total atoms: {features.shape[0]:,}")
print(f"   Total features: {features.shape[1]}")

# Get feature names from feature_engineering.py
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
assert len(feature_names) == features.shape[1], "Feature name count mismatch!"

# Create DataFrame
print("\n4. Creating feature DataFrame...")
df = pd.DataFrame(features, columns=feature_names)
print(f"   DataFrame shape: {df.shape}")

# Compute correlation matrix (this may take a while)
print("\n5. Computing correlation matrix (this may take 2-3 minutes)...")
corr_matrix = df.corr()
print("   Correlation matrix computed!")

# Save correlation matrix
print("\n6. Saving correlation matrix...")
os.makedirs('experiments/analysis', exist_ok=True)
corr_matrix.to_csv('experiments/analysis/feature_correlation_matrix.csv')
print("   Saved: experiments/analysis/feature_correlation_matrix.csv")

# Compute correlation with target
print("\n7. Computing correlation with target (exposure depth)...")
target_corr = df.corrwith(pd.Series(targets))
target_corr_df = pd.DataFrame({
    'feature': target_corr.index,
    'correlation_with_target': target_corr.values
}).sort_values('correlation_with_target', key=abs, ascending=False)

target_corr_df.to_csv('experiments/analysis/feature_target_correlation.csv', index=False)
print("   Saved: experiments/analysis/feature_target_correlation.csv")

print("\n   Top 20 features correlated with target:")
print(target_corr_df.head(20).to_string(index=False))

# Find highly correlated feature pairs (|r| > 0.8)
print("\n8. Finding highly correlated feature pairs (|r| > 0.8)...")
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.8:
            high_corr_pairs.append({
                'feature1': feature_names[i],
                'feature2': feature_names[j],
                'correlation': corr_val
            })

high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', key=abs, ascending=False)
high_corr_df.to_csv('experiments/analysis/high_correlation_pairs.csv', index=False)

print(f"   Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.8)")
print("   Saved: experiments/analysis/high_correlation_pairs.csv")

if len(high_corr_pairs) > 0:
    print("\n   Top 20 most correlated pairs:")
    print(high_corr_df.head(20).to_string(index=False))

# Visualizations
print("\n9. Generating visualizations...")

# Focus on numerical features only (first 35 features) for heatmap
num_numerical = len(SELECTED_NUMERICAL_FEATURES)
print(f"   Creating heatmap for {num_numerical} numerical features...")

plt.figure(figsize=(20, 16))
sns.heatmap(
    corr_matrix.iloc[:num_numerical, :num_numerical],
    cmap='coolwarm',
    center=0,
    vmin=-1, vmax=1,
    xticklabels=feature_names[:num_numerical],
    yticklabels=feature_names[:num_numerical],
    cbar_kws={'label': 'Correlation'},
    square=True
)
plt.title('Numerical Feature Correlation Matrix', fontsize=16, pad=20)
plt.xticks(rotation=90, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('experiments/analysis/numerical_feature_correlation.png', dpi=150, bbox_inches='tight')
print("   Saved: experiments/analysis/numerical_feature_correlation.png")
plt.close()

# Feature-target correlation bar plot (top 30)
print("   Creating feature-target correlation plot...")
top_30 = target_corr_df.head(30)

plt.figure(figsize=(12, 10))
plt.barh(range(len(top_30)), top_30['correlation_with_target'].values)
plt.yticks(range(len(top_30)), top_30['feature'].values)
plt.xlabel('Correlation with Exposure Depth', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 30 Features Correlated with Exposure Depth', fontsize=14, pad=20)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('experiments/analysis/feature_target_correlation.png', dpi=150, bbox_inches='tight')
print("   Saved: experiments/analysis/feature_target_correlation.png")
plt.close()

# Distribution of absolute correlations
print("   Creating correlation distribution plot...")
abs_corrs = target_corr_df['correlation_with_target'].abs()

plt.figure(figsize=(10, 6))
plt.hist(abs_corrs, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Absolute Correlation with Target', fontsize=12)
plt.ylabel('Number of Features', fontsize=12)
plt.title('Distribution of Feature-Target Correlations', fontsize=14)
plt.axvline(x=abs_corrs.median(), color='red', linestyle='--',
            label=f'Median: {abs_corrs.median():.3f}')
plt.axvline(x=abs_corrs.mean(), color='orange', linestyle='--',
            label=f'Mean: {abs_corrs.mean():.3f}')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('experiments/analysis/correlation_distribution.png', dpi=150, bbox_inches='tight')
print("   Saved: experiments/analysis/correlation_distribution.png")
plt.close()

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nTotal features analyzed: {features.shape[1]}")
print(f"Total atoms: {features.shape[0]:,}")
print(f"\nHighly correlated pairs (|r| > 0.8): {len(high_corr_pairs)}")
print(f"\nFeature-target correlation statistics:")
print(f"  Mean absolute correlation: {abs_corrs.mean():.4f}")
print(f"  Median absolute correlation: {abs_corrs.median():.4f}")
print(f"  Max absolute correlation: {abs_corrs.max():.4f}")
print(f"  Features with |r| > 0.1: {(abs_corrs > 0.1).sum()}")
print(f"  Features with |r| > 0.2: {(abs_corrs > 0.2).sum()}")
print(f"  Features with |r| > 0.3: {(abs_corrs > 0.3).sum()}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nOutput files created:")
print("  - experiments/analysis/feature_correlation_matrix.csv")
print("  - experiments/analysis/feature_target_correlation.csv")
print("  - experiments/analysis/high_correlation_pairs.csv")
print("  - experiments/analysis/numerical_feature_correlation.png")
print("  - experiments/analysis/feature_target_correlation.png")
print("  - experiments/analysis/correlation_distribution.png")
print("\nNext step: Review the outputs and identify redundant features to drop.")
print("=" * 80)
