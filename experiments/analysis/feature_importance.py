"""
Feature Importance Analysis using Permutation Importance

This script computes the importance of each feature by measuring how much
the model's R² score degrades when that feature is randomly shuffled.

Usage:
    python experiments/analysis/feature_importance.py [--checkpoint PATH]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset_fixed import ProteinAtomDataset
from src.models.gnn import AtomExposureGNN
from src.data.feature_engineering import (
    SELECTED_NUMERICAL_FEATURES,
    STANDARD_ATOM_TYPES,
    STANDARD_ELEMENTS,
    STANDARD_RESIDUES
)
from torch_geometric.loader import DataLoader


def get_feature_names():
    """Get the complete list of 93 feature names in order (WITH contact_count)."""
    names = []

    # 1. Numerical features (24)
    names.extend(SELECTED_NUMERICAL_FEATURES)

    # 2. Atom types (31)
    names.extend([f'atom_{cat}' for cat in STANDARD_ATOM_TYPES])

    # 3. Elements (5)
    names.extend([f'element_{cat}' for cat in STANDARD_ELEMENTS])

    # 4. Residues (21)
    names.extend([f'residue_{cat}' for cat in STANDARD_RESIDUES])

    # 5. Geometric features (8) - WITH contact_count
    geometric_names = [
        'geom_mean_dist', 'geom_min_dist', 'geom_max_dist', 'geom_std_dist',
        'geom_3rd_nearest_dist', 'geom_dist_to_center', 'geom_radial_position',
        'geom_contact_count_10A'
    ]
    names.extend(geometric_names)

    # 6. Backbone angles (4)
    backbone_names = ['backbone_sin_phi', 'backbone_cos_phi', 'backbone_sin_psi', 'backbone_cos_psi']
    names.extend(backbone_names)

    return names


def compute_r2(model, loader, device):
    """Compute R² score on a dataset."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_preds.append(out.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return r2


def compute_permutation_importance(model, test_dataset, device, n_repeats=3):
    """
    Compute permutation importance for each feature.

    Args:
        model: Trained model
        test_dataset: Test dataset
        device: torch device
        n_repeats: Number of times to repeat the permutation

    Returns:
        DataFrame with feature importances
    """
    # Create loader for baseline
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Compute baseline R²
    baseline_r2 = compute_r2(model, test_loader, device)
    print(f"Baseline R²: {baseline_r2:.4f}")

    feature_names = get_feature_names()
    n_features = len(feature_names)

    # Verify feature count matches
    sample = test_dataset[0]
    actual_features = sample.x.shape[1]
    if actual_features != n_features:
        print(f"Warning: Expected {n_features} features, got {actual_features}")
        # Adjust if needed
        n_features = actual_features
        if n_features > len(feature_names):
            feature_names.extend([f'unknown_{i}' for i in range(len(feature_names), n_features)])
        else:
            feature_names = feature_names[:n_features]

    importances = []
    importance_stds = []

    print(f"\nComputing importance for {n_features} features...")

    for feat_idx in tqdm(range(n_features), desc="Features"):
        repeat_scores = []

        for _ in range(n_repeats):
            # Create a copy of the dataset with permuted feature
            permuted_scores = []

            # Permute feature across all samples
            # We need to create a new loader each time with permuted data
            permuted_loader = create_permuted_loader(test_dataset, feat_idx, device)

            permuted_r2 = compute_r2_with_permuted_feature(
                model, test_dataset, feat_idx, device
            )
            repeat_scores.append(baseline_r2 - permuted_r2)

        importances.append(np.mean(repeat_scores))
        importance_stds.append(np.std(repeat_scores))

    # Create results DataFrame
    results = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'importance_std': importance_stds
    })

    # Sort by importance (descending)
    results = results.sort_values('importance', ascending=False).reset_index(drop=True)

    return results, baseline_r2


def create_permuted_loader(dataset, feat_idx, device):
    """Create a DataLoader with a specific feature permuted."""
    # This is a helper that's not used directly - see compute_r2_with_permuted_feature
    pass


def compute_r2_with_permuted_feature(model, dataset, feat_idx, device):
    """Compute R² with a specific feature permuted across all samples."""
    model.eval()
    all_preds = []
    all_targets = []

    # Collect all feature values for this feature across all proteins
    all_feature_values = []
    for data in dataset:
        all_feature_values.append(data.x[:, feat_idx].numpy())
    all_feature_values = np.concatenate(all_feature_values)

    # Shuffle the values
    np.random.shuffle(all_feature_values)

    # Process each protein with permuted feature
    value_idx = 0
    with torch.no_grad():
        for data in dataset:
            data = data.clone()
            n_nodes = data.x.shape[0]

            # Replace feature with permuted values
            permuted_values = all_feature_values[value_idx:value_idx + n_nodes]
            data.x[:, feat_idx] = torch.tensor(permuted_values, dtype=torch.float32)
            value_idx += n_nodes

            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            all_preds.append(out.cpu().numpy())
            all_targets.append(data.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return r2


def plot_importance(results, save_path, top_n=30):
    """Create bar plot of feature importances."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Take top N features
    top_results = results.head(top_n)

    # Create horizontal bar plot
    y_pos = np.arange(len(top_results))
    bars = ax.barh(y_pos, top_results['importance'],
                   xerr=top_results['importance_std'],
                   color='steelblue', alpha=0.8, capsize=3)

    # Highlight certain features
    for i, (idx, row) in enumerate(top_results.iterrows()):
        if 'contact_count' in row['feature']:
            bars[i].set_color('crimson')
        elif row['feature'].startswith('geom_'):
            bars[i].set_color('forestgreen')
        elif row['feature'].startswith('backbone_'):
            bars[i].set_color('orange')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_results['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance (R² drop when permuted)')
    ax.set_title(f'Top {top_n} Feature Importances (Permutation Importance)')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Numerical/Categorical'),
        Patch(facecolor='forestgreen', label='Geometric'),
        Patch(facecolor='orange', label='Backbone Angles'),
        Patch(facecolor='crimson', label='Contact Count'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved importance plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Feature Importance Analysis')
    parser.add_argument('--checkpoint', type=str,
                        default='experiments/checkpoints/phase15_globalpool/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--n-repeats', type=int, default=3,
                        help='Number of permutation repeats')
    parser.add_argument('--output-dir', type=str, default='experiments/analysis',
                        help='Output directory for results')
    args = parser.parse_args()

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = AtomExposureGNN(
        in_channels=93,              # 93 features (WITH contact_count)
        hidden_channels=136,
        num_layers=4,
        conv_type='gatv2',
        edge_dim=12,
        dropout=0.26,
        use_global_pool=True,
        global_pool_type='mean',
        global_pool_layers='every'
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load test dataset
    print("\nLoading test dataset...")
    # Get project root (2 levels up from this script)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_path = os.path.join(project_root, 'dataset')

    test_dataset = ProteinAtomDataset(
        root=dataset_path,
        split='test',
        feature_config={
            'use_reduced_features': False,
            'include_atom_type': True,
            'include_geometric': True,
            'use_aggregated': False,
            'include_backbone_angles': True,
        }
    )
    print(f"Test dataset: {len(test_dataset)} proteins")

    # Compute permutation importance
    print("\nComputing permutation importance...")
    results, baseline_r2 = compute_permutation_importance(
        model, test_dataset, device, n_repeats=args.n_repeats
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, 'feature_importance.csv')
    results.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    # Create plot
    plot_path = os.path.join(args.output_dir, 'feature_importance.png')
    plot_importance(results, plot_path)

    # Print top features
    print("\n" + "=" * 60)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("=" * 60)
    print(f"{'Rank':<6} {'Feature':<35} {'Importance':>12}")
    print("-" * 60)
    for i, (_, row) in enumerate(results.head(20).iterrows(), 1):
        print(f"{i:<6} {row['feature']:<35} {row['importance']:>12.4f}")

    print("\n" + "=" * 60)
    print("FEATURE GROUP SUMMARY")
    print("=" * 60)

    # Group importance by feature type
    groups = {
        'Geometric': results[results['feature'].str.startswith('geom_')]['importance'].sum(),
        'Backbone Angles': results[results['feature'].str.startswith('backbone_')]['importance'].sum(),
        'Numerical (biochemical)': results[results['feature'].isin(SELECTED_NUMERICAL_FEATURES)]['importance'].sum(),
        'Atom Types': results[results['feature'].str.startswith('atom_')]['importance'].sum(),
        'Elements': results[results['feature'].str.startswith('element_')]['importance'].sum(),
        'Residues': results[results['feature'].str.startswith('residue_')]['importance'].sum(),
    }

    for group, importance in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"{group:<25} {importance:>10.4f}")

    print(f"\nBaseline R²: {baseline_r2:.4f}")
    print(f"Total importance: {results['importance'].sum():.4f}")


if __name__ == '__main__':
    main()
