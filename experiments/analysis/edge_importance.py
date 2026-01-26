"""
Edge Feature Importance Analysis using Zero-Out Method

This script computes the importance of each edge feature by measuring how much
the model's R² score degrades when that edge feature is zeroed out during inference.

Unlike permutation importance for node features, we use zero-out because:
1. Permuting edge features can break graph structure consistency
2. Zero-out is simpler and provides a clear ablation signal

Usage:
    python experiments/analysis/edge_importance.py [--checkpoint PATH]
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
from torch_geometric.loader import DataLoader


# Edge feature names in order (12 total)
EDGE_FEATURE_NAMES = [
    'bond_covalent',      # 0: one-hot
    'bond_peptide',       # 1: one-hot
    'bond_hydrophobic',   # 2: one-hot
    'bond_aromatic',      # 3: one-hot
    'bond_hbond',         # 4: one-hot
    'bond_ionic',         # 5: one-hot
    'bond_ring',          # 6: one-hot
    'distance',           # 7: numerical
    'bond_length',        # 8: numerical
    'normalized_dist',    # 9: distance / 10.0
    'relative_dist',      # 10: distance - 3.8
    'in_radius',          # 11: 1 if distance < 8.0A
]


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


def compute_r2_with_all_edges_zeroed(model, dataset, device):
    """Compute R² with ALL edge features zeroed out - sanity check."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in dataset:
            # Create zeroed edge attributes
            edge_attr_zeroed = torch.zeros_like(data.edge_attr)

            # Move to device
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = edge_attr_zeroed.to(device)
            batch = data.batch.to(device) if data.batch is not None else None
            y = data.y.to(device)

            out = model(x, edge_index, edge_attr, batch)
            all_preds.append(out.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return r2


def compute_r2_with_zeroed_edge_feature(model, dataset, edge_feat_idx, device, debug=False):
    """Compute R² with a specific edge feature zeroed out."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, data in enumerate(dataset):
            # Clone the data to avoid modifying original
            edge_attr_modified = data.edge_attr.clone()

            # Debug: check original values before zeroing
            if debug and i == 0:
                print(f"  Original edge_attr[0, {edge_feat_idx}] = {edge_attr_modified[0, edge_feat_idx].item():.4f}")
                print(f"  Original edge_attr mean for feat {edge_feat_idx} = {edge_attr_modified[:, edge_feat_idx].mean().item():.4f}")

            # Zero out the specific edge feature
            edge_attr_modified[:, edge_feat_idx] = 0.0

            # Debug: verify zeroing worked
            if debug and i == 0:
                print(f"  After zeroing: edge_attr[0, {edge_feat_idx}] = {edge_attr_modified[0, edge_feat_idx].item():.4f}")
                print(f"  After zeroing: mean = {edge_attr_modified[:, edge_feat_idx].mean().item():.4f}")

            # Move to device
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = edge_attr_modified.to(device)
            batch = data.batch.to(device) if data.batch is not None else None
            y = data.y.to(device)

            out = model(x, edge_index, edge_attr, batch)
            all_preds.append(out.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return r2


def compute_edge_importance(model, test_dataset, device, n_repeats=1):
    """
    Compute importance for each edge feature using zero-out method.

    Args:
        model: Trained model
        test_dataset: Test dataset
        device: torch device
        n_repeats: Number of repeats (for zero-out, 1 is enough)

    Returns:
        DataFrame with edge feature importances
    """
    # Create loader for baseline
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Compute baseline R²
    baseline_r2 = compute_r2(model, test_loader, device)
    print(f"Baseline R²: {baseline_r2:.4f}")

    # Print edge feature statistics
    print("\n[DEBUG] Edge feature statistics (first protein):")
    sample_edge_attr = test_dataset[0].edge_attr
    for i, name in enumerate(EDGE_FEATURE_NAMES):
        feat_vals = sample_edge_attr[:, i]
        non_zero = (feat_vals != 0).float().mean().item() * 100
        print(f"  {name:20s}: mean={feat_vals.mean().item():8.4f}, "
              f"std={feat_vals.std().item():8.4f}, "
              f"non-zero={non_zero:5.1f}%")

    n_features = len(EDGE_FEATURE_NAMES)

    # Verify feature count matches
    sample = test_dataset[0]
    actual_features = sample.edge_attr.shape[1]
    if actual_features != n_features:
        print(f"Warning: Expected {n_features} edge features, got {actual_features}")
        n_features = actual_features

    importances = []
    importance_stds = []

    print(f"\nComputing importance for {n_features} edge features...")

    # First, test with ALL edge features zeroed to verify edge features matter at all
    print("\n[DEBUG] Testing with ALL edge features zeroed...")
    all_zeroed_r2 = compute_r2_with_all_edges_zeroed(model, test_dataset, device)
    print(f"[DEBUG] R² with all edge features zeroed: {all_zeroed_r2:.4f}")
    print(f"[DEBUG] Difference from baseline: {baseline_r2 - all_zeroed_r2:.4f}\n")

    for feat_idx in tqdm(range(n_features), desc="Edge Features"):
        repeat_scores = []

        # Debug first feature
        debug_mode = (feat_idx == 0)
        if debug_mode:
            print(f"\n[DEBUG] Analyzing feature {feat_idx} ({EDGE_FEATURE_NAMES[feat_idx]}):")

        for _ in range(n_repeats):
            zeroed_r2 = compute_r2_with_zeroed_edge_feature(
                model, test_dataset, feat_idx, device, debug=debug_mode
            )
            if debug_mode:
                print(f"  R² after zeroing: {zeroed_r2:.4f}")
            repeat_scores.append(baseline_r2 - zeroed_r2)

        importances.append(np.mean(repeat_scores))
        importance_stds.append(np.std(repeat_scores) if n_repeats > 1 else 0.0)

    # Create results DataFrame
    feature_names = EDGE_FEATURE_NAMES[:n_features]
    results = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'importance_std': importance_stds
    })

    # Sort by importance (descending)
    results = results.sort_values('importance', ascending=False).reset_index(drop=True)

    return results, baseline_r2


def plot_importance(results, save_path, baseline_r2):
    """Create bar plot of edge feature importances."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create horizontal bar plot
    y_pos = np.arange(len(results))
    colors = []

    for _, row in results.iterrows():
        if 'bond_' in row['feature'] and row['feature'] != 'bond_length':
            colors.append('steelblue')  # Bond types
        elif row['feature'] in ['distance', 'normalized_dist', 'relative_dist', 'bond_length']:
            colors.append('forestgreen')  # Distance-related
        elif row['feature'] == 'in_radius':
            colors.append('crimson')  # Radius graph
        else:
            colors.append('gray')

    bars = ax.barh(y_pos, results['importance'],
                   xerr=results['importance_std'],
                   color=colors, alpha=0.8, capsize=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(results['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance (R² drop when zeroed)')
    ax.set_title(f'Edge Feature Importance (Zero-Out Method)\nBaseline R²: {baseline_r2:.4f}')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Bond Types'),
        Patch(facecolor='forestgreen', label='Distance Features'),
        Patch(facecolor='crimson', label='Radius Graph'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # Add vertical line at 0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved importance plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Edge Feature Importance Analysis')
    parser.add_argument('--checkpoint', type=str,
                        default='experiments/checkpoints/phase15_globalpool/best_model.pt',
                        help='Path to model checkpoint')
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

    # Compute edge importance
    print("\nComputing edge feature importance...")
    results, baseline_r2 = compute_edge_importance(
        model, test_dataset, device, n_repeats=1
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, 'edge_importance.csv')
    results.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    # Create plot
    plot_path = os.path.join(args.output_dir, 'edge_importance.png')
    plot_importance(results, plot_path, baseline_r2)

    # Print results
    print("\n" + "=" * 60)
    print("EDGE FEATURE IMPORTANCE")
    print("=" * 60)
    print(f"{'Rank':<6} {'Feature':<20} {'Importance':>12}")
    print("-" * 60)
    for i, (_, row) in enumerate(results.iterrows(), 1):
        print(f"{i:<6} {row['feature']:<20} {row['importance']:>12.6f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline R²: {baseline_r2:.4f}")
    print(f"Total edge importance: {results['importance'].sum():.4f}")
    print(f"Most important: {results.iloc[0]['feature']} ({results.iloc[0]['importance']:.4f})")

    # Group analysis
    bond_types = results[results['feature'].str.startswith('bond_') &
                         (results['feature'] != 'bond_length')]
    distance_feats = results[results['feature'].isin(['distance', 'normalized_dist',
                                                       'relative_dist', 'bond_length'])]
    radius_feat = results[results['feature'] == 'in_radius']

    print(f"\nBy category:")
    print(f"  Bond types total: {bond_types['importance'].sum():.4f}")
    print(f"  Distance features total: {distance_feats['importance'].sum():.4f}")
    print(f"  Radius graph (in_radius): {radius_feat['importance'].sum():.4f}")


if __name__ == '__main__':
    main()
