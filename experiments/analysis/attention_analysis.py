"""
Attention Analysis for GATv2 Model

This script analyzes the attention weights learned by the GATv2 model:
1. Attention weight distribution per layer and head
2. Correlation between attention and edge distance
3. Attention patterns by atom exposure level
4. Attention heatmaps for sample proteins

Usage:
    python experiments/analysis/attention_analysis.py [--checkpoint PATH]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset_fixed import ProteinAtomDataset
from src.models.gnn import AtomExposureGNN
from torch_geometric.nn import GATv2Conv


def extract_attention_weights(model, data, device):
    """
    Extract attention weights from all GATv2 layers.

    IMPORTANT: Must replicate the exact forward pass including residual connections!

    Args:
        model: GATv2 model
        data: PyG Data object
        device: torch device

    Returns:
        List of dicts with attention weights per layer
    """
    model.eval()
    data = data.to(device)

    attention_per_layer = []

    with torch.no_grad():
        # Get input projection output (matches model.forward)
        x = data.x
        if model.use_embeddings:
            # Would need element_idx and residue_idx
            pass
        x = model.input_proj(x)
        x = torch.nn.functional.relu(x)

        # Process each layer - MUST match model.forward exactly!
        for i, (conv, bn) in enumerate(zip(model.convs, model.batch_norms)):
            x_in = x  # Save for residual connection

            if isinstance(conv, GATv2Conv):
                # Get attention weights
                out, (edge_index_out, attention) = conv(
                    x, data.edge_index, edge_attr=data.edge_attr,
                    return_attention_weights=True
                )

                attention_per_layer.append({
                    'layer': i,
                    'edge_index': edge_index_out.cpu().numpy(),
                    'attention': attention.cpu().numpy(),  # [num_edges, num_heads]
                    'num_heads': attention.shape[1]
                })

                # Continue forward pass - MUST match model.forward!
                x = bn(out)
                x = torch.nn.functional.elu(x)  # GATv2 uses ELU
                x = torch.nn.functional.dropout(x, p=model.dropout, training=False)

                # CRITICAL: Residual connection (only after first layer)
                if i > 0:
                    x = x + x_in
            else:
                # Non-GATv2 layer
                x = conv(x, data.edge_index, edge_attr=data.edge_attr)
                x = bn(x)
                x = torch.nn.functional.relu(x)
                x = torch.nn.functional.dropout(x, p=model.dropout, training=False)
                if i > 0:
                    x = x + x_in

    return attention_per_layer


def analyze_attention_statistics(attention_per_layer, save_dir):
    """Compute and visualize attention statistics."""
    stats_data = []

    for layer_info in attention_per_layer:
        layer = layer_info['layer']
        attention = layer_info['attention']
        num_heads = layer_info['num_heads']

        for head in range(num_heads):
            head_attention = attention[:, head]
            stats_data.append({
                'layer': layer,
                'head': head,
                'mean': np.mean(head_attention),
                'std': np.std(head_attention),
                'min': np.min(head_attention),
                'max': np.max(head_attention),
                'median': np.median(head_attention),
                'skew': stats.skew(head_attention),
                'entropy': -np.sum(head_attention * np.log(head_attention + 1e-10)) / len(head_attention)
            })

    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(os.path.join(save_dir, 'attention_statistics.csv'), index=False)

    # Plot attention distributions
    fig, axes = plt.subplots(len(attention_per_layer), 1, figsize=(10, 3*len(attention_per_layer)))
    if len(attention_per_layer) == 1:
        axes = [axes]

    for idx, layer_info in enumerate(attention_per_layer):
        ax = axes[idx]
        attention = layer_info['attention']
        num_heads = layer_info['num_heads']

        for head in range(num_heads):
            ax.hist(attention[:, head], bins=50, alpha=0.5, label=f'Head {head}',
                    density=True)

        ax.set_xlabel('Attention Weight')
        ax.set_ylabel('Density')
        ax.set_title(f'Layer {layer_info["layer"]} Attention Distribution')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'attention_distribution.png'), dpi=150)
    plt.close()

    return stats_df


def analyze_attention_vs_distance(attention_per_layer, edge_distances, save_dir):
    """Analyze correlation between attention and edge distance."""
    results = []

    fig, axes = plt.subplots(1, len(attention_per_layer), figsize=(5*len(attention_per_layer), 4))
    if len(attention_per_layer) == 1:
        axes = [axes]

    for idx, layer_info in enumerate(attention_per_layer):
        ax = axes[idx]
        attention = layer_info['attention']
        mean_attention = np.mean(attention, axis=1)  # Average across heads

        # Correlation
        corr, p_value = stats.pearsonr(edge_distances, mean_attention)
        results.append({
            'layer': layer_info['layer'],
            'correlation': corr,
            'p_value': p_value
        })

        # Scatter plot (sample for visibility)
        sample_size = min(10000, len(edge_distances))
        indices = np.random.choice(len(edge_distances), sample_size, replace=False)

        ax.scatter(edge_distances[indices], mean_attention[indices],
                   alpha=0.1, s=1)
        ax.set_xlabel('Edge Distance (Å)')
        ax.set_ylabel('Mean Attention')
        ax.set_title(f'Layer {layer_info["layer"]}\nr={corr:.3f}, p={p_value:.2e}')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'attention_vs_distance.png'), dpi=150)
    plt.close()

    return pd.DataFrame(results)


def analyze_attention_by_edge_type(attention_per_layer, edge_attr, save_dir):
    """Analyze attention by edge type (bond vs radius graph)."""
    # in_radius is index 11
    in_radius = edge_attr[:, 11]
    is_bond = (in_radius == 0)  # Edges not from radius graph are bonds
    is_radius = (in_radius == 1)

    results = []

    for layer_info in attention_per_layer:
        attention = layer_info['attention']
        mean_attention = np.mean(attention, axis=1)

        bond_attention = mean_attention[is_bond]
        radius_attention = mean_attention[is_radius]

        results.append({
            'layer': layer_info['layer'],
            'bond_mean': np.mean(bond_attention) if len(bond_attention) > 0 else np.nan,
            'bond_std': np.std(bond_attention) if len(bond_attention) > 0 else np.nan,
            'radius_mean': np.mean(radius_attention) if len(radius_attention) > 0 else np.nan,
            'radius_std': np.std(radius_attention) if len(radius_attention) > 0 else np.nan,
            'n_bond': len(bond_attention),
            'n_radius': len(radius_attention)
        })

    results_df = pd.DataFrame(results)

    # Box plot
    fig, ax = plt.subplots(figsize=(8, 5))

    layer_data = []
    for layer_info in attention_per_layer:
        attention = layer_info['attention']
        mean_attention = np.mean(attention, axis=1)

        for edge_type, mask in [('Bond', is_bond), ('Radius', is_radius)]:
            for val in mean_attention[mask][:5000]:  # Sample for speed
                layer_data.append({
                    'Layer': layer_info['layer'],
                    'Edge Type': edge_type,
                    'Attention': val
                })

    df = pd.DataFrame(layer_data)
    sns.boxplot(data=df, x='Layer', y='Attention', hue='Edge Type', ax=ax)
    ax.set_title('Attention by Edge Type')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'attention_by_edge_type.png'), dpi=150)
    plt.close()

    return results_df


def analyze_attention_by_exposure(dataset, model, device, save_dir, num_proteins=50):
    """Analyze attention patterns for buried vs exposed atoms across multiple proteins."""
    model.eval()

    # Collect attention and targets across multiple proteins
    all_attention_by_layer = {0: [], 1: [], 2: []}  # 3 layers
    all_targets = []
    all_edge_sources = []
    cumulative_nodes = 0

    num_to_analyze = min(num_proteins, len(dataset))
    indices = np.random.choice(len(dataset), num_to_analyze, replace=False)

    print(f"  Analyzing exposure across {num_to_analyze} proteins...")

    for idx in tqdm(indices, desc="  Proteins"):
        data = dataset[idx]
        attention_per_layer = extract_attention_weights(model, data, device)

        # Collect targets and edge sources with offset
        targets = data.y.cpu().numpy()
        edge_sources = data.edge_index[0].cpu().numpy() + cumulative_nodes

        all_targets.append(targets)
        all_edge_sources.append(edge_sources)

        for layer_info in attention_per_layer:
            layer_idx = layer_info['layer']
            all_attention_by_layer[layer_idx].append(layer_info['attention'])

        cumulative_nodes += data.num_nodes

    # Concatenate all
    all_targets = np.concatenate(all_targets)
    all_edge_sources = np.concatenate(all_edge_sources)

    # Categorize all nodes
    buried = all_targets < 0.3
    intermediate = (all_targets >= 0.3) & (all_targets < 0.7)
    exposed = all_targets >= 0.7

    print(f"  Node distribution: Buried={np.sum(buried)}, Intermediate={np.sum(intermediate)}, Exposed={np.sum(exposed)}")

    results = []

    for layer_idx in range(3):
        attention = np.concatenate(all_attention_by_layer[layer_idx], axis=0)
        mean_attention = np.mean(attention, axis=1)

        # Attention when source is buried/exposed
        for category, mask in [('Buried', buried), ('Intermediate', intermediate), ('Exposed', exposed)]:
            source_in_category = mask[all_edge_sources]
            attention_from_category = mean_attention[source_in_category]

            if len(attention_from_category) > 0:
                results.append({
                    'layer': layer_idx,
                    'source_category': category,
                    'mean_attention': np.mean(attention_from_category),
                    'std_attention': np.std(attention_from_category),
                    'n_edges': len(attention_from_category)
                })
            else:
                results.append({
                    'layer': layer_idx,
                    'source_category': category,
                    'mean_attention': np.nan,
                    'std_attention': np.nan,
                    'n_edges': 0
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_dir, 'attention_by_exposure.csv'), index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot_df = results_df.pivot(index='source_category', columns='layer', values='mean_attention')
    pivot_df = pivot_df.reindex(['Buried', 'Intermediate', 'Exposed'])
    pivot_df.plot(kind='bar', ax=ax)
    ax.set_ylabel('Mean Attention')
    ax.set_xlabel('Source Node Category')
    ax.set_title('Attention by Source Node Exposure Level')
    ax.legend(title='Layer')
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'attention_by_exposure.png'), dpi=150)
    plt.close()

    return results_df


def create_attention_heatmap(attention_per_layer, data, save_dir, max_nodes=50):
    """Create attention heatmap for a small protein."""
    num_nodes = data.num_nodes
    if num_nodes > max_nodes:
        print(f"Skipping heatmap: protein has {num_nodes} nodes (max {max_nodes})")
        return

    edge_index = data.edge_index.cpu().numpy()

    fig, axes = plt.subplots(1, len(attention_per_layer), figsize=(6*len(attention_per_layer), 5))
    if len(attention_per_layer) == 1:
        axes = [axes]

    for idx, layer_info in enumerate(attention_per_layer):
        ax = axes[idx]
        attention = layer_info['attention']
        mean_attention = np.mean(attention, axis=1)

        # Build attention matrix
        attn_matrix = np.zeros((num_nodes, num_nodes))
        for e_idx in range(edge_index.shape[1]):
            src, dst = edge_index[0, e_idx], edge_index[1, e_idx]
            attn_matrix[src, dst] = mean_attention[e_idx]

        sns.heatmap(attn_matrix, ax=ax, cmap='viridis', square=True)
        ax.set_title(f'Layer {layer_info["layer"]} Attention')
        ax.set_xlabel('Target Node')
        ax.set_ylabel('Source Node')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'attention_heatmap.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Attention Analysis for GATv2')
    parser.add_argument('--checkpoint', type=str,
                        default='experiments/baselines/phase13a_best/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='experiments/analysis',
                        help='Output directory for results')
    parser.add_argument('--num-proteins', type=int, default=50,
                        help='Number of proteins to analyze')
    args = parser.parse_args()

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = AtomExposureGNN(
        in_channels=93,
        hidden_channels=128,
        num_layers=3,
        conv_type='gatv2',
        edge_dim=12,
        dropout=0.25
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = ProteinAtomDataset(
        root='dataset/',
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

    # Create output directory
    save_dir = os.path.join(args.output_dir, 'attention_analysis')
    os.makedirs(save_dir, exist_ok=True)

    # Debug: Check model structure
    print("\n[DEBUG] Model structure check:")
    for i, conv in enumerate(model.convs):
        print(f"  Layer {i}: {type(conv).__name__}")
        if hasattr(conv, 'att_src'):
            print(f"    att_src shape: {conv.att_src.shape}")
            print(f"    att_src first values: {conv.att_src.data.flatten()[:4].tolist()}")

    # Debug: Extract attention from ONE protein first
    print("\n[DEBUG] Single protein attention check:")
    test_data = test_dataset[0].to(device)
    single_attention = extract_attention_weights(model, test_dataset[0], device)
    for layer_info in single_attention:
        attn = layer_info['attention']
        print(f"  Layer {layer_info['layer']}:")
        print(f"    Attention shape: {attn.shape}")
        print(f"    Head 0 first 5 values: {attn[:5, 0]}")
        print(f"    Head 1 first 5 values: {attn[:5, 1]}")
        print(f"    Head 2 first 5 values: {attn[:5, 2]}")
        print(f"    Head 3 first 5 values: {attn[:5, 3]}")
        print(f"    Are heads identical? {np.allclose(attn[:, 0], attn[:, 1])}")

    # Collect attention from multiple proteins
    print(f"\nExtracting attention from {args.num_proteins} proteins...")
    all_attention_layers = [[] for _ in range(3)]  # 3 layers
    all_edge_distances = []
    all_edge_attr = []

    # Select proteins
    num_to_analyze = min(args.num_proteins, len(test_dataset))
    indices = np.random.choice(len(test_dataset), num_to_analyze, replace=False)

    for i in tqdm(indices, desc="Proteins"):
        data = test_dataset[i]
        attention_per_layer = extract_attention_weights(model, data, device)

        for layer_idx, layer_info in enumerate(attention_per_layer):
            all_attention_layers[layer_idx].append(layer_info['attention'])

        # Edge distances (index 7) - must be on CPU for numpy
        all_edge_distances.append(data.edge_attr[:, 7].cpu().numpy())
        all_edge_attr.append(data.edge_attr.cpu().numpy())

    # Concatenate
    combined_attention = []
    for layer_idx in range(len(all_attention_layers)):
        if all_attention_layers[layer_idx]:
            combined_attention.append({
                'layer': layer_idx,
                'attention': np.concatenate(all_attention_layers[layer_idx], axis=0),
                'num_heads': all_attention_layers[layer_idx][0].shape[1]
            })

    all_edge_distances = np.concatenate(all_edge_distances)
    all_edge_attr = np.concatenate(all_edge_attr)

    print(f"\nTotal edges analyzed: {len(all_edge_distances):,}")

    # Run analyses
    print("\n1. Analyzing attention statistics...")
    stats_df = analyze_attention_statistics(combined_attention, save_dir)
    print(stats_df.to_string())

    print("\n2. Analyzing attention vs distance...")
    dist_corr_df = analyze_attention_vs_distance(combined_attention, all_edge_distances, save_dir)
    print(dist_corr_df.to_string())

    print("\n3. Analyzing attention by edge type...")
    edge_type_df = analyze_attention_by_edge_type(combined_attention, all_edge_attr, save_dir)
    print(edge_type_df.to_string())

    print("\n4. Analyzing attention by exposure level...")
    # Aggregate across multiple proteins to ensure all exposure categories are represented
    exposure_df = analyze_attention_by_exposure(test_dataset, model, device, save_dir, num_proteins=args.num_proteins)
    print(exposure_df.to_string())

    print("\n5. Creating attention heatmap for small protein...")
    # Find a small protein - try test set first, then train set
    small_protein = None
    small_protein_size = None

    # First try test set
    for i in range(len(test_dataset)):
        num_nodes = test_dataset[i].num_nodes
        if num_nodes < 100:  # Relaxed threshold
            if small_protein is None or num_nodes < small_protein_size:
                small_protein = test_dataset[i]
                small_protein_size = num_nodes
                if num_nodes < 50:
                    break  # Good enough

    # If no small protein in test, try loading train dataset
    if small_protein is None or small_protein_size > 80:
        print("  Looking in train dataset for smaller proteins...")
        try:
            train_dataset = ProteinAtomDataset(
                root='dataset/',
                split='train',
                feature_config={
                    'use_reduced_features': False,
                    'include_atom_type': True,
                    'include_geometric': True,
                    'use_aggregated': False,
                    'include_backbone_angles': True,
                }
            )
            for i in range(min(500, len(train_dataset))):  # Check first 500
                num_nodes = train_dataset[i].num_nodes
                if num_nodes < 50:
                    small_protein = train_dataset[i]
                    small_protein_size = num_nodes
                    break
        except Exception as e:
            print(f"  Could not load train dataset: {e}")

    if small_protein is not None:
        print(f"  Found protein with {small_protein_size} nodes")
        small_attention = extract_attention_weights(model, small_protein, device)
        create_attention_heatmap(small_attention, small_protein, save_dir, max_nodes=small_protein_size + 10)
    else:
        print(f"  No small protein found (smallest: {small_protein_size} nodes)")

    # Summary
    print("\n" + "=" * 60)
    print("ATTENTION ANALYSIS SUMMARY")
    print("=" * 60)

    print("\nKey Findings:")
    print(f"  - Edges analyzed: {len(all_edge_distances):,}")
    print(f"  - Number of layers: {len(combined_attention)}")
    print(f"  - Heads per layer: {combined_attention[0]['num_heads']}")

    print("\nAttention-Distance Correlation:")
    for _, row in dist_corr_df.iterrows():
        sign = "+" if row['correlation'] > 0 else ""
        print(f"  Layer {row['layer']}: r={sign}{row['correlation']:.3f}")

    print("\nAttention by Edge Type (Layer 0):")
    layer0 = edge_type_df[edge_type_df['layer'] == 0].iloc[0]
    print(f"  Bond edges: mean={layer0['bond_mean']:.4f} (n={layer0['n_bond']:,})")
    print(f"  Radius edges: mean={layer0['radius_mean']:.4f} (n={layer0['n_radius']:,})")

    print(f"\nResults saved to: {save_dir}/")
    print("Files generated:")
    print("  - attention_statistics.csv")
    print("  - attention_distribution.png")
    print("  - attention_vs_distance.png")
    print("  - attention_by_edge_type.png")
    print("  - attention_by_exposure.csv")
    print("  - attention_by_exposure.png")
    print("  - attention_heatmap.png (if small protein found)")


if __name__ == '__main__':
    main()
