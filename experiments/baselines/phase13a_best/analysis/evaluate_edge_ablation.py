"""
Edge Ablation Study for GATv2 Model
Tests the contribution of edge structure and edge features to model predictions.

Three modes:
- normal: Standard inference with full graph
- self_loops: Only self-loops (removes graph structure)
- zero_features: Full graph but edge features set to zero
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from src.data.dataset_fixed import ProteinAtomDataset
from src.models.gnn import create_model
from src.utils.config import Config
from src.utils.visualization import (
    plot_predictions,
    plot_error_distribution
)


def create_self_loops(num_nodes, device):
    """Create edge_index with only self-loops."""
    edge_index = torch.arange(num_nodes, device=device)
    edge_index = torch.stack([edge_index, edge_index], dim=0)
    return edge_index


def process_batch(batch, mode):
    """
    Process batch according to ablation mode.

    Args:
        batch: PyTorch Geometric batch
        mode: 'normal', 'self_loops', or 'zero_features'

    Returns:
        Modified batch
    """
    if mode == 'normal':
        # No modification
        return batch

    elif mode == 'self_loops':
        # Replace edge_index with self-loops only
        num_nodes = batch.x.size(0)
        batch.edge_index = create_self_loops(num_nodes, batch.x.device)
        # Zero out edge features for self-loops
        batch.edge_attr = torch.zeros(
            (num_nodes, 12),
            dtype=batch.x.dtype,
            device=batch.x.device
        )
        return batch

    elif mode == 'zero_features':
        # Keep graph structure, zero out edge features
        batch.edge_attr = torch.zeros_like(batch.edge_attr)
        return batch

    else:
        raise ValueError(f"Unknown mode: {mode}")


@torch.no_grad()
def evaluate_with_ablation(model, data_loader, device, mode):
    """
    Evaluate model with edge ablation.

    Args:
        model: GNN model
        data_loader: DataLoader for test set
        device: Device to use
        mode: Ablation mode

    Returns:
        tuple: (metrics dict, y_true array, y_pred array)
    """
    model.eval()
    model = model.to(device)

    all_preds = []
    all_targets = []

    print(f"\nRunning inference with mode: {mode}")
    print("=" * 60)

    for batch in tqdm(data_loader, desc='Evaluating', ncols=80, ascii=True):
        batch = batch.to(device)

        # Apply ablation
        batch = process_batch(batch, mode)

        # Get embedding indices if present
        element_idx = getattr(batch, 'element_idx', None)
        residue_idx = getattr(batch, 'residue_idx', None)

        # Forward pass
        out = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            element_idx=element_idx,
            residue_idx=residue_idx
        )

        # Store predictions and targets
        all_preds.append(out.cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())

    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Compute metrics
    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)

    # Pearson correlation
    pearson_corr, pearson_pval = pearsonr(all_targets, all_preds)

    # Mean and std of errors
    errors = all_preds - all_targets
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    median_ae = np.median(np.abs(errors))

    metrics = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'pearson_corr': float(pearson_corr),
        'pearson_pval': float(pearson_pval),
        'mean_error': float(mean_error),
        'std_error': float(std_error),
        'median_ae': float(median_ae)
    }

    return metrics, all_targets, all_preds


def print_metrics(metrics, mode):
    """Pretty print evaluation metrics."""
    print("\n" + "=" * 60)
    print(f"EVALUATION METRICS - Mode: {mode.upper()}")
    print("=" * 60)
    print(f"Mean Absolute Error (MAE):     {metrics['mae']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
    print(f"R² Score:                       {metrics['r2']:.4f}")
    print(f"Pearson Correlation:            {metrics['pearson_corr']:.4f}")
    print(f"Median Absolute Error:          {metrics['median_ae']:.4f}")
    print(f"Mean Error:                     {metrics['mean_error']:.4f}")
    print(f"Std Error:                      {metrics['std_error']:.4f}")
    print("=" * 60 + "\n")


def save_results(metrics, y_true, y_pred, mode, output_dir, config):
    """Save metrics and generate plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics to JSON
    output = {
        'timestamp': datetime.now().isoformat(),
        'mode': mode,
        'metrics': metrics,
        'config': {
            'experiment_name': getattr(config.experiment, 'name', 'unknown'),
            'model': {
                'conv_type': getattr(config.model, 'conv_type', 'unknown'),
                'hidden_channels': getattr(config.model, 'hidden_channels', None),
                'num_layers': getattr(config.model, 'num_layers', None),
                'in_channels': getattr(config.model, 'in_channels', None),
                'edge_dim': getattr(config.model, 'edge_dim', None),
            }
        }
    }

    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Generate plots
    print("\nGenerating visualizations...")

    # Predictions vs Actual
    plot_predictions(
        y_true,
        y_pred,
        save_path=os.path.join(output_dir, 'predictions_vs_actual.png')
    )

    # Error distribution
    plot_error_distribution(
        y_true,
        y_pred,
        save_path=os.path.join(output_dir, 'error_distribution.png')
    )

    print(f"Saved visualizations to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Edge Ablation Study for GATv2 Model'
    )

    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='configs/phase13a_lower_lr.yaml',
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    # Ablation mode
    parser.add_argument(
        '--mode',
        type=str,
        choices=['normal', 'self_loops', 'zero_features'],
        required=True,
        help='Ablation mode: normal, self_loops, or zero_features'
    )

    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results'
    )

    # Data loading
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (default: from config)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of data loading workers (default: from config)'
    )

    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading config from {args.config}")
    config = Config.from_yaml(args.config)

    # Override batch size if specified
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.num_workers:
        config.data.num_workers = args.num_workers

    # Set output directory
    if args.output_dir is None:
        script_dir = Path(__file__).parent
        args.output_dir = str(script_dir / 'results' / args.mode)

    print("\n" + "=" * 60)
    print("EDGE ABLATION STUDY - GATv2 MODEL")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Check if using aggregated features
    use_aggregated = False
    if hasattr(config, 'features'):
        use_aggregated = getattr(config.features, 'use_aggregated', False)

    # Create model
    print("\nCreating model...")
    model_config = config.model.to_dict()

    # Add embedding parameters if using aggregated features
    if use_aggregated:
        model_config['use_embeddings'] = True
        model_config['num_numerical'] = 31
        model_config['num_elements'] = 5
        model_config['num_residues'] = 21
        model_config['element_embed_dim'] = getattr(
            config.features, 'element_embed_dim', 8
        )
        model_config['residue_embed_dim'] = getattr(
            config.features, 'residue_embed_dim', 11
        )

    model = create_model(model_config)
    num_params = sum(p.numel() for p in model.parameters())
    conv_type = getattr(config.model, 'conv_type', 'gcn').upper()
    print(f"  Architecture: {conv_type}")
    print(f"  Layers: {config.model.num_layers}, Hidden: {config.model.hidden_channels}")
    print(f"  Parameters: {num_params:,}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Build feature configuration
    feature_config = None
    if hasattr(config, 'features'):
        include_backbone_angles = getattr(
            config.features, 'include_backbone_angles', False
        )
        feature_config = {
            'use_reduced_features': getattr(config.features, 'use_reduced', False),
            'include_atom_type': getattr(config.features, 'include_atom_type', True),
            'include_geometric': getattr(config.features, 'include_geometric', True),
            'use_aggregated': use_aggregated,
            'include_backbone_angles': include_backbone_angles,
        }

    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = ProteinAtomDataset(
        root=config.data.root,
        split='test',
        feature_config=feature_config
    )
    print(f"  Test set: {len(test_dataset)} proteins")

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )

    # Run evaluation
    metrics, y_true, y_pred = evaluate_with_ablation(
        model,
        test_loader,
        args.device,
        args.mode
    )

    # Print metrics
    print_metrics(metrics, args.mode)

    # Save results
    save_results(metrics, y_true, y_pred, args.mode, args.output_dir, config)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
