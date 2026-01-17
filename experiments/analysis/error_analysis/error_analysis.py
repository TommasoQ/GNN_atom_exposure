"""
Error Analysis for GNN Atom Exposure Prediction

Analyzes prediction errors by:
1. Exposure range (buried vs exposed)
2. Atom type (C, N, O, S, etc.)
3. Protein size
4. Residue type

Run after training to understand model weaknesses.
"""

import sys
from pathlib import Path

# Add project root to path BEFORE other imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from src.data.dataset_fixed import ProteinAtomDataset
from src.models.gnn import AtomExposureGNN
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def load_model_and_data(checkpoint_path: str = None, device: str = 'cuda'):
    """Load trained model and test data."""
    if checkpoint_path is None:
        checkpoint_path = 'experiments/checkpoints/best_model.pt'
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Checkpoint metrics: {checkpoint.get('metrics', 'N/A')}")
    
    # Create model
    model = AtomExposureGNN(
        in_channels=88,
        hidden_channels=96,
        num_layers=3,
        dropout=0.35,
        conv_type='gine'
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test data
    test_dataset = ProteinAtomDataset(root='dataset/', split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    return model, test_loader, test_dataset


def collect_predictions(model, test_loader, device: str = 'cuda'):
    """Collect all predictions and targets."""
    all_preds = []
    all_targets = []
    all_atom_types = []
    all_protein_ids = []
    all_protein_sizes = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Collecting predictions"):
            batch = batch.to(device)
            preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(batch.y.cpu().numpy().flatten())
            
            # Extract ELEMENT from one-hot encoding
            # Feature order: atom_type (31), element (5), residue (21), geometric (7), numerical (24)
            # Elements are at indices 31-35: C, N, O, S, Other
            element_onehot = batch.x[:, 31:36].cpu().numpy()
            elements = np.argmax(element_onehot, axis=1)
            all_atom_types.append(elements)
            
            # Protein info
            batch_ids = batch.batch.cpu().numpy()
            for i in range(batch.num_graphs):
                mask = batch_ids == i
                size = mask.sum()
                all_protein_sizes.extend([size] * size)
    
    return {
        'predictions': np.concatenate(all_preds),
        'targets': np.concatenate(all_targets),
        'atom_types': np.concatenate(all_atom_types),
        'protein_sizes': np.array(all_protein_sizes)
    }


def analyze_by_exposure_range(data: dict, output_dir: str):
    """Analyze errors by exposure range."""
    preds = data['predictions']
    targets = data['targets']
    errors = preds - targets
    abs_errors = np.abs(errors)
    
    # Define exposure ranges
    ranges = [
        (0.0, 0.2, 'Buried (0-0.2)'),
        (0.2, 0.5, 'Semi-buried (0.2-0.5)'),
        (0.5, 0.8, 'Intermediate (0.5-0.8)'),
        (0.8, 1.2, 'Semi-exposed (0.8-1.2)'),
        (1.2, 2.0, 'Exposed (1.2+)')
    ]
    
    results = []
    for low, high, name in ranges:
        mask = (targets >= low) & (targets < high)
        if mask.sum() == 0:
            continue
        
        mae = np.abs(errors[mask]).mean()
        rmse = np.sqrt((errors[mask] ** 2).mean())
        bias = errors[mask].mean()
        count = mask.sum()
        pct = count / len(targets) * 100
        
        results.append({
            'Range': name,
            'Count': count,
            'Pct': f'{pct:.1f}%',
            'MAE': mae,
            'RMSE': rmse,
            'Bias': bias
        })
    
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("ERROR BY EXPOSURE RANGE")
    print("="*60)
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv(f'{output_dir}/error_by_exposure_range.csv', index=False)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = range(len(df))
    
    axes[0].bar(x, df['MAE'], color='steelblue')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([r.split('(')[0].strip() for r in df['Range']], rotation=45, ha='right')
    axes[0].set_ylabel('MAE')
    axes[0].set_title('Mean Absolute Error by Exposure Range')
    
    axes[1].bar(x, df['Bias'], color=['red' if b < 0 else 'green' for b in df['Bias']])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([r.split('(')[0].strip() for r in df['Range']], rotation=45, ha='right')
    axes[1].set_ylabel('Bias (pred - target)')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_title('Prediction Bias by Exposure Range')
    
    counts = [int(c) for c in df['Count']]
    axes[2].bar(x, counts, color='gray')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([r.split('(')[0].strip() for r in df['Range']], rotation=45, ha='right')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Sample Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_by_exposure_range.png', dpi=150)
    plt.close()
    
    return df


def analyze_by_atom_type(data: dict, output_dir: str):
    """Analyze errors by atom type."""
    preds = data['predictions']
    targets = data['targets']
    atom_types = data['atom_types']
    errors = preds - targets
    
    type_names = ['C', 'N', 'O', 'S', 'Other']
    
    results = []
    for i, name in enumerate(type_names):
        mask = atom_types == i
        if mask.sum() == 0:
            continue
        
        mae = np.abs(errors[mask]).mean()
        rmse = np.sqrt((errors[mask] ** 2).mean())
        bias = errors[mask].mean()
        count = mask.sum()
        pct = count / len(targets) * 100
        avg_exposure = targets[mask].mean()
        
        results.append({
            'Atom': name,
            'Count': count,
            'Pct': f'{pct:.1f}%',
            'Avg Exposure': avg_exposure,
            'MAE': mae,
            'RMSE': rmse,
            'Bias': bias
        })
    
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("ERROR BY ATOM TYPE")
    print("="*60)
    print(df.to_string(index=False))
    
    df.to_csv(f'{output_dir}/error_by_atom_type.csv', index=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(df))
    ax.bar(x, df['MAE'], color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Atom'])
    ax.set_ylabel('MAE')
    ax.set_title('Mean Absolute Error by Atom Type')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_by_atom_type.png', dpi=150)
    plt.close()
    
    return df


def analyze_by_protein_size(data: dict, output_dir: str):
    """Analyze errors by protein size."""
    preds = data['predictions']
    targets = data['targets']
    sizes = data['protein_sizes']
    errors = preds - targets
    
    # Define size bins
    bins = [
        (0, 500, 'Small (<500)'),
        (500, 1000, 'Medium (500-1000)'),
        (1000, 2000, 'Large (1000-2000)'),
        (2000, 10000, 'Very Large (2000+)')
    ]
    
    results = []
    for low, high, name in bins:
        mask = (sizes >= low) & (sizes < high)
        if mask.sum() == 0:
            continue
        
        mae = np.abs(errors[mask]).mean()
        rmse = np.sqrt((errors[mask] ** 2).mean())
        bias = errors[mask].mean()
        count = mask.sum()
        
        results.append({
            'Size': name,
            'Atoms': count,
            'MAE': mae,
            'RMSE': rmse,
            'Bias': bias
        })
    
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("ERROR BY PROTEIN SIZE")
    print("="*60)
    print(df.to_string(index=False))
    
    df.to_csv(f'{output_dir}/error_by_protein_size.csv', index=False)
    
    return df


def create_scatter_plot(data: dict, output_dir: str):
    """Create prediction vs target scatter plot."""
    preds = data['predictions']
    targets = data['targets']
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Subsample for plotting (too many points)
    n = len(preds)
    if n > 50000:
        idx = np.random.choice(n, 50000, replace=False)
        preds_plot = preds[idx]
        targets_plot = targets[idx]
    else:
        preds_plot = preds
        targets_plot = targets
    
    ax.scatter(targets_plot, preds_plot, alpha=0.1, s=1)
    ax.plot([0, 2], [0, 2], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('True Exposure')
    ax.set_ylabel('Predicted Exposure')
    ax.set_title(f'Prediction vs Target (n={n:,})')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.legend()
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/prediction_scatter.png', dpi=150)
    plt.close()


def create_error_distribution(data: dict, output_dir: str):
    """Create error distribution histogram."""
    preds = data['predictions']
    targets = data['targets']
    errors = preds - targets
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Error distribution
    axes[0].hist(errors, bins=100, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].axvline(x=errors.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.4f}')
    axes[0].set_xlabel('Error (pred - target)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Error Distribution')
    axes[0].legend()
    
    # Absolute error distribution
    abs_errors = np.abs(errors)
    axes[1].hist(abs_errors, bins=100, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(x=abs_errors.mean(), color='red', linestyle='--', linewidth=2, label=f'MAE: {abs_errors.mean():.4f}')
    axes[1].axvline(x=np.median(abs_errors), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(abs_errors):.4f}')
    axes[1].set_xlabel('Absolute Error')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Absolute Error Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_distribution.png', dpi=150)
    plt.close()


def main():
    """Run full error analysis."""
    print("="*60)
    print("GNN ATOM EXPOSURE - ERROR ANALYSIS")
    print("="*60)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Output to same folder as this script
    script_dir = Path(__file__).parent
    output_dir = script_dir / 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and data
    print("\nLoading model and test data...")
    model, test_loader, test_dataset = load_model_and_data(device=device)
    print(f"Test set: {len(test_dataset)} proteins")
    
    # Collect predictions
    print("\nCollecting predictions...")
    data = collect_predictions(model, test_loader, device)
    print(f"Total atoms: {len(data['predictions']):,}")
    
    # Run analyses
    print("\nRunning error analyses...")
    
    df_exposure = analyze_by_exposure_range(data, output_dir)
    df_atom = analyze_by_atom_type(data, output_dir)
    df_size = analyze_by_protein_size(data, output_dir)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_scatter_plot(data, output_dir)
    create_error_distribution(data, output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    errors = data['predictions'] - data['targets']
    print(f"Overall MAE:  {np.abs(errors).mean():.4f}")
    print(f"Overall RMSE: {np.sqrt((errors**2).mean()):.4f}")
    print(f"Overall Bias: {errors.mean():.4f}")
    print(f"\nWorst exposure range: {df_exposure.loc[df_exposure['MAE'].idxmax(), 'Range']}")
    print(f"Best exposure range:  {df_exposure.loc[df_exposure['MAE'].idxmin(), 'Range']}")
    
    print(f"\nResults saved to: {output_dir}/")
    print("  - error_by_exposure_range.csv")
    print("  - error_by_atom_type.csv")
    print("  - error_by_protein_size.csv")
    print("  - prediction_scatter.png")
    print("  - error_distribution.png")
    print("  - error_by_exposure_range.png")
    print("  - error_by_atom_type.png")


if __name__ == '__main__':
    main()
