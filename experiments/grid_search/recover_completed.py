"""
Recovery script for grid search experiments.
Processes completed experiments that have best_model.pt but no final_metrics.json.
"""

import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime

# Set matplotlib backend before any imports
import matplotlib
matplotlib.use('Agg')

import torch
from torch_geometric.loader import DataLoader
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_fixed import ProteinAtomDataset
from src.models.gnn import create_model
from src.training.evaluate import evaluate_model
from src.utils.config import Config


def get_experiment_info(exp_name: str) -> dict:
    """Parse experiment name to get scheduler and params."""
    if exp_name.startswith('cosine_'):
        # cosine_lr0.001_minlr1e-06
        parts = exp_name.replace('cosine_', '').split('_')
        lr = float(parts[0].replace('lr', ''))
        min_lr = float(parts[1].replace('minlr', ''))
        return {
            'scheduler': 'cosine_annealing',
            'params': {'lr': lr, 'min_lr': min_lr}
        }
    elif exp_name.startswith('onecycle_'):
        # onecycle_maxlr0.001_pct0.2
        parts = exp_name.replace('onecycle_', '').split('_')
        max_lr = float(parts[0].replace('maxlr', ''))
        pct_start = float(parts[1].replace('pct', ''))
        return {
            'scheduler': 'one_cycle',
            'params': {'max_lr': max_lr, 'pct_start': pct_start}
        }
    return {'scheduler': 'unknown', 'params': {}}


def main():
    results_dir = 'experiments/grid_search/results'
    summary_csv = os.path.join(results_dir, 'summary.csv')
    base_config = Config.from_yaml('configs/config.yaml')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Find experiments with best_model.pt but no final_metrics.json
    experiments_to_recover = []
    for exp_name in os.listdir(results_dir):
        exp_dir = os.path.join(results_dir, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        
        model_path = os.path.join(exp_dir, 'best_model.pt')
        metrics_path = os.path.join(exp_dir, 'final_metrics.json')
        
        if os.path.exists(model_path) and not os.path.exists(metrics_path):
            experiments_to_recover.append(exp_name)
    
    if not experiments_to_recover:
        print("No experiments to recover.")
        return
    
    print(f"Found {len(experiments_to_recover)} experiments to recover:")
    for exp in experiments_to_recover:
        print(f"  - {exp}")
    
    # Load test dataset once
    print("\nLoading test dataset...")
    test_dataset = ProteinAtomDataset(root=base_config.data.root, split='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=base_config.data.batch_size,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True if device == 'cuda' else False
    )
    print(f"  Test: {len(test_dataset)} proteins")
    
    # Get existing results from CSV to avoid duplicates
    existing = set()
    if os.path.exists(summary_csv):
        with open(summary_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add(row['experiment'])
    
    # Process each experiment
    for exp_name in experiments_to_recover:
        if exp_name in existing:
            print(f"\nSkipping {exp_name} (already in CSV)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Recovering: {exp_name}")
        print(f"{'='*60}")
        
        exp_dir = os.path.join(results_dir, exp_name)
        model_path = os.path.join(exp_dir, 'best_model.pt')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Create model and load weights
        model = create_model(base_config.model.to_dict())
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_metrics, y_true, y_pred = evaluate_model(model, test_loader, device)
        
        # Get experiment info
        exp_info = get_experiment_info(exp_name)
        
        # Get training info from checkpoint
        epochs_trained = checkpoint.get('epoch', 0) + 1
        val_losses = checkpoint.get('val_losses', [])
        best_epoch = val_losses.index(min(val_losses)) + 1 if val_losses else epochs_trained
        
        # Print results
        print(f"R²:      {test_metrics['r2']:.4f}")
        print(f"MAE:     {test_metrics['mae']:.4f}")
        print(f"RMSE:    {test_metrics['rmse']:.4f}")
        print(f"Pearson: {test_metrics['pearson_corr']:.4f}")
        
        # Save final_metrics.json
        final_metrics = {
            'experiment': exp_name,
            'scheduler': exp_info['scheduler'],
            'params': exp_info['params'],
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in test_metrics.items()}
        }
        
        metrics_path = os.path.join(exp_dir, 'final_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        print(f"Saved: {metrics_path}")
        
        # Append to summary CSV
        result = {
            'experiment': exp_name,
            'scheduler': exp_info['scheduler'],
            'r2': round(test_metrics['r2'], 4),
            'mae': round(test_metrics['mae'], 4),
            'mse': round(test_metrics['mse'], 4),
            'rmse': round(test_metrics['rmse'], 4),
            'pearson': round(test_metrics['pearson_corr'], 4),
            'median_ae': round(test_metrics['median_ae'], 4),
            'mean_error': round(test_metrics['mean_error'], 4),
            'std_error': round(test_metrics['std_error'], 4),
            'duration_s': 0,  # Unknown for recovered experiments
            'epochs_trained': epochs_trained,
            'best_epoch': best_epoch,
            'params': json.dumps(exp_info['params']),
            'timestamp': datetime.now().isoformat()
        }
        
        fieldnames = [
            'experiment', 'scheduler', 'r2', 'mae', 'mse', 'rmse', 'pearson',
            'median_ae', 'mean_error', 'std_error', 'duration_s',
            'epochs_trained', 'best_epoch', 'params', 'timestamp'
        ]
        
        file_exists = os.path.exists(summary_csv)
        with open(summary_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(result)
        
        print(f"Added to: {summary_csv}")
    
    print("\n" + "="*60)
    print("Recovery complete!")
    print(f"Results saved to: {summary_csv}")
    print("="*60)


if __name__ == '__main__':
    main()
