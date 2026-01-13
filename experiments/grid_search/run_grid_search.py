"""
Grid Search for Scheduler Comparison
=====================================
Compares Cosine Annealing vs One Cycle schedulers with various hyperparameters.

18 experiments total:
- 9 Cosine Annealing: lr × min_lr grid
- 9 One Cycle: max_lr × pct_start grid

Features:
- Automatic Mixed Precision (AMP) for faster training
- Resume capability: skips completed experiments
- Per-experiment checkpoints and validation history
- Comprehensive metrics: R², MAE, RMSE, Pearson, etc.
"""

import os
import sys
import json
import csv
import time
import argparse
from datetime import datetime
from pathlib import Path
from itertools import product
from typing import Dict, Any, List, Optional

# Set matplotlib backend BEFORE importing pyplot (fixes tkinter threading crash)
import matplotlib
matplotlib.use('Agg')

import yaml
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_fixed import ProteinAtomDataset
from src.models.gnn import create_model
from src.training.train import Trainer
from src.training.evaluate import evaluate_model, compute_metrics
from src.utils.config import Config
from src.utils.visualization import plot_training_curves


def load_grid_config(config_path: str) -> Dict[str, Any]:
    """Load grid search configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_experiments(grid_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all experiment configurations from grid."""
    experiments = []
    
    # Cosine Annealing experiments
    cos_lrs = grid_config['cosine_annealing']['learning_rates']
    cos_min_lrs = grid_config['cosine_annealing']['min_lrs']
    
    for lr, min_lr in product(cos_lrs, cos_min_lrs):
        exp_name = f"cosine_lr{lr}_minlr{min_lr:.0e}"
        experiments.append({
            'name': exp_name,
            'scheduler': 'cosine_annealing',
            'learning_rate': lr,
            'min_lr': min_lr,
            'params': {'lr': lr, 'min_lr': min_lr}
        })
    
    # One Cycle experiments
    oc_max_lrs = grid_config['one_cycle']['max_lrs']
    oc_pct_starts = grid_config['one_cycle']['pct_starts']
    
    for max_lr, pct_start in product(oc_max_lrs, oc_pct_starts):
        exp_name = f"onecycle_maxlr{max_lr}_pct{pct_start}"
        experiments.append({
            'name': exp_name,
            'scheduler': 'one_cycle',
            'max_lr': max_lr,
            'pct_start': pct_start,
            'learning_rate': max_lr / 25.0,  # div_factor default
            'params': {'max_lr': max_lr, 'pct_start': pct_start}
        })
    
    return experiments


def get_completed_experiments(summary_csv: str) -> set:
    """Get set of already completed experiment names."""
    completed = set()
    if os.path.exists(summary_csv):
        with open(summary_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(row['experiment'])
    return completed


def save_result_to_csv(summary_csv: str, result: Dict[str, Any]):
    """Append a single result to the summary CSV."""
    file_exists = os.path.exists(summary_csv)
    
    fieldnames = [
        'experiment', 'scheduler', 'r2', 'mae', 'mse', 'rmse', 'pearson',
        'median_ae', 'mean_error', 'std_error', 'duration_s',
        'epochs_trained', 'best_epoch', 'params', 'timestamp'
    ]
    
    with open(summary_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def save_validation_history(output_dir: str, trainer: Trainer):
    """Save per-epoch validation metrics to JSON."""
    # Convert any numpy types to Python native types for JSON serialization
    def to_python(val):
        if isinstance(val, (np.floating, np.integer)):
            return float(val)
        elif isinstance(val, np.ndarray):
            return val.tolist()
        return val
    
    history = {
        'train_losses': [to_python(v) for v in trainer.train_losses],
        'val_losses': [to_python(v) for v in trainer.val_losses],
        'val_maes': [to_python(v) for v in trainer.val_maes],
        'val_rmses': [to_python(v) for v in trainer.val_rmses]
    }
    
    history_path = os.path.join(output_dir, 'validation_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)


def save_final_metrics(output_dir: str, metrics: Dict[str, float], exp_config: Dict[str, Any]):
    """Save final test metrics to JSON."""
    output = {
        'experiment': exp_config['name'],
        'scheduler': exp_config['scheduler'],
        'params': exp_config['params'],
        'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in metrics.items()}
    }
    
    metrics_path = os.path.join(output_dir, 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(output, f, indent=2)


def run_experiment(
    exp_config: Dict[str, Any],
    base_config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    output_dir: str,
    device: str
) -> Dict[str, Any]:
    """Run a single experiment and return results."""
    
    exp_name = exp_config['name']
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"Scheduler: {exp_config['scheduler']}")
    print(f"Parameters: {exp_config['params']}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Create fresh model
    model = create_model(base_config.model.to_dict())
    model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=exp_config['learning_rate'],
        weight_decay=base_config.training.weight_decay
    )
    criterion = nn.MSELoss()
    
    # Create scheduler based on experiment type
    scheduler = None
    scheduler_step_per_batch = False
    num_epochs = base_config.training.num_epochs
    
    if exp_config['scheduler'] == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=exp_config['min_lr']
        )
    elif exp_config['scheduler'] == 'one_cycle':
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=exp_config['max_lr'],
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=exp_config['pct_start'],
            div_factor=25.0,
            final_div_factor=10000.0
        )
        scheduler_step_per_batch = True
    
    # Create experiment output directory
    exp_output_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # Create trainer with AMP enabled
    use_amp = getattr(base_config.training, 'use_amp', True)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=exp_output_dir,
        gradient_clip=getattr(base_config.training, 'gradient_clip', 1.0),
        early_stopping_patience=getattr(base_config.training, 'early_stopping_patience', 7),
        scheduler=scheduler,
        scheduler_step_per_epoch=not scheduler_step_per_batch,
        use_amp=use_amp
    )
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_best=True
    )
    
    # Save training curves
    plot_training_curves(
        trainer.train_losses,
        trainer.val_losses,
        save_path=os.path.join(exp_output_dir, 'training_curves.png')
    )
    
    # Save validation history
    save_validation_history(exp_output_dir, trainer)
    
    # Load best model and evaluate on test set
    best_model_path = os.path.join(exp_output_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        trainer.load_checkpoint(best_model_path)
    
    print("\nEvaluating on test set...")
    test_metrics, y_true, y_pred = evaluate_model(trainer.model, test_loader, device)
    
    # Save final metrics
    save_final_metrics(exp_output_dir, test_metrics, exp_config)
    
    duration = time.time() - start_time
    epochs_trained = len(trainer.train_losses)
    
    # Find best epoch (0-indexed in list, report as 1-indexed)
    best_epoch = trainer.val_losses.index(min(trainer.val_losses)) + 1
    
    # Print results
    print(f"\n{'='*50}")
    print(f"RESULTS: {exp_name}")
    print(f"{'='*50}")
    print(f"R² Score:           {test_metrics['r2']:.4f}")
    print(f"MAE:                {test_metrics['mae']:.4f}")
    print(f"RMSE:               {test_metrics['rmse']:.4f}")
    print(f"Pearson:            {test_metrics['pearson_corr']:.4f}")
    print(f"Median AE:          {test_metrics['median_ae']:.4f}")
    print(f"Duration:           {duration:.1f}s ({duration/60:.1f}min)")
    print(f"Epochs trained:     {epochs_trained}")
    print(f"Best epoch:         {best_epoch}")
    print(f"{'='*50}")
    
    return {
        'experiment': exp_name,
        'scheduler': exp_config['scheduler'],
        'r2': round(test_metrics['r2'], 4),
        'mae': round(test_metrics['mae'], 4),
        'mse': round(test_metrics['mse'], 4),
        'rmse': round(test_metrics['rmse'], 4),
        'pearson': round(test_metrics['pearson_corr'], 4),
        'median_ae': round(test_metrics['median_ae'], 4),
        'mean_error': round(test_metrics['mean_error'], 4),
        'std_error': round(test_metrics['std_error'], 4),
        'duration_s': round(duration, 1),
        'epochs_trained': epochs_trained,
        'best_epoch': best_epoch,
        'params': json.dumps(exp_config['params']),
        'timestamp': datetime.now().isoformat()
    }


def main():
    parser = argparse.ArgumentParser(description='Grid Search for Scheduler Comparison')
    parser.add_argument('--config', type=str, 
                       default='experiments/grid_search/grid_config.yaml',
                       help='Path to grid search config')
    parser.add_argument('--base-config', type=str,
                       default='configs/config.yaml',
                       help='Path to base training config')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    parser.add_argument('--dry-run', action='store_true',
                       help='List experiments without running')
    args = parser.parse_args()
    
    # Load configurations
    grid_config = load_grid_config(args.config)
    base_config = Config.from_yaml(args.base_config)
    
    # Set up output directory
    results_dir = grid_config['output']['results_dir']
    summary_csv = grid_config['output']['summary_csv']
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate all experiments
    experiments = generate_experiments(grid_config)
    
    if args.dry_run:
        print("\n" + "="*70)
        print("GRID SEARCH EXPERIMENTS (Dry Run)")
        print("="*70)
        for i, exp in enumerate(experiments, 1):
            print(f"{i:2d}. {exp['name']}")
            print(f"    Scheduler: {exp['scheduler']}")
            print(f"    Params: {exp['params']}")
        print(f"\nTotal: {len(experiments)} experiments")
        return
    
    # Check for completed experiments
    completed = get_completed_experiments(summary_csv)
    remaining = [e for e in experiments if e['name'] not in completed]
    
    print("\n" + "="*70)
    print("GRID SEARCH FOR SCHEDULER COMPARISON")
    print("="*70)
    print(f"Total experiments:     {len(experiments)}")
    print(f"Already completed:     {len(completed)}")
    print(f"Remaining:             {len(remaining)}")
    print(f"Results directory:     {results_dir}")
    print(f"Summary CSV:           {summary_csv}")
    
    if not remaining:
        print("\nAll experiments already completed!")
        print(f"Results are in: {summary_csv}")
        return
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"\nDevice: {device}")
    
    # Set up for faster training
    cudnn_benchmark = getattr(base_config.training, 'cudnn_benchmark', True)
    if cudnn_benchmark and device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("cuDNN benchmark: Enabled")
    
    use_amp = getattr(base_config.training, 'use_amp', True)
    if use_amp and device == 'cuda':
        print("Mixed precision (AMP): Enabled")
    
    # Set random seed
    seed = grid_config['grid_search']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load datasets once
    print("\nLoading datasets...")
    train_dataset = ProteinAtomDataset(root=base_config.data.root, split='train')
    val_dataset = ProteinAtomDataset(root=base_config.data.root, split='val')
    test_dataset = ProteinAtomDataset(root=base_config.data.root, split='test')
    print(f"  Train: {len(train_dataset)} proteins")
    print(f"  Val:   {len(val_dataset)} proteins")
    print(f"  Test:  {len(test_dataset)} proteins")
    
    # Create data loaders
    batch_size = base_config.data.batch_size
    num_workers = base_config.data.num_workers
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f"\nBatch size: {batch_size}")
    print(f"Steps per epoch: {len(train_loader)}")
    
    # Estimate time
    estimated_time = len(remaining) * 15  # ~15 min per experiment with optimizations
    print(f"\nEstimated time: ~{estimated_time} minutes ({estimated_time/60:.1f} hours)")
    
    # Run experiments
    total_start = time.time()
    
    for i, exp_config in enumerate(remaining, 1):
        print(f"\n[{i}/{len(remaining)}] Starting experiment: {exp_config['name']}")
        
        try:
            result = run_experiment(
                exp_config=exp_config,
                base_config=base_config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                output_dir=results_dir,
                device=device
            )
            
            # Save result immediately (crash-safe)
            save_result_to_csv(summary_csv, result)
            print(f"Result saved to {summary_csv}")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[!] CUDA OOM Error for {exp_config['name']}")
                print("    Try reducing batch_size in configs/config.yaml")
                torch.cuda.empty_cache()
                continue
            else:
                raise
        except Exception as e:
            print(f"\n[!] Error in experiment {exp_config['name']}: {e}")
            continue
    
    total_duration = time.time() - total_start
    
    print("\n" + "="*70)
    print("GRID SEARCH COMPLETE")
    print("="*70)
    print(f"Total time: {total_duration/60:.1f} minutes ({total_duration/3600:.2f} hours)")
    print(f"Results saved to: {summary_csv}")
    print(f"\nRun the analysis script to compare results:")
    print(f"  python experiments/grid_search/analysis/compare_results.py")


if __name__ == '__main__':
    main()
