"""
Main training script for GNN Protein Atom Exposure Prediction
"""
import argparse
import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
import random

from src.data.dataset_fixed import ProteinAtomDataset
from src.models.gnn import create_model
from src.training.train import Trainer
from src.training.evaluate import evaluate_model, print_metrics
from src.utils.config import Config
from src.utils.visualization import (
    plot_training_curves, plot_training_curves_extended, plot_predictions,
    plot_error_distribution, plot_error_by_exposure_range
)
import json
from datetime import datetime


def save_test_metrics(metrics: dict, config, checkpoint_epoch: int, save_path: str):
    """Save test metrics to JSON file for permanent record."""
    # Convert numpy float32 to Python float for JSON serialization
    serializable_metrics = {k: float(v) for k, v in metrics.items()}
    output = {
        'timestamp': datetime.now().isoformat(),
        'test_metrics': serializable_metrics,
        'checkpoint_epoch': checkpoint_epoch,
        'config': {
            'experiment_name': getattr(config.experiment, 'name', 'unknown'),
            'model': {
                'conv_type': getattr(config.model, 'conv_type', 'unknown'),
                'hidden_channels': getattr(config.model, 'hidden_channels', None),
                'num_layers': getattr(config.model, 'num_layers', None),
                'in_channels': getattr(config.model, 'in_channels', None),
                'edge_dim': getattr(config.model, 'edge_dim', None),
            },
            'training': {
                'num_epochs': getattr(config.training, 'num_epochs', None),
                'learning_rate': getattr(config.training, 'learning_rate', None),
                'scheduler': getattr(config.training, 'scheduler', None),
                'weighted_loss': getattr(config.training, 'weighted_loss', False),
            },
            'features': {
                'include_backbone_angles': getattr(config.features, 'include_backbone_angles', False) if hasattr(config, 'features') else False,
            }
        }
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved test metrics to {save_path}")


def set_seed(seed: int, cudnn_benchmark: bool = False, deterministic: bool = False):
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        cudnn_benchmark: If True, enable cudnn.benchmark for faster training
                        (slightly non-deterministic but faster)
        deterministic: If True, force fully deterministic operations
                      (slower but 100% reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # Set CUBLAS workspace config for deterministic cuBLAS operations
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        # Fully deterministic - reproducible but slower
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    elif cudnn_benchmark:
        # Faster but slightly non-deterministic
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        # Default: deterministic cudnn but no benchmark
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):
    """Main training function."""

    # Load configuration
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        from src.utils.config import get_default_config
        config = get_default_config()

    # Override config with command line arguments
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr

    # Set random seed with optional cudnn benchmark for faster training
    cudnn_benchmark = getattr(config.training, 'cudnn_benchmark', False)
    deterministic = getattr(config.training, 'deterministic', False)
    set_seed(config.experiment.seed, cudnn_benchmark=cudnn_benchmark, deterministic=deterministic)
    
    # Check if AMP is enabled
    use_amp = getattr(config.training, 'use_amp', False)

    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")
    if use_amp and device == 'cuda':
        print("Mixed precision training (AMP): Enabled")
    if deterministic:
        print("Deterministic mode: Enabled (fully reproducible, slower)")
    elif cudnn_benchmark:
        print("cuDNN benchmark mode: Enabled (faster, slight variance)")

    print("\n" + "="*60)
    print("GNN PROTEIN ATOM EXPOSURE PREDICTION")
    print("="*60)

    # Check if using aggregated features (need embedding config)
    use_aggregated = False
    if hasattr(config, 'features'):
        use_aggregated = getattr(config.features, 'use_aggregated', False)

    # Create model
    print("\nCreating model...")
    model_config = config.model.to_dict()

    # Add embedding parameters if using aggregated features
    if use_aggregated:
        model_config['use_embeddings'] = True
        model_config['num_numerical'] = 31  # Aggregated features produce 31 numerical
        model_config['num_elements'] = 5    # C, N, O, S, OTHER
        model_config['num_residues'] = 21   # 20 amino acids + OTHER
        model_config['element_embed_dim'] = getattr(config.features, 'element_embed_dim', 8)
        model_config['residue_embed_dim'] = getattr(config.features, 'residue_embed_dim', 11)

    model = create_model(model_config)
    num_params = sum(p.numel() for p in model.parameters())
    conv_type = getattr(config.model, 'conv_type', 'gcn').upper()
    print(f"  Architecture: {conv_type}")
    print(f"  Layers: {config.model.num_layers}, Hidden: {config.model.hidden_channels}")
    if use_aggregated:
        print(f"  Features: Aggregated (31 numerical + embeddings)")
    print(f"  Parameters: {num_params:,}")
    model.to(device)

    # Create optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Loss function: standard MSE or exposure-weighted MSE
    use_weighted_loss = getattr(config.training, 'weighted_loss', False)
    if use_weighted_loss:
        from src.training.train import ExposureWeightedMSELoss
        loss_alpha = getattr(config.training, 'loss_alpha', 5.0)
        loss_threshold = getattr(config.training, 'loss_threshold', 0.5)
        loss_power = getattr(config.training, 'loss_power', 2.0)
        criterion = ExposureWeightedMSELoss(
            alpha=loss_alpha,
            threshold=loss_threshold,
            power=loss_power
        )
        print(f"  Loss: Exposure-Weighted MSE (alpha={loss_alpha}, threshold={loss_threshold}, power={loss_power})")
    else:
        criterion = nn.MSELoss()
        print(f"  Loss: Standard MSE")

    # Get scheduler type (scheduler created later for one_cycle which needs steps_per_epoch)
    scheduler_type = getattr(config.training, 'scheduler', 'reduce_on_plateau')
    scheduler = None
    scheduler_step_per_batch = False  # OneCycleLR steps per batch, others per epoch

    if scheduler_type == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs,
            eta_min=getattr(config.training, 'min_lr', 1e-6)
        )
        print(f"  Scheduler: Cosine Annealing (T_max={config.training.num_epochs})")
    elif scheduler_type == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=getattr(config.training, 'patience', 10),
            min_lr=getattr(config.training, 'min_lr', 1e-6)
        )
        print(f"  Scheduler: ReduceOnPlateau (patience={getattr(config.training, 'patience', 10)})")
    elif scheduler_type == 'one_cycle':
        # OneCycleLR needs steps_per_epoch, will be created after data loaders
        print(f"  Scheduler: One Cycle (will initialize after data loading)")
        scheduler_step_per_batch = True

    # Create trainer (scheduler may be set later for one_cycle)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=config.experiment.checkpoint_dir,
        gradient_clip=getattr(config.training, 'gradient_clip', None),
        early_stopping_patience=getattr(config.training, 'early_stopping_patience', None),
        early_stopping_metric=getattr(config.training, 'early_stopping_metric', 'loss'),
        scheduler=scheduler,
        scheduler_step_per_epoch=not scheduler_step_per_batch,
        use_amp=use_amp
    )

    # Build feature configuration from YAML config (use_aggregated already set above)
    feature_config = None
    if hasattr(config, 'features'):
        include_backbone_angles = getattr(config.features, 'include_backbone_angles', False)
        feature_config = {
            'use_reduced_features': getattr(config.features, 'use_reduced', False),
            'include_atom_type': getattr(config.features, 'include_atom_type', True),
            'include_geometric': getattr(config.features, 'include_geometric', True),
            'use_aggregated': use_aggregated,
            'include_backbone_angles': include_backbone_angles,
        }
        if use_aggregated:
            print(f"\nFeature config: aggregated transforms (31 numerical + embeddings)")
        elif include_backbone_angles:
            print(f"\nFeature config: standard features + backbone angles (phi/psi)")
        else:
            print(f"\nFeature config: {feature_config}")

    # Training
    if not args.eval_only:
        print("\nStarting training...")
        print("\nLoading training and validation datasets...")
        train_dataset = ProteinAtomDataset(root=config.data.root, split='train', feature_config=feature_config)
        val_dataset = ProteinAtomDataset(root=config.data.root, split='val', feature_config=feature_config)
        print(f"  Train: {len(train_dataset)} proteins")
        print(f"  Val:   {len(val_dataset)} proteins")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True if device == 'cuda' else False,
            persistent_workers=True if config.data.num_workers > 0 else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=True if device == 'cuda' else False,
            persistent_workers=True if config.data.num_workers > 0 else False
        )

        # Initialize OneCycleLR if selected (requires steps_per_epoch)
        if scheduler_type == 'one_cycle':
            steps_per_epoch = len(train_loader)
            # Calculate pct_start from warmup_epochs (adaptive to any epoch count)
            warmup_epochs = getattr(config.training, 'warmup_epochs', 15)
            pct_start = warmup_epochs / config.training.num_epochs
            print(f"  Initializing OneCycleLR scheduler:")
            print(f"    - {steps_per_epoch} steps per epoch")
            print(f"    - {warmup_epochs} warmup epochs (pct_start={pct_start:.3f})")
            trainer.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=getattr(config.training, 'max_lr', 0.001),
                epochs=config.training.num_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=pct_start,
                div_factor=getattr(config.training, 'div_factor', 25.0),
                final_div_factor=getattr(config.training, 'final_div_factor', 10000.0),
                last_epoch=-1  # Start from step 0, avoids warning
            )

        print("-" * 60)
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.training.num_epochs,
            save_best=config.experiment.save_best
        )

        # Plot training curves
        plot_training_curves(
            trainer.train_losses,
            trainer.val_losses,
            save_path=os.path.join(config.experiment.log_dir, 'training_curves.png')
        )

    # Load best model for evaluation
    if args.checkpoint:
        print(f"\nLoading checkpoint from {args.checkpoint}...")
        trainer.load_checkpoint(args.checkpoint)
    elif os.path.exists(os.path.join(config.experiment.checkpoint_dir, 'best_model.pt')):
        print("\nLoading best model...")
        trainer.load_checkpoint(os.path.join(config.experiment.checkpoint_dir, 'best_model.pt'))

    # Evaluation
    print("\nLoading test dataset...")
    test_dataset = ProteinAtomDataset(root=config.data.root, split='test', feature_config=feature_config)
    print(f"  Test:  {len(test_dataset)} proteins")

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True if device == 'cuda' else False
    )

    print("\nEvaluating on test set...")
    print("-" * 60)
    # evaluate_model now returns metrics and predictions to avoid a second pass
    test_metrics, y_true, y_pred = evaluate_model(trainer.model, test_loader, device)
    print_metrics(test_metrics)

    # Get checkpoint epoch for logging
    checkpoint_epoch = len(trainer.train_losses) if trainer.train_losses else 0

    # Always save test metrics to JSON
    save_test_metrics(
        test_metrics,
        config,
        checkpoint_epoch,
        save_path=os.path.join(config.experiment.log_dir, 'test_metrics.json')
    )

    # Always generate visualizations (non-blocking, saved to files)
    print("\nGenerating visualizations...")

    # 1. Training curves with R² (if training was done)
    if trainer.train_losses and trainer.val_losses:
        plot_training_curves_extended(
            trainer.train_losses,
            trainer.val_losses,
            trainer.val_r2s if trainer.val_r2s else None,
            save_path=os.path.join(config.experiment.log_dir, 'training_curves.png')
        )

    # 2. Predictions vs Actual scatter plot
    plot_predictions(
        y_true,
        y_pred,
        save_path=os.path.join(config.experiment.log_dir, 'predictions_vs_actual.png')
    )

    # 3. Error distribution histogram
    plot_error_distribution(
        y_true,
        y_pred,
        save_path=os.path.join(config.experiment.log_dir, 'error_distribution.png')
    )

    # 4. Error by exposure range (key diagnostic)
    error_stats = plot_error_by_exposure_range(
        y_true,
        y_pred,
        save_path=os.path.join(config.experiment.log_dir, 'error_by_exposure_range.png')
    )

    # Print error stats summary
    print("\nError by Exposure Range:")
    print("-" * 60)
    print(f"{'Range':<20} {'Count':>10} {'MAE':>10} {'Bias':>10}")
    print("-" * 60)
    for s in error_stats:
        print(f"{s['label'].replace(chr(10), ' '):<20} {s['count']:>10} {s['mae']:>10.4f} {s['bias']:>+10.4f}")

    print("\nDone!")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GNN for Protein Atom Exposure Prediction')

    # Configuration
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')

    # Evaluation
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate (skip training)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for evaluation')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')

    # Device
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage')

    args = parser.parse_args()

    main(args)
