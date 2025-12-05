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

from src.data.dataset import ProteinAtomDataset
from src.models.gnn import create_model
from src.training.train import Trainer
from src.training.evaluate import evaluate_model, print_metrics
from src.utils.config import Config
from src.utils.visualization import plot_training_curves, plot_predictions


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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

    # Set random seed
    set_seed(config.experiment.seed)

    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")

    print("\n" + "="*60)
    print("GNN PROTEIN ATOM EXPOSURE PREDICTION")
    print("="*60)

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = ProteinAtomDataset(root=config.data.root, split='train')
    val_dataset = ProteinAtomDataset(root=config.data.root, split='val')
    test_dataset = ProteinAtomDataset(root=config.data.root, split='test')

    print(f"  Train: {len(train_dataset)} proteins")
    print(f"  Val:   {len(val_dataset)} proteins")
    print(f"  Test:  {len(test_dataset)} proteins")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )

    # Create model
    print("\nCreating model...")
    model = create_model(config.model.to_dict())
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {config.model.model_type.upper()}")
    print(f"  Parameters: {num_params:,}")

    # Create optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    criterion = nn.MSELoss()

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=config.experiment.checkpoint_dir
    )

    # Training
    if not args.eval_only:
        print("\nStarting training...")
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
    print("\nEvaluating on test set...")
    print("-" * 60)
    test_metrics = evaluate_model(trainer.model, test_loader, device)
    print_metrics(test_metrics)

    # Generate predictions and visualizations
    if args.visualize:
        from src.training.evaluate import predict

        print("\nGenerating visualizations...")
        y_pred, y_true = predict(trainer.model, test_loader, device)

        # Plot predictions
        plot_predictions(
            y_true,
            y_pred,
            save_path=os.path.join(config.experiment.log_dir, 'predictions.png')
        )

        from src.utils.visualization import plot_error_distribution
        plot_error_distribution(
            y_true,
            y_pred,
            save_path=os.path.join(config.experiment.log_dir, 'error_distribution.png')
        )

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
