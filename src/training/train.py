"""
Training loop and utilities for GNN models
"""
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import os
import sys
import csv
import json
from typing import Dict, Optional, List
import numpy as np


class Trainer:
    """
    Trainer class for GNN models.

    Args:
        model (nn.Module): GNN model
        optimizer (torch.optim.Optimizer): Optimizer
        criterion (nn.Module): Loss function
        device (str): Device to train on ('cuda' or 'cpu')
        checkpoint_dir (str): Directory to save checkpoints
        scheduler: Learning rate scheduler (optional)
        scheduler_step_per_epoch (bool): If True, step scheduler after each epoch
        use_amp (bool): If True, use automatic mixed precision training
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = 'experiments/checkpoints',
        gradient_clip: Optional[float] = None,
        early_stopping_patience: Optional[int] = None,
        scheduler=None,
        scheduler_step_per_epoch: bool = True,
        use_amp: bool = False
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.gradient_clip = gradient_clip
        self.early_stopping_patience = early_stopping_patience
        self.scheduler = scheduler
        self.scheduler_step_per_epoch = scheduler_step_per_epoch
        self.use_amp = use_amp and device == 'cuda'  # AMP only works on CUDA
        
        # Initialize GradScaler for mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.train_losses = []
        self.val_losses = []
        self.val_maes = []  # Track validation MAE per epoch
        self.val_rmses = []  # Track validation RMSE per epoch
        self.learning_rates = []  # Track LR per epoch
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader (DataLoader): Training data loader

        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0
        num_samples = 0

        pbar = tqdm(train_loader, desc='Training', mininterval=0.5, leave=False, disable=not sys.stdout.isatty(), ncols=80, ascii=True)
        for batch in pbar:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                # Mixed precision training
                with torch.amp.autocast('cuda'):
                    out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    loss = self.criterion(out, batch.y)
                
                # Scaled backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping if enabled (must unscale first)
                if self.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = self.criterion(out, batch.y)
                loss.backward()
                
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                self.optimizer.step()

            # Step scheduler per batch if configured (e.g. OneCycleLR)
            if self.scheduler is not None and not self.scheduler_step_per_epoch:
                self.scheduler.step()

            # Update metrics
            total_loss += loss.item() * batch.num_nodes
            num_samples += batch.num_nodes

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_samples
        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader (DataLoader): Validation data loader

        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        total_loss = 0
        num_samples = 0

        all_preds = []
        all_targets = []

        for batch in tqdm(val_loader, desc='Validation', mininterval=0.5, leave=False, disable=not sys.stdout.isatty(), ncols=80, ascii=True):
            batch = batch.to(self.device)

            # Forward pass with optional AMP
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    loss = self.criterion(out, batch.y)
            else:
                out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = self.criterion(out, batch.y)

            total_loss += loss.item() * batch.num_nodes
            num_samples += batch.num_nodes

            # Store predictions and targets
            all_preds.append(out.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())

        avg_loss = total_loss / num_samples

        # Compute additional metrics
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        mae = np.mean(np.abs(all_preds - all_targets))
        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))

        return {
            'loss': avg_loss,
            'mae': mae,
            'rmse': rmse
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_best: bool = True
    ):
        """
        Full training loop.

        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            num_epochs (int): Number of epochs
            save_best (bool): Whether to save best model
        """
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Track current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_metrics = self.validate(val_loader)
            self.val_losses.append(val_metrics['loss'])
            self.val_maes.append(val_metrics['mae'])
            self.val_rmses.append(val_metrics['rmse'])

            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}")

            # Save best model and check early stopping
            if save_best and val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
                self.save_checkpoint(
                    os.path.join(self.checkpoint_dir, 'best_model.pt'),
                    epoch,
                    val_metrics
                )
                print(f"[+] New best model saved (val_loss: {self.best_val_loss:.4f})")
            else:
                self.epochs_without_improvement += 1

            # Early stopping check
            if self.early_stopping_patience is not None:
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\n[!] Early stopping triggered after {epoch + 1} epochs")
                    print(f"   No improvement for {self.early_stopping_patience} consecutive epochs")
                    print(f"   Best validation loss: {self.best_val_loss:.4f}")
                    break

            # Step scheduler after epoch
            if self.scheduler is not None and self.scheduler_step_per_epoch:
                if hasattr(self.scheduler, 'step'):
                    # ReduceLROnPlateau needs val_loss, others don't
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()

        print("\nTraining completed!")
        
        # Export training history to CSV
        self.export_training_history()

    def export_training_history(self, filename: str = None):
        """Export training history to CSV for analysis."""
        if filename is None:
            filename = os.path.join(self.checkpoint_dir, '..', 'logs', 'training_history.csv')
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_mae', 'val_rmse', 'learning_rate'])
            
            for i in range(len(self.train_losses)):
                lr = self.learning_rates[i] if i < len(self.learning_rates) else None
                writer.writerow([
                    i + 1,
                    f"{self.train_losses[i]:.6f}",
                    f"{self.val_losses[i]:.6f}",
                    f"{self.val_maes[i]:.6f}",
                    f"{self.val_rmses[i]:.6f}",
                    f"{lr:.8f}" if lr else ""
                ])
        
        print(f"Saved training history to {filename}")

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_maes': self.val_maes,
            'val_rmses': self.val_rmses,
            'metrics': metrics
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_maes = checkpoint.get('val_maes', [])
        self.val_rmses = checkpoint.get('val_rmses', [])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Convenience function to train a model.

    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of epochs
        learning_rate (float): Learning rate
        device (str): Device to train on
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    trainer = Trainer(model, optimizer, criterion, device)
    trainer.train(train_loader, val_loader, num_epochs)

    return trainer


if __name__ == '__main__':
    from src.data.dataset import ProteinAtomDataset
    from src.models.gnn import AtomExposureGNN

    # Load dataset
    train_dataset = ProteinAtomDataset(root='dataset/', split='train')
    val_dataset = ProteinAtomDataset(root='dataset/', split='val')

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Create model
    model = AtomExposureGNN(in_channels=80, hidden_channels=128, num_layers=3)

    # Train
    trainer = train_model(model, train_loader, val_loader, num_epochs=10)
