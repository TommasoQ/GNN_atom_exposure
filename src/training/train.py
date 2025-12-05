"""
Training loop and utilities for GNN models
"""
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import os
from typing import Dict, Optional
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
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = 'experiments/checkpoints'
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

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

        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            batch = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            # Compute loss
            loss = self.criterion(out, batch.y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item() * batch.num_graphs
            num_samples += batch.num_graphs

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

        for batch in tqdm(val_loader, desc='Validation'):
            batch = batch.to(self.device)

            # Forward pass
            out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            # Compute loss
            loss = self.criterion(out, batch.y)

            total_loss += loss.item() * batch.num_graphs
            num_samples += batch.num_graphs

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

            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_metrics = self.validate(val_loader)
            self.val_losses.append(val_metrics['loss'])

            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}")

            # Save best model
            if save_best and val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(
                    os.path.join(self.checkpoint_dir, 'best_model.pt'),
                    epoch,
                    val_metrics
                )
                print(f"Saved best model (val_loss: {self.best_val_loss:.4f})")

        print("\nTraining completed!")

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics': metrics
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
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
