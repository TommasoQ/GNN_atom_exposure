"""
Visualization utilities for GNN models and results
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional
import torch
from pathlib import Path


# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None
):
    """
    Plot training and validation loss curves.

    Args:
        train_losses (list): Training losses
        val_losses (list): Validation losses
        save_path (str, optional): Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    plt.show()


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = 'Predicted vs True Atom Exposure'
):
    """
    Plot predicted vs true values.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        save_path (str, optional): Path to save figure
        title (str): Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Calculate R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)

    ax.set_xlabel('True Exposure')
    ax.set_ylabel('Predicted Exposure')
    ax.set_title(f'{title}\nR² = {r2:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved predictions plot to {save_path}")

    plt.show()


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot distribution of prediction errors.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        save_path (str, optional): Path to save figure
    """
    errors = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Histogram
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].set_xlabel('Prediction Error')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Error Distribution\nMean: {errors.mean():.4f}, Std: {errors.std():.4f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot
    axes[1].boxplot(errors, vert=True)
    axes[1].set_ylabel('Prediction Error')
    axes[1].set_title('Error Box Plot')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved error distribution to {save_path}")

    plt.show()


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
):
    """
    Compare metrics across different models.

    Args:
        metrics_dict (dict): Dictionary mapping model names to their metrics
        save_path (str, optional): Path to save figure
    """
    metric_names = ['mae', 'rmse', 'r2']
    models = list(metrics_dict.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, metric in enumerate(metric_names):
        values = [metrics_dict[model][metric] for model in models]
        axes[idx].bar(models, values, alpha=0.7)
        axes[idx].set_title(metric.upper())
        axes[idx].set_ylabel('Value')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics comparison to {save_path}")

    plt.show()


def plot_atom_exposure_protein(
    data,
    predictions: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Visualize atom exposure for a single protein.

    Args:
        data: PyTorch Geometric Data object
        predictions (np.ndarray, optional): Predicted exposure values
        save_path (str, optional): Path to save figure
    """
    coords = data.x[:, [6, 7, 8]].numpy()  # x, y, z coordinates
    true_exposure = data.y.numpy()

    fig = plt.figure(figsize=(15, 5))

    # True exposure
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=true_exposure, cmap='viridis', s=20, alpha=0.6
    )
    ax1.set_title('True Atom Exposure')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.colorbar(scatter1, ax=ax1, label='Exposure')

    # Predicted exposure
    if predictions is not None:
        ax2 = fig.add_subplot(122, projection='3d')
        scatter2 = ax2.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            c=predictions, cmap='viridis', s=20, alpha=0.6
        )
        ax2.set_title('Predicted Atom Exposure')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        plt.colorbar(scatter2, ax=ax2, label='Exposure')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved atom exposure plot to {save_path}")

    plt.show()


if __name__ == '__main__':
    # Test visualizations with dummy data

    # Training curves
    train_losses = np.exp(-np.linspace(0, 2, 50)) + np.random.randn(50) * 0.05
    val_losses = np.exp(-np.linspace(0, 1.8, 50)) + np.random.randn(50) * 0.08
    plot_training_curves(train_losses, val_losses)

    # Predictions
    y_true = np.random.randn(1000) * 2 + 5
    y_pred = y_true + np.random.randn(1000) * 0.5
    plot_predictions(y_true, y_pred)
    plot_error_distribution(y_true, y_pred)
