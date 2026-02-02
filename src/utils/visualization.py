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
        plt.close()
    else:
        plt.show()


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = 'Predicted vs True Atom Exposure',
    show_zero_line: bool = False
):
    """
    Plot predicted vs true values.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        save_path (str, optional): Path to save figure
        title (str): Plot title
        show_zero_line (bool): If True, show y=0 line and count negatives (for raw predictions)
    """
    from sklearn.metrics import r2_score

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.5, s=10)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Show y=0 line for raw predictions (to highlight negatives)
    if show_zero_line:
        ax.axhline(y=0, color='orange', linestyle=':', linewidth=1.5, alpha=0.8, label='y=0')
        n_negative = np.sum(y_pred < 0)
        if n_negative > 0:
            ax.text(0.02, 0.98, f'Negative: {n_negative} ({n_negative/len(y_pred)*100:.1f}%)',
                    transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    r2 = r2_score(y_true, y_pred)
    ax.set_xlabel('True Exposure')
    ax.set_ylabel('Predicted Exposure')
    ax.set_title(f'{title}\nR² = {r2:.4f}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved predictions plot to {save_path}")
        plt.close()
    else:
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
        plt.close()
    else:
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
        plt.close()
    else:
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
        plt.close()
    else:
        plt.show()


def plot_training_curves_extended(
    train_losses: List[float],
    val_losses: List[float],
    val_r2s: List[float] = None,
    save_path: Optional[str] = None
):
    """
    Plot training curves with optional R² subplot.

    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        val_r2s: Validation R² scores per epoch (optional)
        save_path: Path to save figure (non-blocking if provided)
    """
    n_plots = 2 if val_r2s else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # R² plot
    if val_r2s:
        axes[1].plot(epochs[:len(val_r2s)], val_r2s, 'g-', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('R² Score')
        axes[1].set_title('Validation R² Score')
        axes[1].grid(True, alpha=0.3)

        # Mark best R²
        best_idx = np.argmax(val_r2s)
        axes[1].axvline(best_idx + 1, color='orange', linestyle='--', alpha=0.7,
                       label=f'Best: {val_r2s[best_idx]:.4f} @ epoch {best_idx + 1}')
        axes[1].legend()

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved training curves to {save_path}")
    else:
        plt.close()


def plot_error_by_exposure_range(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot prediction errors grouped by exposure range.

    Args:
        y_true: True exposure values
        y_pred: Predicted exposure values
        save_path: Path to save figure (non-blocking)
    """
    # Define exposure bins
    bins = [0, 0.2, 0.5, 0.8, 1.2, float('inf')]
    labels = ['Buried\n(0-0.2)', 'Semi-buried\n(0.2-0.5)', 'Intermediate\n(0.5-0.8)',
              'Semi-exposed\n(0.8-1.2)', 'Exposed\n(1.2+)']

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    # Compute stats per bin
    stats = []
    for i in range(len(bins) - 1):
        mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
        if mask.sum() > 0:
            stats.append({
                'label': labels[i],
                'count': mask.sum(),
                'pct': 100 * mask.sum() / len(y_true),
                'mae': abs_errors[mask].mean(),
                'bias': errors[mask].mean(),
                'std': errors[mask].std()
            })
        else:
            stats.append({
                'label': labels[i],
                'count': 0,
                'pct': 0,
                'mae': 0,
                'bias': 0,
                'std': 0
            })

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.arange(len(labels))

    # MAE by range
    maes = [s['mae'] for s in stats]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(maes)))
    axes[0].bar(x, maes, color=colors, edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel('MAE')
    axes[0].set_title('Mean Absolute Error by Exposure Range')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Bias by range
    biases = [s['bias'] for s in stats]
    colors_bias = ['green' if b < 0 else 'red' for b in biases]
    axes[1].bar(x, biases, color=colors_bias, edgecolor='black', alpha=0.7)
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel('Mean Error (Bias)')
    axes[1].set_title('Prediction Bias by Exposure Range\n(+: overpredicts, -: underpredicts)')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Sample distribution
    counts = [s['pct'] for s in stats]
    axes[2].bar(x, counts, color='steelblue', edgecolor='black')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=9)
    axes[2].set_ylabel('Percentage of Samples')
    axes[2].set_title('Sample Distribution by Exposure Range')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved error by exposure range to {save_path}")
    else:
        plt.close()

    return stats


def plot_per_protein_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protein_ids: List[str],
    batch_ptr: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot distribution of R² scores across individual proteins.

    Args:
        y_true: True exposure values (all atoms concatenated)
        y_pred: Predicted exposure values (all atoms concatenated)
        protein_ids: List of protein IDs
        batch_ptr: Pointer array indicating protein boundaries
        save_path: Path to save figure (non-blocking)
    """
    from sklearn.metrics import r2_score

    r2_scores = []
    valid_proteins = []

    for i, pid in enumerate(protein_ids):
        start = batch_ptr[i] if i < len(batch_ptr) else 0
        end = batch_ptr[i + 1] if i + 1 < len(batch_ptr) else len(y_true)

        y_t = y_true[start:end]
        y_p = y_pred[start:end]

        if len(y_t) > 1 and np.var(y_t) > 0:
            r2 = r2_score(y_t, y_p)
            r2_scores.append(r2)
            valid_proteins.append(pid)

    r2_scores = np.array(r2_scores)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(r2_scores, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(np.median(r2_scores), color='red', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(r2_scores):.4f}')
    axes[0].axvline(np.mean(r2_scores), color='orange', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(r2_scores):.4f}')
    axes[0].set_xlabel('R² Score')
    axes[0].set_ylabel('Number of Proteins')
    axes[0].set_title(f'Per-Protein R² Distribution (n={len(r2_scores)})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot + violin
    parts = axes[1].violinplot(r2_scores, positions=[1], showmeans=True, showmedians=True)
    axes[1].set_ylabel('R² Score')
    axes[1].set_title('R² Score Distribution')
    axes[1].set_xticks([1])
    axes[1].set_xticklabels(['All Proteins'])
    axes[1].grid(True, alpha=0.3, axis='y')

    # Stats annotation
    stats_text = f'Min: {r2_scores.min():.4f}\nMax: {r2_scores.max():.4f}\nStd: {r2_scores.std():.4f}'
    axes[1].text(1.3, np.median(r2_scores), stats_text, fontsize=10, verticalalignment='center')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved per-protein R² distribution to {save_path}")
    else:
        plt.close()

    return r2_scores, valid_proteins


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
