"""
Evaluation metrics and utilities for GNN models
"""
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import numpy as np
import sys
from typing import Dict, Tuple
from tqdm import tqdm


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[Dict[str, Dict[str, float]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset with both raw and clamped metrics.

    Atom exposure cannot be negative, so we compute metrics on both:
    - Raw predictions (for model diagnosis)
    - Clamped predictions (final metrics, clamped to [0, inf))

    Args:
        model (nn.Module): Model to evaluate
        data_loader (DataLoader): Data loader
        device (str): Device to evaluate on

    Returns:
        tuple: (metrics dict with 'raw' and 'clamped' keys,
                y_true array, y_pred array, y_pred_clamped array)
    """
    model.eval()
    model = model.to(device)

    all_preds = []
    all_targets = []

    for batch in tqdm(data_loader, desc='Evaluating', mininterval=0.5, leave=False, disable=not sys.stdout.isatty(), ncols=80, ascii=True):
        batch = batch.to(device)

        # Get embedding indices if present
        element_idx = getattr(batch, 'element_idx', None)
        residue_idx = getattr(batch, 'residue_idx', None)

        # Forward pass
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                    element_idx=element_idx, residue_idx=residue_idx)

        # Store predictions and targets
        all_preds.append(out.cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())

    # Concatenate all predictions and targets
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    # Clamp predictions: exposure cannot be negative
    y_pred_clamped = np.maximum(y_pred, 0.0)

    # Compute metrics for both raw and clamped
    metrics = {
        'raw': compute_metrics(y_true, y_pred),
        'clamped': compute_metrics(y_true, y_pred_clamped)
    }

    # Count negative predictions for diagnostics
    n_negative = np.sum(y_pred < 0)
    metrics['diagnostics'] = {
        'n_negative_predictions': int(n_negative),
        'pct_negative_predictions': float(n_negative / len(y_pred) * 100),
        'min_prediction': float(y_pred.min()),
        'max_prediction': float(y_pred.max())
    }

    return metrics, y_true, y_pred, y_pred_clamped


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        y_true (np.ndarray): Ground truth values
        y_pred (np.ndarray): Predicted values

    Returns:
        dict: Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Pearson correlation
    pearson_corr, pearson_pval = pearsonr(y_true, y_pred)

    # Mean and std of errors
    errors = y_pred - y_true
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    # Median absolute error
    median_ae = np.median(np.abs(errors))

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'pearson_corr': pearson_corr,
        'pearson_pval': pearson_pval,
        'mean_error': mean_error,
        'std_error': std_error,
        'median_ae': median_ae
    }


def print_metrics(metrics: Dict[str, Dict[str, float]]):
    """
    Print evaluation metrics with clear RAW vs CLAMPED separation.

    Args:
        metrics (dict): Dictionary with 'raw', 'clamped', and 'diagnostics' keys
    """
    raw = metrics['raw']
    clamped = metrics['clamped']
    diag = metrics.get('diagnostics', {})

    width = 70

    # === RAW RESULTS (for diagnosis) ===
    print("\n" + "=" * width)
    print("RAW PREDICTIONS (for model diagnosis)")
    print("=" * width)
    print(f"  Mean Absolute Error (MAE):     {raw['mae']:.4f}")
    print(f"  Root Mean Squared Error:       {raw['rmse']:.4f}")
    print(f"  R² Score:                      {raw['r2']:.4f}")
    print(f"  Pearson Correlation:           {raw['pearson_corr']:.4f}")
    print(f"  Median Absolute Error:         {raw['median_ae']:.4f}")
    print(f"  Mean Error (Bias):             {raw['mean_error']:+.4f}")
    print(f"  Std Error:                     {raw['std_error']:.4f}")

    # Diagnostics about negative predictions
    if diag:
        print()
        print(f"  Negative predictions: {diag['n_negative_predictions']} ({diag['pct_negative_predictions']:.2f}%)")
        print(f"  Prediction range: [{diag['min_prediction']:.4f}, {diag['max_prediction']:.4f}]")

    # === CLAMPED RESULTS (final metrics) ===
    print("\n" + "=" * width)
    print("CLAMPED PREDICTIONS (final metrics, exposure >= 0)")
    print("=" * width)
    print(f"  Mean Absolute Error (MAE):     {clamped['mae']:.4f}")
    print(f"  Root Mean Squared Error:       {clamped['rmse']:.4f}")
    print(f"  R² Score:                      {clamped['r2']:.4f}")
    print(f"  Pearson Correlation:           {clamped['pearson_corr']:.4f}")
    print(f"  Median Absolute Error:         {clamped['median_ae']:.4f}")
    print(f"  Mean Error (Bias):             {clamped['mean_error']:+.4f}")
    print(f"  Std Error:                     {clamped['std_error']:.4f}")
    print("=" * width + "\n")


if __name__ == '__main__':
    from src.data.dataset import ProteinAtomDataset
    from src.models.gnn import AtomExposureGNN

    # Load test dataset
    test_dataset = ProteinAtomDataset(root='dataset/', split='test')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Load model (assuming checkpoint exists)
    model = AtomExposureGNN(in_channels=80, hidden_channels=128, num_layers=3)
    # model.load_state_dict(torch.load('experiments/checkpoints/best_model.pt')['model_state_dict'])

    # Evaluate (returns metrics, y_true, y_pred, y_pred_clamped)
    metrics, y_true, y_pred, y_pred_clamped = evaluate_model(model, test_loader)
    print_metrics(metrics)
