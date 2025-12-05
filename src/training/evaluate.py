"""
Evaluation metrics and utilities for GNN models
"""
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import numpy as np
from typing import Dict, Tuple
from tqdm import tqdm


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Args:
        model (nn.Module): Model to evaluate
        data_loader (DataLoader): Data loader
        device (str): Device to evaluate on

    Returns:
        dict: Dictionary of evaluation metrics
    """
    model.eval()
    model = model.to(device)

    all_preds = []
    all_targets = []

    for batch in tqdm(data_loader, desc='Evaluating'):
        batch = batch.to(device)

        # Forward pass
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # Store predictions and targets
        all_preds.append(out.cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())

    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Compute metrics
    metrics = compute_metrics(all_targets, all_preds)

    return metrics


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


@torch.no_grad()
def predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions for a dataset.

    Args:
        model (nn.Module): Model
        data_loader (DataLoader): Data loader
        device (str): Device

    Returns:
        tuple: (predictions, targets)
    """
    model.eval()
    model = model.to(device)

    all_preds = []
    all_targets = []

    for batch in tqdm(data_loader, desc='Predicting'):
        batch = batch.to(device)

        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        all_preds.append(out.cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def print_metrics(metrics: Dict[str, float]):
    """
    Pretty print evaluation metrics.

    Args:
        metrics (dict): Dictionary of metrics
    """
    print("\n" + "=" * 50)
    print("EVALUATION METRICS")
    print("=" * 50)
    print(f"Mean Absolute Error (MAE):     {metrics['mae']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
    print(f"R² Score:                       {metrics['r2']:.4f}")
    print(f"Pearson Correlation:            {metrics['pearson_corr']:.4f}")
    print(f"Median Absolute Error:          {metrics['median_ae']:.4f}")
    print(f"Mean Error:                     {metrics['mean_error']:.4f}")
    print(f"Std Error:                      {metrics['std_error']:.4f}")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    from src.data.dataset import ProteinAtomDataset
    from src.models.gnn import AtomExposureGNN

    # Load test dataset
    test_dataset = ProteinAtomDataset(root='dataset/', split='test')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Load model (assuming checkpoint exists)
    model = AtomExposureGNN(in_channels=80, hidden_channels=128, num_layers=3)
    # model.load_state_dict(torch.load('experiments/checkpoints/best_model.pt')['model_state_dict'])

    # Evaluate
    metrics = evaluate_model(model, test_loader)
    print_metrics(metrics)
