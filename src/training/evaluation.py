import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import pandas as pd


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    scaler_y: StandardScaler,
    device: str = 'cuda'
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate model and return metrics + predictions."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).squeeze().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(batch_y.numpy())
    
    preds_scaled = np.concatenate(all_preds)
    targets_scaled = np.concatenate(all_targets)
    
    preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    targets = scaler_y.inverse_transform(targets_scaled.reshape(-1, 1)).ravel()
    
    errors = targets - preds
    
    metrics = {
        'mae': mean_absolute_error(targets, preds),
        'rmse': np.sqrt(mean_squared_error(targets, preds)),
        'r2': r2_score(targets, preds),
        'error_mean': float(np.mean(errors)),
        'error_std': float(np.std(errors))
    }
    
    return metrics, targets, preds


def plot_error_histogram(
    errors: np.ndarray,
    target_name: str,
    save_path: str
):
    """Plot error distribution histogram with Gaussian fit."""
    mu = float(np.mean(errors))
    sigma = float(np.std(errors))
    n = len(errors)
    se_mu = sigma / np.sqrt(n) if n > 1 else 0.0
    se_sigma = sigma / np.sqrt(2 * (n - 1)) if n > 2 else 0.0
    
    plt.figure(figsize=(8, 5))
    counts, bins, _ = plt.hist(errors, bins=40, density=True, alpha=0.8, label='Errors')
    
    if sigma > 0:
        x = np.linspace(bins[0], bins[-1], 200)
        pdf = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        plt.plot(x, pdf, 'r-', lw=2, label='Gaussian fit')
        
        ax = plt.gca()
        ax.axvline(mu, color='k', linestyle='--', linewidth=1.2)
        ax.axvline(mu - sigma, color='k', linestyle=':', linewidth=1.0, alpha=0.7)
        ax.axvline(mu + sigma, color='k', linestyle=':', linewidth=1.0, alpha=0.7)
        ax.text(
            0.98, 0.95,
            f'μ={mu:.3f}±{se_mu:.3f}\nσ={sigma:.3f}±{se_sigma:.3f}',
            transform=ax.transAxes,
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    plt.title(f'{target_name} Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pred_vs_actual(
    targets: np.ndarray,
    preds: np.ndarray,
    target_name: str,
    save_path: str
):
    """Plot predicted vs actual scatter plot."""
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, alpha=0.6)
    
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    plt.plot(lims, lims, 'r--', linewidth=1.5)
    
    r2 = r2_score(targets, preds)
    plt.title(f'{target_name}: Predicted vs Actual (R²={r2:.3f})')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def aggregate_results(
    results: list[Dict[str, float]],
    target_name: str
) -> Dict[str, float]:
    """Aggregate results from multiple runs."""
    keys = ['mae', 'rmse', 'r2', 'error_std']
    aggregated = {}
    
    for key in keys:
        values = [r[key] for r in results]
        aggregated[f'{key}_mean'] = np.mean(values)
        aggregated[f'{key}_std'] = np.std(values, ddof=1)
    
    return aggregated


def save_results_csv(
    results: list[Dict],
    save_path: str
):
    """Save results to CSV file."""
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)


def plot_sigma_vs_orbit_inclusion(
    sigma_values: list[float],
    percentages: list[int],
    target_name: str,
    save_path: str
):
    """Plot prediction error std vs percentage of orbit included."""
    plt.figure(figsize=(8, 6))
    plt.plot(percentages, sigma_values, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Orbit Included (%)')
    plt.ylabel('Prediction Error σ')
    plt.title(f'{target_name}: Error vs Orbit Coverage')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
