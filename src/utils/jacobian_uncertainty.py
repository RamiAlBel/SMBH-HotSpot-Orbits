"""Jacobian/Hessian-based uncertainty propagation for trained SMBH models.

Given a trained model and input noise levels (σ_r, σ_T, σ_DPA), provides
analytic uncertainty estimates via first-order (Jacobian) and second-order
(Hessian) propagation, with Monte Carlo ground truth for comparison.

All computations happen in normalised space (StandardScaler applied); the
output σ is returned in the original target units.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Low-level autodiff helpers
# ---------------------------------------------------------------------------

def compute_jacobian(model: nn.Module, x_norm_1d: torch.Tensor) -> torch.Tensor:
    """Return ∂y_norm/∂x_norm for a single normalised input vector.

    Args:
        model:      Trained model in eval() mode.
        x_norm_1d: Normalised input, shape (input_dim,). Must be on same
                   device as model.

    Returns:
        Jacobian tensor of shape (input_dim,).
    """
    model.eval()
    x = x_norm_1d.clone().detach().requires_grad_(True)
    y = model(x.unsqueeze(0)).squeeze()
    y.backward()
    return x.grad.detach()


def compute_hessian(model: nn.Module, x_norm_1d: torch.Tensor) -> torch.Tensor:
    """Return ∂²y_norm/∂x_norm² for a single normalised input vector.

    Args:
        model:      Trained model in eval() mode.
        x_norm_1d: Normalised input, shape (input_dim,). CPU tensor recommended
                   for torch.autograd.functional.hessian compatibility.

    Returns:
        Hessian tensor of shape (input_dim, input_dim).
    """
    model.eval()

    def scalar_fn(x: torch.Tensor) -> torch.Tensor:
        return model(x.unsqueeze(0)).squeeze()

    H = torch.autograd.functional.hessian(scalar_fn, x_norm_1d.clone().detach())
    return H.detach()


# ---------------------------------------------------------------------------
# Uncertainty estimators
# ---------------------------------------------------------------------------

def jacobian_sigma(
    J_norm: np.ndarray,
    sigma_obs_orig: np.ndarray,
    scaler_X_scale: np.ndarray,
    scaler_y_scale: float,
) -> float:
    """First-order analytic σ_y from Jacobian in normalised space.

    Math (all in normalised space):
        σ_x_norm_i = σ_obs_orig_i / σ_X_i
        σ_y = σ_y_scale * sqrt( Σ_i J_norm_i² σ_x_norm_i² )

    Args:
        J_norm:          Jacobian ∂y_norm/∂x_norm, shape (input_dim,).
        sigma_obs_orig:  Observational σ in original units, shape (input_dim,).
        scaler_X_scale:  StandardScaler σ_X per feature, shape (input_dim,).
        scaler_y_scale:  StandardScaler σ_y (scalar).

    Returns:
        σ_y in original target units (float).
    """
    sigma_x_norm = sigma_obs_orig / scaler_X_scale
    return float(scaler_y_scale * np.sqrt(np.sum(J_norm ** 2 * sigma_x_norm ** 2)))


def mc_sigma(
    model: nn.Module,
    x_orig_1d: np.ndarray,
    sigma_obs_orig: np.ndarray,
    scaler_X_mean: np.ndarray,
    scaler_X_scale: np.ndarray,
    scaler_y_scale: float,
    n_mc: int = 2000,
    device: str = "cpu",
) -> float:
    """Monte Carlo σ_y: std of model outputs over perturbed inputs.

    Args:
        model:           Trained model in eval() mode.
        x_orig_1d:       Clean input in original units, shape (input_dim,).
        sigma_obs_orig:  Observational σ in original units, shape (input_dim,).
        scaler_X_mean:   StandardScaler μ_X, shape (input_dim,).
        scaler_X_scale:  StandardScaler σ_X, shape (input_dim,).
        scaler_y_scale:  StandardScaler σ_y (scalar).
        n_mc:            Number of Monte Carlo draws.
        device:          Torch device string.

    Returns:
        σ_y in original target units (float).
    """
    model.eval()
    rng = np.random.default_rng()
    x_noisy = x_orig_1d[None] + rng.standard_normal((n_mc, len(x_orig_1d))) * sigma_obs_orig[None]
    x_noisy_norm = (x_noisy - scaler_X_mean) / scaler_X_scale
    x_tensor = torch.tensor(x_noisy_norm, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_norm_samples = model(x_tensor).cpu().numpy().squeeze()
    y_samples = y_norm_samples * scaler_y_scale
    return float(np.std(y_samples))


def analyze_sample(
    model: nn.Module,
    x_orig_1d: np.ndarray,
    sigma_obs_orig: np.ndarray,
    scaler_X_mean: np.ndarray,
    scaler_X_scale: np.ndarray,
    scaler_y_mean: float,
    scaler_y_scale: float,
    n_mc: int = 2000,
    device: str = "cpu",
) -> dict:
    """Full Jacobian + Hessian + MC analysis for one test sample.

    Decision tree:
        ratio ∈ [0.8, 1.0]          → method = "jacobian"
        ratio > 1.0                  → method = "suspicious"
        relative_correction < 0.10   → method = "jacobian+hessian"
        else                         → method = "mc"

    Args:
        model:           Trained model in eval() mode.
        x_orig_1d:       Clean input in original units, shape (input_dim,).
        sigma_obs_orig:  Observational σ in original units, shape (input_dim,).
        scaler_X_mean:   StandardScaler μ_X, shape (input_dim,).
        scaler_X_scale:  StandardScaler σ_X, shape (input_dim,).
        scaler_y_mean:   StandardScaler μ_y (scalar, unused but kept for API symmetry).
        scaler_y_scale:  StandardScaler σ_y (scalar).
        n_mc:            Monte Carlo draws.
        device:          Torch device string.

    Returns:
        dict with keys: sigma_jacobian, sigma_mc, ratio,
        hessian_correction_pct, method, J_norm (numpy array).
    """
    model.eval()

    # Normalise input
    x_norm = (x_orig_1d - scaler_X_mean) / scaler_X_scale
    sigma_x_norm = sigma_obs_orig / scaler_X_scale

    # --- Jacobian ---
    x_norm_tensor = torch.tensor(x_norm, dtype=torch.float32).to(device)
    J_norm = compute_jacobian(model, x_norm_tensor).cpu().numpy()
    sigma_jac = jacobian_sigma(J_norm, sigma_obs_orig, scaler_X_scale, scaler_y_scale)

    # --- MC ground truth ---
    sigma_mc_val = mc_sigma(
        model, x_orig_1d, sigma_obs_orig,
        scaler_X_mean, scaler_X_scale, scaler_y_scale,
        n_mc=n_mc, device=device,
    )

    # --- Ratio ---
    if sigma_mc_val > 1e-12:
        ratio = sigma_jac / sigma_mc_val
    else:
        ratio = float("nan")

    # --- Hessian (in normalised space, on CPU for functional.hessian) ---
    x_norm_cpu = torch.tensor(x_norm, dtype=torch.float32)
    # Temporarily move model to CPU for hessian computation if needed
    original_device = next(model.parameters()).device
    model_cpu = model.to("cpu")
    H_norm = compute_hessian(model_cpu, x_norm_cpu).cpu().numpy()
    model.to(original_device)  # restore

    jacobian_term = float(np.sum(J_norm ** 2 * sigma_x_norm ** 2))
    hessian_correction = 0.5 * float(
        np.sum(H_norm ** 2 * np.outer(sigma_x_norm ** 2, sigma_x_norm ** 2))
    )
    if jacobian_term > 1e-12:
        relative_correction = hessian_correction / jacobian_term
    else:
        relative_correction = float("nan")

    # --- Decision ---
    if not np.isnan(ratio) and 0.8 <= ratio <= 1.0:
        method = "jacobian"
    elif not np.isnan(ratio) and ratio > 1.0:
        method = "suspicious"
    elif not np.isnan(relative_correction) and relative_correction < 0.10:
        method = "jacobian+hessian"
    else:
        method = "mc"

    return {
        "sigma_jacobian": sigma_jac,
        "sigma_mc": sigma_mc_val,
        "ratio": ratio,
        "hessian_correction_pct": relative_correction * 100.0 if not np.isnan(relative_correction) else float("nan"),
        "method": method,
        "J_norm": J_norm,
    }


# ---------------------------------------------------------------------------
# Batch analysis helper
# ---------------------------------------------------------------------------

def analyze_test_set(
    model: nn.Module,
    X_test_raw: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    sigma_obs_orig: np.ndarray,
    scaler_X_mean: np.ndarray,
    scaler_X_scale: np.ndarray,
    scaler_y_mean: float,
    scaler_y_scale: float,
    n_mc: int = 2000,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[list[dict], np.ndarray]:
    """Run analyze_sample over the full test set.

    Args:
        model:          Trained model in eval() mode.
        X_test_raw:     Clean test features in original units, (n_test, input_dim).
        y_test_true:    True targets in original units, (n_test,).
        y_test_pred:    Predicted targets in original units, (n_test,).
        sigma_obs_orig: Observational σ vector, shape (input_dim,).
        scaler_X_mean:  StandardScaler μ_X, shape (input_dim,).
        scaler_X_scale: StandardScaler σ_X, shape (input_dim,).
        scaler_y_mean:  StandardScaler μ_y (scalar).
        scaler_y_scale: StandardScaler σ_y (scalar).
        n_mc:           MC draws per sample.
        device:         Torch device.
        verbose:        Print progress every 50 samples.

    Returns:
        (records, jacobians_array) where:
          records: list of dicts (one per sample) with analysis results.
          jacobians_array: numpy array of shape (n_test, input_dim) — normalised Jacobians.
    """
    n_test = len(X_test_raw)
    records = []
    jacobians = []

    for i in range(n_test):
        if verbose and i % 50 == 0:
            print(f"  Jacobian analysis: sample {i}/{n_test}", flush=True)

        result = analyze_sample(
            model=model,
            x_orig_1d=X_test_raw[i],
            sigma_obs_orig=sigma_obs_orig,
            scaler_X_mean=scaler_X_mean,
            scaler_X_scale=scaler_X_scale,
            scaler_y_mean=scaler_y_mean,
            scaler_y_scale=scaler_y_scale,
            n_mc=n_mc,
            device=device,
        )
        records.append({
            "sample_idx": i,
            "y_true": float(y_test_true[i]),
            "y_pred": float(y_test_pred[i]),
            "sigma_jacobian": result["sigma_jacobian"],
            "sigma_mc": result["sigma_mc"],
            "ratio": result["ratio"],
            "hessian_correction_pct": result["hessian_correction_pct"],
            "method": result["method"],
        })
        jacobians.append(result["J_norm"])

    return records, np.array(jacobians)


def aggregate_jacobian_results(records: list[dict], target_name: str) -> dict:
    """Compute aggregate statistics from per-sample analysis records."""
    ratios = np.array([r["ratio"] for r in records if not np.isnan(r["ratio"])])
    hess_pcts = np.array([r["hessian_correction_pct"] for r in records if not np.isnan(r["hessian_correction_pct"])])
    sigma_jacs = np.array([r["sigma_jacobian"] for r in records])
    sigma_mcs = np.array([r["sigma_mc"] for r in records])
    methods = [r["method"] for r in records]

    n_reliable = sum(1 for m in methods if m == "jacobian")
    frac_reliable = n_reliable / len(methods) if methods else float("nan")

    return {
        "target": target_name,
        "ratio_mean": float(np.mean(ratios)) if len(ratios) else float("nan"),
        "ratio_std": float(np.std(ratios)) if len(ratios) else float("nan"),
        "frac_reliable": frac_reliable,
        "hessian_correction_mean": float(np.mean(hess_pcts)) if len(hess_pcts) else float("nan"),
        "sigma_jacobian_mean": float(np.mean(sigma_jacs)),
        "sigma_mc_mean": float(np.mean(sigma_mcs)),
    }
