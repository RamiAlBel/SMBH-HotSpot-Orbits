#!/usr/bin/env python3
"""Experiment VII: Non-equatorial noise sweep — one SLURM array job per noise combo.

Usage:
    python train.py <task_id>      # task_id in 0..124
    python train.py                # defaults to SLURM_ARRAY_TASK_ID env var
"""
import os
import sys
import csv
from itertools import product
from pathlib import Path

import pandas as pd
import torch

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from src.utils.config import load_config, get_repo_root
from src.models.regression_head import RegressionHead
from src.training.data_loader import build_features_targets_timeseries, prepare_dataloaders
from src.training.trainer import Trainer
from src.training.evaluation import evaluate_model


def get_combo(task_id: int, cfg: dict) -> tuple[float, float, float]:
    """Map flat task_id (0..124) to (sigma_T, sigma_r, sigma_DPA)."""
    T_vals   = cfg['sweep']['sigma_T_values']
    r_vals   = cfg['sweep']['sigma_r_values']
    DPA_vals = cfg['sweep']['sigma_DPA_values']
    combos = list(product(T_vals, r_vals, DPA_vals))
    return combos[task_id]


def main():
    # ------------------------------------------------------------------ setup
    if len(sys.argv) > 1:
        task_id = int(sys.argv[1])
    else:
        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

    config_path = Path(__file__).parent / "config.yaml"
    cfg = load_config(config_path)
    root = get_repo_root()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sigma_T, sigma_r, sigma_DPA = get_combo(task_id, cfg)
    print(f"Task {task_id}: sigma_T={sigma_T}, sigma_r={sigma_r}, sigma_DPA={sigma_DPA}")
    print(f"Device: {device}")

    # ----------------------------------------------------------------- data
    dataset_path = root / cfg['data']['dataset_path']
    df = pd.read_csv(dataset_path)
    print(f"Dataset: {len(df)} rows")

    exp_name = cfg['experiment']['name']
    results_dir = root / "results"
    metrics_dir = results_dir / "metrics" / exp_name
    metrics_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg['training']['seeds'][0]  # single seed per combo

    row = {
        'task_id': task_id,
        'sigma_T': sigma_T,
        'sigma_r': sigma_r,
        'sigma_DPA': sigma_DPA,
    }

    # ---------------------------------------------------------- per target
    for target_cfg in cfg['targets']:
        target_name   = target_cfg['name']
        target_column = target_cfg['column']
        to_radians    = target_cfg['convert_to_radians']

        print(f"\n{'='*55}")
        print(f"Target: {target_name}")
        print(f"{'='*55}")

        features, targets, metadata = build_features_targets_timeseries(
            df, target_name, target_column,
            num_samples=cfg['data']['num_samples'],
            convert_to_radians=to_radians,
            half_orbit=cfg['data']['half_orbit']
        )
        print(f"Features: {features.shape}, Targets: {targets.shape}")

        torch.manual_seed(seed)

        train_loader, val_loader, test_loader, scaler_X, scaler_y, _ = prepare_dataloaders(
            features, targets,
            batch_size=cfg['training']['batch_size'],
            train_ratio=cfg['split']['train'],
            val_ratio=cfg['split']['val'],
            random_seed=seed,
            noise_enabled=True,
            sigma_r=sigma_r,
            sigma_T=sigma_T,
            sigma_DPA=sigma_DPA,
            dpa_length_scale=cfg['noise'].get('dpa_length_scale', 0.0)
        )

        model = RegressionHead(
            input_dim=features.shape[1],
            hidden_dims=tuple(cfg['model']['hidden_dims']),
            num_blocks=cfg['model']['num_blocks'],
            dropout=cfg['model']['dropout']
        )

        trainer = Trainer(
            model, train_loader, val_loader,
            learning_rate=cfg['training']['learning_rate'],
            weight_decay=cfg['training']['weight_decay'],
            early_stop_patience=cfg['training']['early_stop_patience'],
            device=device
        )

        trainer.train(
            epochs=cfg['training']['epochs'],
            use_wandb=False,
            verbose=True
        )

        test_metrics, _, _ = evaluate_model(model, test_loader, scaler_y, device)

        print(f"MAE={test_metrics['mae']:.4f}  RMSE={test_metrics['rmse']:.4f}  "
              f"R²={test_metrics['r2']:.4f}  σ={test_metrics['error_std']:.4f}")

        # Save checkpoint
        ckpt_dir = results_dir / "checkpoints" / exp_name / target_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = f"sT{sigma_T}_sr{sigma_r}_sdpa{sigma_DPA}.pth"
        trainer.save_checkpoint(
            str(ckpt_dir / ckpt_name),
            scaler_X_mean=scaler_X.mean_,
            scaler_X_scale=scaler_X.scale_,
            scaler_y_mean=scaler_y.mean_,
            scaler_y_scale=scaler_y.scale_
        )

        # Accumulate into result row
        for metric, val in test_metrics.items():
            row[f"{target_name}_{metric}"] = val

    # -------------------------------------------------- write per-combo CSV
    out_path = metrics_dir / f"combo_{task_id}.csv"
    fieldnames = list(row.keys())
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    print(f"\nSaved: {out_path}")
    print(f"Task {task_id} complete.")


if __name__ == "__main__":
    main()
