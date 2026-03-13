# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repo contains a machine learning framework for predicting supermassive black hole (SMBH) parameters (spin α, inclination i, theta θ) from hotspot observables (radius r, period T, differential phase angle ΔPA(t)).

There is one directory:
- **`smbh_hotspots_repository/`** — The clean, structured ML framework (primary working directory)

## Installation

```bash
cd /scratch/ralbe/meniar_and_django/smbh_hotspots_repository
pip install -r requirements.txt
```

## Running Experiments

All commands should be run from the repo root (`smbh_hotspots_repository/`), not from the experiment directory. Each experiment has a `train.py` and an optional config argument:

```bash
# Run with default config
cd experiments/experiment_1_eq_avg && python train.py

# Run with a specific config (e.g., no noise)
cd experiments/experiment_1_eq_avg && python train.py config_no_noise.yaml

# Submit to SLURM
cd experiments/experiment_1_eq_avg && sbatch submit.sh

# Submit all experiments at once
cd experiments && bash submit_all.sh
```

Monitor SLURM jobs:
```bash
squeue -u $USER
tail -f results/logs/exp1_eq_avg_*.out
```

## Data Preparation

Raw `.dat` simulation files must be placed before running preprocessing:
- Equatorial (i=0): `data/raw/meniar_files/`
- Equatorial dense: `data/raw/sepray_t_files/`
- Non-equatorial: `data/raw/non_eq/dpa_neq/`

```bash
cd src/preprocessing
python prepare_dataset_i0.py          # → data/processed/dpa_dataset_i0.csv (Exp I)
python prepare_dataset_ultradense.py  # → data/processed/dpa_dataset_ultradense.csv (Exp II-III)
python prepare_dataset_noneq.py       # → data/processed/dpa_dataset_noneq.csv (Exp IV-V)
```

## Architecture

### Core modules (`src/`)

- **`models/regression_head.py`** — `RegressionHead`: MLP with configurable residual blocks, BatchNorm, Dropout. Input: `2 + k` features (r, T, k DPA samples). Output: 1 scalar.
- **`training/data_loader.py`** — `build_features_targets_avg()` (Exp I) and `build_features_targets_timeseries()` (Exp II-V); `prepare_dataloaders()` handles train/val/test splitting, noise injection, and StandardScaler fitting.
- **`training/trainer.py`** — `Trainer` class: Adam optimizer, MSE loss, early stopping, checkpoint saving. Checkpoints include scaler parameters for deployment.
- **`training/evaluation.py`** — Metrics (MAE, RMSE, R², error μ/σ), plotting (error histograms, pred-vs-actual), result aggregation and CSV export.
- **`utils/noise.py`** — `add_noise()`: Gaussian noise on r, T, DPA features. Supports independent Gaussian (default) or GP-based smooth noise via `dpa_length_scale > 0`.
- **`utils/config.py`** — `load_config()` and `get_repo_root()` utilities.
- **`postprocessing/`** — Scripts for corner plots and aggregated experiment result analysis.

### Experiment structure

Each experiment in `experiments/experiment_N_*/` contains:
- `config.yaml` — All hyperparameters (model, training, noise, split, sweep)
- `config_no_noise.yaml` — Same config with `noise.enabled: false`
- `train.py` — Self-contained training script (adds `repo_root` to `sys.path`)
- `submit.sh` / `submit_no_noise.sh` — SLURM submission scripts

### Experiment types

| Exp | Description | Dataset | Input features | Targets |
|-----|-------------|---------|----------------|---------|
| I | Equatorial averaged ΔPA | `dpa_dataset_i0.csv` | r, T, ΔPA_avg | α, i |
| II | Equatorial full orbit | `dpa_dataset_ultradense.csv` | r, T, ΔPA(t)×10 | α, i |
| III | Equatorial partial orbit + sweep | `dpa_dataset_ultradense.csv` | r, T, ΔPA(t)×k | α, i |
| IV | Non-equatorial full orbit | `dpa_dataset_noneq.csv` | r, T, ΔPA(t)×10 | α, i, θ, z |
| V | Non-equatorial partial orbit + sweep | `dpa_dataset_noneq.csv` | r, T, ΔPA(t)×k | α, i, θ, z |

### Data flow

1. Raw `.dat` files → preprocessing scripts → CSVs in `data/processed/`
2. CSV → `build_features_targets_*()` → numpy arrays
3. Arrays → `prepare_dataloaders()` → noise injection → StandardScaler normalization → DataLoaders
4. DataLoaders → `Trainer.train()` → best model (by val loss, early stopping)
5. Best model → `evaluate_model()` → metrics + plots saved to `results/`

### Noise injection

Noise is injected **before** z-normalization. Default sigmas: r=0.1M, T=2 min, ΔPA=5°. Fresh noise is sampled each time `prepare_dataloaders()` is called (i.e., per seed).

### Sweep mode (Experiments III & V)

When `sweep.enabled: true` in config, trains 50 models (10 orbit coverage levels × 5 seeds) and generates σ-vs-orbit-coverage plots. This is compute-intensive (~24-36h on SLURM).

### Output locations

```
results/
├── checkpoints/experiment_N/{target}/model_seed{seed}.pth
├── figures/experiment_N/{target}/error_hist_seed{seed}.png
├── figures/experiment_N/{target}/pred_vs_actual_seed{seed}.png
├── logs/                          # SLURM stdout/stderr
└── metrics/experiment_N/
    ├── {target}_summary.csv       # One row per seed
    ├── {target}_aggregated.csv    # μ±σ across seeds
    └── {target}_sweep.csv         # Sweep mode only
```

### WandB

Enabled by default (`use_wandb: true` in config). Set to `false` to disable, or ensure `wandb login` has been run first.

## Key Configuration Options

```yaml
model:
  hidden_dims: [256, 256]   # Layer widths
  num_blocks: 2             # Residual blocks
  dropout: 0.1

training:
  seeds: [42, 43, 44, 45, 46]   # 5 seeds → μ±σ statistics
  early_stop_patience: 40
  use_wandb: true

noise:
  enabled: true
  sigma_r: 0.1      # Units of M
  sigma_T: 2.0      # Minutes
  sigma_DPA: 5.0    # Degrees

sweep:
  enabled: false    # Set true for σ vs orbit coverage analysis
```
