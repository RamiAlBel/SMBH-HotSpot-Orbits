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

Each experiment has a `train.py` plus one or more config YAMLs. The non-equatorial experiments (4 and 5) ship two configs each — `config_neq30.yaml` (theta in [-30,30] dataset, legacy) and `config_neq45.yaml` (theta in [-45,45] dataset, default). The result path is driven by `experiment.name` inside each config, so two configs in the same folder produce two separate result subdirectories.

```bash
# Equatorial experiments (single config, with optional no-noise variant)
cd experiments/experiment_1_eq_avg && python train.py
cd experiments/experiment_1_eq_avg && python train.py config_no_noise.yaml

# Non-equatorial: pass the variant explicitly
cd experiments/experiment_4_noneq_full && python train.py config_neq45.yaml
cd experiments/experiment_4_noneq_full && python train.py config_neq30_no_noise.yaml

# Submit to SLURM (each variant has its own submit_*.sh)
cd experiments/experiment_4_noneq_full && sbatch submit_neq45.sh

# Submit all main experiments (defaults to neq45 for exp 4 and 5)
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

```
experiments/
├── experiment_1_eq_avg/                  # Exp I
├── experiment_2_eq_full/                 # Exp II
├── experiment_3_eq_partial/              # Exp III (sweep over orbit coverage)
├── experiment_4_noneq_full/              # Exp IV — two datasets (neq30, neq45)
│   ├── config_neq30{,_no_noise}.yaml
│   ├── config_neq45{,_no_noise,_noise10}.yaml
│   └── submit_neq30*.sh / submit_neq45*.sh
├── experiment_5_noneq_partial/           # Exp V — two datasets (neq30, neq45)
│   ├── config_neq30{,_no_noise}.yaml
│   ├── config_neq45{,_no_noise}.yaml
│   └── submit_neq30*.sh / submit_neq45*.sh
└── uncertainty_experiments/              # Noise sweeps + Jacobian-noise studies
    ├── eq_noise_sweep/
    ├── neq30_noise_sweep/
    ├── neq45_noise_sweep/
    ├── eq_jacobian/
    └── neq30_jacobian/
```

Each experiment folder contains a self-contained `train.py` (adds `repo_root` to `sys.path`), one or more `config*.yaml` files, and matching `submit*.sh` SLURM scripts.

### Experiment types

| Exp | Description | Dataset | Input features | Targets |
|-----|-------------|---------|----------------|---------|
| I | Equatorial averaged ΔPA | `dpa_dataset_i0.csv` | r, T, ΔPA_avg | α, i |
| II | Equatorial full orbit | `dpa_dataset_ultradense.csv` | r, T, ΔPA(t)×10 | α, i |
| III | Equatorial partial orbit + sweep | `dpa_dataset_ultradense.csv` | r, T, ΔPA(t)×k | α, i |
| IV | Non-equatorial full orbit | `dpa_dataset_noneq.csv` (neq30) **or** `dpa_dataset_neq_45.csv` (neq45, default) | r, T, ΔPA(t)×10 | α, i, θ, z |
| V | Non-equatorial partial orbit + sweep | `dpa_dataset_noneq.csv` (neq30) **or** `dpa_dataset_neq_45.csv` (neq45, default) | r, T, ΔPA(t)×k | α, i, θ, z |

Datasets: **neq30** has theta in [-30, 30]°; **neq45** has theta in [-45, 45]°. Active runs use **neq45** by default; the neq30 configs are kept for reference but their old results were dropped.

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

The leaf directory under `results/{checkpoints,figures,metrics}/` is the value of `experiment.name` from the config (not the folder name). For exp 4 / 5 this means each variant lands in its own subdirectory: e.g., `experiment_4_noneq_full_neq45/` and `experiment_4_noneq_full_neq45_no_noise/`. Uncertainty runs land under `uncertainty_<name>/`.

```
results/
├── checkpoints/<experiment.name>/{target}/model_seed{seed}.pth
├── figures/<experiment.name>/{target}/error_hist_seed{seed}.png
├── figures/<experiment.name>/{target}/pred_vs_actual_seed{seed}.png
├── logs/                                  # SLURM stdout/stderr
└── metrics/<experiment.name>/
    ├── {target}_summary.csv               # One row per seed
    ├── {target}_aggregated.csv            # μ±σ across seeds
    └── {target}_sweep.csv                 # Sweep mode only
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
