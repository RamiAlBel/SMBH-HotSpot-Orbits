# SMBH Hotspots ML Repository

Machine learning framework for predicting black hole parameters (spin α, inclination i, theta θ) from hotspot observables (radius r, period T, and differential phase angle ΔPA(t)).

## Overview

This repository contains 5 experiments that train neural networks to predict SMBH parameters from hotspot orbital data:

- **Experiment I**: Equatorial orbits with averaged ΔPA → α or i
- **Experiment II**: Equatorial orbits with full ΔPA(t) time series → α or i
- **Experiment III**: Equatorial orbits with half ΔPA(t) (+ σ vs orbit coverage sweep) → α or i
- **Experiment IV**: Non-equatorial orbits with full ΔPA(t) → α, i, θ, or z
- **Experiment V**: Non-equatorial orbits with half ΔPA(t) (+ sweep) → α, i, θ, or z

Each experiment trains 5 models with different random seeds and reports μ±σ statistics.

## Repository Structure

```
smbh_hotspots_repository/
├── src/
│   ├── preprocessing/       # Data preparation scripts
│   ├── models/              # Neural network architectures
│   ├── training/            # Training utilities & data loaders
│   └── utils/               # Configuration & noise injection
├── experiments/
│   ├── experiment_1_eq_avg/
│   ├── experiment_2_eq_full/
│   ├── experiment_3_eq_half/
│   ├── experiment_4_noneq_full/
│   └── experiment_5_noneq_half/
├── data/
│   ├── raw/                 # Raw .dat files from simulations
│   └── processed/           # Generated CSV files
├── results/
│   ├── checkpoints/         # Trained model .pth files
│   ├── figures/             # Error distributions & predictions
│   ├── logs/                # SLURM job logs
│   └── metrics/             # CSV files with results
└── README.md
```

## Installation

```bash
cd /scratch/ralbe/meniar_and_django/smbh_hotspots_repository
pip install -r requirements.txt
```

**Dependencies**: Python 3.8+, PyTorch, pandas, scikit-learn, matplotlib, seaborn, PyYAML

## Data Preparation

### Step 1: Place raw simulation files

Place Aris's `.dat` files in the appropriate directories:
- Equatorial data (i=0): `data/raw/meniar_files/`
- Equatorial dense data: `data/raw/sepray_t_files/`
- Non-equatorial data: `data/raw/non_eq/dpa_neq/`

### Step 2: Run preprocessing scripts

```bash
# Generate dpa_dataset_i0.csv (Experiment I)
cd src/preprocessing
python prepare_dataset_i0.py

# Generate dpa_dataset_ultradense.csv (Experiments II & III)
python prepare_dataset_ultradense.py

# Generate dpa_dataset_noneq.csv (Experiments IV & V)
python prepare_dataset_noneq.py
```

These scripts will create CSV files in `data/processed/` with columns:
- **Experiment I**: r, K, a, i, DPA, Period
- **Experiments II-III**: r, K, a, i, DPA, Perid_fraq, Period
- **Experiments IV-V**: r, K, a, i, theta, DPA, Perid_fraq, Period

## Running Experiments

Each experiment has its own directory with:
- `config.yaml`: Configuration file (model, training, noise parameters)
- `train.py`: Training script
- `submit.sh`: SLURM submission script

### Local Execution

```bash
cd experiments/experiment_1_eq_avg
python train.py
```

### SLURM Execution

```bash
cd experiments/experiment_1_eq_avg
sbatch submit.sh
```

Monitor job status:
```bash
squeue -u $USER
tail -f ../../results/logs/exp1_eq_avg_*.out
```

## Experiment Details

### Experiment I: Equatorial Averaged ΔPA

**Input**: r, T, ΔPA_avg (single value per orbit)
**Targets**: spin (α) or inclination (i)
**Dataset**: `dpa_dataset_i0.csv`

```bash
cd experiments/experiment_1_eq_avg
sbatch submit.sh
```

Results: `results/metrics/experiment_1_eq_avg/`

---

### Experiment II: Equatorial Full Orbit

**Input**: r, T, ΔPA(t₁), ..., ΔPA(t₁₀) (10 samples at Perid_fraq = 0.1, 0.2, ..., 1.0)
**Targets**: spin (α) or inclination (i)
**Dataset**: `dpa_dataset_ultradense.csv`

```bash
cd experiments/experiment_2_eq_full
sbatch submit.sh
```

Results: `results/metrics/experiment_2_eq_full/`

---

### Experiment III: Equatorial Half Orbit + Sweep

**Input**: r, T, ΔPA(t) with k consecutive samples (k=1 to 10)
**Targets**: spin (α) or inclination (i)
**Dataset**: `dpa_dataset_ultradense.csv`

**Mode 1 - Standard half orbit (k=5)**:
Edit `config.yaml` and set `sweep.enabled: false`

**Mode 2 - σ vs orbit coverage sweep**:
Edit `config.yaml` and set `sweep.enabled: true`

This runs 50 models per target (10 values of k × 5 seeds) and generates plots showing prediction error σ vs % of orbit included.

```bash
cd experiments/experiment_3_eq_half
sbatch submit.sh
```

Results:
- Standard mode: `results/metrics/experiment_3_eq_half/{target}_summary.csv`
- Sweep mode: `results/metrics/experiment_3_eq_half/{target}_sweep.csv` + plots

---

### Experiment IV: Non-Equatorial Full Orbit

**Input**: r, T, ΔPA(t₁), ..., ΔPA(t₁₀)
**Targets**: spin (α), inclination (i), theta (θ), or z = r·sin(θ)
**Dataset**: `dpa_dataset_noneq.csv`

```bash
cd experiments/experiment_4_noneq_full
sbatch submit.sh
```

Results: `results/metrics/experiment_4_noneq_full/`

---

### Experiment V: Non-Equatorial Half Orbit + Sweep

**Input**: r, T, ΔPA(t) with k consecutive samples
**Targets**: spin (α), inclination (i), theta (θ), or z
**Dataset**: `dpa_dataset_noneq.csv`

Same two modes as Experiment III (standard or sweep).

```bash
cd experiments/experiment_5_noneq_half
sbatch submit.sh
```

Results: `results/metrics/experiment_5_noneq_half/`

## Configuration

Edit `config.yaml` in each experiment directory to customize:

**Model parameters**:
```yaml
model:
  hidden_dims: [256, 256]
  num_blocks: 2
  dropout: 0.1
```

**Training parameters**:
```yaml
training:
  batch_size: 32
  epochs: 500
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stop_patience: 40
  seeds: [42, 43, 44, 45, 46]
```

**Noise injection** (applied before normalization):
```yaml
noise:
  enabled: true
  sigma_r: 0.1      # units of M
  sigma_T: 2.0      # minutes
  sigma_DPA: 5.0    # degrees
```

## Output Files

### Metrics

For each experiment, results are saved in `results/metrics/experiment_X/`:

**`{target}_summary.csv`**: One row per seed
```csv
target,seed,mae,rmse,r2,error_mean,error_std
spin,42,0.045,0.062,0.954,-0.001,0.059
spin,43,0.048,0.065,0.951,0.002,0.061
...
```

**`{target}_aggregated.csv`**: Aggregated statistics across seeds
```csv
target,mae_mean,mae_std,rmse_mean,rmse_std,r2_mean,r2_std,error_std_mean,error_std_std
spin,0.046,0.003,0.063,0.004,0.952,0.002,0.060,0.002
```

**`{target}_sweep.csv`** (Experiments III & V only):
```csv
percent,num_samples,sigma_mean,sigma_std
10,1,0.15,0.01
20,2,0.12,0.008
...
100,10,0.06,0.004
```

### Figures

Saved in `results/figures/experiment_X/`:
- `{target}/error_hist_seed{seed}.png`: Error distribution histograms with Gaussian fit
- `{target}/pred_vs_actual_seed{seed}.png`: Predicted vs actual scatter plots
- `{target}_sigma_vs_orbit.png`: σ vs % orbit coverage (sweep mode)

### Checkpoints

Trained models saved in `results/checkpoints/experiment_X/{target}/model_seed{seed}.pth`

Each checkpoint contains:
- `model_state_dict`: Model weights
- `scaler_X_mean`, `scaler_X_scale`: Feature normalization parameters
- `scaler_y_mean`, `scaler_y_scale`: Target normalization parameters

## Key Implementation Details

### Sampling Strategy

**Full orbit**: 10 evenly spaced samples at Perid_fraq = 0.1, 0.2, ..., 1.0

**Half orbit**: Random consecutive window of 5 samples from the 10 available

**Variable coverage (sweep)**: Random consecutive window of k samples (k=1 to 10)

### Noise Injection

Gaussian noise is added **before** z-normalization:
- r: σ = 0.1 (in units of black hole mass M)
- T: σ = 2.0 minutes
- ΔPA: σ = 5.0 degrees

Fresh noise is sampled each training epoch.

### Model Architecture

`RegressionHead` with:
- Input: 2 + k features (r, T, DPA samples)
- Hidden layers: [256, 256] with 2 residual blocks
- Batch normalization + dropout (0.1)
- Output: 1 (predicted parameter)

### Train/Val/Test Split

80/10/10 split with random shuffling (seed=42 for reproducibility)

## Interpreting Results

**Good model**: 
- R² > 0.9
- Error distribution centered near 0 (μ ≈ 0)
- Small prediction uncertainty (σ < 0.1 for spin, < 5° for inclination)

**Sweep analysis** (Experiments III & V):
- Plot shows how prediction error decreases with more orbit coverage
- Useful for determining minimum required observation time
- Expected trend: σ decreases as % orbit increases

## Troubleshooting

**Dataset not found**: Run preprocessing scripts in `src/preprocessing/`

**CUDA out of memory**: Reduce `batch_size` in `config.yaml`

**Training too slow**: Check GPU utilization with `nvidia-smi`, consider reducing `epochs` or `num_samples_range` for sweep mode

**NaN losses**: Check data for outliers, reduce `learning_rate`

## Citation

If you use this code, please cite:
```
[Your paper citation here]
```

## Contact

For questions or issues, contact [Your name/email]
