# Quick Start Guide

## Repository Summary

Clean, organized ML framework for SMBH hotspot parameter prediction. **All 5 experiments implemented with full SLURM support.**

### Files Created
- **30 total files** across 19 directories
- **7 core modules** in `src/`
- **15 experiment files** (5 experiments × 3 files each)
- **Comprehensive documentation**

## Immediate Next Steps

### 1. Prepare Datasets

```bash
cd /scratch/ralbe/meniar_and_django/smbh_hotspots_repository

# Copy raw data files (if not already done)
# Then run preprocessing:
cd src/preprocessing
python prepare_dataset_i0.py          # For Experiment I
python prepare_dataset_ultradense.py  # For Experiments II & III  
python prepare_dataset_noneq.py       # For Experiments IV & V
```

### 2. Run an Experiment

**Option A: Test locally (quick verification)**
```bash
cd experiments/experiment_1_eq_avg
python train.py
```

**Option B: Submit to SLURM (production runs)**
```bash
cd experiments/experiment_1_eq_avg
sbatch submit.sh
```

Monitor progress:
```bash
squeue -u $USER
tail -f ../../results/logs/exp1_eq_avg_*.out
```

### 3. Check Results

```bash
# View metrics
cat results/metrics/experiment_1_eq_avg/spin_aggregated.csv

# View figures
ls results/figures/experiment_1_eq_avg/
```

## Experiment Overview

| Exp | Description | Dataset | Targets | Features | Runtime |
|-----|-------------|---------|---------|----------|---------|
| I | Eq. Avg | i0 | α, i | r, T, ΔPA_avg | ~4h |
| II | Eq. Full | ultradense | α, i | r, T, ΔPA(t)×10 | ~6h |
| III | Eq. Half + Sweep | ultradense | α, i | r, T, ΔPA(t)×k | ~24h |
| IV | Non-eq Full | noneq | α, i, θ, z | r, T, ΔPA(t)×10 | ~12h |
| V | Non-eq Half + Sweep | noneq | α, i, θ, z | r, T, ΔPA(t)×k | ~36h |

## Key Features

✅ **Modular design**: Shared core modules, DRY principle  
✅ **YAML configs**: Easy parameter tuning without code changes  
✅ **Multi-seed training**: 5 runs per experiment → μ±σ statistics  
✅ **Noise injection**: Gaussian noise on inputs before normalization  
✅ **Sweep mode**: σ vs orbit coverage analysis (Exp III & V)  
✅ **SLURM ready**: All experiments have submit scripts  
✅ **Clean outputs**: Organized figures, metrics, checkpoints  
✅ **Minimal comments**: Self-documenting code, no "yapping"

## Repository Structure

```
smbh_hotspots_repository/
├── src/                    # Core modules (reusable)
│   ├── models/             # RegressionHead architecture
│   ├── training/           # Trainer, data loader, evaluation
│   ├── utils/              # Config, noise injection
│   └── preprocessing/      # Dataset generation scripts
├── experiments/            # 5 experiments (config + train + submit)
│   ├── experiment_1_eq_avg/
│   ├── experiment_2_eq_full/
│   ├── experiment_3_eq_half/
│   ├── experiment_4_noneq_full/
│   └── experiment_5_noneq_half/
├── data/                   # Raw .dat files → processed .csv
├── results/                # Checkpoints, figures, logs, metrics
└── README.md               # Full documentation
```

## Customization

Edit `config.yaml` in any experiment directory:

**Change model architecture**:
```yaml
model:
  hidden_dims: [512, 512]  # Larger network
  num_blocks: 3            # More residual blocks
```

**Adjust training**:
```yaml
training:
  batch_size: 64
  epochs: 1000
  learning_rate: 0.0005
```

**Modify noise levels**:
```yaml
noise:
  sigma_r: 0.05   # Less noise on radius
  sigma_T: 1.0    # Less noise on period
  sigma_DPA: 3.0  # Less noise on DPA
```

**Enable/disable sweep mode** (Exp III & V):
```yaml
sweep:
  enabled: true   # Run σ vs orbit coverage analysis
```

## Troubleshooting

**Import errors**: Check that you're in the experiment directory when running `train.py`

**Data not found**: Run preprocessing scripts in `src/preprocessing/`

**SLURM jobs fail**: Check logs in `results/logs/` for error messages

**GPU memory**: Reduce `batch_size` in config.yaml

## Next Actions

1. ✅ Run preprocessing scripts to generate datasets
2. ✅ Test Experiment I locally to verify everything works
3. ✅ Submit all experiments to SLURM for production runs
4. ✅ Analyze results in `results/metrics/` and `results/figures/`
5. ✅ Adjust configs as needed and rerun

## Notes

- All scripts are executable (`chmod +x` already applied)
- WandB is disabled by default (set `use_wandb: true` to enable)
- Checkpoints include scalers for deployment/inference
- Sweep mode trains 50-100 models (compute-intensive)

---

**Repository ready for immediate use!** 🚀
