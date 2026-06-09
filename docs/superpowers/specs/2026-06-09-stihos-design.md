# STIHOS Design Spec
**SpaceTime Inference via Hotspot ObservationS**
Date: 2026-06-09

## Context

This repo trains ML models to predict supermassive black hole parameters (spin α, inclination i, and for non-equatorial orbits also theta θ and height z) from hotspot observables (radius r, period T, differential phase angle DPA(t)). The trained checkpoints live in `results/checkpoints/` but there is no user-facing inference package. STIHOS is a thin, pip-installable inference layer that wraps the trained models for end-users who want to submit a single observation and receive parameter estimates with error bars.

## Scope

- **New separate repository** at `/scratch/ralbe/meniar_and_django/stihos/`, pushed to GitHub as `OriginalJesusPenguin/stihos`
- Inference-only: no training, no data preprocessing, no sklearn dependency at runtime
- Bundles three model families (checkpoints copied from `results/checkpoints/`):
  - `eq_avg` ← `experiment_1_eq_avg/` (spin + incl, 3 seeds × 2 targets = 6 files, ~19 MB)
  - `eq_full` ← `experiment_2_eq_full/` (spin + incl, 3 seeds × 2 targets = 6 files, ~19 MB)
  - `noneq_full` ← `experiment_4_noneq_full_neq45/` (spin + incl + theta + z, 3 seeds × 4 targets = 12 files, ~38 MB)
- Total bundled checkpoint size: ~76 MB

## Repository Layout

```
stihos/                        # new git root
├── pyproject.toml
├── README.md
├── stihos/
│   ├── __init__.py            # exposes predict(), PredictionResult
│   ├── _model.py              # RegressionHead (copied verbatim from src/models/regression_head.py)
│   ├── _inference.py          # load_checkpoint(), mc_predict(), _pool_predictions()
│   ├── _units.py              # unit conversion helpers (r, T, DPA → canonical M/min/deg)
│   ├── cli.py                 # argparse CLI, entry point `stihos`
│   └── checkpoints/
│       ├── eq_avg/spin/model_seed{42,43,44}.pth
│       ├── eq_avg/incl/model_seed{42,43,44}.pth
│       ├── eq_full/spin/...
│       ├── eq_full/incl/...
│       ├── noneq_full/spin/...
│       ├── noneq_full/incl/...
│       ├── noneq_full/theta/...
│       └── noneq_full/z/...
└── tests/
    └── test_predict.py        # smoke tests: eq and noneq predict runs, unit conversion
```

## Python API

```python
from stihos import predict

# Equatorial — single average DPA scalar → uses eq_avg models
result = predict(r=7.5, T=70.0, dpa=133.5, orbit="equatorial")

# Equatorial — 10-point timeseries → uses eq_full models
result = predict(r=7.5, T=70.0, dpa=[120.0, 125.3, ...], orbit="equatorial")

# Non-equatorial — 10-point timeseries (orbit defaults to "nonequatorial" if len(dpa)==10 is ambiguous)
result = predict(r=7.5, T=70.0, dpa=[120.0, ...], orbit="nonequatorial")

# Unit overrides (defaults: r_unit="M", T_unit="min", dpa_unit="deg")
result = predict(r=7.5, T=1.17, dpa=2.33, orbit="equatorial",
                 T_unit="hours", dpa_unit="rad")

# MC sample count (default: n_mc=2000)
result = predict(..., n_mc=5000)
```

### DPA auto-detection
- `dpa` is a single float or 1-element list → avg mode (eq_avg models)
- `dpa` is a list/array of exactly 10 values → timeseries mode (eq_full or noneq_full models)
- Any other list length → `ValueError` with a message explaining the constraint (the timeseries models have `input_dim=12` and cannot accept partial orbits)

### PredictionResult

```python
result.spin          # (mean: float, std: float), dimensionless
result.incl          # (mean: float, std: float), degrees
result.theta         # (mean: float, std: float), degrees — nonequatorial only, else None
result.z             # (mean: float, std: float), M units — nonequatorial only, else None
str(result)          # formatted table (see CLI output section)
result.to_dict()     # {'spin': {'mean':..., 'std':...}, ...}
```

## CLI

```bash
# Equatorial avg DPA
stihos --r 7.5 --T 70 --dpa 133.5

# Equatorial full timeseries (space-separated list)
stihos --r 7.5 --T 70 --dpa 120 125 131 136 138 140 141 142 142 143

# Non-equatorial
stihos --r 7.5 --T 70 --dpa 120 125 131 136 138 140 141 142 142 143 --orbit noneq

# Unit and MC overrides
stihos --r 7.5 --T 1.17 --T-unit hours --dpa 2.33 --dpa-unit rad --n-mc 5000
```

Example output:
```
STIHOS — SpaceTime Inference via Hotspot ObservationS
Orbit: equatorial | Mode: avg DPA | n_mc: 2000 (×3 seeds)

 Parameter      Mean        ± Std
 ─────────────────────────────────
 spin α         0.842       0.041
 incl i (°)     23.1        2.3
```

## Inference / MC Scheme

Implemented in `stihos/_inference.py`:

1. **Checkpoint loading** — `load_checkpoint(path)`: `torch.load(path, map_location='cpu', weights_only=False)`. Returns the raw dict. `input_dim` inferred from `ck['scaler_X_mean'].shape[0]`. Constructs `RegressionHead(input_dim, hidden_dims=(256,256), num_blocks=2, dropout=0.1)`, calls `model.load_state_dict(ck['model_state_dict'])`, sets `model.eval()`.

2. **Per-seed MC** — `mc_predict(model, ck, x_orig, n_mc, sigma_obs)`:
   - `x_orig` is a 1-D array `[r(M), T(min), dpa_0, ..., dpa_k]`
   - `sigma_obs` defaults to `[0.1, 2.0, 5.0, ..., 5.0]` (same as training defaults in source repo)
   - Sample `n_mc` noisy copies: `x_noisy = x_orig + rng.standard_normal((n_mc, D)) * sigma_obs`
   - Normalize: `(x_noisy - ck['scaler_X_mean']) / ck['scaler_X_scale']`
   - Batched forward pass (single call, eval mode)
   - Inverse-transform: `y_raw * ck['scaler_y_scale'][0] + ck['scaler_y_mean'][0]`
   - For incl and theta: `np.rad2deg(y)` (these were trained in radians)
   - Returns array of `n_mc` predictions in physical units

3. **Seed pooling** — `_pool_predictions(per_seed_arrays)`: `np.concatenate` all seeds → `(mean, std)` of the pooled `3 × n_mc` values.

Directly mirrors `mc_sigma()` from `src/utils/jacobian_uncertainty.py:88-121` in the source repo. Key differences from that function: adds the mean term (not just std), pools multiple seeds, handles the `rad2deg` conversion for incl/theta.

## Unit Conversion (`stihos/_units.py`)

| Parameter | Default unit | Supported alts | Conversion |
|-----------|-------------|----------------|------------|
| r         | M           | —              | identity   |
| T         | min         | hours          | ×60        |
| dpa       | deg         | rad            | ×180/π     |

Output units are fixed: spin (dimensionless), incl/theta (degrees), z (M).

## pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "stihos"
version = "0.1.0"
description = "SpaceTime Inference via Hotspot ObservationS — SMBH parameter inference from hotspot observables"
requires-python = ">=3.9"
dependencies = ["torch>=1.10", "numpy>=1.21"]

[project.scripts]
stihos = "stihos.cli:main"

[tool.hatch.build.targets.wheel]
include = ["stihos/"]
```

Note: `torch` is large. For users who already have it, this is a fast install. We explicitly exclude sklearn (not needed at inference — scalers stored as numpy arrays in checkpoints).

## GitHub workflow

1. Pre-implementation: `git init` in `/scratch/ralbe/meniar_and_django/stihos/`, `gh repo create OriginalJesusPenguin/stihos --public`, push empty repo.
2. Post-implementation: commit all files + checkpoints, push.

## Verification

```bash
cd /scratch/ralbe/meniar_and_django/stihos
pip install -e .

# Smoke test — Python API
python -c "
from stihos import predict
r = predict(r=7.5, T=70.0, dpa=133.5, orbit='equatorial')
print(r)
assert r.spin is not None and r.incl is not None and r.theta is None
r2 = predict(r=7.5, T=70.0, dpa=[120,125,131,136,138,140,141,142,142,143], orbit='nonequatorial')
assert r2.theta is not None and r2.z is not None
print('all checks passed')
"

# CLI smoke test
stihos --r 7.5 --T 70 --dpa 133.5
stihos --r 7.5 --T 70 --dpa 120 125 131 136 138 140 141 142 142 143 --orbit noneq

# Run unit tests
python -m pytest tests/
```
