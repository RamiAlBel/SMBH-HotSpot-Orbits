"""Generate notebooks/uncertainty_estimation.ipynb from a curated cell list.

Run this script once whenever the notebook structure needs to change.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "uncertainty_estimation.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": text.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


CELLS: list[dict] = []

# ── Title ────────────────────────────────────────────────────────────────────

CELLS.append(md("""# Uncertainty Estimation on SMBH Hotspot Predictions

This notebook compares **three approaches** for predicting how observational
uncertainty on the inputs $(r,\\,T,\\,\\Delta\\mathrm{PA}(t))$ propagates into
the model's output uncertainty on $(\\alpha,\\,i,\\,\\theta)$:

| # | Approach | Where the σ comes from | Input-dependent? | Notes |
|---|----------|------------------------|-------------------|-------|
| 1 | **Trilinear interpolation** | Pre-trained noise-sweep grid (Exp VI / VII): for each cell of $(\\sigma_T,\\sigma_r,\\sigma_{\\Delta\\mathrm{PA}})$ a model was retrained and its test-set residual std was recorded. | No — only depends on the input σ levels. | Includes both data noise and model bias. Bounded by the grid corners. |
| 2 | **Jacobian propagation** | First-order autodiff: $\\sigma_y \\approx \\sqrt{\\sum_i J_i^2 \\sigma_i^2}$ in normalised space, then rescaled. | Yes — depends on the local gradient at this input. | Fast, but the gradient can be unstable far from training data. |
| 3 | **Monte Carlo** | Draw $N_{\\rm MC}=2000$ noisy copies of the input, push them through a no-noise model, take the std of the outputs. | Yes — depends on the input and on the model's local non-linearity. | Robust; no Jacobian instability; cost is one extra forward pass × $N_{\\rm MC}$. |

Each method is evaluated on the three mock observations under
``mock_observations/`` at the standard input sigmas

$$\\sigma_r = 0.1\\,M,\\quad \\sigma_T = 2\\,\\mathrm{min},\\quad \\sigma_{\\Delta\\mathrm{PA}} = 10^\\circ.$$
"""))

# ── 0. Setup ─────────────────────────────────────────────────────────────────

CELLS.append(md("## 0. Setup\n"))
CELLS.append(code("""import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import RegularGridInterpolator

REPO_ROOT = Path("..").resolve()
sys.path.insert(0, str(REPO_ROOT))

from src.models.regression_head import RegressionHead
from src.utils.jacobian_uncertainty import compute_jacobian, jacobian_sigma, mc_sigma

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEG = 180.0 / np.pi
N_MC = 2000

SIGMA_DEFAULT = dict(sigma_r=0.1, sigma_T=2.0, sigma_DPA=10.0)

MOCK_DIR = REPO_ROOT / "mock_observations"
RESULTS  = REPO_ROOT / "results"

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 200)
print(f"device={DEVICE}  N_MC={N_MC}  SIGMA_DEFAULT={SIGMA_DEFAULT}")
"""))

CELLS.append(md("""### Experiment registry

Each entry says which trained model to use, which targets it predicts, and
which noise-sweep grid feeds Approach 1.
"""))

CELLS.append(code("""EXP_REGISTRY = {
    1: dict(
        label="Exp I — eq averaged ΔPA",
        ckpt_dir="experiment_1_eq_avg_no_noise",
        exp_type="avg",
        num_input_features=3,
        targets=[
            # Exp I predicts spin only — the i=0 dataset doesn't constrain
            # inclination, so any incl prediction would be meaningless.
            dict(name="spin", in_radians=False, unit=""),
        ],
        sweep_name=None,  # no matching noise-sweep grid for averaged-DPA inputs
    ),
    2: dict(
        label="Exp II — eq full orbit",
        ckpt_dir="experiment_2_eq_full_no_noise",
        exp_type="ts",
        num_input_features=12,
        targets=[
            dict(name="spin", in_radians=False, unit=""),
            dict(name="incl", in_radians=True,  unit="deg"),
        ],
        sweep_name="uncertainty_eq_noise_sweep",
    ),
    4: dict(
        label="Exp IV — non-eq full orbit (neq45)",
        ckpt_dir="experiment_4_noneq_full_neq45_no_noise",
        exp_type="ts",
        num_input_features=12,
        targets=[
            dict(name="spin",  in_radians=False, unit=""),
            dict(name="incl",  in_radians=True,  unit="deg"),
            dict(name="theta", in_radians=True,  unit="deg"),
        ],
        sweep_name="uncertainty_neq45_noise_sweep",
    ),
}
"""))

# ── 1. Mock observations ─────────────────────────────────────────────────────

CELLS.append(md("""## 1. Mock observations

The files in ``mock_observations/`` follow the same `.dat` format as the raw
simulation files. The truth parameters are encoded in the filename:

```
lc_r{r×10}_K{K×100}_a{a×100}_i{i×10}_th{th_raw}[_extra].dat
        r [M]      K          α          i [°]    θ = th_raw [°]
```

**Note on θ convention.** The mock-observation filenames encode θ
directly — ``_th30.dat`` means truth θ = 30°. This is a different
convention from the training-dataset filenames, where the same suffix
maps to θ = 90° − ``th_raw`` after preprocessing. So we read mock-obs θ
straight off the filename. Exp IV's neq45 training data covers θ ∈
$[-30°,+30°]$, so any mock file outside that range is an extrapolation
test for the model.

Note that `z = r·sin(θ_rad)`, so it can always be reconstructed from the truth
parameters and is omitted as a target here.

Each file is parsed into:

* **`feat_avg`** — a single-row vector $[r,\\,T,\\,\\Delta\\mathrm{PA}_0]$ for Exp I.
* **`feat_ts`** — a 12-vector $[r,\\,T,\\,\\Delta\\mathrm{PA}(0.1),\\dots,\\Delta\\mathrm{PA}(1.0)]$
  for Exps II and IV. Files with fewer than two phase samples can't be used here.
"""))

CELLS.append(code("""def parse_mock_filename(path: Path) -> dict:
    parts = path.stem.split("_")
    return dict(
        r=float(parts[1][1:]) / 10.0,
        K=float(parts[2][1:]) / 100.0,
        a=float(parts[3][1:]) / 100.0,
        i=float(parts[4][1:]) / 10.0,
        theta=float(parts[5][2:]),  # mock-obs filenames already store θ directly
    )


def load_mock_observation(path: Path) -> dict:
    text  = path.read_text()
    lines = text.splitlines()
    period_min = float(lines[2].split()[1])

    arr = np.loadtxt(io.StringIO("\\n".join(lines[3:])))
    if arr.ndim == 1:
        arr = arr[None, :]

    truth = parse_mock_filename(path)
    truth["Period"] = period_min

    dpa_avg  = float(arr[0, -2])
    feat_avg = np.array([truth["r"], period_min, dpa_avg], dtype=np.float32)

    feat_ts = None
    if len(arr) >= 2:
        sample_points = np.linspace(0.1, 1.0, 10)
        phases, dpa = arr[:, 1], arr[:, -2]
        idx = np.argsort(phases)
        ph_s, dp_s = phases[idx], dpa[idx]
        dpa_ts = np.interp(sample_points, ph_s, dp_s, left=dp_s[0], right=dp_s[-1])
        feat_ts = np.concatenate(([truth["r"], period_min], dpa_ts)).astype(np.float32)

    return dict(path=path, truth=truth, dpa_avg=dpa_avg,
                feat_avg=feat_avg, feat_ts=feat_ts, n_phases=len(arr))


mocks = [load_mock_observation(p) for p in sorted(MOCK_DIR.glob("**/*.dat"))]
mock_summary = pd.DataFrame([
    dict(file=m["path"].name, n_phases=m["n_phases"], **{
        "a (truth)": m["truth"]["a"],
        "i (truth, °)": m["truth"]["i"],
        "θ (truth, °)": m["truth"]["theta"],
        "Period (min)": m["truth"]["Period"],
        "feat_ts available": m["feat_ts"] is not None,
    })
    for m in mocks
])
mock_summary
"""))

# ── 2. Models ────────────────────────────────────────────────────────────────

CELLS.append(md("""## 2. Loading the no-noise models

Each `(experiment, target)` checkpoint bundles the trained weights together
with the `StandardScaler` parameters (`X_mean`, `X_scale`, `y_mean`, `y_scale`).
The MC and Jacobian routines need both — they work in normalised space and
only convert back to the original target unit at the very end.
"""))

CELLS.append(code("""def load_model_and_scalers(exp_id: int, target_name: str, seed: int = 42):
    cfg = EXP_REGISTRY[exp_id]
    ckpt_path = (RESULTS / "checkpoints" / cfg["ckpt_dir"]
                 / target_name / f"model_seed{seed}.pth")
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = RegressionHead(input_dim=cfg["num_input_features"]).to(DEVICE)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    scalers = dict(
        X_mean=np.asarray(state["scaler_X_mean"],  dtype=np.float32),
        X_scale=np.asarray(state["scaler_X_scale"], dtype=np.float32),
        y_mean=float(np.asarray(state["scaler_y_mean"]).item()),
        y_scale=float(np.asarray(state["scaler_y_scale"]).item()),
    )
    return model, scalers


def predict_y(model, scalers, x_orig: np.ndarray) -> float:
    x_norm = (x_orig - scalers["X_mean"]) / scalers["X_scale"]
    with torch.no_grad():
        y_norm = model(torch.tensor(x_norm[None], dtype=torch.float32, device=DEVICE))
    return float(y_norm.item() * scalers["y_scale"] + scalers["y_mean"])


def sigma_obs_vector(num_features: int, sigma_r, sigma_T, sigma_DPA) -> np.ndarray:
    sig = np.zeros(num_features, dtype=np.float32)
    sig[0]  = sigma_r
    sig[1]  = sigma_T
    sig[2:] = sigma_DPA
    return sig


def feat_for(exp_cfg, mock):
    if exp_cfg["exp_type"] == "avg":
        return mock["feat_avg"]
    return mock["feat_ts"]  # may be None for the single-row file


def applicable(exp_cfg, mock):
    return exp_cfg["exp_type"] == "avg" or mock["feat_ts"] is not None


def truth_for_target(truth, target_name):
    return {"spin": truth["a"], "incl": truth["i"], "theta": truth["theta"]}.get(target_name)


def display_unit(target):
    return target["unit"] or "—"


# Cache models so each (exp, target) is loaded once.
MODEL_CACHE = {}
def cached_model(exp_id, target_name):
    key = (exp_id, target_name)
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = load_model_and_scalers(exp_id, target_name)
    return MODEL_CACHE[key]
"""))

# ── 3. Approach 1 ────────────────────────────────────────────────────────────

CELLS.append(md(r"""## 3. Approach 1 — Trilinear interpolation over the noise-sweep grid

For Experiment VI (equatorial) and Experiment VII / XII (non-eq), 125 separate
models were trained — one per cell of the $5\times5\times5$ grid in
$(\sigma_T, \sigma_r, \sigma_{\Delta\mathrm{PA}})$. The recorded
`{target}_error_std` is the **test-set residual std** of that model in its
training units (radians for `incl`/`theta`, dimensionless for `spin`).

At inference time we want $\sigma_y$ at an arbitrary $(\sigma_T,\sigma_r,\sigma_{\Delta\mathrm{PA}})$,
which we obtain by trilinear interpolation over the three sigma axes:

$$\sigma_y(\sigma_T, \sigma_r, \sigma_{\Delta\mathrm{PA}}) \;=\; \mathrm{Trilinear}\bigl(\text{grid of error\_std}\bigr).$$

Pros: encodes the actual data + model bias; no autodiff instability.<br>
Cons: independent of the specific input — a clean orbit and a near-degenerate
orbit get the same predicted σ. Out-of-grid sigmas are extrapolated flat.
"""))

CELLS.append(code("""def build_interpolators(sweep_name: str, targets: list[dict]):
    sweep_dir = RESULTS / "metrics" / sweep_name
    df = pd.concat([pd.read_csv(f) for f in sorted(sweep_dir.glob("combo_*.csv"))],
                   ignore_index=True)

    sT = np.sort(df["sigma_T"].unique())
    sr = np.sort(df["sigma_r"].unique())
    sd = np.sort(df["sigma_DPA"].unique())

    interps = {}
    for t in targets:
        col = f"{t['name']}_error_std"
        if col not in df.columns:
            continue
        grid = np.full((len(sT), len(sr), len(sd)), np.nan)
        for _, r in df.iterrows():
            grid[np.searchsorted(sT, r["sigma_T"]),
                 np.searchsorted(sr, r["sigma_r"]),
                 np.searchsorted(sd, r["sigma_DPA"])] = r[col]
        interps[t["name"]] = RegularGridInterpolator(
            (sT, sr, sd), grid, bounds_error=False, fill_value=None,
        )
    return interps, dict(sigma_T=sT, sigma_r=sr, sigma_DPA=sd)


def predict_sigma_trilinear(interp, sigma_r, sigma_T, sigma_DPA, in_radians):
    sigma_y = float(interp([sigma_T, sigma_r, sigma_DPA])[0])
    return sigma_y * (DEG if in_radians else 1.0)


# Build interpolators once per experiment.
SWEEP_INTERPS = {}
for exp_id, cfg in EXP_REGISTRY.items():
    if cfg["sweep_name"] is not None:
        SWEEP_INTERPS[exp_id], _ = build_interpolators(cfg["sweep_name"], cfg["targets"])
        print(f"Built interpolators for exp {exp_id} from {cfg['sweep_name']} "
              f"({list(SWEEP_INTERPS[exp_id])})")
"""))

CELLS.append(md("""### 3.1 Sigma-only summary

Approach 1 doesn't depend on the specific input data point — only on the
input sigma triple. So this table is a single-shot summary of *expected*
output σ at `SIGMA_DEFAULT` for every (exp, target) we cover.
"""))

CELLS.append(code("""rows = []
for exp_id, cfg in EXP_REGISTRY.items():
    if cfg["sweep_name"] is None:
        continue
    for t in cfg["targets"]:
        if t["name"] not in SWEEP_INTERPS[exp_id]:
            continue
        sigma = predict_sigma_trilinear(
            SWEEP_INTERPS[exp_id][t["name"]],
            **SIGMA_DEFAULT, in_radians=t["in_radians"],
        )
        rows.append(dict(
            exp=exp_id, exp_label=cfg["label"],
            target=t["name"], unit=display_unit(t),
            sigma_trilinear=sigma,
        ))
trilinear_summary_df = pd.DataFrame(rows)
trilinear_summary_df
"""))

CELLS.append(md("""### 3.2 Approach 1 on the mock observations

Same predicted σ as above, but paired with each model's actual point
prediction $\\hat y$ on each mock obs. The σ column is identical for every
mock-obs row of the same `(exp, target)` — that's the defining limitation
of this approach.
"""))

CELLS.append(code("""rows = []
for mock in mocks:
    for exp_id, cfg in EXP_REGISTRY.items():
        if not applicable(cfg, mock):
            continue
        x = feat_for(cfg, mock)
        for t in cfg["targets"]:
            model, scalers = cached_model(exp_id, t["name"])
            pred = predict_y(model, scalers, x)
            pred_disp = pred * (DEG if t["in_radians"] else 1.0)
            truth = truth_for_target(mock["truth"], t["name"])

            sigma_tri = (predict_sigma_trilinear(
                            SWEEP_INTERPS[exp_id][t["name"]],
                            **SIGMA_DEFAULT, in_radians=t["in_radians"])
                         if exp_id in SWEEP_INTERPS and t["name"] in SWEEP_INTERPS[exp_id]
                         else np.nan)

            rows.append(dict(
                mock=mock["path"].name, exp=exp_id, target=t["name"],
                unit=display_unit(t), truth=truth, pred=pred_disp,
                sigma_trilinear=sigma_tri,
            ))
trilinear_table = pd.DataFrame(rows)
trilinear_table
"""))

# ── 4. Approach 2 ────────────────────────────────────────────────────────────

CELLS.append(md(r"""## 4. Approach 2 — Jacobian propagation

In normalised space the model is $y_{\rm norm} = f(x_{\rm norm})$. A first-order
Taylor expansion around an observed input gives

$$\sigma_{y_{\rm norm}}^2 \;\approx\; \sum_i \left(\frac{\partial f}{\partial x_{\rm norm,i}}\right)^2 \sigma_{x_{\rm norm,i}}^2.$$

Translating back to original units uses the StandardScaler:
$\sigma_{x_{\rm norm,i}} = \sigma_{\rm obs,i} / \sigma_{X,i}$ and
$\sigma_y = \sigma_{y_{\rm scale}}\cdot\sigma_{y_{\rm norm}}$.

**Caveat:** the gradient of a deep MLP can spike well beyond what local linearity
can carry, especially for inputs far from the training distribution. Expect
this approach to produce occasional unphysical σ values.
"""))

CELLS.append(code("""def predict_sigma_jacobian(model, scalers, x_orig, sigma_obs_orig, in_radians):
    x_norm = (x_orig - scalers["X_mean"]) / scalers["X_scale"]
    x_t    = torch.tensor(x_norm, dtype=torch.float32, device=DEVICE)
    J      = compute_jacobian(model, x_t).cpu().numpy()
    sigma_y = jacobian_sigma(J, sigma_obs_orig, scalers["X_scale"], scalers["y_scale"])
    return sigma_y * (DEG if in_radians else 1.0)


rows = []
for mock in mocks:
    for exp_id, cfg in EXP_REGISTRY.items():
        if not applicable(cfg, mock):
            continue
        x = feat_for(cfg, mock)
        sigma_obs = sigma_obs_vector(cfg["num_input_features"], **SIGMA_DEFAULT)
        for t in cfg["targets"]:
            model, scalers = cached_model(exp_id, t["name"])
            pred = predict_y(model, scalers, x)
            pred_disp = pred * (DEG if t["in_radians"] else 1.0)
            truth = truth_for_target(mock["truth"], t["name"])
            sigma_jac = predict_sigma_jacobian(model, scalers, x, sigma_obs, t["in_radians"])
            rows.append(dict(
                mock=mock["path"].name, exp=exp_id, target=t["name"],
                unit=display_unit(t), truth=truth, pred=pred_disp,
                sigma_jacobian=sigma_jac,
            ))
jacobian_table = pd.DataFrame(rows)
jacobian_table
"""))

# ── 5. Approach 3 ────────────────────────────────────────────────────────────

CELLS.append(md(r"""## 5. Approach 3 — Monte Carlo sampling

For each input we draw $N_{\rm MC}=2000$ noisy copies

$$x^{(k)} = x_{\rm orig} + \boldsymbol\varepsilon_k,\quad \varepsilon_{k,i} \sim \mathcal N(0,\sigma_{\rm obs,i}^2),$$

push them through the no-noise model, and report
$\sigma_y \approx \mathrm{std}_k\bigl(y^{(k)}\bigr)\cdot \sigma_{y_{\rm scale}}$.

This approach respects the model's local non-linearity (no Taylor truncation)
and is what we typically trust when the Jacobian and the trilinear estimate
disagree.
"""))

CELLS.append(code("""def predict_sigma_mc(model, scalers, x_orig, sigma_obs_orig, in_radians, n_mc=N_MC):
    sigma_y = mc_sigma(
        model, x_orig.astype(np.float32), sigma_obs_orig,
        scalers["X_mean"], scalers["X_scale"], scalers["y_scale"],
        n_mc=n_mc, device=DEVICE,
    )
    return sigma_y * (DEG if in_radians else 1.0)


rows = []
for mock in mocks:
    for exp_id, cfg in EXP_REGISTRY.items():
        if not applicable(cfg, mock):
            continue
        x = feat_for(cfg, mock)
        sigma_obs = sigma_obs_vector(cfg["num_input_features"], **SIGMA_DEFAULT)
        for t in cfg["targets"]:
            model, scalers = cached_model(exp_id, t["name"])
            pred = predict_y(model, scalers, x)
            pred_disp = pred * (DEG if t["in_radians"] else 1.0)
            truth = truth_for_target(mock["truth"], t["name"])
            sigma_mc_val = predict_sigma_mc(model, scalers, x, sigma_obs, t["in_radians"])
            rows.append(dict(
                mock=mock["path"].name, exp=exp_id, target=t["name"],
                unit=display_unit(t), truth=truth, pred=pred_disp,
                sigma_mc=sigma_mc_val,
            ))
mc_table = pd.DataFrame(rows)
mc_table
"""))

# ── 6. Method-vs-method per-σ comparison ────────────────────────────────────

CELLS.append(md("""## 6. Side-by-side comparison of the three σ estimates

Combining all three methods on the same `(mock, exp, target)` grid, with the
truth and the model's point prediction for context. All σ values are in the
displayed unit (degrees for angles, dimensionless for spin).
"""))

CELLS.append(code("""key = ["mock", "exp", "target", "unit", "truth", "pred"]
combined = (trilinear_table
            .merge(jacobian_table, on=key, how="outer")
            .merge(mc_table,        on=key, how="outer")
            .sort_values(["mock", "exp", "target"])
            .reset_index(drop=True))

# Add a |pred − truth| column for quick eyeballing of where the model is off.
combined["|pred − truth|"] = (combined["pred"] - combined["truth"]).abs()

combined[
    ["mock", "exp", "target", "unit", "truth", "pred", "|pred − truth|",
     "sigma_trilinear", "sigma_jacobian", "sigma_mc"]
]
"""))

CELLS.append(md(r"""### 6.1 Disagreement ratios

For each row, the ratio $\sigma_{\rm Jacobian} / \sigma_{\rm MC}$ tells us
how far the first-order Taylor approximation drifts from the empirical
sampling estimate. Values close to 1 mean the two agree; values $\gg 1$ are
exactly the Jacobian-blow-up regime that motivated Approach 3.
"""))

CELLS.append(code("""ratio = combined.assign(
    jac_over_mc=lambda d: d["sigma_jacobian"] / d["sigma_mc"],
    tri_over_mc=lambda d: d["sigma_trilinear"] / d["sigma_mc"],
)[
    ["mock", "exp", "target", "sigma_trilinear", "sigma_jacobian", "sigma_mc",
     "tri_over_mc", "jac_over_mc"]
]
ratio
"""))

# ── 7. Mission-style all-parameter MC table ─────────────────────────────────

CELLS.append(md(r"""## 7. All-parameter MC table on the EHT mock observation

The three files under ``mock_observations/`` are not three separate missions
— they are three input modalities of the *same* Sgr A* EHT-like scenario,
each tailored to one of the three experiment variants:

| Experiment | Input modality | Mock file |
|---|---|---|
| Exp I  (eq, averaged ΔPA) | single ΔPA point | ``lc_r90_K64_a-21_i0_th90_0.dat`` |
| Exp II (eq, full orbit)   | 10-sample ΔPA(t), $\theta = 0$ | ``lc_r90_K64_a-21_i250_th90.dat`` |
| Exp IV (non-eq, full orbit) | 10-sample ΔPA(t), $\theta \neq 0$ | ``lc_r90_K64_a-21_i250_th30.dat`` |

Inputs go through one shared MC pipeline:

* $N_{\rm MC} = 2000$ Gaussian-perturbed copies of the input vector.
* No-noise checkpoints, so the variance reflects only the propagated
  observational uncertainty plus the model's local non-linearity.
* **Spin samples are clipped to $[-1,+1]$** before computing the interval.
* The 2σ interval is the empirical 2.5 / 97.5 percentile of the (clipped)
  sample distribution, naturally asymmetric near $a = \pm 1$.

The Truth row uses the inclination and θ encoded in the **Exp IV** file
(the most complete modality), matching the practice in the LaTeX paper
table where the truth row is taken from the Exp IV input.

Future missions (ngEHT, BHEX, …) will be added by appending entries to
``MISSIONS`` once their mock files exist.
"""))

CELLS.append(code(r"""def mc_predict_with_interval(model, scalers, x_orig, sigma_obs_orig,
                              n_mc=N_MC, in_radians=False, clamp=None,
                              rng_seed=None):
    # Returns (pred_mean, ci_lo, ci_hi) using empirical 2.5/97.5 percentiles
    # of the MC sample distribution. clamp=(lo, hi) clips the samples in the
    # original target unit before computing statistics (use for spin).
    rng = np.random.default_rng(rng_seed)
    x_noisy = x_orig[None] + rng.standard_normal((n_mc, len(x_orig))) * sigma_obs_orig[None]
    x_norm  = (x_noisy - scalers["X_mean"]) / scalers["X_scale"]
    with torch.no_grad():
        y_norm = model(torch.tensor(x_norm, dtype=torch.float32, device=DEVICE))
    y = y_norm.cpu().numpy().squeeze() * scalers["y_scale"] + scalers["y_mean"]
    if clamp is not None:
        y = np.clip(y, clamp[0], clamp[1])

    factor = DEG if in_radians else 1.0
    pred   = float(np.mean(y))                   * factor
    lo, hi = np.percentile(y, [2.5, 97.5]) * factor
    return pred, float(lo), float(hi)


def fmt_pred(value: float, decimals: int = 2) -> str:
    return f"{value:+.{decimals}f}"


def fmt_ci(lo: float, hi: float, decimals: int = 2) -> str:
    return f"[{lo:+.{decimals}f}, {hi:+.{decimals}f}]"


# ── Mission registry ────────────────────────────────────────────────────────
# Each mission pairs a label with the file-per-experiment routing and a
# sigma triple. ngEHT and BHEX entries can be appended once their mock
# observation files become available.
MOCK_BY_NAME = {m["path"].name: m for m in mocks}

MISSIONS = [
    dict(
        test=1,
        mission="EHT",
        sigma=dict(sigma_r=0.1, sigma_T=2.0, sigma_DPA=10.0),
        files={
            1: "lc_r90_K64_a-21_i0_th90_0.dat",   # Exp I: avg DPA, single row
            2: "lc_r90_K64_a-21_i250_th90.dat",   # Exp II: eq full orbit (theta=0)
            4: "lc_r90_K64_a-21_i250_th30.dat",   # Exp IV: non-eq full orbit (theta != 0)
        },
        truth_from=4,  # which experiment's mock file supplies the Truth row
    ),
    # dict(test=2, mission="ngEHT", ...),
    # dict(test=3, mission="BHEX",  ...),
]


def _exp_target(cfg, name):
    return next((t for t in cfg["targets"] if t["name"] == name), None)


def build_mission_table(missions, n_mc=N_MC):
    rows = []
    for m in missions:
        sigma_kw = m["sigma"]
        files    = m["files"]

        # Use the Exp IV mock for nominal observation parameters, falling back
        # to the first available file if Exp IV isn't routed.
        ref_id   = m.get("truth_from", next(iter(files)))
        ref_mock = MOCK_BY_NAME[files[ref_id]]
        if ref_mock["feat_ts"] is not None:
            dpa_nominal = float(np.mean(ref_mock["feat_ts"][2:]))
        else:
            dpa_nominal = ref_mock["dpa_avg"]
        obs_dpa = f"{dpa_nominal:.1f} ± {sigma_kw['sigma_DPA']:.1f}"
        obs_r   = f"{ref_mock['truth']['r']:.1f} ± {sigma_kw['sigma_r']:.2f}"
        obs_P   = f"{ref_mock['truth']['Period']:.1f} ± {sigma_kw['sigma_T']:.1f}"

        # One row per (mission, exp), in canonical Exp I → II → IV order.
        for k, exp_id in enumerate(sorted(files)):
            cfg  = EXP_REGISTRY[exp_id]
            mock = MOCK_BY_NAME[files[exp_id]]
            if not applicable(cfg, mock):
                continue
            x         = feat_for(cfg, mock)
            sigma_obs = sigma_obs_vector(cfg["num_input_features"], **sigma_kw)
            first     = (k == 0)

            row = {
                "Test":    str(m["test"]) if first else "",
                "Mission": m["mission"]   if first else "",
                "ΔPA [°]": obs_dpa        if first else "",
                "r [M]":   obs_r          if first else "",
                "P [min]": obs_P          if first else "",
                "Model":   f"Exp {['','I','II','III','IV'][exp_id]}",
            }

            for tname, decimals, clamp in (("spin", 2, (-1.0, 1.0)),
                                            ("incl", 1, None),
                                            ("theta", 1, None)):
                target = _exp_target(cfg, tname)
                if target is None:
                    row[f"{tname} pred"] = "—"
                    row[f"{tname} 2σ"]   = "—"
                    continue
                model, scalers = cached_model(exp_id, tname)
                pred, lo, hi = mc_predict_with_interval(
                    model, scalers, x, sigma_obs,
                    n_mc=n_mc, in_radians=target["in_radians"], clamp=clamp,
                    rng_seed=42,
                )
                row[f"{tname} pred"] = fmt_pred(pred, decimals)
                row[f"{tname} 2σ"]   = fmt_ci(lo, hi, decimals)
            rows.append(row)

        # Truth row from the chosen reference file.
        rows.append({
            "Test": "", "Mission": "Truth",
            "ΔPA [°]": "", "r [M]": "", "P [min]": "", "Model": "",
            "spin pred":  fmt_pred(ref_mock["truth"]["a"],     2), "spin 2σ":  "—",
            "incl pred":  fmt_pred(ref_mock["truth"]["i"],     1), "incl 2σ":  "—",
            "theta pred": fmt_pred(ref_mock["truth"]["theta"], 1), "theta 2σ": "—",
        })
    return pd.DataFrame(rows)


mission_table = build_mission_table(MISSIONS)


def _style(df):
    def row_style(r):
        if r["Mission"] == "Truth":
            return ["font-style: italic; background-color: #f5f5f5"] * len(r)
        return [""] * len(r)
    return (df.style
              .hide(axis="index")
              .apply(row_style, axis=1)
              .set_table_styles([
                  {"selector": "th",
                   "props": "text-align: center; background-color: #fafafa; "
                            "border-bottom: 2px solid #ddd;"},
                  {"selector": "td",
                   "props": "text-align: center; padding: 4px 10px;"},
              ]))


_style(mission_table)
"""))

# ── 8. MC convergence plot ──────────────────────────────────────────────────

CELLS.append(md(r"""## 8. Convergence of the MC estimate with $N_{\rm MC}$

To check that $N_{\rm MC} = 2000$ is enough, we redo the MC propagation while
sweeping $N_{\rm MC}$ from $10$ up to $10\,000$ and watch how the mean and
the empirical 2σ interval (2.5 / 97.5 percentiles) stabilise. Because we
generate the full 10 000-sample stream once and look at successive prefixes
of it, the curves are monotonic in the sense that increasing $N$ strictly
adds more samples — no fresh randomness between points.

The dashed red line in each panel is the truth taken from the corresponding
mock-observation file.
"""))

CELLS.append(code(r"""import matplotlib.pyplot as plt

N_MAX_GRID = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]


def mc_convergence(model, scalers, x_orig, sigma_obs_orig, n_max,
                    in_radians=False, clamp=None, rng_seed=42, ns=None):
    rng = np.random.default_rng(rng_seed)
    eps = rng.standard_normal((n_max, len(x_orig))) * sigma_obs_orig[None]
    x_noisy = x_orig[None] + eps
    x_norm  = (x_noisy - scalers["X_mean"]) / scalers["X_scale"]
    with torch.no_grad():
        y_norm = model(torch.tensor(x_norm, dtype=torch.float32, device=DEVICE))
    y_all = y_norm.cpu().numpy().squeeze() * scalers["y_scale"] + scalers["y_mean"]
    if clamp is not None:
        y_all = np.clip(y_all, clamp[0], clamp[1])

    if ns is None:
        ns = N_MAX_GRID
    factor = DEG if in_radians else 1.0
    means, los, his = [], [], []
    for n in ns:
        s = y_all[:n]
        means.append(float(np.mean(s)) * factor)
        lo, hi = np.percentile(s, [2.5, 97.5]) * factor
        los.append(float(lo))
        his.append(float(hi))
    return list(ns), means, los, his


def truth_value(mock, tname):
    return {"spin": mock["truth"]["a"],
            "incl": mock["truth"]["i"],
            "theta": mock["truth"]["theta"]}[tname]


def plot_convergence_for_mission(mission):
    sigma_kw = mission["sigma"]
    files    = mission["files"]
    ordered_exps = [eid for eid in sorted(files)
                    if applicable(EXP_REGISTRY[eid], MOCK_BY_NAME[files[eid]])]
    targets = ["spin", "incl", "theta"]

    fig, axes = plt.subplots(len(ordered_exps), len(targets),
                              figsize=(4.4 * len(targets), 2.8 * len(ordered_exps)),
                              squeeze=False, sharex=True)
    fig.suptitle(f"MC convergence — {mission['mission']} mission "
                 f"(σ_r={sigma_kw['sigma_r']}, σ_T={sigma_kw['sigma_T']}, "
                 f"σ_DPA={sigma_kw['sigma_DPA']})", y=1.02)

    for r, exp_id in enumerate(ordered_exps):
        cfg  = EXP_REGISTRY[exp_id]
        mock = MOCK_BY_NAME[files[exp_id]]
        x         = feat_for(cfg, mock)
        sigma_obs = sigma_obs_vector(cfg["num_input_features"], **sigma_kw)

        for c, tname in enumerate(targets):
            ax = axes[r, c]
            target = _exp_target(cfg, tname)
            if target is None:
                ax.set_visible(False)
                continue
            model, scalers = cached_model(exp_id, tname)
            ns, means, los, his = mc_convergence(
                model, scalers, x, sigma_obs, n_max=10_000,
                in_radians=target["in_radians"],
                clamp=(-1.0, 1.0) if tname == "spin" else None,
                rng_seed=42,
            )
            ns = np.asarray(ns)
            means, los, his = map(np.asarray, (means, los, his))
            ax.fill_between(ns, los, his, alpha=0.25, color="tab:blue",
                            label="2σ band")
            ax.plot(ns, means, "o-", color="tab:blue", label="MC mean", lw=1.4, ms=4)
            ax.axhline(truth_value(mock, tname), color="tab:red", ls="--",
                       lw=1.2, label="truth")
            ax.set_xscale("log")
            unit = " [°]" if target["in_radians"] else ""
            ax.set_ylabel(f"{tname}{unit}")
            ax.set_title(f"Exp {['','I','II','III','IV'][exp_id]} — {tname}",
                          fontsize=10)
            if r == len(ordered_exps) - 1:
                ax.set_xlabel(r"$N_{\rm MC}$")
            ax.grid(True, alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    return fig


for m in MISSIONS:
    plot_convergence_for_mission(m)
plt.show()
"""))

# ── 9. Method comparison: σ_y vs input ΔPA noise ────────────────────────────

CELLS.append(md(r"""## 9. How does each method's σ scale with input noise?

Approaches 1 (trilinear) and 2 (Jacobian) are deterministic — there is no
$N_{\rm MC}$ to converge over. The right diagnostic for them is to *sweep
the input noise* and watch how the predicted output σ moves. Here we
sweep $\sigma_{\Delta\mathrm{PA}}$ from $0^\circ$ to $20^\circ$ while
holding $\sigma_r = 0.1\,M$ and $\sigma_T = 2\,\mathrm{min}$ fixed, and
overlay all three methods.

The vertical grey line marks the default $\sigma_{\Delta\mathrm{PA}} = 10^\circ$
used in §7. The Jacobian curve (red) should be linear in σ_input; the
trilinear curve (blue) follows the noise-sweep grid; the MC curve (green)
is the empirical sampling estimate.
"""))

CELLS.append(code(r"""def method_sigma(method, exp_id, tname, sigma_kw, mock, n_mc=2000):
    cfg    = EXP_REGISTRY[exp_id]
    target = _exp_target(cfg, tname)
    if target is None:
        return np.nan
    x         = feat_for(cfg, mock)
    sigma_obs = sigma_obs_vector(cfg["num_input_features"], **sigma_kw)
    model, scalers = cached_model(exp_id, tname)

    if method == "trilinear":
        if exp_id not in SWEEP_INTERPS or tname not in SWEEP_INTERPS[exp_id]:
            return np.nan
        return predict_sigma_trilinear(SWEEP_INTERPS[exp_id][tname],
                                        sigma_kw["sigma_r"],
                                        sigma_kw["sigma_T"],
                                        sigma_kw["sigma_DPA"],
                                        target["in_radians"])
    if method == "jacobian":
        return predict_sigma_jacobian(model, scalers, x, sigma_obs, target["in_radians"])
    if method == "mc":
        return predict_sigma_mc(model, scalers, x, sigma_obs, target["in_radians"], n_mc=n_mc)
    raise ValueError(method)


def plot_method_sweep_for_mission(mission,
                                   varying="sigma_DPA",
                                   sweep=np.linspace(0.0, 20.0, 21)):
    files    = mission["files"]
    base_sig = dict(mission["sigma"])
    ordered_exps = [eid for eid in sorted(files)
                    if applicable(EXP_REGISTRY[eid], MOCK_BY_NAME[files[eid]])]
    targets = ["spin", "incl", "theta"]

    fig, axes = plt.subplots(len(ordered_exps), len(targets),
                              figsize=(4.4 * len(targets), 2.8 * len(ordered_exps)),
                              squeeze=False, sharex=True)
    var_label = {"sigma_r": r"$\sigma_r$ [M]",
                 "sigma_T": r"$\sigma_T$ [min]",
                 "sigma_DPA": r"$\sigma_{\Delta\mathrm{PA}}$ [°]"}[varying]
    fig.suptitle(f"σ_y vs {var_label} — {mission['mission']} mission "
                 f"(other σ at default)", y=1.02)

    for r, exp_id in enumerate(ordered_exps):
        cfg  = EXP_REGISTRY[exp_id]
        mock = MOCK_BY_NAME[files[exp_id]]
        for c, tname in enumerate(targets):
            ax = axes[r, c]
            target = _exp_target(cfg, tname)
            if target is None:
                ax.set_visible(False)
                continue
            tri, jac, mc_v = [], [], []
            for v in sweep:
                sk = dict(base_sig)
                sk[varying] = float(v)
                tri.append(method_sigma("trilinear", exp_id, tname, sk, mock))
                jac.append(method_sigma("jacobian",  exp_id, tname, sk, mock))
                mc_v.append(method_sigma("mc",        exp_id, tname, sk, mock, n_mc=2000))
            sweep_a = np.asarray(sweep)
            if not np.all(np.isnan(tri)):
                ax.plot(sweep_a, tri, "o-", color="tab:blue",   label="Approach 1 (trilinear)", lw=1.4, ms=4)
            ax.plot(sweep_a, jac,  "s-", color="tab:red",    label="Approach 2 (Jacobian)",   lw=1.4, ms=4)
            ax.plot(sweep_a, mc_v, "^-", color="tab:green",  label="Approach 3 (MC, N=2000)", lw=1.4, ms=4)
            ax.axvline(base_sig[varying], color="grey", ls="--", lw=1, alpha=0.7)
            unit = " [°]" if target["in_radians"] else ""
            ax.set_ylabel(f"σ({tname}){unit}")
            ax.set_title(f"Exp {['','I','II','III','IV'][exp_id]} — {tname}",
                          fontsize=10)
            if r == len(ordered_exps) - 1:
                ax.set_xlabel(var_label)
            ax.grid(True, alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    return fig


for m in MISSIONS:
    plot_method_sweep_for_mission(m, varying="sigma_DPA",
                                   sweep=np.linspace(0.0, 20.0, 21))
plt.show()
"""))

# ── Write ────────────────────────────────────────────────────────────────────

NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.x",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(NB, indent=1))
print(f"Wrote {NB_PATH} ({len(CELLS)} cells)")
