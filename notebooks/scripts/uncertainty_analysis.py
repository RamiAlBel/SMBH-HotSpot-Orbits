"""End-to-end uncertainty estimation on mock observations.

Implements three approaches for predicting output uncertainty σ_y on
(spin α, inclination i, theta θ) given input observational uncertainties
σ_obs = (σ_r, σ_T, σ_DPA):

  1. Trilinear interpolation over the noise-sweep grid (Exp VI / VII).
  2. First-order Jacobian propagation (autodiff in normalised space).
  3. Monte Carlo sampling of the no-noise model with perturbed inputs.

Each method is evaluated on the three mock observations under
``mock_observations/`` at the standard input sigmas
σ_r = 0.1 M, σ_T = 2 min, σ_DPA = 10°.

Run from the repo root:
    python notebooks/scripts/uncertainty_analysis.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import RegularGridInterpolator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.regression_head import RegressionHead
from src.utils.jacobian_uncertainty import (
    compute_jacobian,
    jacobian_sigma,
    mc_sigma,
)


# ── Configuration ───────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEG = 180.0 / np.pi
N_MC = 2000

SIGMA_DEFAULT = dict(sigma_r=0.1, sigma_T=2.0, sigma_DPA=10.0)

MOCK_DIR = REPO_ROOT / "mock_observations"
RESULTS = REPO_ROOT / "results"

# One entry per experiment we want to evaluate.
#   exp_type:     'avg' (Exp 1) or 'ts' (Exp 2/4)
#   ckpt_dir:     no-noise checkpoint root under results/checkpoints/
#   targets:      list of dicts (name, in_radians)
#   sweep_name:   noise-sweep grid used by Approach 1, or None
#   accepts_th:   whether the model trained on neq data (rejects single-row mock obs?)
EXP_REGISTRY = {
    1: dict(
        label="Exp I — eq averaged ΔPA",
        ckpt_dir="experiment_1_eq_avg_no_noise",
        exp_type="avg",
        num_input_features=3,
        targets=[
            dict(name="spin", in_radians=False, unit=""),
            dict(name="incl", in_radians=False, unit="deg"),
        ],
        sweep_name=None,  # exp 1 has no matching noise-sweep grid
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


# ── 1. Mock observation loader ──────────────────────────────────────────────

def parse_mock_filename(path: Path) -> dict:
    """Extract truth parameters encoded in the filename.

    Format: lc_r{r10}_K{K100}_a{a100}_i{i10}_th{th_raw}[_extra].dat
    Convention (per src/preprocessing/prepare_dataset_noneq.py):
        r     = r_raw  / 10           [M]
        K     = K_raw  / 100
        a     = a_raw  / 100          (spin)
        i     = i_raw  / 10           [deg]
        theta = 90 - th_raw           [deg]   (polar angle from spin axis)
    """
    parts = path.stem.split("_")
    return dict(
        r=float(parts[1][1:]) / 10.0,
        K=float(parts[2][1:]) / 100.0,
        a=float(parts[3][1:]) / 100.0,
        i=float(parts[4][1:]) / 10.0,
        theta=90.0 - float(parts[5][2:]),
    )


def load_mock_observation(path: Path) -> dict:
    """Read a mock-observation .dat file.

    Returns a dict with:
        truth:         {r, K, a, i, theta, Period}
        dpa_avg:       single ΔPA value at the first phase (used by Exp 1)
        feat_avg:      [r, Period, dpa_avg]  (Exp 1 input vector)
        feat_ts:       [r, Period, DPA_0.1, …, DPA_1.0]  (Exp 2/4 input vector),
                       or None if the file has fewer than 2 phase samples.
        n_phases:      number of orbit-phase rows in the file.
    """
    text = path.read_text()
    lines = text.splitlines()
    period_min = float(lines[2].split()[1])

    arr = np.loadtxt(io.StringIO("\n".join(lines[3:])))
    if arr.ndim == 1:
        arr = arr[None, :]

    truth = parse_mock_filename(path)
    truth["Period"] = period_min

    # Exp 1 uses a single DPA value (the first row).
    dpa_avg = float(arr[0, -2])
    feat_avg = np.array([truth["r"], period_min, dpa_avg], dtype=np.float32)

    # Exp 2/4 use a 10-sample DPA(t) interpolated onto phase=[0.1, …, 1.0].
    feat_ts = None
    if len(arr) >= 2:
        sample_points = np.linspace(0.1, 1.0, 10)
        phases = arr[:, 1]
        dpa = arr[:, -2]
        idx = np.argsort(phases)
        phases_s, dpa_s = phases[idx], dpa[idx]
        dpa_ts = np.interp(
            sample_points, phases_s, dpa_s,
            left=dpa_s[0], right=dpa_s[-1],
        )
        feat_ts = np.concatenate(([truth["r"], period_min], dpa_ts)).astype(np.float32)

    return dict(
        path=path,
        truth=truth,
        dpa_avg=dpa_avg,
        feat_avg=feat_avg,
        feat_ts=feat_ts,
        n_phases=len(arr),
    )


def load_all_mock_observations() -> list[dict]:
    paths = sorted(MOCK_DIR.glob("**/*.dat"))
    return [load_mock_observation(p) for p in paths]


# ── 2. Model registry ───────────────────────────────────────────────────────

def load_model_and_scalers(exp_id: int, target_name: str, seed: int = 42):
    """Load a no-noise checkpoint + bundled scalers for one (exp, target)."""
    cfg = EXP_REGISTRY[exp_id]
    ckpt_path = (
        RESULTS / "checkpoints" / cfg["ckpt_dir"] / target_name / f"model_seed{seed}.pth"
    )
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = RegressionHead(input_dim=cfg["num_input_features"]).to(DEVICE)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    scalers = dict(
        X_mean=np.asarray(state["scaler_X_mean"], dtype=np.float32),
        X_scale=np.asarray(state["scaler_X_scale"], dtype=np.float32),
        y_mean=float(np.asarray(state["scaler_y_mean"]).item()),
        y_scale=float(np.asarray(state["scaler_y_scale"]).item()),
    )
    return model, scalers


def predict(model, scalers, x_orig: np.ndarray) -> float:
    """Run the model in original (un-normalised) units."""
    x_norm = (x_orig - scalers["X_mean"]) / scalers["X_scale"]
    with torch.no_grad():
        y_norm = model(torch.tensor(x_norm[None], dtype=torch.float32, device=DEVICE))
    return float(y_norm.item() * scalers["y_scale"] + scalers["y_mean"])


def sigma_obs_vector(num_features: int, sigma_r: float, sigma_T: float, sigma_DPA: float) -> np.ndarray:
    """Per-feature observational σ in original units.

    Feature layout: [r, Period, DPA_avg]               (Exp 1, len=3)
                    [r, Period, DPA_0.1, …, DPA_1.0]    (Exp 2/4, len=12)
    """
    sig = np.zeros(num_features, dtype=np.float32)
    sig[0] = sigma_r
    sig[1] = sigma_T
    sig[2:] = sigma_DPA
    return sig


# ── 3. Approach 1: Trilinear noise-sweep interpolation ──────────────────────

def build_interpolators(sweep_name: str, targets: list[dict]) -> dict:
    """Return {target_name: RegularGridInterpolator over (σ_T, σ_r, σ_DPA)}.

    Each combo CSV contains a single row per (σ_T, σ_r, σ_DPA) cell with
    columns {target}_error_std (output σ in the model's training units).
    """
    sweep_dir = RESULTS / "metrics" / sweep_name
    rows = []
    for f in sorted(sweep_dir.glob("combo_*.csv")):
        rows.append(pd.read_csv(f))
    df = pd.concat(rows, ignore_index=True)

    sT_axis = np.sort(df["sigma_T"].unique())
    sr_axis = np.sort(df["sigma_r"].unique())
    sd_axis = np.sort(df["sigma_DPA"].unique())

    interps = {}
    for t in targets:
        col = f"{t['name']}_error_std"
        if col not in df.columns:
            continue
        # Build (n_T, n_r, n_DPA) grid in lexicographic order.
        grid = np.full((len(sT_axis), len(sr_axis), len(sd_axis)), np.nan)
        for _, r in df.iterrows():
            iT = np.searchsorted(sT_axis, r["sigma_T"])
            iR = np.searchsorted(sr_axis, r["sigma_r"])
            iD = np.searchsorted(sd_axis, r["sigma_DPA"])
            grid[iT, iR, iD] = r[col]
        interps[t["name"]] = RegularGridInterpolator(
            (sT_axis, sr_axis, sd_axis), grid,
            bounds_error=False, fill_value=None,  # extrapolate flat
        )
    return interps, dict(sigma_T=sT_axis, sigma_r=sr_axis, sigma_DPA=sd_axis)


def predict_sigma_trilinear(interp, sigma_r: float, sigma_T: float, sigma_DPA: float,
                            in_radians: bool) -> float:
    """Query σ in model units, convert to display unit (deg if radians)."""
    sigma_y = float(interp([sigma_T, sigma_r, sigma_DPA])[0])
    return sigma_y * (DEG if in_radians else 1.0)


# ── 4. Approach 2: Jacobian-based σ ─────────────────────────────────────────

def predict_sigma_jacobian(model, scalers, x_orig: np.ndarray,
                           sigma_obs_orig: np.ndarray, in_radians: bool) -> float:
    """First-order σ via autodiff Jacobian in normalised space."""
    x_norm = (x_orig - scalers["X_mean"]) / scalers["X_scale"]
    x_t = torch.tensor(x_norm, dtype=torch.float32, device=DEVICE)
    J = compute_jacobian(model, x_t).cpu().numpy()
    sigma_y = jacobian_sigma(J, sigma_obs_orig, scalers["X_scale"], scalers["y_scale"])
    return sigma_y * (DEG if in_radians else 1.0)


# ── 5. Approach 3: Monte Carlo σ ────────────────────────────────────────────

def predict_sigma_mc(model, scalers, x_orig: np.ndarray,
                     sigma_obs_orig: np.ndarray, in_radians: bool,
                     n_mc: int = N_MC) -> float:
    sigma_y = mc_sigma(
        model, x_orig.astype(np.float32), sigma_obs_orig,
        scalers["X_mean"], scalers["X_scale"], scalers["y_scale"],
        n_mc=n_mc, device=DEVICE,
    )
    return sigma_y * (DEG if in_radians else 1.0)


# ── 6. Driver: build all tables ─────────────────────────────────────────────

def display_unit(target: dict) -> str:
    return target["unit"] or "—"


def truth_for_target(truth: dict, target_name: str) -> float | None:
    return {"spin": truth["a"], "incl": truth["i"], "theta": truth["theta"]}.get(target_name)


def applicable_to(exp_cfg: dict, mock: dict) -> tuple[bool, str | None]:
    """Decide whether a given mock obs can be fed into the experiment.

    Exp 1 (avg) wants a single DPA → works for the degenerate single-row file.
    Exp 2/4 (ts) need ≥2 phase rows → skip the single-row file.
    """
    if exp_cfg["exp_type"] == "avg":
        return True, mock["feat_avg"]
    return (mock["feat_ts"] is not None), mock["feat_ts"]


def feature_vector(exp_cfg: dict, mock: dict) -> np.ndarray:
    return mock["feat_avg"] if exp_cfg["exp_type"] == "avg" else mock["feat_ts"]


def evaluate_all(mocks: list[dict]) -> pd.DataFrame:
    """Run all 3 methods on each (mock, exp, target) at SIGMA_DEFAULT."""
    sigma_kw = SIGMA_DEFAULT
    rows = []

    # Pre-build sweep interpolators per experiment.
    sweep_interps = {}
    for exp_id, cfg in EXP_REGISTRY.items():
        if cfg["sweep_name"] is not None:
            sweep_interps[exp_id], _ = build_interpolators(cfg["sweep_name"], cfg["targets"])

    for mock in mocks:
        mock_label = mock["path"].name
        for exp_id, cfg in EXP_REGISTRY.items():
            ok, x = applicable_to(cfg, mock)
            if not ok:
                continue
            sigma_obs = sigma_obs_vector(cfg["num_input_features"], **sigma_kw)
            for t in cfg["targets"]:
                model, scalers = load_model_and_scalers(exp_id, t["name"])
                pred = predict(model, scalers, x)
                # Display: convert prediction from rad to deg when applicable.
                pred_disp = pred * (DEG if t["in_radians"] else 1.0)
                truth = truth_for_target(mock["truth"], t["name"])

                # Approach 1
                if exp_id in sweep_interps and t["name"] in sweep_interps[exp_id]:
                    sigma_tri = predict_sigma_trilinear(
                        sweep_interps[exp_id][t["name"]],
                        sigma_kw["sigma_r"], sigma_kw["sigma_T"], sigma_kw["sigma_DPA"],
                        t["in_radians"],
                    )
                else:
                    sigma_tri = np.nan

                # Approach 2
                sigma_jac = predict_sigma_jacobian(
                    model, scalers, x, sigma_obs, t["in_radians"],
                )
                # Approach 3
                sigma_mc = predict_sigma_mc(
                    model, scalers, x, sigma_obs, t["in_radians"],
                )

                rows.append(dict(
                    mock=mock_label,
                    exp=exp_id,
                    exp_label=cfg["label"],
                    target=t["name"],
                    unit=display_unit(t),
                    truth=truth,
                    pred=pred_disp,
                    sigma_trilinear=sigma_tri,
                    sigma_jacobian=sigma_jac,
                    sigma_mc=sigma_mc,
                ))
    return pd.DataFrame(rows)


# ── Sigma-only summary (no mock obs needed): Approach 1 across exps ─────────

def trilinear_summary(sigma_kw: dict) -> pd.DataFrame:
    rows = []
    for exp_id, cfg in EXP_REGISTRY.items():
        if cfg["sweep_name"] is None:
            continue
        interps, _ = build_interpolators(cfg["sweep_name"], cfg["targets"])
        for t in cfg["targets"]:
            if t["name"] not in interps:
                continue
            sigma = predict_sigma_trilinear(
                interps[t["name"]],
                sigma_kw["sigma_r"], sigma_kw["sigma_T"], sigma_kw["sigma_DPA"],
                t["in_radians"],
            )
            rows.append(dict(
                exp=exp_id, exp_label=cfg["label"], target=t["name"],
                unit=display_unit(t), sigma_trilinear=sigma,
            ))
    return pd.DataFrame(rows)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"device={DEVICE}  N_MC={N_MC}")
    print(f"SIGMA_DEFAULT = {SIGMA_DEFAULT}\n")

    print("Loading mock observations…")
    mocks = load_all_mock_observations()
    for m in mocks:
        t = m["truth"]
        print(f"  {m['path'].name:50s}  n_phases={m['n_phases']:3d}  "
              f"a={t['a']:+.2f}  i={t['i']:5.1f}°  θ={t['theta']:5.1f}°  P={t['Period']:.2f} min")
    print()

    print("─── Approach 1 (trilinear) — sigma-only summary at SIGMA_DEFAULT ───")
    print(trilinear_summary(SIGMA_DEFAULT).to_string(index=False))
    print()

    print("─── All three methods on each (mock × exp × target) at SIGMA_DEFAULT ───")
    df = evaluate_all(mocks)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
