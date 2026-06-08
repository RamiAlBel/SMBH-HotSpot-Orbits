"""
Self-contained script to reproduce Part 3 (MC spin table) and Part 4/cell-42
(combined all-parameter table) from the uncertainty_estimation notebook,
using the updated observation parameters.

Run from the repo root:
    python3 run_mc_tables.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from src.models.regression_head import RegressionHead
from src.training.data_loader import (
    build_features_targets_avg,
    build_features_targets_timeseries,
    prepare_dataloaders,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEVICE}  repo_root={repo_root}")

# ── Mock observations (updated parameters) ───────────────────────────────────
MOCK_OBS = [
    dict(mission="EHT",   r=9.0, P=90.0, a_truth=-0.18, sigma_r=1.0,  sigma_T=1.0,  sigma_DPA=15.0),
    dict(mission="ngEHT", r=5.5, P=54.0, a_truth=0.95,  sigma_r=0.35, sigma_T=0.5,  sigma_DPA=12.0),
    dict(mission="BHEX",  r=4.5, P=30.0, a_truth=-0.78, sigma_r=0.1,  sigma_T=0.33, sigma_DPA=2.0),
]

# ── Experiment configs ────────────────────────────────────────────────────────
EXP13_CONFIGS = {
    1:  dict(name="experiment_1_eq_avg",      dataset="dpa_dataset_i0.csv",        exp_type="avg"),
    2:  dict(name="experiment_2_eq_full",      dataset="dpa_dataset_ultradense.csv", exp_type="ts"),
    10: dict(name="experiment_10_neq45_full",  dataset="dpa_dataset_neq_45.csv",    exp_type="ts"),
}

DATASET_PATHS = {
    1:  repo_root / "data" / "processed" / "dpa_dataset_i0.csv",
    2:  repo_root / "data" / "processed" / "dpa_dataset_ultradense.csv",
    10: repo_root / "data" / "processed" / "dpa_dataset_neq_45.csv",
}

SPLIT_CFG = dict(train_ratio=0.8, val_ratio=0.1, random_seed=42, batch_size=256)

# ── Utility: load model ───────────────────────────────────────────────────────
def load_model(ckpt: dict, input_dim: int, device: str) -> RegressionHead:
    model = RegressionHead(
        input_dim=input_dim, hidden_dims=(256, 256), num_blocks=2, dropout=0.1,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

# ── Find nearest dataset point ────────────────────────────────────────────────
def get_all_features_avg(dataset_path):
    df = pd.read_csv(dataset_path)
    features, targets = build_features_targets_avg(df, "spin", "a", False)
    _, _, _, _, _, (_, _, _) = prepare_dataloaders(features, targets, noise_enabled=False, **SPLIT_CFG)
    return df, features

def get_all_features_ts(dataset_path, num_samples=10):
    df = pd.read_csv(dataset_path)
    features, targets, _ = build_features_targets_timeseries(
        df, "spin", "a", num_samples=num_samples, convert_to_radians=False,
        half_orbit=False, random_seed=42,
    )
    return df, features

def find_nearest(df, features_all, r_obs, P_obs, a_obs):
    d = np.sqrt(
        ((df["r"]      - r_obs)  / 1.0)  ** 2 +
        ((df["Period"] - P_obs)  / 10.0) ** 2 +
        ((df["a"]      - a_obs)  / 0.5)  ** 2
    )
    g = int(d.idxmin())
    return {
        "r": float(df.loc[g, "r"]), "Period": float(df.loc[g, "Period"]),
        "a": float(df.loc[g, "a"]), "dist": float(d.iloc[g]),
        "global_idx": g, "x_raw": features_all[g].copy(),
    }

# ── Build matched_points ──────────────────────────────────────────────────────
matched_points = {}
for exp_id, exp_cfg in EXP13_CONFIGS.items():
    ds_path = DATASET_PATHS[exp_id]
    if exp_cfg["exp_type"] == "avg":
        df_full, X_full = get_all_features_avg(ds_path)
    else:
        df_full, X_full = get_all_features_ts(ds_path, num_samples=10)
    print(f"Exp {exp_id} ({exp_cfg['name']}) — {len(df_full)} rows")
    for mock in MOCK_OBS:
        nearest = find_nearest(df_full, X_full, mock["r"], mock["P"], mock["a_truth"])
        matched_points[(mock["mission"], exp_id)] = nearest
        print(f"  {mock['mission']:6s} → matched r={nearest['r']:.2f}  "
              f"P={nearest['Period']:.1f}  a={nearest['a']:+.3f}")
    print()

# ── MC utility ────────────────────────────────────────────────────────────────
def mc_sigma_clipped(model, x_orig_1d, sigma_obs, sx_mean, sx_scale, sy_mean, sy_scale,
                     n_mc=2000, device="cpu"):
    model.eval()
    rng = np.random.default_rng()
    x_noisy = x_orig_1d[None] + rng.standard_normal((n_mc, len(x_orig_1d))) * sigma_obs[None]
    x_norm = (x_noisy - sx_mean) / sx_scale
    with torch.no_grad():
        y_s = model(torch.tensor(x_norm, dtype=torch.float32).to(device)).cpu().numpy().squeeze()
    y_s = np.clip(y_s * sy_scale + sy_mean, -1.0, 1.0)
    return float(np.std(y_s))

def mc_all(model, x_raw, sigma_vec, sx_mean, sx_scale, sy_mean, sy_scale,
           n_mc=2000, device="cpu", clip=None):
    model.eval()
    rng = np.random.default_rng()
    x_noisy = x_raw[None] + rng.standard_normal((n_mc, len(x_raw))) * sigma_vec[None]
    x_norm  = (x_noisy - sx_mean) / sx_scale
    with torch.no_grad():
        y_s = model(torch.tensor(x_norm, dtype=torch.float32).to(device)).cpu().numpy().squeeze()
    y_s = y_s * sy_scale + sy_mean
    if clip is not None:
        y_s = np.clip(y_s, clip[0], clip[1])
    x1 = torch.tensor(((x_raw - sx_mean) / sx_scale)[None], dtype=torch.float32).to(device)
    with torch.no_grad():
        y_p = float(model(x1).cpu().item()) * sy_scale + sy_mean
    if clip is not None:
        y_p = float(np.clip(y_p, clip[0], clip[1]))
    return float(y_p), float(np.std(y_s))

# ── Run MC for spin-only (exps 1, 2, 10) — used in cell-29 table ─────────────
N_MC = 2000
mc_results = {}
print("Running MC for spin (exps 1, 2, 10)...")
for mock in MOCK_OBS:
    mission = mock["mission"]
    for exp_id, exp_cfg in EXP13_CONFIGS.items():
        exp_name = exp_cfg["name"]
        exp_type = exp_cfg["exp_type"]
        key = (mission, exp_id)
        if key not in matched_points:
            continue
        mp = matched_points[key]
        n_dpa = 1 if exp_type == "avg" else 10
        ckpt_path = repo_root / "results" / "checkpoints" / exp_name / "spin" / "model_seed42.pth"
        if not ckpt_path.exists():
            print(f"  Missing: {ckpt_path}")
            continue
        ckpt_e = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_e = load_model(ckpt_e, 2 + n_dpa, DEVICE)
        sx_mean  = ckpt_e["scaler_X_mean"]
        sx_scale = ckpt_e["scaler_X_scale"]
        sy_scale = float(np.array(ckpt_e["scaler_y_scale"]).ravel()[0])
        sy_mean  = float(np.array(ckpt_e["scaler_y_mean"]).ravel()[0])
        sigma_vec = np.array([mock["sigma_r"], mock["sigma_T"]] + [mock["sigma_DPA"]] * n_dpa,
                             dtype=np.float32)
        sigma_mc = mc_sigma_clipped(model_e, mp["x_raw"], sigma_vec,
                                    sx_mean, sx_scale, sy_mean, sy_scale, n_mc=N_MC, device=DEVICE)
        x_norm = (mp["x_raw"] - sx_mean) / sx_scale
        with torch.no_grad():
            a_pred = float(np.clip(
                model_e(torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                        ).cpu().item() * sy_scale + sy_mean, -1.0, 1.0))
        mc_results[key] = {"a_pred": a_pred, "sigma_mc": sigma_mc}
        print(f"  {mission:6s} Exp{exp_id}: a_pred={a_pred:+.3f}  2σ=[{a_pred-2*sigma_mc:+.3f},{a_pred+2*sigma_mc:+.3f}]")

# ── Combined all-parameter MC (cell-42 table: exps 1, 2, 10 / all targets) ───
COMBINED_EXP_CFG = {
    1:  dict(name="experiment_1_eq_avg",      targets=["spin"],                   n_dpa=1),
    2:  dict(name="experiment_2_eq_full",      targets=["spin", "incl"],           n_dpa=10),
    10: dict(name="experiment_10_neq45_full",  targets=["spin", "incl", "theta"], n_dpa=10),
}

COMBINED_TARGET_INFO = {
    "spin":  dict(rad=False, clip=(-1.0, 1.0), col="a",     unit=""),
    "incl":  dict(rad=True,  clip=None,        col="i",     unit="deg"),
    "theta": dict(rad=True,  clip=None,        col="theta", unit="deg"),
    "z":     dict(rad=False, clip=None,        col=None,    unit="M"),
}

mc_combined = {}
print("\nRunning combined MC (all targets)...")
for mock in MOCK_OBS:
    mission = mock["mission"]
    for exp_id, ecfg in COMBINED_EXP_CFG.items():
        mp = matched_points.get((mission, exp_id))
        if mp is None:
            continue
        x_raw = mp["x_raw"]
        n_dpa = ecfg["n_dpa"]
        sigma_vec = np.array([mock["sigma_r"], mock["sigma_T"]] + [mock["sigma_DPA"]] * n_dpa,
                             dtype=np.float32)
        for target in ecfg["targets"]:
            tcfg = COMBINED_TARGET_INFO[target]
            ckpt_path = (repo_root / "results" / "checkpoints" / ecfg["name"]
                         / target / "model_seed42.pth")
            if not ckpt_path.exists():
                print(f"  Missing: {ckpt_path}")
                continue
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            mdl = load_model(ck, 2 + n_dpa, DEVICE)
            sx_m = ck["scaler_X_mean"]; sx_s = ck["scaler_X_scale"]
            sy_s = float(np.array(ck["scaler_y_scale"]).ravel()[0])
            sy_m = float(np.array(ck["scaler_y_mean"]).ravel()[0])
            y_pred, sigma_mc = mc_all(mdl, x_raw, sigma_vec, sx_m, sx_s, sy_m, sy_s,
                                      n_mc=N_MC, device=DEVICE, clip=tcfg["clip"])
            if tcfg["rad"]:
                y_pred   = np.rad2deg(y_pred)
                sigma_mc = np.rad2deg(sigma_mc)
            mc_combined[(mission, exp_id, target)] = {"pred": float(y_pred), "sigma": float(sigma_mc)}
    print(f"  {mission}: done")

# ── Collect truth values from exp 10 matched points ──────────────────────────
df_neq45_truth = pd.read_csv(DATASET_PATHS[10])
truth_vals = {}
for mock in MOCK_OBS:
    mission = mock["mission"]
    mp10 = matched_points.get((mission, 10))
    if mp10 is None:
        truth_vals[mission] = {"spin": mock["a_truth"], "incl": float("nan"),
                               "theta": float("nan"), "z": float("nan")}
        continue
    row = df_neq45_truth.iloc[mp10["global_idx"]]
    i_deg  = float(row["i"])
    th_deg = float(row["theta"])
    z_val  = float(row["r"]) * np.sin(np.deg2rad(th_deg))
    truth_vals[mission] = {"spin": mock["a_truth"], "incl": i_deg, "theta": th_deg, "z": z_val}

print("\nTruth values (from Exp IV matched simulations):")
for mission, tv in truth_vals.items():
    print(f"  {mission}: a={tv['spin']:+.2f}  i={tv['incl']:.1f}°  theta={tv['theta']:.1f}°")

# ── LaTeX: combined all-param table (matching user's requested format) ─────────
# Format: spin, incl, theta (no z, matching the user's table structure)
EXP_LABEL_MAP = {1: "Exp~I", 2: "Exp~II", 10: "Exp~IV"}

MOCK_DISPLAY_C = [
    dict(test_no=1, mission="EHT",   dpa_ctr=140, r_ctr="9.0", r_sig="1.0",  P_ctr="90", P_sig="1"),
    dict(test_no=2, mission="ngEHT", dpa_ctr=172, r_ctr="5.5", r_sig="0.35", P_ctr="54", P_sig="0.5"),
    dict(test_no=3, mission="BHEX",  dpa_ctr=80,  r_ctr="4.5", r_sig="0.1",  P_ctr="30", P_sig="0.33"),
]

def pm(c, s):
    return "$" + str(c) + r" \pm " + str(s) + "$"

def cell_val(pred, sigma):
    lo = pred - 2 * sigma
    hi = pred + 2 * sigma
    return f"${pred:+.1f}$", "$[" + f"{lo:+.1f}" + r",\;" + f"{hi:+.1f}" + "]$"

def cell_clipped(pred, sigma, clip):
    lo = float(np.clip(pred - 2*sigma, clip[0], clip[1]))
    hi = float(np.clip(pred + 2*sigma, clip[0], clip[1]))
    p  = float(np.clip(pred, clip[0], clip[1]))
    return f"${p:+.2f}$", "$[" + f"{lo:+.2f}" + r",\;" + f"{hi:+.2f}" + "]$"

lines = []
lines.append(r"\begin{table*}[t!]")
lines.append(r"    \centering")
lines.append(r"    \footnotesize ")
lines.append(r"    \setlength{\tabcolsep}{5pt} ")
lines.append(
    r"    \caption{All-parameter estimation for 3 \sgra mock observations using three Exp variants.}"
)
lines.append(r"    \label{tab:all_params_mc}")
lines.append(r"    \begin{tabular}{c c | c c c | c | cc | cc | cc}")
lines.append(r"        \toprule")
lines.append(r"        \multicolumn{2}{c|}{Mission} &")
lines.append(r"        \multicolumn{3}{c|}{Observation parameters} &")
lines.append(r"        & \multicolumn{2}{c|}{Spin $a_*$} & \multicolumn{2}{c|}{Incl.\ $i$ [deg]} & \multicolumn{2}{c}{Elev.\ $\theta_z$ [deg]} \\")
lines.append(r"        \cmidrule(lr){1-2}\cmidrule(lr){3-5}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}")
lines.append(r"        Test & Mission & $\Delta\mathrm{PA}\,[^\circ]$ & $r\,[M]$ & $P\,[\mathrm{min}]$ & Model & Pred. & $2\sigma$ & Pred. & $2\sigma$ & Pred. & $2\sigma$ \\")
lines.append(r"        \midrule")

for i_m, md in enumerate(MOCK_DISPLAY_C):
    mission = md["mission"]
    tv      = truth_vals[mission]
    dpa_sig = next(m["sigma_DPA"] for m in MOCK_OBS if m["mission"] == mission)

    dpa_s = pm(md["dpa_ctr"], int(dpa_sig) if dpa_sig == int(dpa_sig) else dpa_sig)
    r_s   = pm(md["r_ctr"], md["r_sig"])
    P_s   = pm(md["P_ctr"], md["P_sig"])

    n_rows = len(COMBINED_EXP_CFG)
    mr = lambda n, v: r"\multirow{" + str(n) + r"}{*}{" + str(v) + "}"

    for k, (exp_id, ecfg) in enumerate(COMBINED_EXP_CFG.items()):
        exp_label = EXP_LABEL_MAP[exp_id]
        cols = []
        for target in ["spin", "incl", "theta"]:  # exclude z
            key = (mission, exp_id, target)
            tcfg = COMBINED_TARGET_INFO[target]
            if target in ecfg["targets"] and key in mc_combined:
                r2 = mc_combined[key]
                p, s = r2["pred"], r2["sigma"]
                if tcfg["clip"]:
                    p_s, iv_s = cell_clipped(p, s, tcfg["clip"])
                else:
                    # round incl/theta to 1 decimal
                    lo = p - 2*s; hi = p + 2*s
                    p_s  = f"${p:+.1f}$"
                    iv_s = "$[" + f"{lo:+.1f}" + r",\;" + f"{hi:+.1f}" + "]$"
                cols.append((p_s, iv_s))
            else:
                cols.append(("---", "---"))

        data_cells = " & ".join(f"{p} & {iv}" for p, iv in cols)

        if k == 0:
            row  = "        " + mr(n_rows, md["test_no"]) + " & " + mr(n_rows, mission)
            row += "\n            & " + mr(n_rows, dpa_s)
            row += "\n            & " + mr(n_rows, r_s)
            row += "\n            & " + mr(n_rows, P_s)
            row += "\n            & " + exp_label + " & " + data_cells + r" \\"
        else:
            row = "        & & & & & " + exp_label + " & " + data_cells + r" \\"
        lines.append(row)

    def fmt_truth(v, dec=2):
        if np.isnan(v): return "---"
        return f"${v:+.{dec}f}$"
    truth_row = (
        r"        \multicolumn{6}{l|}{\textit{Truth}} & "
        + fmt_truth(tv["spin"]) + r" & --- & "
        + fmt_truth(tv["incl"], 1) + r" & --- & "
        + fmt_truth(tv["theta"], 1) + r" & \\"
    )
    lines.append(truth_row)

    if i_m < len(MOCK_DISPLAY_C) - 1:
        lines.append(r"        \midrule")

lines.append(r"        \bottomrule")
lines.append(r"    \end{tabular}")
lines.append(
    r"    \tablefoot{Observational uncertainties are $1\sigma$ Gaussian applied to $r$, $P$, and $\Delta\mathrm{PA}$. "
    r"Spin samples clamped to $[-1,1]$ before computing $\sigma$. "
    r"All $2\sigma$ intervals are Monte Carlo propagated ($N=2000$). "
    r"Spin is clamped to $[-1,1]$. "
    r"Truth values for $i$ and $\theta$ are from the nearest matched simulation in the Exp~IV dataset.}"
)
lines.append(r"\end{table*}")

print("\n\n" + "="*70)
print("LATEX TABLE (combined all-param, no z column):")
print("="*70)
print("\n".join(lines))
