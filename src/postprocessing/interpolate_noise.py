#!/usr/bin/env python3
"""Trilinear interpolation tool for Experiment VI/VII noise propagation grids.

Given user-supplied measurement uncertainties (sigma_r, sigma_T, sigma_DPA),
estimates the expected prediction error (σ) for each target via trilinear
interpolation over the 5×5×5 noise grid trained in the selected experiment.

Usage — command-line:
    cd /scratch/ralbe/meniar_and_django/smbh_hotspots_repository
    python src/postprocessing/interpolate_noise.py --exp 6 \\
        --sigma_r 0.3 --sigma_T 1.2 --sigma_DPA 7.0
    python src/postprocessing/interpolate_noise.py --exp 7 \\
        --sigma_r 0.3 --sigma_T 1.2 --sigma_DPA 7.0

Usage — interactive (no --sigma_* arguments):
    python src/postprocessing/interpolate_noise.py --exp 6
    python src/postprocessing/interpolate_noise.py --exp 7
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from src.utils.config import get_repo_root

METRIC = "error_std"

EXP_CONFIGS = {
    6: {
        "name":    "experiment_6_eq_noise_sweep",
        "targets": ["spin", "incl"],
        "label":   "Experiment VI (equatorial)",
        "grid_desc": "sigma_T ∈ [0,2] min, sigma_r ∈ [0,2] M, sigma_DPA ∈ [0,20] deg",
    },
    7: {
        "name":    "experiment_7_noneq_noise_sweep",
        "targets": ["spin", "incl", "theta", "z"],
        "label":   "Experiment VII (non-equatorial)",
        "grid_desc": "sigma_T ∈ [0,2] min, sigma_r ∈ [0,2] M, sigma_DPA ∈ [0,20] deg",
    },
}

# Per-target display: (human label, unit_suffix, apply_rad_conversion)
# apply_rad_conversion: if True, σ from the grid is in radians and is
# converted to degrees via np.rad2deg(σ).
TARGET_DISPLAY = {
    "spin":  ("spin α",        "",    False),
    "incl":  ("inclination i", "deg", True),
    "theta": ("theta θ",       "deg", True),
    "z":     ("z",             "",    False),
}


def load_interpolators(root: Path, exp_name: str,
                       targets: list[str]) -> dict:
    """Build one RegularGridInterpolator per target from the merged sweep CSVs."""
    interps = {}
    for target in targets:
        csv_path = root / "results" / "metrics" / exp_name / f"{target}_noise_sweep.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing {csv_path}.\n"
                "Run src/postprocessing/plot_noise_sweep.py first to merge the combo CSVs."
            )
        df = pd.read_csv(csv_path)

        T_vals   = sorted(df['sigma_T'].unique())
        r_vals   = sorted(df['sigma_r'].unique())
        DPA_vals = sorted(df['sigma_DPA'].unique())

        # Build (nT, nr, nDPA) array
        arr = np.full((len(T_vals), len(r_vals), len(DPA_vals)), np.nan)
        T_idx   = {v: i for i, v in enumerate(T_vals)}
        r_idx   = {v: i for i, v in enumerate(r_vals)}
        DPA_idx = {v: i for i, v in enumerate(DPA_vals)}
        metric_col = f"{target}_{METRIC}"
        for _, row in df.iterrows():
            arr[T_idx[row['sigma_T']],
                r_idx[row['sigma_r']],
                DPA_idx[row['sigma_DPA']]] = row[metric_col]

        if np.any(np.isnan(arr)):
            n_nan = np.sum(np.isnan(arr))
            print(f"WARNING ({target}): {n_nan} missing grid points — "
                  "interpolation near these points may be unreliable.")

        interps[target] = RegularGridInterpolator(
            (T_vals, r_vals, DPA_vals),
            arr,
            method='linear',
            bounds_error=False,   # clamp instead of raising
            fill_value=None       # extrapolate at boundaries
        )
        interps[f"{target}_bounds"] = {
            'sigma_T':   (min(T_vals),   max(T_vals)),
            'sigma_r':   (min(r_vals),   max(r_vals)),
            'sigma_DPA': (min(DPA_vals), max(DPA_vals)),
        }

    return interps


def warn_if_out_of_bounds(sigma_T: float, sigma_r: float, sigma_DPA: float,
                          bounds: dict):
    msgs = []
    if not (bounds['sigma_T'][0] <= sigma_T <= bounds['sigma_T'][1]):
        msgs.append(f"sigma_T={sigma_T} outside grid [{bounds['sigma_T'][0]}, {bounds['sigma_T'][1]}]")
    if not (bounds['sigma_r'][0] <= sigma_r <= bounds['sigma_r'][1]):
        msgs.append(f"sigma_r={sigma_r} outside grid [{bounds['sigma_r'][0]}, {bounds['sigma_r'][1]}]")
    if not (bounds['sigma_DPA'][0] <= sigma_DPA <= bounds['sigma_DPA'][1]):
        msgs.append(f"sigma_DPA={sigma_DPA} outside grid [{bounds['sigma_DPA'][0]}, {bounds['sigma_DPA'][1]}]")
    if msgs:
        print("WARNING: extrapolating — " + "; ".join(msgs))


def query(interps: dict, targets: list[str],
          sigma_T: float, sigma_r: float, sigma_DPA: float):
    print(f"\nInput uncertainties:")
    print(f"  sigma_r   = {sigma_r} M")
    print(f"  sigma_T   = {sigma_T} min")
    print(f"  sigma_DPA = {sigma_DPA} deg")
    print()

    point = np.array([[sigma_T, sigma_r, sigma_DPA]])
    for target in targets:
        warn_if_out_of_bounds(sigma_T, sigma_r, sigma_DPA,
                              interps[f"{target}_bounds"])
        sigma_out = float(interps[target](point)[0])
        label, unit, rad_conv = TARGET_DISPLAY.get(target, (target, "", False))
        if rad_conv:
            value_deg = np.rad2deg(sigma_out)
            print(f"  Estimated σ_{label:<16s} = {value_deg:.4f} {unit}  "
                  f"(trilinear interpolation)")
        else:
            print(f"  Estimated σ_{label:<16s} = {sigma_out:.4f}  "
                  f"(trilinear interpolation)")


def interactive_loop(interps: dict, targets: list[str], cfg: dict):
    print(f"\n{cfg['label']} — noise propagation tool")
    print(f"Grid: {cfg['grid_desc']}")
    print("Type 'quit' to exit.\n")
    while True:
        try:
            raw = input("Enter sigma_r sigma_T sigma_DPA (space-separated): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if raw.lower() in ('quit', 'exit', 'q'):
            break
        parts = raw.split()
        if len(parts) != 3:
            print("  Please enter exactly 3 numbers.")
            continue
        try:
            sigma_r, sigma_T, sigma_DPA = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            print("  Invalid numbers.")
            continue
        query(interps, targets, sigma_T, sigma_r, sigma_DPA)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate prediction σ for each target given measurement uncertainties."
    )
    parser.add_argument(
        '--exp', type=int, choices=[6, 7], required=True,
        help="Which experiment to use (6 = equatorial, 7 = non-equatorial)."
    )
    parser.add_argument('--sigma_r',   type=float, default=None, help="Uncertainty in r [M]")
    parser.add_argument('--sigma_T',   type=float, default=None, help="Uncertainty in T [min]")
    parser.add_argument('--sigma_DPA', type=float, default=None, help="Uncertainty in DPA [deg]")
    args = parser.parse_args()

    cfg     = EXP_CONFIGS[args.exp]
    root    = get_repo_root()
    interps = load_interpolators(root, cfg["name"], cfg["targets"])

    if args.sigma_r is not None and args.sigma_T is not None and args.sigma_DPA is not None:
        query(interps, cfg["targets"], args.sigma_T, args.sigma_r, args.sigma_DPA)
    else:
        interactive_loop(interps, cfg["targets"], cfg)


if __name__ == "__main__":
    main()
