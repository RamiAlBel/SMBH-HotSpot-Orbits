#!/usr/bin/env python3
"""Trilinear interpolation tool for Experiment VI noise propagation grid.

Given user-supplied measurement uncertainties (sigma_r, sigma_T, sigma_DPA),
estimates the expected prediction error (σ) for spin and inclination via
trilinear interpolation over the 5×5×5 noise grid trained in Experiment VI.

Usage — command-line:
    cd /scratch/ralbe/meniar_and_django/smbh_hotspots_repository
    python src/postprocessing/interpolate_noise.py \\
        --sigma_r 0.3 --sigma_T 1.2 --sigma_DPA 7.0

Usage — interactive (no arguments):
    python src/postprocessing/interpolate_noise.py
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

EXP_NAME = "experiment_6_eq_noise_sweep"
TARGETS  = ["spin", "incl"]
METRIC   = "error_std"


def load_interpolators(root: Path) -> dict[str, RegularGridInterpolator]:
    """Build one RegularGridInterpolator per target from the merged sweep CSVs."""
    interps = {}
    for target in TARGETS:
        csv_path = root / "results" / "metrics" / EXP_NAME / f"{target}_noise_sweep.csv"
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
        # Store grid bounds for clamping warnings
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


def query(interps: dict, sigma_T: float, sigma_r: float, sigma_DPA: float):
    print(f"\nInput uncertainties:")
    print(f"  sigma_r   = {sigma_r} M")
    print(f"  sigma_T   = {sigma_T} min")
    print(f"  sigma_DPA = {sigma_DPA} deg")
    print()

    for target in TARGETS:
        warn_if_out_of_bounds(sigma_T, sigma_r, sigma_DPA,
                              interps[f"{target}_bounds"])
        point = np.array([[sigma_T, sigma_r, sigma_DPA]])
        sigma_out = float(interps[target](point)[0])
        label = "spin α" if target == "spin" else "inclination i"
        if target == "spin":
            print(f"  Estimated σ_{label:<14s} = {sigma_out:.4f}  "
                  f"(trilinear interpolation)")
        else:
            sigma_deg = np.rad2deg(sigma_out / 10.0)  # CSV i-values are 10x actual degrees
            print(f"  Estimated σ_{label:<14s} = {sigma_deg:.4f} deg  "
                  f"(trilinear interpolation)")


def interactive_loop(interps: dict):
    print("\nExperiment VI — noise propagation tool")
    print("Grid: sigma_T ∈ [0,2] min, sigma_r ∈ [0,2] M, sigma_DPA ∈ [0,20] deg")
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
        query(interps, sigma_T, sigma_r, sigma_DPA)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate prediction σ for spin and inclination given measurement uncertainties."
    )
    parser.add_argument('--sigma_r',   type=float, default=None, help="Uncertainty in r [M]")
    parser.add_argument('--sigma_T',   type=float, default=None, help="Uncertainty in T [min]")
    parser.add_argument('--sigma_DPA', type=float, default=None, help="Uncertainty in DPA [deg]")
    args = parser.parse_args()

    root = get_repo_root()
    interps = load_interpolators(root)

    if args.sigma_r is not None and args.sigma_T is not None and args.sigma_DPA is not None:
        query(interps, args.sigma_T, args.sigma_r, args.sigma_DPA)
    else:
        interactive_loop(interps)


if __name__ == "__main__":
    main()
