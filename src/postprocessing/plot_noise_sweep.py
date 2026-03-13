#!/usr/bin/env python3
"""Post-processing for Experiment VI: merge combo CSVs, then plot heatmaps and marginals.

Run after all 125 SLURM array jobs have completed:

    cd /scratch/ralbe/meniar_and_django/smbh_hotspots_repository
    python src/postprocessing/plot_noise_sweep.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from src.utils.config import get_repo_root

EXP_NAME = "experiment_6_eq_noise_sweep"
TARGETS   = ["spin", "incl"]
METRIC    = "error_std"   # the σ of (pred − true) on the test set


# ------------------------------------------------------------------ helpers

def load_and_merge(root: Path) -> pd.DataFrame:
    metrics_dir = root / "results" / "metrics" / EXP_NAME
    combo_files = sorted(metrics_dir.glob("combo_*.csv"),
                         key=lambda p: int(p.stem.split('_')[1]))

    if not combo_files:
        raise FileNotFoundError(
            f"No combo_*.csv files found in {metrics_dir}.\n"
            "Run all 125 SLURM array jobs first."
        )

    missing = [i for i in range(125)
               if not (metrics_dir / f"combo_{i}.csv").exists()]
    if missing:
        print(f"WARNING: {len(missing)} combo CSVs missing: {missing[:10]}{'...' if len(missing)>10 else ''}")

    df = pd.concat([pd.read_csv(f) for f in combo_files], ignore_index=True)
    df = df.sort_values(['sigma_T', 'sigma_r', 'sigma_DPA']).reset_index(drop=True)

    # Save merged CSVs
    for target in TARGETS:
        cols = ['sigma_T', 'sigma_r', 'sigma_DPA'] + \
               [c for c in df.columns if c.startswith(f"{target}_")]
        df[cols].to_csv(metrics_dir / f"{target}_noise_sweep.csv", index=False)
        print(f"Saved {target}_noise_sweep.csv  ({len(df)} rows)")

    return df


def get_grid_axes(df: pd.DataFrame) -> tuple:
    T_vals   = sorted(df['sigma_T'].unique())
    r_vals   = sorted(df['sigma_r'].unique())
    DPA_vals = sorted(df['sigma_DPA'].unique())
    return T_vals, r_vals, DPA_vals


def pivot_3d(df: pd.DataFrame, col: str,
             T_vals, r_vals, DPA_vals) -> np.ndarray:
    """Return (nT, nr, nDPA) array of `col` values."""
    arr = np.full((len(T_vals), len(r_vals), len(DPA_vals)), np.nan)
    T_idx   = {v: i for i, v in enumerate(T_vals)}
    r_idx   = {v: i for i, v in enumerate(r_vals)}
    DPA_idx = {v: i for i, v in enumerate(DPA_vals)}
    for _, row in df.iterrows():
        i = T_idx[row['sigma_T']]
        j = r_idx[row['sigma_r']]
        k = DPA_idx[row['sigma_DPA']]
        arr[i, j, k] = row[col]
    return arr


# ---------------------------------------------------------------- plotting

def plot_heatmaps(arr: np.ndarray, target: str, metric_col: str,
                  T_vals, r_vals, DPA_vals, fig_dir: Path):
    """3 heatmaps: average over each axis in turn."""
    pairs = [
        # (row_axis_label, row_vals, col_axis_label, col_vals, mean_axis_idx)
        (r"$\sigma_r$ [M]",   r_vals,   r"$\sigma_T$ [min]", T_vals,   2),  # avg over DPA
        (r"$\sigma_r$ [M]",   r_vals,   r"$\sigma_{DPA}$ [deg]", DPA_vals, 0),  # avg over T
        (r"$\sigma_T$ [min]", T_vals,   r"$\sigma_{DPA}$ [deg]", DPA_vals, 1),  # avg over r
    ]
    fname_suffixes = ["r_vs_T", "r_vs_DPA", "T_vs_DPA"]

    for (ylabel, yvals, xlabel, xvals, avg_ax), suffix in zip(pairs, fname_suffixes):
        data = np.nanmean(arr, axis=avg_ax)  # shape depends on which axis was averaged

        # Make sure data is (len(yvals), len(xvals))
        # arr axes: (nT, nr, nDPA)
        # avg over DPA (ax=2) → (nT, nr)  want rows=r, cols=T → transpose
        # avg over T   (ax=0) → (nr, nDPA) already (r, DPA)
        # avg over r   (ax=1) → (nT, nDPA) want rows=T, cols=DPA → already ok
        if avg_ax == 2:
            data = data.T  # now (nr, nT)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            data,
            xticklabels=[f"{v}" for v in xvals],
            yticklabels=[f"{v}" for v in yvals],
            annot=True, fmt=".3f", cmap="YlOrRd",
            ax=ax, cbar_kws={'label': rf"$\sigma_{{{target}}}$ (avg over 3rd dim)"}
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        target_label = r"\alpha" if target == "spin" else "i"
        ax.set_title(rf"Exp VI — $\sigma_{{{target_label}}}$ heatmap")
        fig.tight_layout()
        out = fig_dir / f"{target}_heatmap_{suffix}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out.name}")


def plot_marginals(arr: np.ndarray, target: str,
                   T_vals, r_vals, DPA_vals, fig_dir: Path):
    """One figure with 3 subplots: σ_error vs each input σ."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    target_label = r"\alpha" if target == "spin" else "i"

    # σ vs sigma_r  (mean over T & DPA axes)
    marginal_r = np.nanmean(arr, axis=(0, 2))   # shape (nr,)
    axes[0].plot(r_vals, marginal_r, 'o-', color='steelblue')
    axes[0].set_xlabel(r"$\sigma_r$ [M]")
    axes[0].set_ylabel(rf"$\sigma_{{{target_label}}}$ (avg)")
    axes[0].set_title(rf"$\sigma_{{{target_label}}}$ vs $\sigma_r$")
    axes[0].grid(True, alpha=0.4)

    # σ vs sigma_T  (mean over r & DPA axes)
    marginal_T = np.nanmean(arr, axis=(1, 2))   # shape (nT,)
    axes[1].plot(T_vals, marginal_T, 'o-', color='tomato')
    axes[1].set_xlabel(r"$\sigma_T$ [min]")
    axes[1].set_ylabel(rf"$\sigma_{{{target_label}}}$ (avg)")
    axes[1].set_title(rf"$\sigma_{{{target_label}}}$ vs $\sigma_T$")
    axes[1].grid(True, alpha=0.4)

    # σ vs sigma_DPA  (mean over T & r axes)
    marginal_DPA = np.nanmean(arr, axis=(0, 1))  # shape (nDPA,)
    axes[2].plot(DPA_vals, marginal_DPA, 'o-', color='seagreen')
    axes[2].set_xlabel(r"$\sigma_{DPA}$ [deg]")
    axes[2].set_ylabel(rf"$\sigma_{{{target_label}}}$ (avg)")
    axes[2].set_title(rf"$\sigma_{{{target_label}}}$ vs $\sigma_{{DPA}}$")
    axes[2].grid(True, alpha=0.4)

    fig.suptitle(f"Experiment VI — marginal noise propagation ({target})", y=1.02)
    fig.tight_layout()
    out = fig_dir / f"{target}_marginal.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------- main

def main():
    root = get_repo_root()
    df = load_and_merge(root)
    T_vals, r_vals, DPA_vals = get_grid_axes(df)

    fig_dir = root / "results" / "figures" / EXP_NAME
    fig_dir.mkdir(parents=True, exist_ok=True)

    for target in TARGETS:
        metric_col = f"{target}_{METRIC}"
        print(f"\n--- Plots for {target} ({metric_col}) ---")
        arr = pivot_3d(df, metric_col, T_vals, r_vals, DPA_vals)
        plot_heatmaps(arr, target, metric_col, T_vals, r_vals, DPA_vals, fig_dir)
        plot_marginals(arr, target, T_vals, r_vals, DPA_vals, fig_dir)

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
