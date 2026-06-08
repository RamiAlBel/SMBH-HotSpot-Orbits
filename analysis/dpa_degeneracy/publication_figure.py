#!/usr/bin/env python3
"""Publication-ready degeneracy figure.

Reuses the hero-pair selection already written to
results/analysis/dpa_degeneracy/figures/<scenario>_hero_pairs.csv and renders a
clean grid (2 columns): pairs of very different (a, i, theta) that produce
near-identical DPA(t) curves within a sigma_DPA error bar. The x-axis is shared
physical time (phase x matched period); a constant +/-2 deg error bar is drawn
on every sample of both curves. Left-column panels carry the y-axis on the
left, right-column panels on the right, and the columns/rows sit flush.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "data" / "processed" / "dpa_dataset_neq_45.csv"
ANALYSIS_DIR = REPO_ROOT / "results" / "analysis" / "dpa_degeneracy"
DPA_COLS = [f"DPA_{i/10:.1f}" for i in range(1, 11)]
PHASE = np.arange(1, 11) / 10.0

# Colorbrewer RdBu endpoints: classic, calm, high-contrast.
BLUE, RED = "#2166ac", "#b2182b"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=["dpa_only", "full_obs"], default="full_obs")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--out-dir", type=Path, default=ANALYSIS_DIR / "figures")
    p.add_argument("--pairs", type=int, nargs="+", default=[0, 1, 2, 3],
                   help="row indices into <scenario>_hero_pairs.csv")
    p.add_argument("--sigma-dpa", type=float, default=2.0, help="error bar [deg]")
    p.add_argument("--normalize", action="store_true",
                   help="plot t/T (fills width) instead of absolute minutes")
    args = p.parse_args()

    heroes = pd.read_csv(args.out_dir / f"{args.scenario}_hero_pairs.csv")
    curves = pd.read_csv(args.dataset)[DPA_COLS].to_numpy(np.float32)

    rows = [heroes.iloc[i] for i in args.pairs]
    periods = [0.5 * (r.T_q + r.T_m) for r in rows]
    ncols = 2

    def col_ticks(xmax, col):
        if args.normalize:
            ticks = np.arange(0, 1.01, 0.25)
        else:
            ticks = np.arange(0, xmax, 20.0)
        return ticks[ticks > 0] if col == 1 else ticks  # drop 0 on the right

    # Each column shares its own x-range so both columns fill the width.
    col_xmax = {c: (1.0 if args.normalize
                    else max(p for k, p in enumerate(periods) if k % ncols == c))
                for c in range(ncols)}

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "font.size": 13, "axes.labelsize": 14, "legend.fontsize": 9.5,
        "xtick.labelsize": 12, "ytick.labelsize": 11,
        "axes.linewidth": 0.9, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 4.5, "ytick.major.size": 4.5,
    })
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.6, 3.3 * nrows + 0.3),
                             squeeze=False,
                             gridspec_kw={"wspace": 0.0, "hspace": 0.0})
    axes = axes.ravel()
    sd = args.sigma_dpa

    for k, (ax, row, period) in enumerate(zip(axes, rows, periods)):
        col, last_row = k % ncols, k // ncols == nrows - 1
        cq = curves[int(row.query_idx)]
        cm = np.roll(curves[int(row.match_idx)], int(row.roll))
        t = PHASE if args.normalize else PHASE * period

        lab_a = f"A:  $a={row.a_q:+.2f}$,  $i={row.i_q:.0f}\\degree$,  $\\theta={row.theta_q:+.0f}\\degree$"
        lab_b = f"B:  $a={row.a_m:+.2f}$,  $i={row.i_m:.0f}\\degree$,  $\\theta={row.theta_m:+.0f}\\degree$"
        ax.errorbar(t, cq, yerr=sd, fmt="o-", color=BLUE, lw=1.8, ms=5,
                    capsize=2.5, elinewidth=0.9, ecolor=BLUE, label=lab_a, zorder=3)
        ax.errorbar(t, cm, yerr=sd, fmt="s--", color=RED, lw=1.8, ms=5,
                    mfc="white", mew=1.4, capsize=2.5, elinewidth=0.9, ecolor=RED,
                    alpha=0.9, label=lab_b, zorder=2)

        ax.set_xlim(0, col_xmax[col] * 1.04)
        ax.margins(y=0.14)
        ax.grid(axis="y", color="0.7", lw=0.5, alpha=0.35, zorder=0)
        leg = ax.legend(loc="lower center", frameon=True, framealpha=0.92,
                        handlelength=1.6, borderpad=0.35, edgecolor="0.8")
        leg.get_frame().set_linewidth(0.6)

        tag = chr(ord("a") + k)
        ax.text(0.028, 0.95, f"({tag})", transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="left")
        if not args.normalize:
            ax.text(0.10, 0.95, f"$T\\approx{period:.0f}$ min", transform=ax.transAxes,
                    fontsize=10.5, color="0.4", va="top", ha="left")

        # y-axis: left column ticks on the left, right column ticks on the right
        ax.yaxis.set_major_locator(MaxNLocator(5, prune="both"))
        if col == 1:
            ax.yaxis.tick_right()

        # x ticks: per-column shared range; only bottom row labelled
        ax.set_xticks(col_ticks(col_xmax[col], col))
        ax.tick_params(labelbottom=last_row)

    for ax in axes[len(rows):]:
        ax.axis("off")

    fig.supxlabel(r"$t/T$" if args.normalize else "time  [min]", fontsize=14)
    fig.tight_layout()
    # One shared y-axis label on the left; the right column shows ticks only.
    fig.text(0.005, 0.5, r"$\Delta$PA  [deg]", rotation=90,
             va="center", ha="left", fontsize=14)
    suffix = "_norm" if args.normalize else ""
    for ext in ("png", "pdf"):
        out = args.out_dir / f"{args.scenario}_degeneracy_publication{suffix}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
