"""Violin variants of plot_mocks_tier_grid_absolute.py.

Same MC machinery, but each (target, array, mock) prediction is shown as a
violin of the full Monte-Carlo sample distribution instead of a mean ±2σ error
bar. Ground truth stays a dotted horizontal line and a thin line tracks the
per-array medians. "Tier" wording is replaced by "Array" in all figure text.

Produces two figures under notebooks/figures/:
  • mocks_array_grid_3x3_violin.{pdf,png}  — the full 3×3 grid, as now.
  • mocks_array_theta_violin.{pdf,png}     — only the θ_z panel (Exp IV).

Run from any cwd:
    python notebooks/scripts/plot_mocks_array_grid_violin.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from uncertainty_analysis import (   # noqa: E402
    EXP_REGISTRY, REPO_ROOT, DEVICE, DEG, N_MC,
    load_all_mock_observations, sigma_obs_vector, truth_for_target,
)
from plot_mocks_tier_grid_absolute import (   # noqa: E402
    PLOT_MOCKS, TIERS, X_POS, ROWS, COLS, GT_LINESTYLE,
    cached_model, feat_for, applicable, _exp_target, _make_panel,
    _draw_mock_truth_panel,
)

VIOLIN_WIDTH = 0.30
MOCK_DX = 0.34   # centre-to-centre offset between the two mocks' violins

PUB_RC = {
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11.5,
    "axes.linewidth": 0.9, "axes.edgecolor": "0.15",
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 4.0, "ytick.major.size": 4.0,
    "xtick.minor.size": 2.2, "ytick.minor.size": 2.2,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.color": "0.15", "ytick.color": "0.15",
    "legend.frameon": False, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


def mc_samples(model, scalers, x_orig, sigma_obs_orig, in_radians, clamp,
               rng_seed=42):
    """Full vector of N_MC predicted values (display units)."""
    rng = np.random.default_rng(rng_seed)
    x_noisy = (x_orig[None]
               + rng.standard_normal((N_MC, len(x_orig))) * sigma_obs_orig[None])
    x_norm = (x_noisy - scalers["X_mean"]) / scalers["X_scale"]
    with torch.no_grad():
        y_norm = model(torch.tensor(x_norm, dtype=torch.float32, device=DEVICE))
    y = y_norm.cpu().numpy().squeeze() * scalers["y_scale"] + scalers["y_mean"]
    if clamp is not None:
        y = np.clip(y, clamp[0], clamp[1])
    return y * (DEG if in_radians else 1.0)


def samples_at_arrays(mock, exp_id, tname, in_radians, clamp):
    """(list of N_MC-sample arrays, one per array tier) and the ground truth."""
    cfg = EXP_REGISTRY[exp_id]
    model, scalers = cached_model(exp_id, tname)
    x = feat_for(cfg, mock)
    truth = truth_for_target(mock["truth"], tname)
    samples = [
        mc_samples(model, scalers, x,
                   sigma_obs_vector(cfg["num_input_features"], **kw),
                   in_radians, clamp)
        for _tag, kw in TIERS
    ]
    return samples, truth


def draw_cell(ax, mock_by_name, exp_id, tname, in_radians, clamp):
    cfg = EXP_REGISTRY[exp_id]
    n = len(PLOT_MOCKS)
    for k, spec in enumerate(PLOT_MOCKS):
        mock = mock_by_name[spec["files"][exp_id]]
        if not applicable(cfg, mock):
            continue
        samples, truth = samples_at_arrays(mock, exp_id, tname, in_radians, clamp)
        dx = (k - (n - 1) / 2) * MOCK_DX
        positions = X_POS + dx
        ax.axhline(truth, color=spec["colour"], lw=1.3, alpha=0.85,
                   linestyle=GT_LINESTYLE, zorder=1.3)
        parts = ax.violinplot(samples, positions=positions, widths=VIOLIN_WIDTH,
                              showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(spec["colour"])
            body.set_edgecolor(spec["colour"])
            body.set_alpha(0.32)
            body.set_linewidth(0.8)
        medians = [float(np.median(a)) for a in samples]
        ax.plot(positions, medians, "-", color=spec["colour"], lw=1.0,
                marker="o", ms=3.6, markerfacecolor=spec["colour"],
                markeredgecolor="white", markeredgewidth=0.5, zorder=2.5)


def _draw_array_table_panel(fig, ref_ax):
    ax = _make_panel(fig, ref_ax)
    ax.text(0.50, 0.90, "Instrument arrays",
            fontsize=10.0, fontweight="bold", va="top", ha="center",
            transform=ax.transAxes, color="0.12")
    col_x = [0.13, 0.39, 0.62, 0.87]
    headers = ["Array", r"$\sigma_{\Delta\mathrm{PA}}$", r"$\sigma_{r}$",
               r"$\sigma_{T}$"]
    for x, h in zip(col_x, headers):
        ax.text(x, 0.66, h, fontsize=9.4, fontweight="bold", ha="center",
                va="center", transform=ax.transAxes, color="0.12")
    ax.plot([0.05, 0.95], [0.55, 0.55], color="0.65", lw=0.6,
            transform=ax.transAxes, clip_on=False)
    cell_y, cell_dy = 0.43, 0.11
    for tag, kw in TIERS:
        cells = [tag, rf"${kw['sigma_DPA']:g}^{{\circ}}$",
                 rf"${kw['sigma_r']:g}\,M$", rf"${kw['sigma_T']:g}\,$min"]
        for x, cell in zip(col_x, cells):
            ax.text(x, cell_y, cell, fontsize=9.2, ha="center", va="center",
                    transform=ax.transAxes, color="0.20")
        cell_y -= cell_dy


# ── Figure 1: full 3×3 grid ───────────────────────────────────────────────────

def make_grid_figure(out_dir, mock_by_name):
    EXCLUDE = {(1, 0)}
    present = [
        [(r, c) not in EXCLUDE
         and _exp_target(EXP_REGISTRY[COLS[c][0]], ROWS[r][0]) is not None
         for c in range(len(COLS))]
        for r in range(len(ROWS))
    ]
    col_bottom = [max(r for r in range(len(ROWS)) if present[r][c])
                  for c in range(len(COLS))]
    row_left = [min(c for c in range(len(COLS)) if present[r][c])
                for r in range(len(ROWS))]

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(
            len(ROWS), len(COLS), figsize=(10.4, 7.6),
            sharex=True, sharey="row", squeeze=False,
            gridspec_kw=dict(wspace=0.0, hspace=0.0),
        )
        axes[0, 0].set_xticks(X_POS)
        axes[0, 0].set_xticklabels([tag for tag, _ in TIERS])
        axes[0, 0].set_xlim(X_POS[0] - 0.5, X_POS[-1] + 0.5)

        for r_idx, (tname, ylabel, in_radians, clamp) in enumerate(ROWS):
            for c_idx, (exp_id, exp_label) in enumerate(COLS):
                ax = axes[r_idx, c_idx]
                if not present[r_idx][c_idx]:
                    ax.axis("off")
                    continue
                draw_cell(ax, mock_by_name, exp_id, tname, in_radians, clamp)
                ax.grid(True, which="major", axis="y", color="0.88", lw=0.5,
                        zorder=0.3)
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax.xaxis.set_tick_params(which="minor", bottom=False, top=False)
                if r_idx == 0:
                    ax.set_title(exp_label, pad=6)
                ax.tick_params(axis="x", labelbottom=(r_idx == col_bottom[c_idx]),
                               labelsize=9.2)
                if c_idx == row_left[r_idx]:
                    ax.tick_params(axis="y", labelleft=True)
                    ax.set_ylabel(ylabel, labelpad=4)
                else:
                    ax.tick_params(axis="y", labelleft=False)

        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.99))
        _draw_mock_truth_panel(fig, axes[1, 0], axes[2, 0], mock_by_name)
        _draw_array_table_panel(fig, axes[2, 1])

        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"mocks_array_grid_3x3_violin.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("saved", out_dir / "mocks_array_grid_3x3_violin.{pdf,png}")


# ── Figure 2: θ_z only (Exp IV) ───────────────────────────────────────────────

def make_theta_figure(out_dir, mock_by_name):
    tname, ylabel, in_radians, clamp = next(r for r in ROWS if r[0] == "theta")
    exp_id, exp_label = next(c for c in COLS if c[0] == 4)

    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(5.4, 4.4))
        draw_cell(ax, mock_by_name, exp_id, tname, in_radians, clamp)

        ax.set_xticks(X_POS)
        ax.set_xticklabels([tag for tag, _ in TIERS])
        ax.set_xlim(X_POS[0] - 0.5, X_POS[-1] + 0.5)
        ax.set_xlabel("Instrument array")
        ax.set_ylabel(ylabel, labelpad=4)
        ax.set_title(rf"{exp_label}:  $\hat{{\theta}}_z$", pad=6)
        ax.grid(True, which="major", axis="y", color="0.88", lw=0.5, zorder=0.3)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_tick_params(which="minor", bottom=False, top=False)

        handles = [Patch(facecolor=s["colour"], edgecolor=s["colour"], alpha=0.32,
                         label=s["display"]) for s in PLOT_MOCKS]
        handles += [
            Line2D([0], [0], color="0.35", lw=1.2, marker="o", ms=4,
                   markerfacecolor="0.35", markeredgecolor="white",
                   label="MC median"),
            Line2D([0], [0], color="0.35", lw=1.4, linestyle=GT_LINESTYLE,
                   label="truth"),
        ]
        ax.legend(handles=handles, loc="best", fontsize=8.6, frameon=True,
                  framealpha=0.9, edgecolor="0.8", handlelength=1.7)

        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"mocks_array_theta_violin.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("saved", out_dir / "mocks_array_theta_violin.{pdf,png}")


if __name__ == "__main__":
    out = REPO_ROOT / "notebooks" / "figures"
    mocks = load_all_mock_observations()
    mock_by_name = {m["path"].name: m for m in mocks}
    make_grid_figure(out, mock_by_name)
    make_theta_figure(out, mock_by_name)
