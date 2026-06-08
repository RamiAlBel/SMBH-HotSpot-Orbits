"""Polished publication-style figure: MC inference error vs. instrument tier.

Produces ``notebooks/figures/mocks_tier_grid_3x3.{pdf,png}`` showing two mock
observations (relabelled "Mock 1" / "Mock 2") evaluated across four instrument
tiers (EHT / ngEHT / Y26 / BHEX) for three regression targets and three
experiment configurations.

Run from any cwd:
    python notebooks/scripts/plot_mocks_tier_grid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import FancyBboxPatch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from uncertainty_analysis import (   # noqa: E402
    EXP_REGISTRY,
    REPO_ROOT,
    DEVICE,
    DEG,
    N_MC,
    load_all_mock_observations,
    load_model_and_scalers,
    sigma_obs_vector,
    truth_for_target,
    applicable_to,
    feature_vector,
)

# ── Inputs ──────────────────────────────────────────────────────────────────

TIER_SETTINGS = {
    "EHT":   dict(sigma_r=1.0,  sigma_T=1.0, sigma_DPA=14.0),
    "ngEHT": dict(sigma_r=0.65, sigma_T=1.0, sigma_DPA=9.0),
    "Y26":   dict(sigma_r=0.5,  sigma_T=3.0, sigma_DPA=5.0),
    "BHEX":  dict(sigma_r=0.1,  sigma_T=0.5, sigma_DPA=1.5),
}
TIER_ORDER = ["EHT", "ngEHT", "BHEX"]
TIERS = [(t, TIER_SETTINGS[t]) for t in TIER_ORDER]
X_POS = np.arange(len(TIERS))

# Each scenario routes one filename per experiment.
PLOT_MOCKS = [
    dict(
        display="Mock 1",
        colour="#0072B2",
        files={
            1: "lc_r60_K60_a80_i0_th90_0.dat",
            2: "lc_r60_K60_a80_i300_th90.dat",
            4: "lc_r60_K60_a80_i300_th80.dat",
        },
    ),
    dict(
        display="Mock 2",
        colour="#D55E00",
        files={
            1: "lc_r45_K64_a-55_i0_th90_0.dat",
            2: "lc_r45_K64_a-55_i100_th90.dat",
            4: "lc_r45_K64_a-55_i100_th120.dat",
        },
    ),
]
TRUTH_FROM_EXP = 4

ROWS = [
    ("spin",  r"$\Delta\hat{a}_{\ast}$",                   False, (-1.0, 1.0)),
    ("incl",  r"$\Delta\hat{i}\ \mathrm{[deg]}$",          True,  None),
    ("theta", r"$\Delta\hat{\theta}_{z}\ \mathrm{[deg]}$", True,  None),
]
COLS = [
    (1, "Exp I"),
    (2, "Exp II"),
    (4, "Exp IV"),
]

# ── Helpers ─────────────────────────────────────────────────────────────────

_MODEL_CACHE: dict = {}


def cached_model(exp_id: int, target_name: str):
    key = (exp_id, target_name)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_model_and_scalers(exp_id, target_name)
    return _MODEL_CACHE[key]


def _exp_target(cfg, name):
    return next((t for t in cfg["targets"] if t["name"] == name), None)


def applicable(cfg, mock):
    return applicable_to(cfg, mock)[0]


def feat_for(cfg, mock):
    return feature_vector(cfg, mock)


def mc_predict_with_interval(model, scalers, x_orig, sigma_obs_orig,
                             n_mc=N_MC, in_radians=False, clamp=None,
                             rng_seed=None):
    rng = np.random.default_rng(rng_seed)
    x_noisy = (x_orig[None]
               + rng.standard_normal((n_mc, len(x_orig))) * sigma_obs_orig[None])
    x_norm  = (x_noisy - scalers["X_mean"]) / scalers["X_scale"]
    with torch.no_grad():
        y_norm = model(torch.tensor(x_norm, dtype=torch.float32, device=DEVICE))
    y = y_norm.cpu().numpy().squeeze() * scalers["y_scale"] + scalers["y_mean"]
    if clamp is not None:
        y = np.clip(y, clamp[0], clamp[1])
    factor = DEG if in_radians else 1.0
    pred   = float(np.mean(y)) * factor
    lo, hi = np.percentile(y, [2.5, 97.5]) * factor
    return pred, float(lo), float(hi)


def mc_error_at_tiers(mock, exp_id, tname, in_radians, clamp):
    cfg            = EXP_REGISTRY[exp_id]
    model, scalers = cached_model(exp_id, tname)
    x              = feat_for(cfg, mock)
    t              = truth_for_target(mock["truth"], tname)
    em, el, eh = [], [], []
    for _tag, kw in TIERS:
        sigma_obs = sigma_obs_vector(cfg["num_input_features"], **kw)
        m, lo, hi = mc_predict_with_interval(
            model, scalers, x, sigma_obs,
            n_mc=N_MC, in_radians=in_radians, clamp=clamp, rng_seed=42,
        )
        em.append(m - t); el.append(lo - t); eh.append(hi - t)
    return np.array(em), np.array(el), np.array(eh)


# ── Figure ──────────────────────────────────────────────────────────────────

def make_figure(out_dir: Path):
    mocks = load_all_mock_observations()
    mock_by_name = {m["path"].name: m for m in mocks}

    # (row, col) cells to skip even if the underlying target exists.
    EXCLUDE = {(1, 0)}   # Exp I inclination prediction is not informative.
    present = [
        [(r, c) not in EXCLUDE
         and _exp_target(EXP_REGISTRY[COLS[c][0]], ROWS[r][0]) is not None
         for c in range(len(COLS))]
        for r in range(len(ROWS))
    ]
    col_bottom = [max(r for r in range(len(ROWS)) if present[r][c])
                  for c in range(len(COLS))]
    row_left   = [min(c for c in range(len(COLS)) if present[r][c])
                  for r in range(len(ROWS))]

    pub_rc = {
        "font.family":       "DejaVu Sans",
        "mathtext.fontset":  "dejavusans",
        "font.size":         10,
        "axes.labelsize":    11,
        "axes.titlesize":    11.5,
        "axes.linewidth":    0.9,
        "axes.edgecolor":    "0.15",
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.top":         True,
        "ytick.right":       True,
        "xtick.major.size":  4.0,
        "ytick.major.size":  4.0,
        "xtick.minor.size":  2.2,
        "ytick.minor.size":  2.2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.color":       "0.15",
        "ytick.color":       "0.15",
        "legend.frameon":    False,
        "savefig.dpi":       300,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    }

    with plt.rc_context(pub_rc):
        fig, axes = plt.subplots(
            len(ROWS), len(COLS),
            figsize=(10.4, 7.6),
            sharex=True, sharey="row", squeeze=False,
            gridspec_kw=dict(wspace=0.0, hspace=0.0),
        )

        axes[0, 0].set_xticks(X_POS)
        axes[0, 0].set_xticklabels([tag for tag, _ in TIERS])
        axes[0, 0].set_xlim(X_POS[0] - 0.5, X_POS[-1] + 0.5)

        legend_handles_seen: dict = {}

        for r_idx, (tname, ylabel, in_radians, clamp) in enumerate(ROWS):
            for c_idx, (exp_id, exp_label) in enumerate(COLS):
                ax = axes[r_idx, c_idx]
                if not present[r_idx][c_idx]:
                    ax.axis("off")
                    continue
                cfg = EXP_REGISTRY[exp_id]

                for k, spec in enumerate(PLOT_MOCKS):
                    mock = mock_by_name[spec["files"][exp_id]]
                    if not applicable(cfg, mock):
                        continue
                    em, el, eh = mc_error_at_tiers(
                        mock, exp_id, tname, in_radians, clamp,
                    )
                    dx = (k - (len(PLOT_MOCKS) - 1) / 2) * 0.12
                    line = ax.errorbar(
                        X_POS + dx, em, yerr=[em - el, eh - em],
                        fmt="o-", ms=4.6, lw=1.25, capsize=2.8, elinewidth=1.0,
                        color=spec["colour"], markerfacecolor=spec["colour"],
                        markeredgecolor="white", markeredgewidth=0.6,
                        label=spec["display"],
                    )
                    if spec["display"] not in legend_handles_seen:
                        legend_handles_seen[spec["display"]] = line

                ax.axhline(0, color="0.10", lw=1.2, alpha=0.9, zorder=1.6)
                ax.grid(True, which="major", axis="y", color="0.88", lw=0.5,
                        zorder=0.3)
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax.xaxis.set_tick_params(which="minor", bottom=False, top=False)

                if r_idx == 0:
                    ax.set_title(exp_label, pad=6)

                if r_idx == col_bottom[c_idx]:
                    ax.tick_params(axis="x", labelbottom=True, labelsize=9.2)
                else:
                    ax.tick_params(axis="x", labelbottom=False)

                if c_idx == row_left[r_idx]:
                    ax.tick_params(axis="y", labelleft=True)
                    ax.set_ylabel(ylabel, labelpad=4)
                else:
                    ax.tick_params(axis="y", labelleft=False)

        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.99))

        # Combined legend + truth-table panel spans (1, 0) and (2, 0).
        _draw_mock_truth_panel(fig, axes[1, 0], axes[2, 0], mock_by_name)
        _draw_tier_table_panel(fig, axes[2, 1])

        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"mocks_tier_grid_3x3.{ext}",
                        dpi=300, bbox_inches="tight")
        print("saved", out_dir / "mocks_tier_grid_3x3.{pdf,png}")


def _make_panel(fig, ref_ax, pad_left=0.010, pad_right=0.080,
                pad_top=0.045, pad_bottom=0.018):
    p = ref_ax.get_position()
    ax = fig.add_axes([
        p.x0 + pad_left,
        p.y0 + pad_bottom,
        (p.x1 - p.x0) - pad_left - pad_right,
        (p.y1 - p.y0) - pad_top - pad_bottom,
    ])
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch(
        (0.005, 0.005), 0.99, 0.99,
        boxstyle="round,pad=0.012,rounding_size=0.040",
        transform=ax.transAxes,
        linewidth=0.7, edgecolor="0.55", facecolor="0.965",
    ))
    return ax


def _draw_mock_truth_panel(fig, ref_top, ref_bottom, mock_by_name):
    """Side-by-side comparison: each mock is a column, parameters are rows.

    Layout:
        ┌──────────────────────────────────┐
        │            Mock 1     Mock 2     │
        │            ●──        ●──        │
        │       ─────────────────────      │
        │   r    6.0 M     4.5 M           │
        │   P    55 min    30 min          │
        │   a*   +0.80     −0.55           │
        │   i    30°       10°             │
        │   θ    +10°      −30°            │
        └──────────────────────────────────┘
    """
    p_top = ref_top.get_position()
    p_bot = ref_bottom.get_position()
    pad_left, pad_right = 0.010, 0.080
    pad_top, pad_bottom = 0.045, 0.018
    ax = fig.add_axes([
        p_top.x0 + pad_left,
        p_bot.y0 + pad_bottom,
        (p_top.x1 - p_top.x0) - pad_left - pad_right,
        (p_top.y1 - p_bot.y0) - pad_top - pad_bottom,
    ])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch(
        (0.005, 0.005), 0.99, 0.99,
        boxstyle="round,pad=0.012,rounding_size=0.022",
        transform=ax.transAxes,
        linewidth=0.7, edgecolor="0.55", facecolor="0.97",
    ))

    x_label = 0.22
    x_mocks = [0.52, 0.82]
    glyph_half = 0.07

    # ── Header row: Mock names ───────────────────────────────────────────
    for x, spec in zip(x_mocks, PLOT_MOCKS):
        ax.text(x, 0.92, spec["display"], fontsize=11.6, fontweight="bold",
                color=spec["colour"], va="center", ha="center",
                transform=ax.transAxes)

    # ── Marker swatches under each mock name ─────────────────────────────
    for x, spec in zip(x_mocks, PLOT_MOCKS):
        ax.plot([x - glyph_half, x + glyph_half], [0.83, 0.83],
                color=spec["colour"], lw=1.6, marker="o", markersize=7,
                markerfacecolor=spec["colour"],
                markeredgecolor="white", markeredgewidth=0.7,
                transform=ax.transAxes, clip_on=False)

    # ── Divider rule below the swatches ──────────────────────────────────
    ax.plot([0.05, 0.95], [0.74, 0.74], color="0.55", lw=0.7,
            transform=ax.transAxes, clip_on=False)

    # ── Parameter rows ───────────────────────────────────────────────────
    truths = [mock_by_name[spec["files"][TRUTH_FROM_EXP]]["truth"]
              for spec in PLOT_MOCKS]
    rows = [
        (r"$r$",        "r",      lambda v: rf"${v:.1f}\,M$"),
        (r"$P$",        "Period", lambda v: rf"${v:.0f}\,$min"),
        (r"$a_{\ast}$", "a",      lambda v: rf"${v:+.2f}$"),
        (r"$i$",        "i",      lambda v: rf"${v:.0f}^{{\circ}}$"),
        (r"$\theta$",   "theta",  lambda v: rf"${v:+.0f}^{{\circ}}$"),
    ]
    y_start, dy = 0.63, 0.115
    for k, (label, key, fmt) in enumerate(rows):
        y = y_start - k * dy
        ax.text(x_label, y, label, fontsize=11.0, color="0.25",
                va="center", ha="right", transform=ax.transAxes,
                fontstyle="italic")
        for x, tr, spec in zip(x_mocks, truths, PLOT_MOCKS):
            ax.text(x, y, fmt(tr[key]), fontsize=10.6, color=spec["colour"],
                    va="center", ha="center", transform=ax.transAxes)


def _draw_tier_table_panel(fig, ref_ax):
    ax = _make_panel(fig, ref_ax)
    ax.text(0.50, 0.90, "Instrument tiers",
            fontsize=10.0, fontweight="bold", va="top", ha="center",
            transform=ax.transAxes, color="0.12")

    col_x = [0.13, 0.39, 0.62, 0.87]
    headers = [
        "Tier",
        r"$\sigma_{\Delta\mathrm{PA}}$",
        r"$\sigma_{r}$",
        r"$\sigma_{T}$",
    ]
    for x, h in zip(col_x, headers):
        ax.text(x, 0.66, h, fontsize=9.4, fontweight="bold",
                ha="center", va="center", transform=ax.transAxes,
                color="0.12")
    ax.plot([0.05, 0.95], [0.55, 0.55], color="0.65", lw=0.6,
            transform=ax.transAxes, clip_on=False)

    cell_y, cell_dy = 0.43, 0.11
    for tag, kw in TIERS:
        cells = [
            tag,
            rf"${kw['sigma_DPA']:g}^{{\circ}}$",
            rf"${kw['sigma_r']:g}\,M$",
            rf"${kw['sigma_T']:g}\,$min",
        ]
        for x, cell in zip(col_x, cells):
            ax.text(x, cell_y, cell, fontsize=9.2, ha="center", va="center",
                    transform=ax.transAxes, color="0.20")
        cell_y -= cell_dy


if __name__ == "__main__":
    out = REPO_ROOT / "notebooks" / "figures"
    make_figure(out)
