"""Sibling of plot_exp5_neq45_noise10_sweeps.py.

Recreates the 3x1 sweep figure (sigma vs % orbit used; rows = spin, incl,
theta) but swaps the two-condition (noisy/clean) comparison for FOUR noise
conditions trained with the instrument error budgets:

    No noise, EHT, ngEHT, BHEX

Encoding:
    line style -> experiment  (Exp V solid, Exp III dashed)
    colour/marker -> noise condition

Exp III is equatorial, so it has no theta row (only spin + incl).

Output: notebooks/figures/exp5_neq45_instruments_sweep_3x1.{pdf,png}
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from pathlib import Path

REPO_ROOT    = Path('/scratch/ralbe/meniar_and_django/smbh_hotspots_repository')
METRICS      = REPO_ROOT / 'results' / 'metrics'
NOTEBOOK_DIR = REPO_ROOT / 'notebooks' / 'figures'

mpl.rcParams.update({
    "font.size": 14, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "figure.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42,
})

DEG = 180.0 / np.pi
FIG_W, FIG_H = 5.6, 3.0

EXP5_PREFIX = 'experiment_5_noneq_partial_neq45_'
EXP3_PREFIX = 'experiment_3_eq_partial_'

# label, dir-suffix, colour, marker  (ordered most-noise -> least -> none)
# Cohesive viridis sequential ramp for the three instruments (dark->light tracks
# decreasing noise), neutral gray for the no-noise reference.
CONDITIONS = [
    ('EHT',      'eht',      '#3b528b', 'o'),   # sigma_r=1.0, sigma_DPA=14, sigma_T=1
    ('ngEHT',    'ngeht',    '#21918c', '^'),   # sigma_r=0.65, sigma_DPA=9, sigma_T=1
    ('BHEX',     'bhex',     '#5ec962', 'D'),   # sigma_r=0.1, sigma_DPA=1.5, sigma_T=0.5
    ('No noise', 'no_noise', '#9e9e9e', 's'),
]

KW = dict(capsize=2.5, lw=1.6, markersize=4.5, elinewidth=0.9, alpha=0.9)
# Shaded-band variant: line + markers, std drawn as a translucent fill_between.
KW_BAND = dict(lw=1.6, markersize=4.5, alpha=0.95)
BAND_ALPHA = 0.16


def _save(fig, name):
    p = NOTEBOOK_DIR / name
    fig.savefig(p.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(p.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f'Saved {p}.pdf/.png')


def load_sweep(exp_name, target):
    df = pd.read_csv(METRICS / exp_name / f'{target}_sweep.csv')
    if 'error_std_mean' in df.columns:
        df = df.rename(columns={'error_std_mean': 'mean_col', 'error_std_std': 'std_col'})
    else:
        df = df.rename(columns={'sigma_mean': 'mean_col', 'sigma_std': 'std_col'})
    return df


def draw_curve(ax, exp_prefix, suffix, colour, marker, target, scale, ls, band=False):
    df = load_sweep(exp_prefix + suffix, target)
    x = df['percent']
    y = df['mean_col'] * scale
    e = df['std_col'] * scale
    if band:
        # Translucent +/-sigma band instead of error caps.
        ax.fill_between(x, y - e, y + e, color=colour, alpha=BAND_ALPHA, lw=0)
        ax.plot(x, y, color=colour, ls=ls, marker=marker, **KW_BAND)
    else:
        ax.errorbar(x, y, yerr=e, color=colour, ls=ls, marker=marker, **KW)


def draw_panel(ax, target, scale, include_exp3, band=False):
    for _label, suffix, colour, marker in CONDITIONS:
        if include_exp3:
            draw_curve(ax, EXP3_PREFIX, suffix, colour, marker, target, scale, '--', band=band)
        draw_curve(ax, EXP5_PREFIX, suffix, colour, marker, target, scale, '-', band=band)


def build_figure(band=False):
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, FIG_H * 3), sharex=True)

    draw_panel(axes[0], 'spin', 1.0, include_exp3=True, band=band)
    axes[0].set_ylabel(r'$\hat{\sigma}_\alpha$')

    draw_panel(axes[1], 'incl', DEG, include_exp3=True, band=band)
    axes[1].set_ylabel(r'$\hat{\sigma}_i$ (°)')

    draw_panel(axes[2], 'theta', DEG, include_exp3=False, band=band)   # equatorial Exp III has no theta
    axes[2].set_ylabel(r'$\hat{\sigma}_\theta$ (°)')
    axes[2].set_xlabel('% of orbit used')

    style_handles = [
        Line2D([0], [0], color='0.2', ls='--', lw=2.0, label='Exp III'),
        Line2D([0], [0], color='0.2', ls='-',  lw=2.0, label='Exp V'),
    ]
    cond_handles = [
        Line2D([0], [0], color=c, marker=m, ls='', markersize=6, label=lab)
        for lab, _s, c, m in CONDITIONS
    ]
    leg1 = axes[0].legend(handles=style_handles, loc='upper right', fontsize=9,
                          framealpha=0.9, borderpad=0.5, title='Experiment')
    axes[0].add_artist(leg1)
    axes[0].legend(handles=cond_handles, loc='lower left', fontsize=9,
                   framealpha=0.9, borderpad=0.5, title='Noise')

    plt.tight_layout()
    return fig


fig = build_figure(band=False)
_save(fig, 'exp5_neq45_instruments_sweep_3x1')
plt.close(fig)

fig_band = build_figure(band=True)
_save(fig_band, 'exp5_neq45_instruments_sweep_3x1_band')
plt.close(fig_band)

print('Done.')
