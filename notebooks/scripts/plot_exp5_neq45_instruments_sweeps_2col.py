"""Two-column variant of plot_exp5_neq45_instruments_sweeps.py.

Same sweep data (sigma vs % orbit used; rows = spin, incl, theta) but instead of
overlaying the two experiments in each panel, they are split across columns:

    left  column -> "Equatorial orbits"      (Exp III, equatorial: spin + incl only)
    right column -> "Non-equatorial orbits"  (Exp V, non-equatorial: spin + incl + theta)

The experiment is no longer encoded by line style (every curve is solid); only
colour/marker encodes the noise condition. The equatorial column has no theta
row, so its bottom panel is left empty.

Output: notebooks/figures/exp5_neq45_instruments_sweep_3x2.{pdf,png}
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
FIG_W, FIG_H = 6.19, 3.0   # 7.28 * 0.85 ≈ 15 % narrower

EXP5_PREFIX = 'experiment_5_noneq_partial_neq45_'
EXP3_PREFIX = 'experiment_3_eq_partial_'

# label, dir-suffix, colour, marker  (ordered most-noise -> least -> none)
# Cohesive viridis sequential ramp for the three instruments (dark->light tracks
# decreasing noise), neutral gray for the no-noise reference. Labels carry the
# (approximate) instrument era.
CONDITIONS = [
    ('EHT (2019)',          'eht',      '#3b528b', 'o'),   # sigma_r=1.0, sigma_DPA=14, sigma_T=1
    ('ngEHT (early 2030s)', 'ngeht',    '#21918c', '^'),   # sigma_r=0.65, sigma_DPA=9, sigma_T=1
    ('BHEX (2031)',            'bhex',     '#5ec962', 'D'),   # sigma_r=0.1, sigma_DPA=1.5, sigma_T=0.5
    ('No observational error', 'no_noise', '#9e9e9e', 's'),
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


def draw_curve(ax, exp_prefix, suffix, colour, marker, target, scale, band=False):
    df = load_sweep(exp_prefix + suffix, target)
    x = df['percent']
    y = df['mean_col'] * scale
    e = df['std_col'] * scale
    if band:
        ax.fill_between(x, y - e, y + e, color=colour, alpha=BAND_ALPHA, lw=0)
        ax.plot(x, y, color=colour, ls='-', marker=marker, **KW_BAND)
    else:
        ax.errorbar(x, y, yerr=e, color=colour, ls='-', marker=marker, **KW)


def draw_panel(ax, exp_prefix, target, scale, band=False):
    for _label, suffix, colour, marker in CONDITIONS:
        draw_curve(ax, exp_prefix, suffix, colour, marker, target, scale, band=band)


def build_figure(band=False):
    fig, axes = plt.subplots(3, 2, figsize=(FIG_W * 2, FIG_H * 3),
                             sharex=True, sharey='row')

    # Left column: equatorial (Exp III) -> spin + incl only.
    draw_panel(axes[0, 0], EXP3_PREFIX, 'spin', 1.0, band=band)
    draw_panel(axes[1, 0], EXP3_PREFIX, 'incl', DEG, band=band)
    axes[2, 0].axis('off')   # equatorial has no theta
    axes[0, 0].set_title('Equatorial hot-spot orbits', fontsize=16, pad=10)

    # Right column: non-equatorial (Exp V) -> spin + incl + theta.
    draw_panel(axes[0, 1], EXP5_PREFIX, 'spin', 1.0, band=band)
    draw_panel(axes[1, 1], EXP5_PREFIX, 'incl', DEG, band=band)
    draw_panel(axes[2, 1], EXP5_PREFIX, 'theta', DEG, band=band)
    axes[0, 1].set_title('Non-equatorial hot-spot orbits', fontsize=16, pad=10)

    axes[0, 0].set_ylabel(r'$\hat{\sigma}_\alpha$')
    axes[1, 0].set_ylabel(r'$\hat{\sigma}_i$ (°)')
    axes[2, 1].set_ylabel(r'$\hat{\sigma}_\theta$ (°)')
    axes[2, 1].yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2))
    axes[2, 1].tick_params(labelleft=True)

    for ax in axes.flat:
        if ax.axison:
            ax.set_ylim(bottom=0)

    axes[1, 0].set_xlabel('% of orbit used')   # bottom of left column (theta off)
    axes[1, 0].tick_params(labelbottom=True)
    axes[2, 1].set_xlabel('% of orbit used')

    cond_handles = [
        Line2D([0], [0], color=c, marker=m, ls='-', markersize=6, label=lab)
        for lab, _s, c, m in CONDITIONS
    ]
    # Legend lives in the empty bottom-left panel (equatorial has no theta row).
    axes[2, 0].legend(handles=cond_handles, loc='center', fontsize=12,
                      framealpha=0.9, borderpad=0.8, title='Instrument',
                      title_fontsize=13)

    plt.tight_layout()
    return fig


fig = build_figure(band=False)
_save(fig, 'exp5_neq45_instruments_sweep_3x2')
plt.close(fig)

fig_band = build_figure(band=True)
_save(fig_band, 'exp5_neq45_instruments_sweep_3x2_band')
plt.close(fig_band)

print('Done.')
