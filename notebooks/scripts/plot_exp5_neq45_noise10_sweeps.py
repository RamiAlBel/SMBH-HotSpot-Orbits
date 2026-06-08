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
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})

COLOR_NOISY = '#5B84B1'
COLOR_CLEAN = '#FC766AFF'
DEG = 180.0 / np.pi

FIG_W, FIG_H = 5, 3.0

# Exp V noisy now points at the sigma_DPA=10 deg run finished 2026-05-15.
EXP5_NOISY = 'experiment_5_noneq_partial_neq45_noise10'
EXP5_CLEAN = 'experiment_5_noneq_partial_neq45_no_noise'
EXP3_NOISY = 'experiment_3_eq_partial'
EXP3_CLEAN = 'experiment_3_eq_partial_no_noise'


def _save(fig, name):
    p = NOTEBOOK_DIR / name
    fig.savefig(p.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(p.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f'Saved {p}.pdf/.png')


def load_sweep(exp_name, target):
    p = METRICS / exp_name / f'{target}_sweep.csv'
    df = pd.read_csv(p)
    if 'error_std_mean' in df.columns:
        df = df.rename(columns={'error_std_mean': 'mean_col', 'error_std_std': 'std_col'})
    else:
        df = df.rename(columns={'sigma_mean': 'mean_col', 'sigma_std': 'std_col'})
    return df


KW = dict(capsize=3, lw=1.8, markersize=5)


def draw_4curve(ax, target, scale_exp3=1.0, scale_exp5=1.0):
    exp3       = load_sweep(EXP3_NOISY, target)
    exp3_clean = load_sweep(EXP3_CLEAN, target)
    exp5       = load_sweep(EXP5_NOISY, target)
    exp5_clean = load_sweep(EXP5_CLEAN, target)

    ax.errorbar(exp3_clean['percent'], exp3_clean['mean_col'] * scale_exp3,
                yerr=exp3_clean['std_col'] * scale_exp3,
                color=COLOR_CLEAN, ls='--', marker='s', **KW)
    ax.errorbar(exp3['percent'], exp3['mean_col'] * scale_exp3,
                yerr=exp3['std_col'] * scale_exp3,
                color=COLOR_NOISY, ls='--', marker='o', **KW)
    ax.errorbar(exp5_clean['percent'], exp5_clean['mean_col'] * scale_exp5,
                yerr=exp5_clean['std_col'] * scale_exp5,
                color=COLOR_CLEAN, ls='-', marker='s', **KW)
    ax.errorbar(exp5['percent'], exp5['mean_col'] * scale_exp5,
                yerr=exp5['std_col'] * scale_exp5,
                color=COLOR_NOISY, ls='-', marker='o', **KW)


def draw_2curve(ax, target, scale=1.0):
    exp5       = load_sweep(EXP5_NOISY, target)
    exp5_clean = load_sweep(EXP5_CLEAN, target)

    ax.errorbar(exp5_clean['percent'], exp5_clean['mean_col'] * scale,
                yerr=exp5_clean['std_col'] * scale,
                color=COLOR_CLEAN, ls='-', marker='s', **KW)
    ax.errorbar(exp5['percent'], exp5['mean_col'] * scale,
                yerr=exp5['std_col'] * scale,
                color=COLOR_NOISY, ls='-', marker='o', **KW)


fig, axes = plt.subplots(3, 1, figsize=(FIG_W, FIG_H * 3), sharex=True)

draw_4curve(axes[0], 'spin', scale_exp3=1.0, scale_exp5=1.0)
axes[0].set_ylabel(r'$\hat{\sigma}_\alpha$')

draw_4curve(axes[1], 'incl', scale_exp3=DEG, scale_exp5=DEG)
axes[1].set_ylabel(r'$\hat{\sigma}_i$ (°)')

draw_2curve(axes[2], 'theta', scale=DEG)
axes[2].set_ylabel(r'$\hat{\sigma}_\theta$ (°)')
axes[2].set_xlabel('% of orbit used')

legend_handles = [
    Line2D([0], [0], color='black',       ls='-',  lw=2.0, label='Exp V'),
    Line2D([0], [0], color='black',       ls='--', lw=2.0, label='Exp III'),
    Line2D([0], [0], color=COLOR_CLEAN,   ls='-',  lw=2.0, label='No noise'),
    Line2D([0], [0], color=COLOR_NOISY,   ls='-',  lw=2.0, label='With noise'),
]
axes[0].legend(handles=legend_handles, loc='upper right', fontsize=9,
               framealpha=0.9, borderpad=0.5)

plt.tight_layout()
_save(fig, 'exp5_neq45_noise10_sweep_3x1')
plt.close(fig)

print('Done.')
