import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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


def _save(fig, name):
    p = NOTEBOOK_DIR / name
    fig.savefig(p.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(p.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f'Saved {p}.pdf/.png')


def load_sweep(exp_name, target):
    p = METRICS / exp_name / f'{target}_sweep.csv'
    df = pd.read_csv(p)
    # exp3 (eq partial) uses sigma_mean/sigma_std; exp5 neq45 uses error_std_mean/error_std_std
    if 'error_std_mean' in df.columns:
        df = df.rename(columns={'error_std_mean': 'mean_col', 'error_std_std': 'std_col'})
    else:
        df = df.rename(columns={'sigma_mean': 'mean_col', 'sigma_std': 'std_col'})
    return df


# ── Combined 4-curve sweep plot (Exp III + Exp V) ────────────────────────────

def plot_combined_4curve(target, ylabel, scale_exp3=1.0, scale_exp5=1.0,
                         show_legend=False, figsize=(FIG_W, FIG_H)):
    exp3        = load_sweep('experiment_3_eq_partial',                  target)
    exp3_clean  = load_sweep('experiment_3_eq_partial_no_noise',         target)
    exp5        = load_sweep('experiment_5_noneq_partial_neq45',         target)
    exp5_clean  = load_sweep('experiment_5_noneq_partial_neq45_no_noise', target)

    fig, ax = plt.subplots(figsize=figsize)

    kw = dict(capsize=3, lw=1.8, markersize=5)

    ax.errorbar(exp3_clean['percent'], exp3_clean['mean_col'] * scale_exp3,
                yerr=exp3_clean['std_col'] * scale_exp3,
                color=COLOR_CLEAN, ls='--', marker='s',
                label='Exp III, No Noise', **kw)

    ax.errorbar(exp3['percent'], exp3['mean_col'] * scale_exp3,
                yerr=exp3['std_col'] * scale_exp3,
                color=COLOR_NOISY, ls='--', marker='o',
                label='Exp III, With Noise', **kw)

    ax.errorbar(exp5_clean['percent'], exp5_clean['mean_col'] * scale_exp5,
                yerr=exp5_clean['std_col'] * scale_exp5,
                color=COLOR_CLEAN, ls='-', marker='s',
                label='Exp V, No Noise', **kw)

    ax.errorbar(exp5['percent'], exp5['mean_col'] * scale_exp5,
                yerr=exp5['std_col'] * scale_exp5,
                color=COLOR_NOISY, ls='-', marker='o',
                label='Exp V, With Noise', **kw)

    ax.set_xlabel('% of orbit used')
    ax.set_ylabel(ylabel)

    if show_legend:
        handles = [
            Line2D([0], [0], color=COLOR_CLEAN, ls='--', lw=2.0, label='Exp III, No Noise'),
            Line2D([0], [0], color=COLOR_NOISY, ls='--', lw=2.0, label='Exp III, With Noise'),
            Line2D([0], [0], color=COLOR_CLEAN, ls='-',  lw=2.0, label='Exp V, No Noise'),
            Line2D([0], [0], color=COLOR_NOISY, ls='-',  lw=2.0, label='Exp V, With Noise'),
        ]
        ax.legend(handles=handles, loc='upper right', fontsize=10,
                  framealpha=0.9, borderpad=0.5)

    plt.tight_layout()
    return fig


# ── Single 2-curve sweep plot (Exp V / exp5_neq45 only) ──────────────────────

def plot_combined_2curve(target, ylabel, scale=1.0, figsize=(FIG_W, FIG_H)):
    exp5       = load_sweep('experiment_5_noneq_partial_neq45',          target)
    exp5_clean = load_sweep('experiment_5_noneq_partial_neq45_no_noise', target)

    fig, ax = plt.subplots(figsize=figsize)

    kw = dict(capsize=3, lw=1.8, markersize=5)

    ax.errorbar(exp5_clean['percent'], exp5_clean['mean_col'] * scale,
                yerr=exp5_clean['std_col'] * scale,
                color=COLOR_CLEAN, ls='-', marker='s',
                label='No Noise', **kw)

    ax.errorbar(exp5['percent'], exp5['mean_col'] * scale,
                yerr=exp5['std_col'] * scale,
                color=COLOR_NOISY, ls='-', marker='o',
                label='With Noise', **kw)

    ax.set_xlabel('% of orbit used')
    ax.set_ylabel(ylabel)

    pass  # no legend on theta

    plt.tight_layout()
    return fig


# ── Generate plots ────────────────────────────────────────────────────────────

# Spin: 4-curve (Exp III + Exp V), no unit conversion needed
fig = plot_combined_4curve('spin',
                           ylabel=r'$\hat{\sigma}_\alpha$',
                           scale_exp3=1.0, scale_exp5=1.0,
                           show_legend=False)
_save(fig, 'exp5_neq45_spin_sweep')
plt.close(fig)

# Inclination: 4-curve (Exp III + Exp V), both in radians → degrees
fig = plot_combined_4curve('incl',
                           ylabel=r'$\hat{\sigma}_i$ (°)',
                           scale_exp3=DEG, scale_exp5=DEG,
                           show_legend=True)
_save(fig, 'exp5_neq45_incl_sweep')
plt.close(fig)

# Theta: 2-curve (Exp V only, neq45), radians → degrees
fig = plot_combined_2curve('theta',
                           ylabel=r'$\hat{\sigma}_\theta$ (°)',
                           scale=DEG)
_save(fig, 'exp5_neq45_theta_sweep')
plt.close(fig)

print('Done.')
