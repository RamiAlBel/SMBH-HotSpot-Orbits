"""Generate clean (no-noise) versions of experiment 4 (neq45 dataset) plots,
matching the existing exp4_neq45_*_noisy figures in notebooks/figures/."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error

REPO_ROOT    = Path('/scratch/ralbe/meniar_and_django/smbh_hotspots_repository')
METRICS      = REPO_ROOT / 'results' / 'metrics'
NOTEBOOK_DIR = REPO_ROOT / 'notebooks' / 'figures'

mpl.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})

COLOR_CLEAN = '#FC766AFF'
DEG = 180.0 / np.pi

EXP_CLEAN = 'experiment_4_noneq_full_neq45_no_noise'


def _save(fig, name):
    p = NOTEBOOK_DIR / name
    fig.savefig(p.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(p.with_suffix('.png'), dpi=300, bbox_inches='tight')


def _gaussian_fit_and_plot(ax, err, color, bins=60):
    n = len(err)
    counts, edges, _ = ax.hist(err, bins=bins, color=color, alpha=0.75, edgecolor='none')
    mu, sigma = stats.norm.fit(err)
    se_mu    = sigma / np.sqrt(n)
    se_sigma = sigma / np.sqrt(2 * n)
    bw = edges[1] - edges[0]
    xr = np.linspace(edges[0], edges[-1], 400)
    ax.plot(xr, stats.norm.pdf(xr, mu, sigma) * n * bw, color='black', lw=2)
    ax.axvline(0,  color='gray', lw=1.4, ls='--', alpha=0.8)
    ax.axvline(mu, color='red',  lw=1.8, ls='-')
    ax.errorbar(mu, counts.max() * 0.04, xerr=se_mu,
                fmt='none', color='red', capsize=5, lw=2, zorder=5)
    eqn = (f'$\\mu = {mu:+.3f} \\pm {se_mu:.3f}$\n'
           f'$\\sigma = {sigma:.3f} \\pm {se_sigma:.3f}$')
    ax.text(0.97, 0.96, eqn, transform=ax.transAxes, ha='right', va='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.35', fc='white', alpha=0.92, ec='lightgray'))
    return mu, sigma, se_mu, se_sigma


def pred_vs_true_single(y_true, y_pred, symbol, error_label, color,
                        unit='', force_positive_lim=False, figsize=(9, 4)):
    unit_str = f' ({unit})' if unit else ''
    xlabel = f'${symbol}_{{\\mathrm{{GT}}}}${unit_str}'
    ylabel = f'$\\hat{{{symbol}}}${unit_str}'
    err = y_pred - y_true
    if force_positive_lim:
        all_v = np.concatenate([y_true, y_pred])
        lo, hi = np.nanmin(all_v), np.nanmax(all_v)
        buf = (hi - lo) * 0.05
        lim = [lo - buf, hi + buf]
    else:
        lim_abs = np.nanmax(np.abs(np.concatenate([y_true, y_pred])))
        buf = lim_abs * 0.05
        lim = [-lim_abs - buf, lim_abs + buf]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    ax = axes[0]
    ax.scatter(y_true, y_pred, color=color, s=6, alpha=0.4, edgecolor='none', rasterized=True)
    ax.plot(lim, lim, 'k--', lw=1.2, alpha=0.7)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.text(0.05, 0.95, f'$R^2={r2:.3f}$\nMAE={mae:.3f}',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9, ec='lightgray'))

    ax2 = axes[1]
    _gaussian_fit_and_plot(ax2, err, color)
    ax2.set_xlabel(error_label); ax2.set_ylabel('Count')
    plt.tight_layout()
    return fig


def load_test_details(exp_name, incl_scale=1.0, theta_scale=1.0):
    dfs = []
    for seed in [42, 43, 44]:
        p = METRICS / exp_name / f'test_details_seed{seed}.csv'
        if p.exists():
            df = pd.read_csv(p); df['seed'] = seed; dfs.append(df)
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    for col in ['incl_true', 'incl_pred', 'incl_error']:
        if col in df.columns: df[col] = df[col] * incl_scale
    for col in ['theta_true', 'theta_pred', 'theta_error']:
        if col in df.columns: df[col] = df[col] * theta_scale
    return df


def corner_plot_errors_matrix(df, targets, labels, panel_size=1.7, hist_color=COLOR_CLEAN):
    cols = [f'{t}_error' for t in targets if f'{t}_error' in df.columns]
    if not cols:
        return None
    plot_df = df[cols].rename(
        columns={f'{t}_error': labels[t] for t in targets if f'{t}_error' in df.columns})

    hb_ref = [None]

    def diag_hist(x, **kw):
        ax = plt.gca()
        n = len(x.dropna())
        counts, edges, _ = ax.hist(x.dropna(), bins=40, color=hist_color,
                                   edgecolor='black', lw=0.4, alpha=0.85)
        mu, sigma = stats.norm.fit(x.dropna())
        se_mu    = sigma / np.sqrt(n)
        se_sigma = sigma / np.sqrt(2 * n)
        bw = edges[1] - edges[0]
        xr = np.linspace(edges[0], edges[-1], 300)
        ax.plot(xr, stats.norm.pdf(xr, mu, sigma) * n * bw, 'k-', lw=1.6)
        ax.axvline(0,  color='black', lw=1.3, ls='--')
        ax.axvline(mu, color='red',   lw=1.8)
        ax.errorbar(mu, counts.max() * 0.04, xerr=se_mu,
                    fmt='none', color='red', capsize=3, lw=1.6)
        ax.text(0.96, 0.96,
                f'$\\mu={mu:+.2f}\\pm{se_mu:.2f}$\n$\\sigma={sigma:.2f}\\pm{se_sigma:.2f}$',
                transform=ax.transAxes, ha='right', va='top', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.88, ec='lightgray'))
        ax.grid(False)

    def lower_hex(x, y, **kw):
        ax = plt.gca()
        hb = ax.hexbin(x, y, gridsize=28, cmap='magma_r', bins='log', mincnt=1)
        ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.6)
        ax.axvline(0, color='black', lw=0.8, ls='--', alpha=0.6)
        ax.grid(False)
        if hb_ref[0] is None:
            hb_ref[0] = hb

    g = sns.PairGrid(plot_df, corner=True, diag_sharey=False, height=panel_size)
    g.map_diag(diag_hist)
    g.map_lower(lower_hex)
    if hb_ref[0] is not None:
        cbar_ax = g.fig.add_axes([0.78, 0.57, 0.025, 0.28])
        cb = g.fig.colorbar(hb_ref[0], cax=cbar_ax)
        cb.set_label(r'$\log_{10}(N)$', rotation=270, labelpad=15)
    plt.tight_layout()
    return g.fig


df = load_test_details(EXP_CLEAN, incl_scale=DEG, theta_scale=DEG)
assert df is not None, f'no clean results found for {EXP_CLEAN}'
print(f'Loaded {len(df)} rows from {EXP_CLEAN}')

# 1) pred_vs_true plots — spin, incl, theta, z
fig = pred_vs_true_single(df['spin_true'].values, df['spin_pred'].values,
                          symbol=r'\alpha', error_label=r'$\Delta\alpha$',
                          color=COLOR_CLEAN, figsize=(9, 4))
_save(fig, 'exp4_neq45_spin_pred_vs_true_clean'); plt.close(fig)

fig = pred_vs_true_single(df['incl_true'].values, df['incl_pred'].values,
                          symbol='i', error_label=r'$\Delta i$ (°)', unit='°',
                          color=COLOR_CLEAN, force_positive_lim=True, figsize=(9, 4))
_save(fig, 'exp4_neq45_incl_pred_vs_true_clean'); plt.close(fig)

fig = pred_vs_true_single(df['theta_true'].values, df['theta_pred'].values,
                          symbol=r'\theta', error_label=r'$\Delta\theta$ (°)', unit='°',
                          color=COLOR_CLEAN, force_positive_lim=True, figsize=(9, 4))
_save(fig, 'exp4_neq45_theta_pred_vs_true_clean'); plt.close(fig)

fig = pred_vs_true_single(df['z_true'].values, df['z_pred'].values,
                          symbol='z', error_label=r'$\Delta z$ (M)', unit='M',
                          color=COLOR_CLEAN, force_positive_lim=True, figsize=(9, 4))
_save(fig, 'exp4_neq45_z_pred_vs_true_clean'); plt.close(fig)

# 2) corner plots — theta and z variants
LABELS = {
    'spin':  r'$\Delta\alpha$',
    'incl':  r'$\Delta i$ (°)',
    'theta': r'$\Delta\theta$ (°)',
    'z':     r'$\Delta z$ (M)',
}
for extra in ('theta', 'z'):
    fig = corner_plot_errors_matrix(df, ['spin', 'incl', extra], LABELS,
                                    panel_size=1.7, hist_color=COLOR_CLEAN)
    _save(fig, f'exp4_neq45_corner_{extra}_clean'); plt.close(fig)

print('done')
