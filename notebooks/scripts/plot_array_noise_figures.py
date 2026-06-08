"""Regenerate the per-figure "noisy" diagnostics for each instrument array
(EHT, ngEHT, BHEX) instead of the single default-noise condition.

It reuses the same plotting logic as ``notebooks/extra_plots.ipynb`` (cells 0-5)
but loops over the array-specific result directories that were trained/evaluated
with each instrument's noise budget:

    experiment_1_eq_avg_{eht,ngeht,bhex}            -> exp1_spin_pred_vs_true_<array>
    experiment_2_eq_full_{eht,ngeht,bhex}          -> exp2_spin_pred_vs_true_<array>
                                                      exp2_incl_pred_vs_true_<array>
                                                      exp2_corner_<array>
    experiment_4_noneq_full_neq45_{eht,ngeht,bhex} -> exp4_neq45_corner_theta_<array>

Note: the θ corner is built from Experiment IV (the only non-equatorial run with
per-sample test details — Exp V stores sweep summaries only). The requested
"exp5_corner_theta" maps to this exp4_neq45 θ corner.

Output: notebooks/figures/<name>_<array>.{pdf,png}

Run:
    python notebooks/scripts/plot_array_noise_figures.py
"""
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
    "font.size": 14, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "figure.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42,
})

DEG = 180.0 / np.pi

# Instrument arrays: (dir-suffix, plot colour) — viridis tones matching the
# instrument sweep figure (dark->light tracks decreasing noise).
ARRAYS = [
    ('eht',   '#3b528b'),
    ('ngeht', '#21918c'),
    ('bhex',  '#5ec962'),
]


# ── helpers (mirrored from extra_plots.ipynb cell 0) ─────────────────────────

def _save(fig, name):
    p = NOTEBOOK_DIR / name
    fig.savefig(p.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(p.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print('saved', p.with_suffix('.pdf').name, '/', p.with_suffix('.png').name)


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
    eqn = (rf'$\mu = {mu:+.3f} \pm {se_mu:.3f}$' + '\n'
           rf'$\sigma = {sigma:.3f} \pm {se_sigma:.3f}$')
    ax.text(0.97, 0.96, eqn, transform=ax.transAxes, ha='right', va='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.35', fc='white', alpha=0.92, ec='lightgray'))
    return mu, sigma, se_mu, se_sigma


def pred_vs_true_single(y_true, y_pred, symbol, error_label, color,
                        unit='', force_positive_lim=False, figsize=(9, 4)):
    """Two-panel (1x2) figure: scatter on left, error histogram on right."""
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
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(0.05, 0.95, f'$R^2={r2:.3f}$\nMAE={mae:.3f}',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9, ec='lightgray'))

    ax2 = axes[1]
    _gaussian_fit_and_plot(ax2, err, color)
    ax2.set_xlabel(error_label)
    ax2.set_ylabel('Count')

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


def load_predictions(exp_name, target, seed=43):
    p = METRICS / exp_name / f'{target}_predictions_seed{seed}.csv'
    if not p.exists():
        return None, None
    arr = np.loadtxt(p, delimiter=',', skiprows=1)
    return arr[:, 0], arr[:, 1]


def corner_plot_errors_matrix(df, targets, labels, panel_size=1.3, hist_color='#5B84B1'):
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
                rf'$\mu={mu:+.2f}\pm{se_mu:.2f}$' + '\n' + rf'$\sigma={sigma:.2f}\pm{se_sigma:.2f}$',
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


def exp2_corner(df_det, color):
    """Inline Δα vs Δi hexbin corner (mirrors extra_plots.ipynb cell 3)."""
    fig = plt.figure(figsize=(5, 3.5))
    gs = fig.add_gridspec(2, 3, width_ratios=(4, 1, 0.6), height_ratios=(1, 4),
                          wspace=0.08, hspace=0.08)
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_cb    = fig.add_subplot(gs[1, 2])

    x = df_det['spin_error'].values
    y = df_det['incl_error'].values

    hb = ax_main.hexbin(x, y, gridsize=30, cmap='magma_r', bins='log', mincnt=1)
    ax_main.axhline(0, color='gray', ls='--', lw=1.0, alpha=0.8)
    ax_main.axvline(0, color='gray', ls='--', lw=1.0, alpha=0.8)
    ax_main.set_xlabel(r'$\Delta\alpha$')
    ax_main.set_ylabel(r'$\Delta i$ (°)')

    cb = fig.colorbar(hb, cax=ax_cb)
    cb.set_label(r'$\log_{10}(N)$', rotation=270, labelpad=15)

    hkw = dict(bins=50, color=color, edgecolor='black', lw=0.4, alpha=0.85)
    ax_histx.hist(x, **hkw)
    ax_histy.hist(y, orientation='horizontal', **hkw)
    ax_histx.axvline(0, color='black', lw=1.2, ls='--')
    ax_histy.axhline(0, color='black', lw=1.2, ls='--')
    ax_histx.tick_params(labelbottom=False)
    ax_histy.tick_params(labelleft=False)
    return fig


# ── main ─────────────────────────────────────────────────────────────────────

EXP4_LABELS = {
    'spin':  r'$\Delta\alpha$',
    'incl':  r'$\Delta i$ (°)',
    'theta': r'$\Delta\theta$ (°)',
}


def main():
    for suffix, color in ARRAYS:
        print(f'\n=== {suffix.upper()} ===')

        # Exp I — spin pred-vs-true (seed 43).
        yt, yp = load_predictions(f'experiment_1_eq_avg_{suffix}', 'spin', seed=43)
        if yt is not None:
            fig = pred_vs_true_single(yt, yp, symbol=r'\alpha',
                                      error_label=r'$\Delta\alpha$', color=color,
                                      figsize=(9, 4))
            _save(fig, f'exp1_spin_pred_vs_true_{suffix}'); plt.close(fig)

        # Exp II — spin + incl pred-vs-true, and Δα-Δi corner (all seeds).
        df2 = load_test_details(f'experiment_2_eq_full_{suffix}', incl_scale=DEG)
        if df2 is not None:
            fig = pred_vs_true_single(df2['spin_true'].values, df2['spin_pred'].values,
                                      symbol=r'\alpha', error_label=r'$\Delta\alpha$',
                                      color=color, figsize=(9, 4))
            _save(fig, f'exp2_spin_pred_vs_true_{suffix}'); plt.close(fig)

            fig = pred_vs_true_single(df2['incl_true'].values, df2['incl_pred'].values,
                                      symbol='i', error_label=r'$\Delta i$ (°)', unit='°',
                                      color=color, force_positive_lim=True, figsize=(9, 4))
            _save(fig, f'exp2_incl_pred_vs_true_{suffix}'); plt.close(fig)

            fig = exp2_corner(df2, color)
            _save(fig, f'exp2_corner_{suffix}'); plt.close(fig)

        # Exp IV — θ corner (spin, incl, theta). Maps to the requested
        # "exp5_corner_theta" (Exp V has no per-sample details).
        df4 = load_test_details(f'experiment_4_noneq_full_neq45_{suffix}',
                                incl_scale=DEG, theta_scale=DEG)
        if df4 is not None:
            fig = corner_plot_errors_matrix(df4, ['spin', 'incl', 'theta'],
                                            EXP4_LABELS, panel_size=1.7,
                                            hist_color=color)
            _save(fig, f'exp4_neq45_corner_theta_{suffix}'); plt.close(fig)


if __name__ == '__main__':
    main()
    print('\nDone.')
