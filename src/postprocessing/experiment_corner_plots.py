import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
from typing import List


# Pretty labels for targets and errors (matching other postprocessing)
TARGET_LABEL = {
    "spin": r"$\alpha$",
    "incl": r"$i$",
    "theta": r"$\theta$",
    "z": r"$z$",
}

ERROR_LABEL = {
    "spin": r"$\Delta\alpha$",
    "incl": r"$\Delta i$",
    "theta": r"$\Delta\theta$",
    "z": r"$\Delta z$",
}


def set_plot_style():
    """Apply publication-quality styling to matplotlib and seaborn."""
    mpl.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })
    # Set seaborn style to align with our custom matplotlib rcParams
    sns.set_style("ticks", {"axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.3})


def load_test_details(exp_name: str, root: Path) -> List[pd.DataFrame]:
    """Load all per-seed test_details CSVs for a given experiment."""
    metrics_dir = root / "results" / "metrics" / exp_name
    if not metrics_dir.exists():
        return []
    dfs = []
    for csv_path in sorted(metrics_dir.glob("test_details_seed*.csv")):
        df = pd.read_csv(csv_path)
        df["seed"] = int(csv_path.stem.split("seed")[-1])
        dfs.append(df)
    return dfs


def corner_plot_errors_2d(
    df: pd.DataFrame,
    x_err: str,
    y_err: str,
    title: str,
    out_path: Path,
    x_label: str,
    y_label: str,
):
    """
    2D corner-style density plot: error vs error with marginal histograms.
    """
    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        wspace=0.05,
        hspace=0.05,
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)

    x = df[x_err].values
    y = df[y_err].values

    # Main panel: 2D density via hexbin
    hb = ax_main.hexbin(
        x, y,
        gridsize=50,
        cmap="magma_r",
        bins="log",
        mincnt=1,
    )
    ax_main.axhline(0.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
    ax_main.axvline(0.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
    ax_main.set_xlabel(x_label)
    ax_main.set_ylabel(y_label)

    # Colorbar for density
    cbar = fig.colorbar(hb, ax=ax_main, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_{10}(N)$", rotation=270, labelpad=20)

    # Marginals
    hist_kwargs = dict(bins=50, color="#5B84B1", edgecolor="black", linewidth=0.5, alpha=0.8)
    ax_histx.hist(x, **hist_kwargs)
    ax_histy.hist(y, orientation="horizontal", **hist_kwargs)

    # Clean up marginal axes
    ax_histx.set_visible(False)
    ax_histy.set_visible(False)

    fig.suptitle(title, fontsize=14, y=0.92)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def calculate_binned_stats(x, y, n_bins=30):
    """Helper to calculate binned means and standard deviations."""
    bins = np.linspace(np.nanmin(x), np.nanmax(x), n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    
    bin_indices = np.digitize(x, bins) - 1
    
    mean_err = np.full(n_bins, np.nan)
    std_err = np.full(n_bins, np.nan)
    
    for i in range(n_bins):
        mask = bin_indices == i
        if np.any(mask):
            mean_err[i] = np.nanmean(y[mask])
            std_err[i] = np.nanstd(y[mask])
            
    valid = ~np.isnan(mean_err)
    return centers[valid], mean_err[valid], std_err[valid]


def error_vs_true_plot(df: pd.DataFrame, target: str, out_path: Path):
    """Plot error vs true value for a single target."""
    true_col = f"{target}_true"
    err_col = f"{target}_error"
    if true_col not in df.columns or err_col not in df.columns:
        return

    x = df[true_col].values
    y = df[err_col].values

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Background scatter (rasterized=True prevents huge file sizes)
    ax.scatter(x, y, s=6, alpha=0.15, color="#5B84B1", edgecolor="none", rasterized=True, zorder=1)

    # Binned statistics
    centers, mean_err, std_err = calculate_binned_stats(x, y)
    if len(centers) > 0:
        ax.plot(centers, mean_err, color="#FC766AFF", linewidth=2.5, label="Binned Mean", zorder=3)
        ax.fill_between(
            centers,
            mean_err - std_err,
            mean_err + std_err,
            color="#FC766AFF",
            alpha=0.25,
            label=r"$\pm 1\sigma$",
            zorder=2
        )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7, zorder=4)
    ax.set_xlabel(rf"{TARGET_LABEL.get(target, target)}$_\mathrm{{GT}}$")
    ax.set_ylabel(ERROR_LABEL.get(target, err_col))
    ax.set_title(rf"{ERROR_LABEL.get(target, err_col)} vs {TARGET_LABEL.get(target, target)}$_\mathrm{{GT}}$")
    
    ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="lightgray")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def error_vs_other_true_plot(df: pd.DataFrame, error_target: str, cond_target: str, out_path: Path):
    """Plot error of one target vs true value of another target."""
    true_col = f"{cond_target}_true"
    err_col = f"{error_target}_error"
    if true_col not in df.columns or err_col not in df.columns:
        return

    x = df[true_col].values
    y = df[err_col].values

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.scatter(x, y, s=6, alpha=0.15, color="#5B84B1", edgecolor="none", rasterized=True, zorder=1)

    centers, mean_err, std_err = calculate_binned_stats(x, y)
    if len(centers) > 0:
        ax.plot(centers, mean_err, color="#FC766AFF", linewidth=2.5, label="Binned Mean", zorder=3)
        ax.fill_between(
            centers,
            mean_err - std_err,
            mean_err + std_err,
            color="#FC766AFF",
            alpha=0.25,
            label=r"$\pm 1\sigma$",
            zorder=2
        )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7, zorder=4)
    ax.set_xlabel(rf"{TARGET_LABEL.get(cond_target, cond_target)}$_\mathrm{{GT}}$")
    ax.set_ylabel(ERROR_LABEL.get(error_target, err_col))
    ax.set_title(rf"{ERROR_LABEL.get(error_target, err_col)} vs {TARGET_LABEL.get(cond_target, cond_target)}$_\mathrm{{GT}}$")
    
    ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="lightgray")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def corner_plot_errors_matrix(
    df: pd.DataFrame,
    targets: List[str],
    title: str,
    out_path: Path,
) -> None:
    """Multi-dimensional corner plot for ERRORS utilizing seaborn PairGrid."""
    # Filter for the ERROR columns
    cols = [f"{t}_error" for t in targets]
    valid_cols = [c for c in cols if c in df.columns]
    
    if not valid_cols:
        return

    # Create a subset dataframe with nicely formatted column names for automatic labeling
    plot_df = df[valid_cols].copy()
    rename_dict = {
        f"{t}_error": ERROR_LABEL.get(t, f"\\Delta {t}") 
        for t in targets if f"{t}_error" in valid_cols
    }
    plot_df.rename(columns=rename_dict, inplace=True)

    # Initialize a Seaborn PairGrid using the 'corner=True' parameter
    g = sns.PairGrid(plot_df, corner=True, diag_sharey=False, height=2.5)

    # Custom mapping function for the diagonal (histograms)
    def diag_hist(x, **kwargs):
        ax = plt.gca()
        ax.hist(x, bins=40, color="#5B84B1", edgecolor="black", linewidth=0.5, alpha=0.8)
        # Line at 0.0 (ideal error)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
        # Line at the mean error (shows bias)
        ax.axvline(np.nanmean(x), color="#FC766AFF", linestyle="-", linewidth=2)
        ax.grid(False) 

    # We need to capture one hexbin collection to create the shared colorbar
    hb_collection = None

    # Custom mapping function for the lower triangle (hexbins)
    def lower_hex(x, y, **kwargs):
        nonlocal hb_collection
        ax = plt.gca()
        hb = ax.hexbin(x, y, gridsize=35, cmap="magma_r", bins="log", mincnt=1)
        
        # Crosshairs at 0 error
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.grid(False)
        
        if hb_collection is None:
            hb_collection = hb

    # Apply the mappings
    g.map_diag(diag_hist)
    g.map_lower(lower_hex)

    # Add a global colorbar in the empty upper-right space
    if hb_collection is not None:
        # [left, bottom, width, height] of the colorbar axis
        cbar_ax = g.fig.add_axes([0.75, 0.55, 0.03, 0.3])
        cbar = g.fig.colorbar(hb_collection, cax=cbar_ax)
        cbar.set_label(r"$\log_{10}(N)$", rotation=270, labelpad=20)

    # Overall title and saving
    g.fig.suptitle(title, fontsize=18, y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def make_plots_for_experiment_2(root: Path):
    """Corner + error-vs-true plots for Experiment II (spin/incl)."""
    for suffix, label in [("", "with_noise"), ("_no_noise", "no_noise")]:
        exp_name = f"experiment_2_eq_full{suffix}"
        dfs = load_test_details(exp_name, root)
        if not dfs:
            continue
        df_all = pd.concat(dfs, ignore_index=True)

        for col in ["incl_true", "incl_error"]:
            if col in df_all.columns:
                df_all[col] = df_all[col] * (180.0 / (10.0 * np.pi))

        exp_fig_dir = root / "results" / "figures" / "comparisons" / exp_name

        out_corner = exp_fig_dir / f"spin_incl_corner_{label}.png"
        corner_plot_errors_2d(
            df_all,
            "spin_error",
            "incl_error",
            r"Exp II (" + label.replace("_", " ") + r"): $\Delta\alpha$ vs $\Delta i$",
            out_corner,
            x_label=ERROR_LABEL["spin"],
            y_label=ERROR_LABEL["incl"],
        )

        targets = ["spin", "incl"]
        base_dir = exp_fig_dir

        for target in targets:
            out_err = base_dir / f"{target}_error_vs_{target}_true_{label}.png"
            error_vs_true_plot(df_all, target, out_err)

        for err_t in targets:
            for cond_t in targets:
                if err_t == cond_t:
                    continue
                out_cross = base_dir / f"{err_t}_error_vs_{cond_t}_true_{label}.png"
                error_vs_other_true_plot(df_all, err_t, cond_t, out_cross)


def make_plots_for_experiment_4(root: Path):
    """Corner + error-vs-true plots for Experiment IV (spin/incl/theta/z)."""
    for suffix, label in [("", "with_noise"), ("_no_noise", "no_noise")]:
        exp_name = f"experiment_4_noneq_full{suffix}"
        dfs = load_test_details(exp_name, root)
        if not dfs:
            continue
        df_all = pd.concat(dfs, ignore_index=True)

        angle_cols = ["incl_true", "incl_error", "theta_true", "theta_error"]
        for col in angle_cols:
            if col in df_all.columns:
                df_all[col] = df_all[col] * 180.0 / np.pi

        base_dir = root / "results" / "figures" / "comparisons" / exp_name

        corner_targets = ["spin", "incl", "theta", "z"]
        corner_out = base_dir / f"corner_errors_{label}.png"
        corner_plot_errors_matrix(
            df_all,
            corner_targets,
            title=r"Exp IV Errors (" + label.replace("_", " ") + r"): $\Delta\alpha, \Delta i, \Delta\theta, \Delta z$",
            out_path=corner_out,
        )

        targets = ["spin", "incl", "theta", "z"]

        for target in targets:
            true_col = f"{target}_true"
            err_col = f"{target}_error"
            if true_col not in df_all.columns or err_col not in df_all.columns:
                continue
            out_err = base_dir / f"{target}_error_vs_{target}_true_{label}.png"
            error_vs_true_plot(df_all, target, out_err)

        for err_t in targets:
            for cond_t in targets:
                if err_t == cond_t:
                    continue
                out_cross = base_dir / f"{err_t}_error_vs_{cond_t}_true_{label}.png"
                error_vs_other_true_plot(df_all, err_t, cond_t, out_cross)


def main():
    set_plot_style()
    repo_root = Path(__file__).parent.parent.parent
    print("Generating corner and error-vs-true plots for Experiments II and IV")
    make_plots_for_experiment_2(repo_root)
    make_plots_for_experiment_4(repo_root)
    print("Done.")


if __name__ == "__main__":
    main()