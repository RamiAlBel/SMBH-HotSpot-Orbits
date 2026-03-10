import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['text.usetex'] = False  # Set to True if LaTeX is installed

COLORS = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff"]

METRICS_DIR = Path(__file__).parent.parent.parent / "results" / "metrics"
FIGURES_DIR = Path(__file__).parent.parent.parent / "results" / "figures"

EXPERIMENT_NAMES = {
    "experiment_1_eq_avg": "Exp I (Eq. Avg)",
    "experiment_2_eq_full": "Exp II (Eq. Full)",
    "experiment_3_eq_half": "Exp III (Eq. Half)",
    "experiment_4_noneq_full": "Exp IV (Non-Eq. Full)",
    "experiment_5_noneq_half": "Exp V (Non-Eq. Half)",
}

TARGET_NAMES = {
    "spin": r"Spin ($\alpha$)",
    "incl": r"Inclination ($i$)",
    "theta": r"Theta ($\theta$)",
    "z": r"Z-coordinate",
}

TARGET_SIGMA_LABELS = {
    "spin": r"$\hat{\sigma}_a$",
    "incl": r"$\hat{\sigma}_i$",
    "theta": r"$\hat{\sigma}_\theta$",
    "z": r"$\hat{\sigma}_z$",
}

TARGET_UNITS = {
    "spin": "",
    "incl": r"$^\circ$",
    "theta": r"$^\circ$",
    "z": "M",
}

# Experiments where angles are in radians (need conversion)
RADIANS_EXPERIMENTS = ["experiment_2_eq_full", "experiment_3_eq_half", 
                       "experiment_4_noneq_full", "experiment_5_noneq_half"]

# Valid targets per experiment
EXPERIMENT_TARGETS = {
    "experiment_1_eq_avg": ["spin"],  # incl is always 0
    "experiment_2_eq_full": ["spin", "incl"],
    "experiment_3_eq_half": ["spin", "incl"],  # no theta, no z
    "experiment_4_noneq_full": ["spin", "incl", "theta", "z"],
    "experiment_5_noneq_half": ["spin", "incl", "theta", "z"],
}


def convert_to_degrees(value: float, target: str, exp_name: str) -> float:
    """Convert radians to degrees if needed."""
    if target in ["incl", "theta"] and exp_name in RADIANS_EXPERIMENTS:
        return np.rad2deg(value)
    return value


def load_aggregated_metrics(exp_name: str, noise_suffix: str = "") -> Dict[str, pd.DataFrame]:
    """Load aggregated metrics for a full-orbit experiment."""
    exp_dir = METRICS_DIR / f"{exp_name}{noise_suffix}"
    metrics = {}
    
    if not exp_dir.exists():
        return metrics
    
    valid_targets = EXPERIMENT_TARGETS.get(exp_name, ["spin", "incl", "theta", "z"])
    
    for target in valid_targets:
        agg_path = exp_dir / f"{target}_aggregated.csv"
        if agg_path.exists():
            df = pd.read_csv(agg_path)
            # Convert radians to degrees for angle metrics
            for col in ["mae_mean", "mae_std", "rmse_mean", "rmse_std", "error_std_mean", "error_std_std"]:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: convert_to_degrees(x, target, exp_name))
            metrics[target] = df
    
    return metrics


def load_sweep_metrics(exp_name: str, noise_suffix: str = "") -> Dict[str, pd.DataFrame]:
    """Load sweep metrics for a half-orbit experiment."""
    exp_dir = METRICS_DIR / f"{exp_name}{noise_suffix}"
    metrics = {}
    
    if not exp_dir.exists():
        return metrics
    
    valid_targets = EXPERIMENT_TARGETS.get(exp_name, ["spin", "incl", "theta", "z"])
    
    for target in valid_targets:
        sweep_path = exp_dir / f"{target}_sweep.csv"
        if sweep_path.exists():
            df = pd.read_csv(sweep_path)
            # Convert radians to degrees for angle metrics
            angle_cols = ["sigma_mean", "sigma_std", "mae_mean", "mae_std", "rmse_mean", "rmse_std", "error_std_mean", "error_std_std"]
            for col in angle_cols:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: convert_to_degrees(x, target, exp_name))
            metrics[target] = df
    
    return metrics


def plot_target_noise_comparison_aggregated(
    exp_name: str, 
    target: str, 
    noise_metrics: pd.DataFrame, 
    no_noise_metrics: pd.DataFrame,
    output_dir: Path
) -> None:
    """Compare noise vs no-noise for a single target (aggregated metrics)."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metric_names = ["mae_mean", "rmse_mean", "error_std_mean", "r2_mean"]
    metric_labels = ["MAE", "RMSE", r"$\sigma$", r"$R^2$"]
    
    for ax_idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[ax_idx]
        
        noise_val = noise_metrics[metric].values[0]
        no_noise_val = no_noise_metrics[metric].values[0]
        
        x = np.array([0, 1])
        vals = [noise_val, no_noise_val]
        colors = [COLORS[0], COLORS[1]]
        labels_bar = ["With Noise", "No Noise"]
        
        bars = ax.bar(x, vals, color=colors, alpha=0.8, width=0.6)
        
        # Add error bars for metrics with std (not R²)
        if metric in ["mae_mean", "rmse_mean", "error_std_mean"]:
            std_col = f"{metric.replace('_mean', '_std')}"
            if std_col in noise_metrics.columns and std_col in no_noise_metrics.columns:
                noise_std = noise_metrics[std_col].values[0]
                no_noise_std = no_noise_metrics[std_col].values[0]
                ax.errorbar(x, vals, yerr=[noise_std, no_noise_std], 
                           fmt='none', color='black', capsize=5, linewidth=1.5)
        
        unit_label = ""
        if label in ["MAE", "RMSE", r"$\sigma$"]:
            unit_label = f" ({TARGET_UNITS[target]})" if TARGET_UNITS[target] else ""
        
        ax.set_ylabel(f"{label}{unit_label}")
        ax.set_title(f"{label}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_bar)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, vals)):
            height = bar.get_height()
            if label == r"$R^2$":
                text = f'{val:.3f}'
            elif val < 0.01:
                text = f'{val:.4f}'
            else:
                text = f'{val:.3f}'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   text, ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(f"{EXPERIMENT_NAMES[exp_name]}: {TARGET_NAMES[target]} - Noise Comparison", 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{exp_name}_{target}_noise_comparison.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def _sweep_sigma_cols(df: pd.DataFrame):
    """Return (mean_col, std_col) for sweep σ, supporting old and new format."""
    if 'error_std_mean' in df.columns:
        return 'error_std_mean', 'error_std_std'
    return 'sigma_mean', 'sigma_std'


def plot_target_sweep_comparison(
    exp_name: str,
    target: str,
    noise_metrics: pd.DataFrame,
    no_noise_metrics: pd.DataFrame,
    output_dir: Path
) -> None:
    """Compare noise vs no-noise sweep for a single target."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    mean_col, std_col = _sweep_sigma_cols(noise_metrics)
    
    ax.errorbar(noise_metrics['percent'], noise_metrics[mean_col], 
                yerr=noise_metrics[std_col], 
                marker='o', label='With Noise', color=COLORS[0], 
                capsize=4, linewidth=2.5, markersize=8)
    ax.errorbar(no_noise_metrics['percent'], no_noise_metrics[mean_col], 
                yerr=no_noise_metrics[std_col], 
                marker='s', label='No Noise', color=COLORS[1], 
                capsize=4, linewidth=2.5, markersize=8)
    
    ax.set_xlabel("% of Orbit Used", fontsize=11)
    
    unit_str = f" ({TARGET_UNITS[target]})" if TARGET_UNITS[target] else ""
    ax.set_ylabel(f"{TARGET_SIGMA_LABELS[target]}{unit_str}", fontsize=11)
    
    ax.set_title(f"{EXPERIMENT_NAMES[exp_name]}: {TARGET_NAMES[target]}", 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{exp_name}_{target}_sweep_comparison.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def _get_sweep_100pct_metrics(exp_name: str, target: str, noise_suffix: str) -> Optional[pd.DataFrame]:
    """Get pseudo-aggregated metrics from 100% orbit sweep row (for exp 3, 5). Fallback when no aggregated file."""
    sweep = load_sweep_metrics(exp_name, noise_suffix)
    if not sweep or target not in sweep:
        return None
    df = sweep[target]
    row_100 = df[df['percent'] == 100]
    if row_100.empty:
        return None
    r = row_100.iloc[0]
    if 'mae_mean' in df.columns:
        return pd.DataFrame([{
            'mae_mean': r['mae_mean'], 'mae_std': r.get('mae_std', 0),
            'rmse_mean': r['rmse_mean'], 'rmse_std': r.get('rmse_std', 0),
            'r2_mean': r['r2_mean'], 'r2_std': r.get('r2_std', 0),
            'error_std_mean': r['error_std_mean'], 'error_std_std': r.get('error_std_std', 0),
        }])
    return pd.DataFrame([{
        'mae_mean': np.nan, 'mae_std': np.nan,
        'rmse_mean': np.nan, 'rmse_std': np.nan,
        'r2_mean': np.nan, 'r2_std': np.nan,
        'error_std_mean': r['sigma_mean'], 'error_std_std': r.get('sigma_std', 0),
    }])


def plot_all_experiments_per_target(target: str, output_dir: Path) -> None:
    """Compare all experiments for a single target (aggregated + 100% sweep fallback for exp 5)."""
    all_experiments = ["experiment_1_eq_avg", "experiment_2_eq_full", "experiment_4_noneq_full", "experiment_5_noneq_half"]
    valid_exps = [e for e in all_experiments if target in EXPERIMENT_TARGETS.get(e, [])]
    
    if not valid_exps:
        return
    
    results = {"noise": {}, "no_noise": {}}
    
    for exp in valid_exps:
        noise_m = load_aggregated_metrics(exp)
        no_noise_m = load_aggregated_metrics(exp, "_no_noise")
        if noise_m and target in noise_m:
            results["noise"][exp] = noise_m[target]
        elif exp == "experiment_5_noneq_half":
            m = _get_sweep_100pct_metrics(exp, target, "")
            if m is not None:
                results["noise"][exp] = m
        if no_noise_m and target in no_noise_m:
            results["no_noise"][exp] = no_noise_m[target]
        elif exp == "experiment_5_noneq_half":
            m = _get_sweep_100pct_metrics(exp, target, "_no_noise")
            if m is not None:
                results["no_noise"][exp] = m
    
    if not results["noise"] and not results["no_noise"]:
        return
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metrics = ["mae_mean", "rmse_mean", "error_std_mean", "r2_mean"]
    labels = ["MAE", "RMSE", r"$\sigma$", r"$R^2$"]
    
    for ax_idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[ax_idx]
        
        exp_labels = []
        noise_vals = []
        no_noise_vals = []
        
        for exp in valid_exps:
            if exp in results["noise"] or exp in results["no_noise"]:
                exp_labels.append(EXPERIMENT_NAMES[exp])
                noise_vals.append(results["noise"][exp][metric].values[0] if exp in results["noise"] else np.nan)
                no_noise_vals.append(results["no_noise"][exp][metric].values[0] if exp in results["no_noise"] else np.nan)
        
        x = np.arange(len(exp_labels))
        width = 0.35
        
        ax.bar(x - width/2, noise_vals, width, label="With Noise", color=COLORS[0], alpha=0.8)
        ax.bar(x + width/2, no_noise_vals, width, label="No Noise", color=COLORS[1], alpha=0.8)
        
        unit_label = ""
        if label in ["MAE", "RMSE", r"$\sigma$"]:
            unit_label = f" ({TARGET_UNITS[target]})" if TARGET_UNITS[target] else ""
        
        ax.set_ylabel(f"{label}{unit_label}")
        ax.set_title(f"{label}")
        ax.set_xticks(x)
        ax.set_xticklabels(exp_labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f"Cross-Experiment Comparison: {TARGET_NAMES[target]}", 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"all_experiments_{target}_comparison.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_sweep_experiments_per_target(target: str, output_dir: Path) -> None:
    """Compare sweep experiments (3 and 5) for a single target."""
    experiments = ["experiment_3_eq_half", "experiment_5_noneq_half"]
    
    # Filter experiments that have this target
    valid_exps = [exp for exp in experiments if target in EXPERIMENT_TARGETS.get(exp, [])]
    
    if not valid_exps:
        return
    
    # Load metrics for valid experiments only
    exp_metrics = {}
    for exp in valid_exps:
        exp_metrics[exp] = {
            "noise": load_sweep_metrics(exp),
            "no_noise": load_sweep_metrics(exp, "_no_noise")
        }
    
    # Check if target exists in any of the experiments
    has_data = False
    for exp in valid_exps:
        if target in exp_metrics[exp]["noise"] or target in exp_metrics[exp]["no_noise"]:
            has_data = True
            break
    
    if not has_data:
        return
    
    n_plots = len(valid_exps)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    exp_titles = {
        "experiment_3_eq_half": "Exp III (Equatorial Half-Orbit)",
        "experiment_5_noneq_half": "Exp V (Non-Equatorial Half-Orbit)"
    }
    
    for ax_idx, exp in enumerate(valid_exps):
        ax = axes[ax_idx]
        
        noise_df = exp_metrics[exp]["noise"].get(target)
        no_noise_df = exp_metrics[exp]["no_noise"].get(target)
        
        mean_col, std_col = ('error_std_mean', 'error_std_std') if noise_df is not None and 'error_std_mean' in noise_df.columns else ('sigma_mean', 'sigma_std')
        if noise_df is not None:
            ax.errorbar(noise_df['percent'], noise_df[mean_col], yerr=noise_df[std_col], 
                        marker='o', label='With Noise', color=COLORS[0], 
                        capsize=4, linewidth=2.5, markersize=8)
        if no_noise_df is not None:
            mc, sc = ('error_std_mean', 'error_std_std') if 'error_std_mean' in no_noise_df.columns else ('sigma_mean', 'sigma_std')
            ax.errorbar(no_noise_df['percent'], no_noise_df[mc], yerr=no_noise_df[sc], 
                        marker='s', label='No Noise', color=COLORS[1], 
                        capsize=4, linewidth=2.5, markersize=8)
        
        ax.set_xlabel("% of Orbit Used")
        unit_str = f" ({TARGET_UNITS[target]})" if TARGET_UNITS[target] else ""
        ax.set_ylabel(f"{TARGET_SIGMA_LABELS[target]}{unit_str}")
        ax.set_title(exp_titles[exp])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Sweep Experiments: {TARGET_NAMES[target]} (Noise vs No-Noise)", 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sweep_experiments_{target}_comparison.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all postprocessing figures from metrics."""
    
    print("=" * 60)
    print("Generating postprocessing figures from metrics")
    print("=" * 60)
    
    comparison_dir = FIGURES_DIR / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Per-target noise vs no-noise for full-orbit experiments (1, 2, 4)
    print("\n[1/5] Generating per-target noise comparisons for full-orbit experiments...")
    for exp in ["experiment_1_eq_avg", "experiment_2_eq_full", "experiment_4_noneq_full"]:
        noise_metrics = load_aggregated_metrics(exp)
        no_noise_metrics = load_aggregated_metrics(exp, "_no_noise")
        
        common_targets = [t for t in noise_metrics.keys() if t in no_noise_metrics]
        
        for target in common_targets:
            plot_target_noise_comparison_aggregated(
                exp, target, noise_metrics[target], no_noise_metrics[target], comparison_dir
            )
    
    # 2. Per-target noise vs no-noise for sweep experiments (3, 5)
    print("\n[2/5] Generating per-target sweep comparisons for half-orbit experiments...")
    for exp in ["experiment_3_eq_half", "experiment_5_noneq_half"]:
        noise_metrics = load_sweep_metrics(exp)
        no_noise_metrics = load_sweep_metrics(exp, "_no_noise")
        
        common_targets = [t for t in noise_metrics.keys() if t in no_noise_metrics]
        
        for target in common_targets:
            plot_target_sweep_comparison(
                exp, target, noise_metrics[target], no_noise_metrics[target], comparison_dir
            )
    
    # 3. Cross-experiment comparison per target (full-orbit)
    print("\n[3/5] Generating cross-experiment comparisons per target...")
    # Only use targets that exist in at least one full-orbit experiment
    all_targets = set()
    for exp in ["experiment_1_eq_avg", "experiment_2_eq_full", "experiment_4_noneq_full"]:
        all_targets.update(EXPERIMENT_TARGETS.get(exp, []))
    
    for target in sorted(all_targets):
        plot_all_experiments_per_target(target, comparison_dir)
    
    # 4. Cross-experiment sweep comparison per target
    print("\n[4/5] Generating sweep experiment comparisons per target...")
    # Only use targets that exist in at least one sweep experiment
    sweep_targets = set()
    for exp in ["experiment_3_eq_half", "experiment_5_noneq_half"]:
        sweep_targets.update(EXPERIMENT_TARGETS.get(exp, []))
    
    for target in sorted(sweep_targets):
        plot_sweep_experiments_per_target(target, comparison_dir)
    
    # 5. Summary table
    print("\n[5/5] Generating summary statistics...")
    print("\nMetrics Summary (converted to proper units):")
    print("-" * 60)
    
    for exp in EXPERIMENT_NAMES.keys():
        noise_m = load_aggregated_metrics(exp)
        no_noise_m = load_aggregated_metrics(exp, "_no_noise")
        
        if noise_m or no_noise_m:
            print(f"\n{EXPERIMENT_NAMES[exp]}:")
            valid_targets = EXPERIMENT_TARGETS.get(exp, [])
            for target in valid_targets:
                if target in noise_m and target in no_noise_m:
                    noise_mae = noise_m[target]['mae_mean'].values[0]
                    no_noise_mae = no_noise_m[target]['mae_mean'].values[0]
                    unit = TARGET_UNITS[target]
                    unit_str = f" {unit}" if unit else ""
                    print(f"  {TARGET_NAMES[target]}: MAE {noise_mae:.4f} (noise) vs {no_noise_mae:.4f} (no noise){unit_str}")
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {comparison_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
