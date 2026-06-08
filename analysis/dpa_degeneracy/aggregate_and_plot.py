#!/usr/bin/env python3
"""Aggregate partial match files and build the degeneracy hero figure.

Merges all partial CSVs for a scenario, dedupes (i,j)/(j,i) pairs, selects a
handful of "hero" pairs that have a small roll-aligned DPA distance but very
different (a, i, theta), and plots each as an overlay of the two phase-aligned
DPA curves annotated with both parameter sets.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "data" / "processed" / "dpa_dataset_neq_45.csv"
ANALYSIS_DIR = REPO_ROOT / "results" / "analysis" / "dpa_degeneracy"
DPA_COLS = [f"DPA_{i/10:.1f}" for i in range(1, 11)]
PHASE = np.arange(1, 11) / 10.0


def load_pairs(partials_dir, scenario):
    files = sorted(partials_dir.glob(f"{scenario}_chunk*.csv"))
    if not files:
        raise SystemExit(f"No partial files for scenario '{scenario}' in {partials_dir}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    # Dedupe symmetric pairs, keep the smallest dpa_l1 representative.
    lo = np.minimum(df.query_idx, df.match_idx)
    hi = np.maximum(df.query_idx, df.match_idx)
    df["pair_key"] = lo.astype(str) + "_" + hi.astype(str)
    df = df.sort_values("dpa_l1").drop_duplicates("pair_key", keep="first")
    return df.reset_index(drop=True)


def select_heroes(df, n, l1_cap, min_param_dist):
    """Pairs with small DPA distance and large param distance, no index reused."""
    cap = l1_cap
    for _ in range(8):  # relax the L1 cap if too few candidates qualify
        cand = df[(df.dpa_l1 <= cap) & (df.param_dist >= min_param_dist)]
        cand = cand.sort_values("param_dist", ascending=False)
        chosen, used = [], set()
        for _, row in cand.iterrows():
            if row.query_idx in used or row.match_idx in used:
                continue
            chosen.append(row)
            used.update([row.query_idx, row.match_idx])
            if len(chosen) == n:
                break
        if len(chosen) == n:
            break
        cap *= 1.5
    return pd.DataFrame(chosen), cap


def plot_heroes(heroes, curves, scenario, l1_cap_used, out_path):
    n = len(heroes)
    if n == 0:
        raise SystemExit("No hero pairs found; loosen --l1-cap / --min-param-dist.")
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 3.6 * nrows),
                             squeeze=False)
    axes = axes.ravel()

    title_map = {
        "dpa_only": "DPA-shape degeneracy (r, T free to differ)",
        "full_obs": "Full-observable degeneracy (similar r, T, and DPA curve)",
    }

    for ax, (_, row) in zip(axes, heroes.iterrows()):
        cq = curves[int(row.query_idx)]
        cm = np.roll(curves[int(row.match_idx)], int(row.roll))
        ax.plot(PHASE, cq, "o-", color="#1f77b4", lw=2, ms=4, label="hotspot A")
        ax.plot(PHASE, cm, "s--", color="#d62728", lw=2, ms=4,
                label=f"hotspot B (rolled +{int(row.roll)})")
        ax.set_xlabel("orbital phase")
        ax.set_ylabel(r"$\Delta$PA  [deg]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")

        txt = (
            f"A:  a={row.a_q:+.2f}  i={row.i_q:.1f}°  "
            f"θ={row.theta_q:+.1f}°  r={row.r_q:.1f}  T={row.T_q:.1f}\n"
            f"B:  a={row.a_m:+.2f}  i={row.i_m:.1f}°  "
            f"θ={row.theta_m:+.1f}°  r={row.r_m:.1f}  T={row.T_m:.1f}\n"
            f"mean |ΔDPA| = {row.dpa_l1:.2f}°   "
            f"param-dist = {row.param_dist:.2f}"
        )
        ax.set_title(txt, fontsize=7.5, loc="left", family="monospace")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title_map[scenario], fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"saved figure -> {out_path}  (L1 cap used: {l1_cap_used:.2f} deg/sample)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=["dpa_only", "full_obs"], required=True)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--partials-dir", type=Path, default=ANALYSIS_DIR / "partials")
    p.add_argument("--out-dir", type=Path, default=ANALYSIS_DIR / "figures")
    p.add_argument("--n-heroes", type=int, default=6)
    p.add_argument("--l1-cap", type=float, default=2.0,
                   help="max mean |dDPA| (deg/sample) for a hero pair")
    p.add_argument("--min-param-dist", type=float, default=0.45,
                   help="min normalised (a,i,theta) distance for a hero pair")
    args = p.parse_args()

    pairs = load_pairs(args.partials_dir, args.scenario)
    print(f"[{args.scenario}] {len(pairs)} unique pairs; "
          f"dpa_l1 min={pairs.dpa_l1.min():.3f} median={pairs.dpa_l1.median():.3f}")

    heroes, cap_used = select_heroes(pairs, args.n_heroes, args.l1_cap, args.min_param_dist)
    cols = ["query_idx", "match_idx", "dpa_l1", "roll", "param_dist", "dr", "dT",
            "a_q", "i_q", "theta_q", "r_q", "T_q", "a_m", "i_m", "theta_m", "r_m", "T_m"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    heroes_csv = args.out_dir / f"{args.scenario}_hero_pairs.csv"
    heroes[cols].to_csv(heroes_csv, index=False)
    print(heroes[["dpa_l1", "param_dist", "a_q", "a_m", "i_q", "i_m",
                  "theta_q", "theta_m"]].to_string(index=False))

    df = pd.read_csv(args.dataset)
    curves = df[DPA_COLS].to_numpy(dtype=np.float32)
    plot_heroes(heroes, curves, args.scenario, cap_used,
                args.out_dir / f"{args.scenario}_degeneracy.png")


if __name__ == "__main__":
    main()
