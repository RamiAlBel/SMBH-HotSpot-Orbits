#!/usr/bin/env python3
"""Find roll-invariant degenerate DPA-curve pairs.

For every query hotspot we find the curves whose DPA(t) curve is most similar
*after optimal cyclic phase alignment* (np.roll over the 10 phase samples), and
record how different their physical parameters (a, i, theta) are. Pairs with a
small DPA distance but a large parameter distance are the degeneracies we want
to surface.

Two scenarios share the same core:
  * dpa_only : compare against the whole dataset (r, T free to differ).
  * full_obs : compare only against curves with similar r AND Period, so the
               *entire* observable vector the model sees (r, T, DPA curve) is
               degenerate.

The work is split into chunks of query curves so it can run as a SLURM array
job; each task writes one partial CSV. Use --local to run every chunk in one
process.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "data" / "processed" / "dpa_dataset_neq_45.csv"
DEFAULT_OUT = REPO_ROOT / "results" / "analysis" / "dpa_degeneracy" / "partials"

DPA_COLS = [f"DPA_{i/10:.1f}" for i in range(1, 11)]
N_PHASE = len(DPA_COLS)

# Parameter ranges used to normalise the (a, i, theta) distance to [0,1] per dim.
PARAM_RANGES = {"a": 2.0, "i": 60.0, "theta": 90.0}


def param_distance(a1, i1, t1, a2, i2, t2):
    """Normalised Euclidean distance in (a, i, theta) space (per-dim scaled)."""
    da = (a1 - a2) / PARAM_RANGES["a"]
    di = (i1 - i2) / PARAM_RANGES["i"]
    dt = (t1 - t2) / PARAM_RANGES["theta"]
    return np.sqrt(da * da + di * di + dt * dt)


def build_rolled(curves):
    """Return array (N_PHASE, N, N_PHASE): all cyclic rolls of every curve."""
    n = curves.shape[0]
    rolled = np.empty((N_PHASE, n, N_PHASE), dtype=np.float32)
    for s in range(N_PHASE):
        rolled[s] = np.roll(curves, s, axis=1)
    return rolled


def find_for_chunk(df, query_idx, scenario, k, tol_r, tol_t):
    """Compute the k best roll-invariant matches for each query curve."""
    curves = df[DPA_COLS].to_numpy(dtype=np.float32)
    a = df["a"].to_numpy()
    incl = df["i"].to_numpy()
    theta = df["theta"].to_numpy()
    r = df["r"].to_numpy()
    period = df["Period"].to_numpy()
    n = curves.shape[0]

    # Pre-roll the whole database once: rolled[s, j] == np.roll(curves[j], s).
    # Comparing query curve i against rolled[s, j] for all s gives
    # min_s mean_k |c_i[k] - roll(c_j, s)[k]|, i.e. the optimal phase alignment.
    rolled = build_rolled(curves)

    rows = []
    for qi in query_idx:
        cq = curves[qi]

        if scenario == "full_obs":
            cand = np.where(
                (np.abs(r - r[qi]) <= tol_r) & (np.abs(period - period[qi]) <= tol_t)
            )[0]
        else:  # dpa_only
            cand = np.arange(n)

        cand = cand[cand != qi]
        if cand.size == 0:
            continue

        # diff: (N_PHASE_rolls, n_cand, N_PHASE) -> mean over phase -> (rolls, n_cand)
        diff = np.abs(rolled[:, cand, :] - cq).mean(axis=2)
        best_d = diff.min(axis=0)          # deg/sample, optimal roll per candidate
        best_roll = diff.argmin(axis=0)

        order = np.argsort(best_d)[:k]
        for o in order:
            j = cand[o]
            rows.append(
                {
                    "query_idx": int(qi),
                    "match_idx": int(j),
                    "dpa_l1": float(best_d[o]),
                    "roll": int(best_roll[o]),
                    "param_dist": float(
                        param_distance(a[qi], incl[qi], theta[qi], a[j], incl[j], theta[j])
                    ),
                    "dr": float(r[qi] - r[j]),
                    "dT": float(period[qi] - period[j]),
                    # raw params, query side
                    "a_q": float(a[qi]), "i_q": float(incl[qi]), "theta_q": float(theta[qi]),
                    "r_q": float(r[qi]), "T_q": float(period[qi]),
                    # raw params, match side
                    "a_m": float(a[j]), "i_m": float(incl[j]), "theta_m": float(theta[j]),
                    "r_m": float(r[j]), "T_m": float(period[j]),
                }
            )
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=["dpa_only", "full_obs"], required=True)
    p.add_argument("--chunk", type=int, default=0, help="0-based chunk index")
    p.add_argument("--n-chunks", type=int, default=1)
    p.add_argument("--k", type=int, default=10, help="matches kept per query curve")
    p.add_argument("--tol-r", type=float, default=0.3, help="full_obs: |dr| tolerance (M)")
    p.add_argument("--tol-t", type=float, default=3.0, help="full_obs: |dT| tolerance (min)")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--local", action="store_true", help="run all chunks in this process")
    args = p.parse_args()

    df = pd.read_csv(args.dataset)
    n = len(df)
    all_chunks = np.array_split(np.arange(n), args.n_chunks)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    chunks = range(args.n_chunks) if args.local else [args.chunk]
    for c in chunks:
        t0 = time.time()
        query_idx = all_chunks[c]
        out = find_for_chunk(df, query_idx, args.scenario, args.k, args.tol_r, args.tol_t)
        out_path = args.out_dir / f"{args.scenario}_chunk{c:03d}.csv"
        out.to_csv(out_path, index=False)
        print(
            f"[{args.scenario}] chunk {c}/{args.n_chunks}: "
            f"{len(query_idx)} queries -> {len(out)} pairs, "
            f"{time.time()-t0:.1f}s -> {out_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
