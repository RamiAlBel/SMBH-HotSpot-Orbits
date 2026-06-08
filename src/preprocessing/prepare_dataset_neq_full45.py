"""Merge dataset_noneq (inner disk, |θ|≤30°) + dataset_noneq45 (outer ring,
30°<|θ|≤45°) into a single CSV covering the full θ ∈ [-45°, +45°] range."""
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

REPO_ROOT = Path("/scratch/ralbe/meniar_and_django/smbh_hotspots_repository")
SOURCES = [
    REPO_ROOT / "data/raw/dataset_noneq",
    REPO_ROOT / "data/raw/dataset_noneq45",
]
OUT_PATH = REPO_ROOT / "data/processed/dpa_dataset_neq_45.csv"

files = []
for src in SOURCES:
    found = sorted(src.rglob("lc_*.dat"))
    print(f"  {src.name}: {len(found)} files")
    files.extend(found)
print(f"Total files: {len(files)}")

sample_points = np.linspace(0.1, 1.0, 10)
orbit_data = []
skipped_count = 0

for path in tqdm.tqdm(files, total=len(files), desc="Processing"):
    text = path.read_text()
    lines = text.splitlines()
    period_min = float(lines[2].split()[1])

    arr = np.loadtxt(io.StringIO("\n".join(lines[3:])))

    parts = path.stem.split("_")
    r_raw = float(parts[1][1:])
    K = float(parts[2][1:]) / 100.0
    a = float(parts[3][1:]) / 100.0
    i_raw = float(parts[4][1:])
    th_raw = float(parts[5][2:])

    r = r_raw / 10.0
    i = i_raw / 10.0
    theta = 90.0 - th_raw

    perid_fraq = arr[:, 1]
    dPA = arr[:, -2]

    if len(perid_fraq) < 10:
        skipped_count += 1
        continue

    phases_sorted_idx = np.argsort(perid_fraq)
    phases_sorted = perid_fraq[phases_sorted_idx]
    dpa_sorted = dPA[phases_sorted_idx]

    dpa_interpolated = np.interp(
        sample_points, phases_sorted, dpa_sorted,
        left=dpa_sorted[0], right=dpa_sorted[-1]
    )

    if np.isnan(dpa_interpolated).any() or np.isinf(dpa_interpolated).any():
        skipped_count += 1
        continue

    orbit_row = {
        'r': r, 'K': K, 'a': a, 'i': i, 'theta': theta, 'Period': period_min,
    }

    if any(np.isnan(v) or np.isinf(v) for v in [r, K, a, i, theta, period_min]):
        skipped_count += 1
        continue

    for idx, phase_val in enumerate(sample_points):
        orbit_row[f'DPA_{phase_val:.1f}'] = dpa_interpolated[idx]

    orbit_data.append(orbit_row)

df = pd.DataFrame(orbit_data)
print(f"\nProcessed {len(files)} files; skipped {skipped_count}; kept {len(df)} orbits")
print(f"theta range: [{df['theta'].min():.0f}, {df['theta'].max():.0f}], "
      f"{df['theta'].nunique()} unique values")
print(f"NaN/Inf check: {df.isna().sum().sum()} NaN, "
      f"{np.isinf(df.select_dtypes(include=[np.number])).sum().sum()} Inf")

os.makedirs(OUT_PATH.parent, exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved {OUT_PATH} ({len(df)} rows)")
