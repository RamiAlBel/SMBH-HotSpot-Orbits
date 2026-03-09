import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

BASE = Path("/scratch/ralbe/meniar_and_django/smbh_hotspots_repository/data/raw/dataset_noneq")
files = sorted(BASE.rglob("*.dat"))

print(f"Processing {len(files)} files...")

orbit_data = []
sample_points = np.linspace(0.1, 1.0, 10)
skipped_count = 0

for path in tqdm.tqdm(files, total=len(files), desc="Processing files"):
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
    
    # Check for NaN/Inf in interpolated values
    if np.isnan(dpa_interpolated).any() or np.isinf(dpa_interpolated).any():
        skipped_count += 1
        continue

    orbit_row = {
        'r': r,
        'K': K,
        'a': a,
        'i': i,
        'theta': theta,
        'Period': period_min,
    }
    
    # Check orbit parameters for NaN/Inf
    if any(np.isnan(v) or np.isinf(v) for v in [r, K, a, i, theta, period_min]):
        skipped_count += 1
        continue
    
    for idx, phase_val in enumerate(sample_points):
        orbit_row[f'DPA_{phase_val:.1f}'] = dpa_interpolated[idx]
    
    orbit_data.append(orbit_row)

df_orbits = pd.DataFrame(orbit_data)

print(f"\nProcessed {len(files)} files")
print(f"Skipped {skipped_count} files (too few samples or NaN/Inf values)")
print(f"Created dataset with {len(df_orbits)} valid orbits")
print(f"Columns: {list(df_orbits.columns)}")
print("\nFirst few rows:")
print(df_orbits.head())

# Final check for any remaining NaN/Inf
nan_check = df_orbits.isna().sum().sum()
inf_check = np.isinf(df_orbits.select_dtypes(include=[np.number])).sum().sum()
print(f"\nFinal validation: {nan_check} NaN values, {inf_check} Inf values in dataframe")

out_path = '../../data/processed/dpa_dataset_noneq.csv'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df_orbits.to_csv(out_path, index=False)
print(f"\nDataset saved to '{out_path}'")
