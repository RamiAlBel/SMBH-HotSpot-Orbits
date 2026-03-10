import io
import os
from glob import glob

import numpy as np
import pandas as pd
import tqdm


files = sorted(
    glob('/scratch/ralbe/meniar_and_django/smbh_hotspots_repository/data/raw/dataset_dense/*.dat')
)

print(f"Processing {len(files)} files for ultradense equatorial dataset...")

orbit_rows = []
sample_points = np.linspace(0.1, 1.0, 10)
skipped_count = 0

for file in tqdm.tqdm(files, total=len(files), desc="Processing ultradense files"):
    try:
        with open(file, 'r') as f:
            lines = f.readlines()

        # Line 2 has period info, line 3 has the two period values
        period_line = lines[2].strip().split()
        period_min = float(period_line[1])

        # Extract parameters from filename: lc_r10_K100_a-50_i0.dat
        parts = file.split('/')[-1].split('.')[0].split('_')
        r = float(parts[1][1:])
        K = float(parts[2][1:]) / 100.0
        a = float(parts[3][1:]) / 100.0
        i = float(parts[4][1:])

        # Load numeric data starting from line 3
        arr = np.loadtxt(io.StringIO("\n".join(lines[3:])))

        if arr.shape[0] < 10:
            skipped_count += 1
            continue

        perid_fraq = arr[:, 1]
        dpa = arr[:, -2]

        # Sort by phase
        idx = np.argsort(perid_fraq)
        phases_sorted = perid_fraq[idx]
        dpa_sorted = dpa[idx]

        # Interpolate DPA at fixed sample points 0.1 ... 1.0
        dpa_interp = np.interp(
            sample_points, phases_sorted, dpa_sorted,
            left=dpa_sorted[0], right=dpa_sorted[-1]
        )

        # Basic NaN/Inf checks
        if np.isnan(dpa_interp).any() or np.isinf(dpa_interp).any():
            skipped_count += 1
            continue

        orbit = {
            "r": r,
            "K": K,
            "a": a,
            "i": i,
            "Period": period_min,
        }
        for idx_sp, sp in enumerate(sample_points):
            orbit[f"DPA_{sp:.1f}"] = dpa_interp[idx_sp]

        orbit_rows.append(orbit)

    except Exception as e:
        print(f"Error processing file {file}: {str(e)}")
        skipped_count += 1
        continue

df_orbits = pd.DataFrame(orbit_rows)

print(f"\nProcessed {len(files)} files")
print(f"Skipped {skipped_count} files (too few samples or invalid data)")
print(f"Created dataset with {len(df_orbits)} orbits")
print(f"Columns: {list(df_orbits.columns)}")
print("\nFirst few rows:")
print(df_orbits.head())

nan_count = df_orbits.isna().sum().sum()
inf_count = np.isinf(df_orbits.select_dtypes(include=[np.number])).sum().sum()
print(f"\nFinal validation: {nan_count} NaN values, {inf_count} Inf values in dataframe")

output_path = '../../data/processed/dpa_dataset_ultradense.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_orbits.to_csv(output_path, index=False)
print(f"\nDataset saved to '{output_path}'")
