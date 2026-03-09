import numpy as np
# import matplotlib.pyplot as plt
#import ehtim as eh
import pandas as pd
import time
import tqdm
import os
from glob import glob
df= ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#5d21d0", "#ff9408", "#dc0ab4", "#b3d4ff", 
     "#00bfa0","#b30000", "#7c1158", "#4421af", "#1a53ff", "#0d88e6", "#00b7c7", "#5ad45a", "#8be04e",
     "#ebdc78","#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", 
     "#8bd3c7","#54bebe", "#76c8c8", "#98d1d1", "#badbdb", "#dedad2", "#e4bcad", "#df979e", "#d7658b", 
     "#c80064"] #dutch field


# Create a list to hold all data rows
data_rows = []

files = sorted(glob('/scratch/ralbe/meniar_and_django/smbh_hotspots_repository/data/raw/dataset_dense/*.dat'))
for file in tqdm.tqdm(files, total=len(files)):
    try:
        with open(file, 'r') as f:
            lines = f.readlines()
        
        # Line 2 has period info, line 3 has the two period values
        period_line = lines[2].strip().split()
        period_min = float(period_line[1])
        
        # Extract parameters from filename: lc_r10_K100_a-50_i0.dat
        parts = file.split('/')[-1].split('.')[0].split('_')
        r = float(parts[1][1:])
        K = float(parts[2][1:]) / 100
        a = float(parts[3][1:]) / 100
        i = float(parts[4][1:])
        
        # Read data starting from line 3 (index 3)
        data_lines = lines[3:]
        
        for line in data_lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            cols = line.split()
            if len(cols) < 13:
                continue
            
            try:
                perid_fraq = float(cols[1])  # Column 1 is Perid_fraq
                dpa_val = float(cols[12])    # Column 12 is DPA
                
                data_rows.append({
                    'r': r,
                    'K': K,
                    'a': a,
                    'i': i,
                    'DPA': dpa_val,
                    'Perid_fraq': perid_fraq,
                    'Period': period_min
                })
            except (ValueError, IndexError):
                continue
                
    except Exception as e:
        print(f"Error processing file {file}: {str(e)}")
        continue

# Create a pandas DataFrame
import pandas as pd
df_dpa = pd.DataFrame(data_rows)

# Display dataset info
print(f"\nDataset Information:")
print(f"Total number of rows: {len(df_dpa)}")
if len(df_dpa) > 0:
    print("\nFirst few rows:")
    print(df_dpa.head())
    print("\nColumn statistics:")
    print(df_dpa.describe())
else:
    print("\nWarning: No data was extracted!") 
output_path = '../../data/processed/dpa_dataset_ultradense.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_dpa.to_csv(output_path, index=False)
print(f"\nDataset saved to '{output_path}'")
