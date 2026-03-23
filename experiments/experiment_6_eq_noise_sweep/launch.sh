#!/bin/bash
# Compute number of sweep combos from config, then submit the array job.
# Usage: bash launch.sh [--concurrency N]   (default: run all at once)

CONCURRENCY=${1:-0}  # 0 means no throttle

cd "$(dirname "$0")"

N=$(python3 -c "
import yaml
from itertools import product
cfg = yaml.safe_load(open('config.yaml'))
s = cfg['sweep']
print(len(list(product(s['sigma_T_values'], s['sigma_r_values'], s['sigma_DPA_values']))))
")

ARRAY_SPEC="0-$((N-1))"
if [ "$CONCURRENCY" -gt 0 ] 2>/dev/null; then
    ARRAY_SPEC="${ARRAY_SPEC}%${CONCURRENCY}"
fi

echo "Submitting exp6 array: tasks ${ARRAY_SPEC} (${N} combos)"
sbatch --array="${ARRAY_SPEC}" submit.sh
