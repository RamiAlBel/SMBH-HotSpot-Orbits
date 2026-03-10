#!/bin/bash

# Submit all experiments, with and without noise

set -euo pipefail

cd "$(dirname "$0")"

echo "Submitting ALL experiments (noise + no-noise)..."

# With noise
sbatch experiment_1_eq_avg/submit.sh
sbatch experiment_2_eq_full/submit.sh
sbatch experiment_3_eq_half/submit.sh
sbatch experiment_4_noneq_full/submit.sh
sbatch experiment_5_noneq_half/submit.sh

# Without noise
sbatch experiment_1_eq_avg/submit_no_noise.sh
sbatch experiment_2_eq_full/submit_no_noise.sh
sbatch experiment_3_eq_half/submit_no_noise.sh
sbatch experiment_4_noneq_full/submit_no_noise.sh
sbatch experiment_5_noneq_half/submit_no_noise.sh

echo "Done submitting all experiments."

