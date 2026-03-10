#!/bin/bash

# Submit all experiments WITH noise (Exp I–V)

set -euo pipefail

# Always run from the directory containing this script
cd "$(dirname "$0")"

echo "Submitting noisy experiments (I–V)..."

sbatch experiment_1_eq_avg/submit.sh
sbatch experiment_2_eq_full/submit.sh
sbatch experiment_3_eq_half/submit.sh
sbatch experiment_4_noneq_full/submit.sh
sbatch experiment_5_noneq_half/submit.sh

echo "Done submitting noisy experiments."

