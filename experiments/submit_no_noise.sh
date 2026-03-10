#!/bin/bash

# Submit all experiments WITHOUT noise (Exp II–V)

set -euo pipefail

# Always run from the directory containing this script
cd "$(dirname "$0")"

echo "Submitting no-noise experiments (II–V)..."

sbatch experiment_2_eq_full/submit_no_noise.sh
sbatch experiment_3_eq_half/submit_no_noise.sh
sbatch experiment_4_noneq_full/submit_no_noise.sh
sbatch experiment_5_noneq_half/submit_no_noise.sh

echo "Done submitting no-noise experiments."

