#!/bin/bash

# Submit all main experiments (Exp I–V) WITHOUT noise.
# Non-equatorial experiments default to the neq45 dataset.

set -euo pipefail

cd "$(dirname "$0")"

echo "Submitting no-noise experiments (I–V)..."

sbatch experiment_1_eq_avg/submit_no_noise.sh
sbatch experiment_2_eq_full/submit_no_noise.sh
sbatch experiment_3_eq_partial/submit_no_noise.sh
sbatch experiment_4_noneq_full/submit_neq45_no_noise.sh
sbatch experiment_5_noneq_partial/submit_neq45_no_noise.sh

echo "Done submitting no-noise experiments."
