#!/bin/bash

# Submit all main experiments (Exp I–V) WITH noise.
# Non-equatorial experiments default to the neq45 dataset.

set -euo pipefail

cd "$(dirname "$0")"

echo "Submitting noisy experiments (I–V)..."

sbatch experiment_1_eq_avg/submit.sh
sbatch experiment_2_eq_full/submit.sh
sbatch experiment_3_eq_partial/submit.sh
sbatch experiment_4_noneq_full/submit_neq45.sh
sbatch experiment_5_noneq_partial/submit_neq45.sh

echo "Done submitting noisy experiments."
