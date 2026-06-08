#!/bin/bash

# Submit all main experiments (Exp I–V), with and without noise.
# Non-equatorial experiments default to the neq45 dataset (theta in [-45,45]).
# To submit the legacy neq30 variants, run the corresponding submit_neq30*.sh manually.

set -euo pipefail

cd "$(dirname "$0")"

echo "Submitting ALL main experiments (noise + no-noise)..."

# With noise
sbatch experiment_1_eq_avg/submit.sh
sbatch experiment_2_eq_full/submit.sh
sbatch experiment_3_eq_partial/submit.sh
sbatch experiment_4_noneq_full/submit_neq45.sh
sbatch experiment_5_noneq_partial/submit_neq45.sh

# Without noise
sbatch experiment_1_eq_avg/submit_no_noise.sh
sbatch experiment_2_eq_full/submit_no_noise.sh
sbatch experiment_3_eq_partial/submit_no_noise.sh
sbatch experiment_4_noneq_full/submit_neq45_no_noise.sh
sbatch experiment_5_noneq_partial/submit_neq45_no_noise.sh

echo "Done submitting all main experiments."
