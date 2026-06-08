#!/bin/bash
#SBATCH --job-name=dpa_degen
#SBATCH --output=/scratch/ralbe/meniar_and_django/smbh_hotspots_repository/results/logs/dpa_degen_%A_%a.out
#SBATCH --error=/scratch/ralbe/meniar_and_django/smbh_hotspots_repository/results/logs/dpa_degen_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=titans
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --array=0-19

# DPA degeneracy search as a SLURM array job.
#   each array task = one chunk of query curves, runs BOTH scenarios.
#   chunk index = SLURM_ARRAY_TASK_ID, n_chunks = array size (20 here).
# After the array finishes, build the figures with:
#   python aggregate_and_plot.py --scenario dpa_only
#   python aggregate_and_plot.py --scenario full_obs

N_CHUNKS=20

source /scratch/ralbe/miniconda3/etc/profile.d/conda.sh
conda activate meniar

cd /scratch/ralbe/meniar_and_django/smbh_hotspots_repository/analysis/dpa_degeneracy

echo "===== DPA degeneracy: chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS} ====="
echo "Start: $(date)"

for SCENARIO in dpa_only full_obs; do
    python find_degenerate_pairs.py \
        --scenario "$SCENARIO" \
        --chunk "$SLURM_ARRAY_TASK_ID" \
        --n-chunks "$N_CHUNKS"
done

echo "End: $(date)"
echo "===== chunk ${SLURM_ARRAY_TASK_ID} complete ====="
