#!/bin/bash

#SBATCH --job-name=exp3_mc
#SBATCH --output=/scratch/ralbe/meniar_and_django/smbh_hotspots_repository/results/logs/exp3_mc_%j.out
#SBATCH --error=/scratch/ralbe/meniar_and_django/smbh_hotspots_repository/results/logs/exp3_mc_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=titans
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

source /scratch/ralbe/miniconda3/etc/profile.d/conda.sh
conda activate meniar

cd /scratch/ralbe/meniar_and_django/smbh_hotspots_repository/experiments/experiment_3_eq_partial

echo "===== Experiment 3 MC: partial-orbit checkpoints for uncertainty analysis ====="
echo "Start time: $(date)"
echo ""

python train.py --mc

echo ""
echo "End time: $(date)"
echo "===== Experiment 3 MC complete ====="
