#!/bin/bash

#SBATCH --job-name=exp5_noneq_half
#SBATCH --output=/scratch/ralbe/meniar_and_django/smbh_hotspots_repository/results/logs/exp5_noneq_half_%j.out
#SBATCH --error=/scratch/ralbe/meniar_and_django/smbh_hotspots_repository/results/logs/exp5_noneq_half_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=titans
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

source /scratch/ralbe/miniconda3/etc/profile.d/conda.sh
conda activate meniar

cd /scratch/ralbe/meniar_and_django/smbh_hotspots_repository/experiments/experiment_5_noneq_half

echo "===== Experiment 5: Non-Equatorial Half Orbit with Sweep ====="
echo "Start time: $(date)"
echo ""

python train.py

echo ""
echo "End time: $(date)"
echo "===== Experiment 5 complete ====="
