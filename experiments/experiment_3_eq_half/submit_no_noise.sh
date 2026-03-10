#!/bin/bash

#SBATCH --job-name=exp3_eq_half_no_noise
#SBATCH --output=/scratch/ralbe/meniar_and_django/smbh_hotspots_repository/results/logs/exp3_eq_half_no_noise_%j.out
#SBATCH --error=/scratch/ralbe/meniar_and_django/smbh_hotspots_repository/results/logs/exp3_eq_half_no_noise_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=titans
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

source /scratch/ralbe/miniconda3/etc/profile.d/conda.sh
conda activate meniar

cd /scratch/ralbe/meniar_and_django/smbh_hotspots_repository/experiments/experiment_3_eq_half

echo "===== Experiment 3: Equatorial Half Orbit DPA(t) - NO NOISE ====="
echo "Start time: $(date)"
echo ""

python train.py config_no_noise.yaml

echo ""
echo "End time: $(date)"
echo "===== Experiment 3 (no noise) complete ====="
