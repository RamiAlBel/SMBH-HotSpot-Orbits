#!/bin/bash

#SBATCH --job-name=exp5_neq45_bhex
#SBATCH --output=/scratch/ralbe/meniar_and_django/smbh_hotspots_repository/results/logs/exp5_neq45_bhex_%j.out
#SBATCH --error=/scratch/ralbe/meniar_and_django/smbh_hotspots_repository/results/logs/exp5_neq45_bhex_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=titans
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

source /scratch/ralbe/miniconda3/etc/profile.d/conda.sh
conda activate meniar

cd /scratch/ralbe/meniar_and_django/smbh_hotspots_repository/experiments/experiment_5_noneq_partial

echo "===== Experiment 5: Non-Equatorial Partial Orbit (neq45) - BHEX ====="
echo "Start time: $(date)"
echo ""

python train.py config_bhex.yaml

echo ""
echo "End time: $(date)"
echo "===== Experiment 5 (bhex) complete ====="
