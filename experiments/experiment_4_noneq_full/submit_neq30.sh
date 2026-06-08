#!/bin/bash

#SBATCH --job-name=exp4_neq30
#SBATCH --output=/scratch/ralbe/meniar_and_django/smbh_hotspots_repository/results/logs/exp4_neq30_%j.out
#SBATCH --error=/scratch/ralbe/meniar_and_django/smbh_hotspots_repository/results/logs/exp4_neq30_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=titans
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

source /scratch/ralbe/miniconda3/etc/profile.d/conda.sh
conda init
conda activate meniar

cd /scratch/ralbe/meniar_and_django/smbh_hotspots_repository/experiments/experiment_4_noneq_full

echo "===== Experiment 4: Non-Equatorial Full Orbit (neq30) ====="
echo "Start time: $(date)"
echo ""

python train.py config_neq30.yaml

echo ""
echo "End time: $(date)"
echo "===== Experiment 4 neq30 complete ====="
