#!/bin/bash

#SBATCH --job-name=exp1_eq_avg_bhex
#SBATCH --output=/scratch/ralbe/meniar_and_django/smbh_hotspots_repository/results/logs/exp1_eq_avg_bhex_%j.out
#SBATCH --error=/scratch/ralbe/meniar_and_django/smbh_hotspots_repository/results/logs/exp1_eq_avg_bhex_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=titans
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

source /scratch/ralbe/miniconda3/etc/profile.d/conda.sh
conda activate meniar

cd /scratch/ralbe/meniar_and_django/smbh_hotspots_repository/experiments/experiment_1_eq_avg

echo "===== Experiment 1: Equatorial Averaged DPA - BHEX ====="
echo "Start time: $(date)"
echo ""

python train.py config_bhex.yaml

echo ""
echo "End time: $(date)"
echo "===== Experiment 1 (bhex) complete ====="
