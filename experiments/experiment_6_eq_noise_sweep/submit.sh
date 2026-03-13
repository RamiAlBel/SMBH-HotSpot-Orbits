#!/bin/bash
#SBATCH --job-name=exp6_noise_sweep
#SBATCH --output=/scratch/ralbe/meniar_and_django/smbh_hotspots_repository/results/logs/exp6_noise_sweep_%A_%a.out
#SBATCH --error=/scratch/ralbe/meniar_and_django/smbh_hotspots_repository/results/logs/exp6_noise_sweep_%A_%a.err
#SBATCH --array=0-124%32
#SBATCH --time=02:00:00
#SBATCH --partition=titans
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

source /scratch/ralbe/miniconda3/etc/profile.d/conda.sh
conda activate meniar

cd /scratch/ralbe/meniar_and_django/smbh_hotspots_repository/experiments/experiment_6_eq_noise_sweep

echo "===== Experiment 6: Noise Sweep (array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}) ====="
echo "Start time: $(date)"

python train.py $SLURM_ARRAY_TASK_ID

echo "End time: $(date)"
echo "===== Task ${SLURM_ARRAY_TASK_ID} complete ====="
