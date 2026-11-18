#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --cpus-per-task=5
#SBATCH --time=04:00:00
#SBATCH --job-name=HiFT_crop_uav123_train
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# Phase 1: crop511 + json for uav123 train into training_dataset/night_ft/uav123 (estimated 5300 MB). CPU only. Limit 105000 MB for home.
set -e
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}
python -I tools/crop_dataset.py uav123 train training_dataset/night_ft/uav123 105000 5300 5
