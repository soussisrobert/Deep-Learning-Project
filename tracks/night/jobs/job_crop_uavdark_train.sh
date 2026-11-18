#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --cpus-per-task=5
#SBATCH --time=04:00:00
#SBATCH --job-name=HiFT_crop_uavdark_train
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# Phase 1: crop511 + json for uavdark train into training_dataset/night_ft/uavdark135 (estimated 4700 MB). CPU only. Limit 105000 MB for home.
set -e
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}
python -I tools/crop_dataset.py uavdark train training_dataset/night_ft/uavdark135 105000 4700 5
