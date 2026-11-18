#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=HiFT_overlap
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# Overlap of candidate training sequences with NAT2021 / NAT2021L (thumbnail correlation + high-pass check). CPU only.
set -e
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}
python -I tools/overlap_check.py 0.95 0.5 \
    DarkTrack2021=training_dataset/DarkTrack2021/extracted/data_seq \
    UAV123=training_dataset/UAV123/extracted/UAV123/data_seq/UAV123
