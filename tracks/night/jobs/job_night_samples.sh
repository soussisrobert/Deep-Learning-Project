#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --job-name=HiFT_night_samples
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# Phase 2: sample crops + augmentation statistics with the run C settings (all options on). CPU only.
set -e
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}
python -I tools/night_samples.py experiments/night_ft/config.yaml experiments/night_ft/run_C.yaml
