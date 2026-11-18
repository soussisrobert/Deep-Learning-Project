#!/bin/bash
#SBATCH --partition=<your-partition>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --job-name=HiFT_E19
#SBATCH --output=output/output_%x_%j.out
#SBATCH --error=error/error_%x_%j.err

# E19: first.pth none vs gamma 0.6 vs anchor+gamma, paired bootstraps from existing per-sequence CSVs (CPU only, no tracking).
set -e
module load anaconda
eval "$(conda shell.bash hook)"
conda activate HiFT
cd ${HIFT_ROOT}/tools
python -I illumination/E19_settings_bootstrap.py
